"""H16 FFN with GPU GeLU/multiply and output-sharded Down weights.

Gate/Up FP32 outputs return to GPU. Optional rank staging first stores all
GPU X/Z in bank0 DRAM; subsequent local reads fill the common GEMV slots.
The direct WRGB path remains available only for baseline reproduction.
"""
from collections import Counter
from pathlib import Path
from rank_staging import stage_input, input_offset, reservation

HORIZONS = (16,)
CASES = ("gate_up_gpu", "down_gpu_input", "combined_timing_chain")
CHANNELS, BANKS, BURST, ROW = 16, 16, 32, 2048
GU_BASE, DOWN_BASE, WEIGHT_END = 262144, 327680, 360448


def check_horizon(horizon):
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return horizon


def address(channel, bank, offset):
    if not (0 <= channel < 16 and 0 <= bank < 16 and offset >= 0):
        raise ValueError("invalid rank1 bank address")
    row, within = divmod(offset, ROW)
    column, byte = divmod(within, BURST)
    bg, ba = divmod(bank, 4)
    return (row << 20) | (ba << 18) | (bg << 16) | (1 << 15) | (column << 9) | (channel << 5) | byte


def check(value, limit, name):
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"invalid {name}")


def decode_address(value):
    return (value >> 5 & 15, value >> 15 & 1,
            (value >> 16 & 3)*4 + (value >> 18 & 3),
            (value >> 20)*ROW + (value >> 9 & 63)*BURST + (value & 31))


def gu_weight_location(projection, reduction, feature):
    """Gate/Up [input_fragment64][projection2][feature_slot16][lane16]."""
    if projection not in ("gate", "up"):
        raise ValueError("invalid Gate/Up projection")
    check(reduction, 1024, "GU reduction")
    check(feature, 4096, "GU feature")
    channel, local = divmod(feature, 256)
    feature_slot, bank = divmod(local, 16)
    fragment, lane = divmod(reduction, 16)
    column = (fragment*2 + (projection == "up"))*16 + feature_slot
    return address(channel, bank, GU_BASE + column*BURST + lane*2)


def decode_gu_weight(value):
    channel, rank, bank, offset = decode_address(value)
    if rank != 1 or not GU_BASE <= offset < DOWN_BASE or offset % 2:
        raise ValueError("not a BF16 Gate/Up weight")
    column, lane_byte = divmod(offset-GU_BASE, BURST)
    group, feature_slot = divmod(column, 16)
    fragment, projection = divmod(group, 2)
    return ("up" if projection else "gate", fragment*16+lane_byte//2,
            channel*256+feature_slot*16+bank)


def down_weight_location(reduction, output):
    """Output-sharded [input_fragment256][output_slot4][input_lane16]."""
    check(reduction, 4096, "Down reduction")
    check(output, 1024, "Down output")
    channel, local = divmod(output, 64)
    bank, output_slot = divmod(local, 4)
    fragment, lane = divmod(reduction, 16)
    return address(channel, bank, DOWN_BASE+(fragment*4+output_slot)*BURST+lane*2)


def decode_down_weight(value):
    channel, rank, bank, offset = decode_address(value)
    if rank != 1 or not DOWN_BASE <= offset < WEIGHT_END or offset % 2:
        raise ValueError("not a BF16 Down weight")
    column, lane_byte = divmod(offset-DOWN_BASE, BURST)
    fragment, output_slot = divmod(column, 4)
    return fragment*16+lane_byte//2, channel*64+bank*4+output_slot


def gpu_input_packet(operator, channel, fragment, token, rank_staging=True):
    if operator not in ("gate_up", "down"):
        raise ValueError("invalid GPU input operator")
    width = 1024 if operator == "gate_up" else 4096
    check(channel, CHANNELS, "channel")
    check(fragment, width//16, "input fragment")
    check(token, 16, "input token")
    return dict(source_tensor="X" if operator == "gate_up" else "Z_gpu",
                source_byte_offset=(token*width+fragment*16)*2, bytes=32,
                destination=address(channel, 0, input_offset(fragment, token, 16) if rank_staging else token*32),
                stage_destination=address(channel, 0, input_offset(fragment, token, 16)),
                gemv_destination=address(channel, 0, token*32), srf_row=token,
                storage_before_srf="bank0 DRAM" if rank_staging else None, slot=fragment%2,
                target_bank_mask=65535, token=token, feature_first=fragment*16)


def layout(horizon, rank_staging=False):
    h = check_horizon(horizon)
    return dict(horizon=h, token_microbatches=1, active_physical_rank=1,
        channels=16, active_ranks=16, banks_per_rank=16, hidden=1024, intermediate=4096,
        gu_output_features_per_rank=256, gu_output_features_per_bank=16,
        down_output_features_per_rank=64, down_output_features_per_bank=4,
        down_reduction_features_per_rank=4096, weight_copies=1, weight_bytes=24*1024**2,
        gu_base=GU_BASE, down_base=DOWN_BASE, layer_stride_bytes_per_bank=WEIGHT_END,
        gu_weight_bytes_per_bank=65536, down_weight_bytes_per_bank=32768,
        gu_weight_layout="[input_fragment64][Gate/Up2][feature_slot16][input_lane16] BF16",
        gu_feature_owner="channel=feature//256; bank=feature%16; feature_slot=(feature%256)//16",
        down_weight_layout="[input_fragment256][output_slot4][input_lane16] BF16",
        down_output_owner="channel=output//64; bank=(output%64)//4; output_slot=output%4",
        gpu_z_layout="[token16][intermediate4096] BF16",
        gpu_z_unique_bytes=h*4096*2, gpu_z_copies_sent=16,
        z_residency=("GPU producer; whole Z copied to rank1 bank0 DRAM staging" if rank_staging else "GPU; no PIM DRAM Z allocation"),
        input_staging=reservation() if rank_staging else None,
        bank_input_layout="[token16][feature_lane16] BF16 in shared bank GEMV slots",
        input_broadcast_extra_cycles=0,
        down_output_layout="rank r returns complete Y[:,64*r:64*(r+1)] FP32",
        down_rank_partial_reduction=False,
        buffers=dict(GU_accumulator_per_bank=128*h, Down_accumulator_per_bank=16*h,
            bank_input_double=64*h, separate_rank_input_bytes=0,
            bank_Z_column_staging=0, rank_full_Z=0, rank_full_GU=0))


class Writer:
    def __init__(self, path, horizon):
        self.horizon = horizon
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8", newline="\n")
        self.counts, self.segments = Counter(), []
        self.input_fragments, self.exports = [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, op, value, buffer=None):
        self.stream.write(f"{op} {value}"+("" if buffer is None else f" {buffer}")+"\n")
        self.counts[op] += 1
        self.line_no += 1

    def fence(self):
        marker = self.marker
        self.marker += 1
        self.emit("BARRIER", marker)
        return marker

    def phase(self, name, body, **metadata):
        before, first = self.counts.copy(), self.line_no+1
        body()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
            line_start=first, line_end=self.line_no, counts=dict(self.counts-before), **metadata))
        self.boundary = end

    def ranks(self, op, offset, buffer=None):
        for channel in range(16):
            self.emit(op, address(channel, 0, offset), buffer)

    def drain(self, cycles):
        self.fence()


def compute(w, operator, rank_staging=False):
    gu = operator == "gate_up"
    fragments, slots = (64, 32) if gu else (256, 4)
    base = GU_BASE if gu else DOWN_BASE
    name = operator+("_local_input_and_compute" if rank_staging else "_input_and_compute")
    if rank_staging:
        stage_input(w, operator, w.horizon, fragments)
    def body():
        for fragment in range(fragments):
            buffer, first = fragment%2, w.line_no+1
            if rank_staging:
                w.ranks("PIM_STAGE_FRAGMENT", input_offset(fragment, 0, w.horizon), buffer)
            else:
                for token in range(w.horizon):
                    w.ranks("PIM_WR_GB", token*32, buffer)
            last, mac_first = w.line_no, w.line_no+1
            for slot in range(slots):
                w.ranks("PIM_MAC_AB", base+(fragment*slots+slot)*32, buffer)
            w.input_fragments.append(dict(phase=name, fragment=fragment, buffer=buffer,
                input_source="bank0_dram_staging" if rank_staging else "gpu_wrgb", source_tensor="X" if gu else "Z_gpu",
                source_tensor_width=1024 if gu else 4096, source_feature_first=fragment*16,
                same_source_fragment_all_ranks=True, input_line_first=first, input_line_last=last,
                line_start=first, line_end=last, mac_line_first=mac_first, mac_line_last=w.line_no,
                bursts_per_rank=w.horizon, bytes_per_rank=w.horizon*32,
                useful_input_bytes_per_rank=w.horizon*32, command_count=16 if rank_staging else w.horizon*16,
                internal_load_bursts=w.horizon*16 if rank_staging else 0,
                consumer_mac_count=slots*16, slots_per_fragment=slots,
                input_broadcast_extra_cycles=0))
        w.drain(w.horizon)
    w.phase(name, body, kind="local_compute" if rank_staging else "pipeline", M=w.horizon,
        fragment_count=fragments, slots_per_fragment=slots, weight_base=base,
        input_source="bank0_dram_staging" if rank_staging else "gpu_wrgb", source_tensor="X" if gu else "Z_gpu")
    def release():
        for slot in (0, 1):
            w.ranks("PIM_GEMV_RELEASE", 0, slot)
    w.phase(name+"_gemv_release", release, kind="gemv_release")


def export(w, operator):
    gu = operator == "gate_up"
    outputs_per_rank = 512 if gu else 64
    name = operator+"_result_export"
    def body():
        first = w.line_no+1
        for offset in range(0, w.horizon*outputs_per_rank*4, 32):
            w.ranks("PIM_RD_RES", offset)
        w.exports.append(dict(phase=name, line_start=first, line_end=w.line_no,
            output_count_per_rank=outputs_per_rank, bytes_fp32=w.horizon*outputs_per_rank*4*16,
            rank_partials=0, destination="GPU", projection_count=2 if gu else 1,
            output_semantics="complete Gate and Up features" if gu else "complete output features"))
        w.drain(4)
    w.phase(name, body, kind="export", output_count_per_rank=outputs_per_rank,
            rank_partials=0, destination="GPU")


def generate(path, horizon, case="combined_timing_chain", local_ck=2, rank_staging=False):
    info = layout(horizon, rank_staging=rank_staging)
    if case not in CASES or type(local_ck) is not int or local_ck <= 0:
        raise ValueError("invalid FFN case or local transfer cycles")
    gu, down = case != "down_gpu_input", case != "gate_up_gpu"
    w = Writer(path, horizon)
    try:
        if gu:
            compute(w, "gate_up", rank_staging=rank_staging)
            export(w, "gate_up")
        if gu and down:
            w.phase("gpu_gelu_multiply_boundary", lambda: None, kind="excluded_boundary",
                measured_latency_us=None, operation="Z=BF16(GeLU(Gate)*Up) on GPU",
                policy="Input assumed ready after an unmeasured GPU operation; no kernel timing claim")
        if down:
            compute(w, "down", rank_staging=rank_staging)
            export(w, "down")
    finally:
        w.stream.close()
    dq = dict(GU_input_gpu_to_pim=gu*horizon*1024*2*16,
        GU_gate_up_pim_to_gpu=gu*horizon*4096*2*4,
        Down_Z_gpu_to_pim=down*horizon*4096*2*16,
        Down_output_pim_to_gpu=down*horizon*1024*4)
    dq["total"] = sum(dq.values())
    input_bytes = dq["GU_input_gpu_to_pim"]+dq["Down_Z_gpu_to_pim"]
    internal = dict(input_multicast_source_bytes=input_bytes,
        input_multicast_destination_bank_bytes=input_bytes*16,
        input_multicast_additional_service_cycles=0,
        input_staging_read=input_bytes if rank_staging else 0,
        input_staging_write=input_bytes if rank_staging else 0,
        Z_dram_read=dq["Down_Z_gpu_to_pim"] if rank_staging else 0,
        Z_dram_write=dq["Down_Z_gpu_to_pim"] if rank_staging else 0, Z_scatter_source=0)
    return dict(case=case, horizon=horizon, action_horizon=horizon,
        whole_batch=True, token_microbatches=1, padded_compute_tokens=horizon,
        layout=info, buffers=info["buffers"], counts=dict(w.counts), segments=w.segments,
        input_fragments=w.input_fragments, exports=w.exports,
        staging_inputs=getattr(w, "staging_inputs", []), rank_staging=rank_staging,
        expected_dq_bytes=dq, expected_internal_bytes=internal,
        weight_array_bytes=(gu*16+down*8)*1024**2,
        mac_service_ck=horizon, input_broadcast_extra_cycles=0,
        rank_local_z_sram_bytes=0, rank_bulk_GU_staging_bytes=0,
        required_backend=dict(pim_horizon=horizon, pim_ffn_dram=False, pim_rank_rope=False,
            pim_mac_scope="rank", pim_input_broadcast=False, nCCDAB=horizon,
            ffn_z_preinstalled=False),
        nonlinear_latency_ck=None, gpu_partial_reduction_required=False,
        timing_chain=gu and down, integrated_chain=False, numeric_compute=False,
        exclusions=["GPU GeLU/multiply/BF16 cast and producer availability",
                    "GPU source/arrival and norm/residual/conditioning arithmetic"],
        standalone_preconditions="X ready on GPU" if gu else "BF16 Z from GPU GeLU(Gate)*Up ready",
        padding_policy="No FFN padding; all 16 tokens execute in one MAC batch")

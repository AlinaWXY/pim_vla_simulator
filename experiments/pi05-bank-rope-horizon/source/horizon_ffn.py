"""Whole-horizon rank-feature FFN with streamed, bank-DRAM Z.

Weights are single-copy and unchanged. H changes token buffers and MAC
service, not head count or token microbatching. Only Z columns have padding.
"""
from collections import Counter
from pathlib import Path

HORIZONS = (16, 20, 30, 40)
CASES = ("gate_up_z_store", "down_tiled_dram", "combined_timing_chain")
CHANNELS, BANKS, BURST, ROW = 16, 16, 32, 2048
GU_BASE, DOWN_BASE, Z_BASE = 262144, 327680, 360448


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


def z_location(horizon, token, feature):
    check_horizon(horizon)
    if not (0 <= token < horizon and 0 <= feature < 4096):
        raise ValueError("invalid Z token or feature")
    blocks = (horizon+15)//16
    channel, local = divmod(feature, 256)
    slot, bank = divmod(local, 16)
    block, lane = divmod(token, 16)
    return address(channel, bank, Z_BASE+(slot*blocks+block)*32+lane*2)


def layout(horizon):
    h = check_horizon(horizon)
    blocks = (h+15)//16
    valid = [min(16, h-16*b) for b in range(blocks)]
    gather = [((8*n+31)//32)*32 for n in valid]
    return dict(horizon=h, token_microbatches=1, active_physical_rank=1,
        channels=16, active_ranks=16, banks_per_rank=16, hidden=1024, intermediate=4096,
        local_intermediate=256, weight_copies=1, weight_bytes=24*1024**2,
        gu_base=GU_BASE, down_base=DOWN_BASE, z_base=Z_BASE,
        gu_weight_bytes_per_bank=65536, down_weight_bytes_per_bank=32768,
        z_blocks_per_feature=blocks, z_columns_per_bank=16*blocks,
        z_valid_tokens_per_column=valid, z_gu_gather_port_bytes_per_column=gather,
        z_bytes_per_bank=512*blocks, z_valid_bytes_per_bank=32*h,
        z_padding_bytes_per_bank=512*blocks-32*h, z_reserved_bytes_per_bank=2048,
        layer_stride_bytes_per_bank=Z_BASE+2048,
        z_layout="[feature_slot16][token_block ceil(H/16)][token_lane16] BF16",
        z_column_formula="feature_slot*ceil(H/16)+token_block",
        rank_input_layout="[token H][feature_lane16] BF16; tail column padding discarded",
        buffers=dict(GU_accumulator_per_bank=128*h, Down_tiled_accumulator_per_bank=64*h,
            rank_input_double=64*h, bank_column_staging=32,
            rank_nonlinear_input_staging=128, rank_nonlinear_output_staging=32,
            rank_full_Z=0, rank_full_GU=0))


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

    def columns(self, op, feature, buffer=None):
        for block in range((self.horizon+15)//16):
            column = feature*((self.horizon+15)//16)+block
            for bank in range(16):
                for channel in range(16):
                    self.emit(op, address(channel, bank, Z_BASE+column*32), buffer)

    def drain(self, cycles):
        self.fence()
        self.emit("WAIT", cycles)


def compute(w, tile=None):
    gu = tile is None
    fragments, slots = (64, 32) if gu else (16, 16)
    base = GU_BASE if gu else DOWN_BASE+tile*8192
    name = "input_and_compute" if gu else f"tile_{tile}_input_and_compute"
    def body():
        for fragment in range(fragments):
            buffer, first = fragment%2, w.line_no+1
            if gu:
                for token in range(w.horizon):
                    w.ranks("PIM_WR_GB", token*32, buffer)
            else:
                w.columns("PIM_Z_LOAD", fragment, buffer)
            last, mac_first = w.line_no, w.line_no+1
            for slot in range(slots):
                w.ranks("PIM_MAC_AB", base+(fragment*slots+slot)*32, buffer)
            bursts = w.horizon if gu else 16*((w.horizon+15)//16)
            w.input_fragments.append(dict(phase=name, tile=tile, fragment=fragment, buffer=buffer,
                input_source="gpu_wrgb" if gu else "bank_dram_z",
                input_line_first=first, input_line_last=last, line_start=first, line_end=last,
                mac_line_first=mac_first, mac_line_last=w.line_no,
                bursts_per_rank=bursts, bytes_per_rank=bursts*32,
                useful_input_bytes_per_rank=w.horizon*32,
                command_count=bursts*16, consumer_mac_count=slots*16, slots_per_fragment=slots))
        w.drain(w.horizon)
    w.phase(name, body, kind="pipeline", tile=tile, M=w.horizon,
        fragment_count=fragments, slots_per_fragment=slots, weight_base=base,
        input_source="gpu_wrgb" if gu else "bank_dram_z")


def export(w, tile):
    name = f"tile_{tile}_result_export"
    def body():
        first = w.line_no+1
        for offset in range(0, w.horizon*256*4, 32):
            w.ranks("PIM_RD_RES", offset)
        w.exports.append(dict(phase=name, tile=tile, output_start=tile*256,
            output_count_per_rank=256, line_start=first, line_end=w.line_no,
            bytes_fp32=w.horizon*256*4*16, rank_partials=16))
        w.drain(4)
    w.phase(name, body, kind="export", tile=tile, output_start=tile*256,
        output_count_per_rank=256, rank_partials=16)


def generate(path, horizon, case="combined_timing_chain", local_ck=2):
    info = layout(horizon)
    if case not in CASES or type(local_ck) is not int or local_ck <= 0:
        raise ValueError("invalid FFN case or local transfer cycles")
    gu, down = case != "down_tiled_dram", case != "gate_up_z_store"
    w = Writer(path, horizon)
    try:
        if gu:
            compute(w)
            def stores():
                for feature in range(16):
                    w.columns("PIM_Z_STORE", feature)
            w.phase("z_stream_store", stores, kind="native_z_store",
                nonlinear_latency_ck=None, z_blocks_per_feature=info["z_blocks_per_feature"])
        if down:
            for tile in range(4):
                compute(w, tile)
                export(w, tile)
    finally:
        w.stream.close()
    dq = dict(pim_input_write=gu*horizon*32768, pim_result_read=down*horizon*65536,
        dram_read=0, dram_write=0)
    dq["total"] = sum(dq.values())
    physical_z = 256*info["z_bytes_per_bank"]
    valid_z = horizon*4096*2
    gu_gather = 4096*sum(info["z_gu_gather_port_bytes_per_column"])
    internal = dict(GU_gather_to_nonlinear=gu*gu_gather,
        GU_gather_valid=gu*horizon*4096*8, GU_gather_padding=gu*(gu_gather-horizon*4096*8),
        Z_return_to_producer_bank=gu*physical_z, Z_dram_write=gu*physical_z,
        Z_dram_read=down*4*physical_z, Z_transpose_to_rank_input=down*4*physical_z,
        Z_write_valid=gu*valid_z, Z_write_padding=gu*(physical_z-valid_z),
        Z_read_valid=down*4*valid_z, Z_read_padding=down*4*(physical_z-valid_z))
    return dict(case=case, horizon=horizon, action_horizon=horizon,
        whole_batch=True, token_microbatches=1, padded_compute_tokens=horizon,
        layout=info, buffers=info["buffers"], counts=dict(w.counts), segments=w.segments,
        input_fragments=w.input_fragments, exports=w.exports,
        expected_dq_bytes=dq, expected_internal_bytes=internal,
        weight_array_bytes=(gu*16+down*8)*1024**2,
        mac_service_ck=horizon, local_ck=local_ck,
        z_base=Z_BASE, z_bytes_per_bank=info["z_bytes_per_bank"],
        z_blocks_per_feature=info["z_blocks_per_feature"], z_reserved_bytes_per_bank=2048,
        rank_local_z_sram_bytes=0, rank_bulk_GU_staging_bytes=0,
        required_backend=dict(pim_horizon=horizon, pim_ffn_dram=True, pim_rank_rope=False,
            pim_mac_scope="rank", pim_input_broadcast=False, nCCDAB=horizon,
            ffn_z_preinstalled=not gu),
        nonlinear_latency_ck=None, gpu_partial_reduction_latency_ck=None,
        timing_chain=gu and down, integrated_chain=False, numeric_compute=False,
        exclusions=["GELU/multiply/BF16 cast and producer availability", "GPU final rank-partial reduction",
                    "GPU source/arrival and norm/residual/conditioning arithmetic"],
        padding_policy="Only Z feature columns and GU rank-port bursts pad; all H tokens execute in one MAC batch")

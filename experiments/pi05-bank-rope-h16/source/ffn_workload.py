"""GU and Down service traces using only physical PIM rank1/channel.

GELU/multiply/cast and the final GPU reduction remain explicitly unmeasured.
Down begins with rank-local BF16 Z ready; it is not a numerical GU->Down chain.
"""

from collections import Counter
from pathlib import Path

import ffn_layout as L

OPERATORS = ("gate_up", "down_tiled", "down_full")
MAC_SERVICE_CK, MOVEMENT_TAIL_CK = 16, 4
GATHER_BYTES_PER_RANK = 2 * L.HORIZON * L.LOCAL_INTERMEDIATE * 4
GATHER_BURSTS_PER_RANK = GATHER_BYTES_PER_RANK // L.BURST_BYTES
Z_BYTES_PER_RANK = L.HORIZON * L.LOCAL_INTERMEDIATE * L.BF16_BYTES


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.counts, self.segments = Counter(), []
        self.input_fragments, self.exports = [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, op, address, buffer=None):
        self.stream.write(f"{op} {address}" + ("" if buffer is None else f" {buffer}") + "\n")
        self.counts[op] += 1
        self.line_no += 1

    def fence(self):
        marker = self.marker
        self.marker += 1
        self.emit("BARRIER", marker)
        return marker

    def wait(self, cycles):
        if cycles:
            self.emit("WAIT", cycles)

    def phase(self, name, callback, **metadata):
        before, first = self.counts.copy(), self.line_no+1
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
            line_start=first, line_end=self.line_no, counts=dict(self.counts-before), **metadata))
        self.boundary = end

    def ranks(self, command, offset, buffer=None):
        for channel in range(L.CHANNELS):
            self.emit(command, L.physical_address(channel, L.PIM_RANK, 0, 0, offset), buffer)

    def drain_mac(self):
        # Legacy non-RoPE MACAB retires at issue. Preserve its explicit tail.
        self.fence()
        self.wait(MAC_SERVICE_CK)

    def drain_movement(self):
        self.fence()
        self.wait(MOVEMENT_TAIL_CK)

    def close(self):
        self.stream.close()


def matrix_for(case, attention_bytes_per_bank=L.ATTENTION_BYTES_PER_BANK):
    region = L.build_layout(attention_bytes_per_bank)["regions"][case]
    macs = L.HORIZON * region["K"] * region["N"] * region.get("reduction_groups", 1)
    return {**region, "slots": region["slots_per_bank"], "logical_macs": macs, "padded_macs": macs}


def emit_fragment(w, *, phase, fragment, slots, base, tile, source):
    buffer, first = fragment % 2, w.line_no+1
    command = "PIM_WR_GB" if source == "gpu_wrgb" else "PIM_LOCAL_GB"
    for query in range(L.HORIZON):
        w.ranks(command, query * L.BURST_BYTES, buffer)
    last, mac_first = w.line_no, w.line_no+1
    for slot in range(slots):
        w.ranks("PIM_MAC_AB", base+(fragment*slots+slot)*L.BURST_BYTES, buffer)
    w.input_fragments.append(dict(phase=phase, tile=tile, fragment=fragment, buffer=buffer,
        input_source=source, input_command=command, line_start=first, line_end=last,
        input_line_first=first, input_line_last=last,
        mac_line_first=mac_first, mac_line_last=w.line_no,
        command_count=L.HORIZON*L.TOTAL_RANKS,
        bursts_per_rank=L.HORIZON, bytes_per_rank=L.HORIZON*L.BURST_BYTES,
        consumer_mac_count=slots*L.TOTAL_RANKS, slots_per_fragment=slots,
        local_source_fragment=fragment if source == "rank_local_sram" else None))


def pipeline(w, *, name, base, fragments, slots, tile, source):
    def body():
        for fragment in range(fragments):
            emit_fragment(w, phase=name, fragment=fragment, slots=slots,
                          base=base, tile=tile, source=source)
        w.drain_mac()
    w.phase(name, body, kind="pipeline", tile=tile, fragment_count=fragments,
            slots_per_fragment=slots, input_source=source, M=L.HORIZON,
            weight_base=base, weight_bursts_per_channel=fragments*slots)


def export_results(w, *, name, slots, tile, output_start):
    def body():
        first = w.line_no+1
        for offset in range(slots * L.HORIZON * L.BANKS_PER_RANK * 4 // L.BURST_BYTES):
            w.ranks("PIM_RD_RES", offset * L.BURST_BYTES)
        w.exports.append(dict(phase=name, tile=tile, output_start=output_start,
            output_count_per_rank=slots*L.BANKS_PER_RANK, line_start=first, line_end=w.line_no,
            bytes_fp32=slots*L.HORIZON*L.BANKS_PER_RANK*4*L.TOTAL_RANKS,
            rank_partials=L.TOTAL_RANKS, accumulator_slots_reused_from_zero=True))
        w.drain_movement()
    w.phase(name, body, kind="export", tile=tile, slots=slots,
            output_start=output_start, output_count_per_rank=slots*L.BANKS_PER_RANK,
            output_source="rank_partial_accumulators", rank_partials=L.TOTAL_RANKS)


def generate(path, case="gate_up", local_ck=2,
             attention_bytes_per_bank=L.ATTENTION_BYTES_PER_BANK):
    if case not in OPERATORS:
        raise ValueError(case)
    if not isinstance(local_ck, int) or local_ck <= 0:
        raise ValueError("local_ck must be a positive memory-clock count")
    matrix = matrix_for(case, attention_bytes_per_bank)
    w = Writer(path)
    try:
        if case == "gate_up":
            pipeline(w, name="input_and_compute", base=matrix["base"], fragments=64,
                     slots=32, tile=None, source="gpu_wrgb")
            gather_ck = GATHER_BURSTS_PER_RANK * local_ck
            w.phase("local_gather", lambda: w.wait(gather_ck), kind="local_gather",
                    analytical_wait=True, assumed_service_ck=gather_ck,
                    wait_cycles=gather_ck, budget_ck=gather_ck,
                    bytes_per_rank=GATHER_BYTES_PER_RANK,
                    bytes_system=GATHER_BYTES_PER_RANK*L.TOTAL_RANKS,
                    parallel_ranks=L.TOTAL_RANKS, port_bytes=32, port_cycles=local_ck,
                    excluded_operations=["GELU", "multiply", "cast"])
        else:
            tiled = case == "down_tiled"
            tiles, slots = (4, 16) if tiled else (1, 64)
            for tile in range(tiles):
                prefix = f"tile_{tile}_" if tiled else ""
                base = matrix["base"] + tile * 16 * slots * L.BURST_BYTES
                pipeline(w, name=prefix+"input_and_compute", base=base, fragments=16,
                         slots=slots, tile=tile, source="rank_local_sram")
                export_results(w, name=prefix+"result_export", slots=slots,
                               tile=tile, output_start=tile*slots*L.BANKS_PER_RANK)
    finally:
        w.close()
    counts = dict(w.counts)
    dq = dict(pim_input_write=w.counts["PIM_WR_GB"]*L.BURST_BYTES,
              pim_result_read=w.counts["PIM_RD_RES"]*L.BURST_BYTES,
              dram_read=0, dram_write=0)
    dq["total"] = sum(dq.values())
    return dict(case=case, horizon=L.HORIZON, action_horizon=L.HORIZON,
        mac_service_ck=MAC_SERVICE_CK, mode="native_service", counts=counts,
        segments=w.segments, matrix=matrix, input_fragments=w.input_fragments, exports=w.exports,
        expected_dq_bytes=dq, physical_active_rank=L.PIM_RANK,
        local_ck=local_ck, pim_local_transfer_cycles=local_ck,
        local_input_command="PIM_LOCAL_GB", local_input_external_dq_bytes=0,
        local_input_rank_selected=True, global_wrgb_rank_broadcast=False,
        input_double_buffer_bytes_per_rank=2*L.HORIZON*L.BURST_BYTES,
        accumulator_bytes_per_bank=matrix["accumulator_bytes_per_bank"],
        local_input_bytes=w.counts["PIM_LOCAL_GB"]*L.BURST_BYTES,
        wrgb_input_dq_bytes=dq["pim_input_write"], result_dq_bytes=dq["pim_result_read"],
        assumed_local_gather_ck=GATHER_BURSTS_PER_RANK*local_ck if case == "gate_up" else None,
        local_gather_included=case == "gate_up",
        local_gather_model="Analytical WAIT for independent32B/local_ck gather port per rank; no nonlinear arithmetic included",
        nonlinear_latency_ck=None, nonlinear_status="unmeasured; GELU/multiply/cast excluded, not zero cost",
        rank_local_z_sram_bytes=Z_BYTES_PER_RANK,
        gpu_partial_count=L.TOTAL_RANKS if case != "gate_up" else 0,
        gpu_partial_reduction_latency_ck=None,
        attention_reserved_bytes_per_bank=attention_bytes_per_bank,
        required_backend=dict(pim_rank_rope=False, pim_input_broadcast=False,
                              pim_mac_scope="rank", nCCDAB=MAC_SERVICE_CK),
        output_policy=("GU operands remain local for nonlinear boundary" if case == "gate_up"
                       else "Sixteen FP32 rank partials exported; GPU reduction not simulated"),
        preconditions=("Weights resident; full16-token X available at each rank input boundary"
                       if case == "gate_up" else
                       "Weights resident;8KiB BF16 Z ready per rank after unmeasured GELU/multiply/cast"),
        exclusions=["GELU, elementwise multiply and cast", "GPU final partial reduction",
                    "GPU source/arrival DMA overhead", "GPU global-memory materialization/readback",
                    "Norm/residual and conditioning arithmetic"],
        integrated_chain=False, numeric_compute=False)

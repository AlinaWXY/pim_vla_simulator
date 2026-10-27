"""Native trace for token-sharded projection, local RoPE, Q forwarding and QK.

All requests target physical rank1.  New PIM_* local operations require
MemoryConfig(pim_rank_rope=True); they are never replaced by external ST/LD.
"""

from collections import Counter
from pathlib import Path

import attention_layout as L

OPERATORS = ("chain", "projection", "rope", "qk")
CONTROL_OPERATORS = ()
SCORE_TILE_TOKENS = 16


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.counts = Counter()
        self.segments, self.input_fragments, self.exports, self.local_writes = [], [], [], []
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

    def phase(self, name, callback, **metadata):
        before, first = self.counts.copy(), self.line_no + 1
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
                                  line_start=first, line_end=self.line_no,
                                  counts=dict(self.counts - before), **metadata))
        self.boundary = end

    def ranks(self, command, offset, buffer=None):
        for channel in range(L.CHANNELS):
            self.emit(command, L.physical_address(channel, L.PIM_RANK, 0, 0, offset), buffer)

    def banks(self, command, offset, buffer=None):
        for location in L.active_bank_locations(offset):
            self.emit(command, L.physical_address(*location), buffer)

    def close(self):
        self.stream.close()


def projection(w, schedule):
    def compute():
        for fragment in range(L.HIDDEN // L.LANES):
            buffer = fragment % 2 if schedule == "double" else 0
            first = w.line_no + 1
            for local_query in range(L.LOCAL_QUERIES):
                w.ranks("PIM_WR_GB", local_query * L.BURST_BYTES, buffer)
            last = w.line_no
            mac_first = w.line_no + 1
            for output_column in range(len(L.PROJECTIONS) * L.FEATURES_PER_BANK):
                offset = (fragment * len(L.PROJECTIONS) * L.FEATURES_PER_BANK
                          + output_column) * L.BURST_BYTES
                w.ranks("PIM_MAC_AB", offset, buffer)
            w.input_fragments.append(dict(fragment=fragment, buffer=buffer,
                line_start=first, line_end=last, mac_line_first=mac_first,
                mac_line_last=w.line_no, external_bytes=(last-first+1) * L.BURST_BYTES,
                scope="rank1 shared input, eight local tokens, Q/K reuse"))
            if schedule == "serial":
                w.fence()
    w.phase("projection_input_compute", compute, kind="projection", mac_service_ck=8,
            native_mac_completion=True)

    def spill():
        first = w.line_no + 1
        for local_query in range(L.LOCAL_QUERIES):
            for kind in ("q_local", "k_suffix"):
                w.banks("PIM_SPILL", L.work_offset(kind, local_query))
        w.local_writes.append(dict(phase="projection_local_spill", line_start=first,
                                   line_end=w.line_no, bytes=(w.line_no-first+1) * L.BURST_BYTES,
                                   source="FP32 accumulators rounded to BF16", external_bytes=0))
    w.phase("projection_local_spill", spill, kind="local_write")


def rope(w, schedule):
    def body():
        for local_query in range(L.LOCAL_QUERIES):
            qaddr = L.work_offset("q_local", local_query)
            kaddr = L.work_offset("k_suffix", local_query)
            w.banks("PIM_ROPE_COEF", L.work_offset("coef", local_query))
            w.banks("PIM_ROPE_LOAD", qaddr, 0)
            w.banks("PIM_ROPE_EXEC", qaddr, 0)
            if schedule == "double":
                # K can enter the other input column while Q is in flight.
                w.banks("PIM_ROPE_LOAD", kaddr, 1)
                w.banks("PIM_ROPE_EXEC", kaddr, 1)
                w.banks("PIM_ROPE_STORE", qaddr)
                w.banks("PIM_ROPE_STORE", kaddr)
            else:
                w.banks("PIM_ROPE_STORE", qaddr)
                w.banks("PIM_ROPE_LOAD", kaddr, 0)
                w.banks("PIM_ROPE_EXEC", kaddr, 0)
                w.banks("PIM_ROPE_STORE", kaddr)
        # No per-token global barriers; native buffer/row/pipeline hazards apply.
    w.phase("rope_local", body, kind="rope", coefficient_dtype="BF16",
            pairs_per_exec=L.PAIRS_PER_BANK, executions_per_bank=2*L.LOCAL_QUERIES,
            theoretical_pair_issues_per_bank=2*L.LOCAL_QUERIES*L.PAIRS_PER_BANK,
            timing="II and latency are native configuration parameters, not calibrated silicon")


def q_exchange(w):
    def export():
        first = w.line_no + 1
        for local_query in range(L.LOCAL_QUERIES):
            w.banks("PIM_Q_EXPORT", L.work_offset("q_local", local_query))
        w.exports.append(dict(phase="q_export_to_gpu", line_start=first,
                              line_end=w.line_no, bytes=(w.line_no-first+1)*L.BURST_BYTES,
                              local_tap=True, source_retained=True))
    w.phase("q_export_to_gpu", export, kind="external_read",
            native_effect="RD plus retained local Q-buffer row")

    def remote():
        for local_query in range(L.LOCAL_QUERIES):
            # Target channel c receives the matching bank from c^1.
            # Address is a route/slot tag; this operation does not write DRAM.
            w.banks("PIM_Q_REMOTE", L.work_offset("q_local", local_query), 0)
    w.phase("q_remote_buffer_fill", remote, kind="external_write",
            destination="peer rank bank-local Q buffer, not remote DRAM",
            source_relation="source_address=destination_address xor32",
            source_ready="previous global completion barrier")


def qk(w):
    # All16 queries remain in their 512B bank buffer through this single K scan.
    # Output tiles bound rank score storage; they do not repeat Q or K reads.
    for tile, start in enumerate(range(0, L.LOCAL_TOKENS, SCORE_TILE_TOKENS)):
        stop = min(start + SCORE_TILE_TOKENS, L.LOCAL_TOKENS)

        def compute(start=start, stop=stop):
            for token in range(start, stop):
                w.ranks("PIM_MAC_QK", L.qk_column_offset(token), 0)
        w.phase(f"qk_tile_{tile}_compute", compute, kind="qk",
                token_slot_start=start, token_slot_stop=stop, mac_service_ck=16,
                active_queries=16, bank_feature_lanes=16,
                reduction_scope="all16 banks via BG and rank accumulators")

        def export(start=start, stop=stop):
            first = w.line_no + 1
            for offset in range(start * L.QUERIES * 4, stop * L.QUERIES * 4, L.BURST_BYTES):
                w.ranks("PIM_RD_BG", offset)
            w.exports.append(dict(phase=f"qk_tile_{tile}_export", line_start=first,
                line_end=w.line_no, bytes=(w.line_no-first+1)*L.BURST_BYTES,
                scope="complete rank scores, no cross-rank numerical reduction",
                maximum_staged_bytes_per_rank=(stop-start) * L.QUERIES * 4))
        w.phase(f"qk_tile_{tile}_export", export, kind="external_read",
                token_slot_start=start, token_slot_stop=stop)


def generate(path, case="chain", schedule="double"):
    if case not in OPERATORS or schedule not in ("double", "serial"):
        raise ValueError((case, schedule))
    w = Writer(path)
    try:
        if case in ("chain", "projection"):
            projection(w, schedule)
        if case in ("chain", "rope"):
            rope(w, schedule)
        if case in ("chain", "qk"):
            q_exchange(w)
            qk(w)
    finally:
        w.close()
    counts = dict(w.counts)
    dq = dict(x_gpu_to_pim=w.counts["PIM_WR_GB"] * L.BURST_BYTES,
              q_pim_to_gpu=w.counts["PIM_Q_EXPORT"] * L.BURST_BYTES,
              q_gpu_to_remote_buffer=w.counts["PIM_Q_REMOTE"] * L.BURST_BYTES,
              score_pim_to_gpu=w.counts["PIM_RD_BG"] * L.BURST_BYTES)
    dq["total"] = sum(dq.values())
    buffers = dict(L.manifest()["buffers"])
    # Native implementation uses two bounded output entries to allow the next
    # EXEC while the preceding completed output waits for its local STORE.
    buffers["rope_output_per_bank"] = 2 * L.BURST_BYTES
    buffers["rope_output_entries_per_bank"] = 2
    return dict(case=case, variant="H=16 bank-local RoPE", schedule=schedule,
                action_horizon=L.QUERIES, attention_heads=L.HEADS,
                counts=counts, segments=w.segments, input_fragments=w.input_fragments,
                exports=w.exports, local_writes=w.local_writes,
                expected_dq_bytes=dq, buffers=buffers,
                expected_local_bytes=dict(
                    projection_spill=w.counts["PIM_SPILL"] * L.BURST_BYTES,
                    rope_coeff_read=w.counts["PIM_ROPE_COEF"] * L.BURST_BYTES,
                    rope_input_read=w.counts["PIM_ROPE_LOAD"] * L.BURST_BYTES,
                    rope_output_write=w.counts["PIM_ROPE_STORE"] * L.BURST_BYTES,
                    qk_k_read=w.counts["PIM_MAC_QK"] * L.BANKS_PER_RANK * L.BURST_BYTES),
                qk_query_outer_rescans=False, q_remote_dram_writes=0,
                physical_active_rank=L.PIM_RANK, score_padding_exported=True,
                score_tile_tokens=SCORE_TILE_TOKENS,
                required_backend=dict(pim_rank_rope=True, pim_input_broadcast=False,
                                      pim_mac_scope="rank", nCCDAB=8, nCCDQK=16),
                initialization_excluded=L.manifest()["initialization_bytes"],
                standalone_preconditions={"projection": "X available; weights resident",
                    "rope": "BF16 Q/K local spill and coefficient row already resident; not an integrated chain",
                    "qk": "post-RoPE local Q and all K already resident; not an integrated chain",
                    "chain": "X available at controller; weights, coefficient tables, prefix K and zero padding resident"}[case],
                unknown_latency_us=dict(GPU_softmax=None, GPU_forward_dispatch=None,
                                        GPU_source_X_read=None),
                numeric_compute=False,
                numeric_scope="Trace validates ownership and timing dependencies, not bitwise model inference")

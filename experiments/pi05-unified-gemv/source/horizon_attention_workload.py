"""Native trace for token-sharded projection, local RoPE, Q forwarding and QK.

All requests target physical rank1.  New PIM_* local operations require
MemoryConfig(pim_rank_rope=True); they are never replaced by external ST/LD.
"""

from collections import Counter
from pathlib import Path

import horizon_attention_layout as L
from rank_staging import stage_input, input_offset, source_bank, fragment_metadata, fragment_batches, prefetch_metadata

OPERATORS = ("chain", "projection", "rope", "qk")
CONTROL_OPERATORS = ()
SCORE_TILE_TOKENS = 16
RANK_BANK_OPERATIONS = {"PIM_SPILL", "PIM_ROPE_COEF", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC",
                        "PIM_ROPE_STORE", "PIM_Q_EXPORT", "PIM_Q_REMOTE"}


class Writer:
    def __init__(self, path, rank_descriptors=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.counts = Counter()
        self.expanded_counts = Counter()
        self.rank_descriptors = rank_descriptors
        self.rank_batches = []
        self.segments, self.input_fragments, self.exports, self.local_writes = [], [], [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, op, address, buffer=None):
        self.stream.write(f"{op} {address}" + ("" if buffer is None else f" {buffer}") + "\n")
        self.counts[op] += 1
        if op.startswith("PIM_RANK_"):
            self.expanded_counts[op.replace("PIM_RANK_", "PIM_", 1)] += L.BANKS_PER_RANK
        else:
            self.expanded_counts[op] += 1
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

    def ranks(self, command, offset, buffer=None, bank=0):
        for channel in range(L.CHANNELS):
            self.emit(command, L.physical_address(channel, L.PIM_RANK, bank//4, bank%4, offset), buffer)

    def banks(self, command, offset, buffer=None):
        if self.rank_descriptors:
            if command not in RANK_BANK_OPERATIONS:
                raise ValueError(f"Unsupported rank batch operation: {command}")
            first = self.line_no+1
            descriptor = command.replace("PIM_", "PIM_RANK_", 1)
            self.ranks(descriptor, offset, buffer)
            self.rank_batches.append(dict(command=descriptor, child_command=command,
                line_start=first, line_end=self.line_no, physical_rank=1,
                canonical_bank=0, child_banks=list(range(L.BANKS_PER_RANK)),
                children_per_descriptor=L.BANKS_PER_RANK, offset_bytes=offset,
                buffer=buffer, completion="after all sixteen child operations complete"))
            return
        for location in L.active_bank_locations(offset):
            self.emit(command, L.physical_address(*location), buffer)

    def close(self):
        self.stream.close()


def projection(w, schedule, rank_staging=False, staging_banks=1, qkv_fusion=False, whole_srf=False):
    columns, weight_base = (48, 393216) if qkv_fusion else (32, 0)
    views = 4 if qkv_fusion or whole_srf else 2
    if rank_staging:
        stage_input(w, "projection", L.LOCAL_QUERIES, L.HIDDEN//L.LANES, staging_banks, views=views)
        if qkv_fusion:
            from v_token_layout import input_contract
            w.staging_inputs[-1]["logical_input"] = input_contract()
    def compute():
        for batch in fragment_batches(L.HIDDEN//L.LANES, rank_staging or whole_srf or qkv_fusion, views=views):
            inputs = {}
            for fragment in batch:
                buffer = fragment%views if rank_staging or schedule == "double" else 0
                first = w.line_no+1
                if rank_staging:
                    w.ranks("PIM_STAGE_FRAGMENT", input_offset(fragment, 0, L.LOCAL_QUERIES, staging_banks), buffer,
                            bank=source_bank(fragment, staging_banks))
                else:
                    for local_query in range(L.LOCAL_QUERIES):
                        w.ranks("PIM_WR_GB", local_query*L.BURST_BYTES, buffer)
                inputs[fragment] = (buffer, first, w.line_no)
            for fragment in batch:
                buffer, first, last = inputs[fragment]
                mac_first = w.line_no+1
                for output_column in range(columns):
                    offset = weight_base + (fragment*columns + output_column)*L.BURST_BYTES
                    w.ranks("PIM_MAC_AB", offset, buffer)
                w.input_fragments.append(dict(fragment=fragment, buffer=buffer,
                    line_start=first, line_end=last, mac_line_first=mac_first,
                    mac_line_last=w.line_no, external_bytes=0 if rank_staging else (last-first+1)*L.BURST_BYTES,
                    input_source=("bank0_dram_staging" if staging_banks == 1 else "striped_dram_staging") if rank_staging else "gpu_wrgb",
                    command_count=16 if rank_staging else L.LOCAL_QUERIES*16,
                    internal_load_bursts=L.LOCAL_QUERIES*16 if rank_staging else 0,
                    scope=f"rank1 shared input, {L.LOCAL_QUERIES} local tokens, "+("Q/K/V reuse" if qkv_fusion else "Q/K reuse"),
                    **(prefetch_metadata(batch, L.LOCAL_QUERIES, views=views) if rank_staging else {}),
                    **(fragment_metadata(fragment, L.LOCAL_QUERIES, staging_banks) if rank_staging else {})))
                if schedule == "serial":
                    w.fence()
    w.phase("projection_local_input_compute" if rank_staging else "projection_input_compute", compute, kind="projection", mac_service_ck=L.LOCAL_QUERIES,
            native_mac_completion=True)

    if qkv_fusion:
        from pv_token_workload import local_cache_spill
        local_cache_spill(w, qkv_fusion=True)

    def spill():
        first = w.line_no + 1
        for local_query in range(L.LOCAL_QUERIES):
            for kind in ("q_local", "k_suffix"):
                w.banks("PIM_SPILL", L.work_offset(kind, local_query))
        w.local_writes.append(dict(phase="projection_local_spill", line_start=first,
                                   line_end=w.line_no, bytes=(w.line_no-first+1) * L.BURST_BYTES * (L.BANKS_PER_RANK if w.rank_descriptors else 1),
                                   source="FP32 accumulators rounded to BF16", external_bytes=0))
    w.phase("projection_local_spill", spill, kind="local_write")
    def release_projection():
        for slot in range(views):
            w.ranks("PIM_GEMV_RELEASE", 0, slot)
    w.phase("projection_gemv_release", release_projection, kind="gemv_release")


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
                              line_end=w.line_no, bytes=(w.line_no-first+1)*L.BURST_BYTES*(L.BANKS_PER_RANK if w.rank_descriptors else 1),
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
    # All H queries remain in their H*32B bank buffer through this single K scan.
    # Output tiles bound rank score storage; they do not repeat Q or K reads.
    for tile, start in enumerate(range(0, L.LOCAL_TOKENS, SCORE_TILE_TOKENS)):
        stop = min(start + SCORE_TILE_TOKENS, L.LOCAL_TOKENS)

        def compute(start=start, stop=stop):
            for token in range(start, stop):
                w.ranks("PIM_MAC_QK", L.qk_column_offset(token), 0)
        w.phase(f"qk_tile_{tile}_compute", compute, kind="qk",
                token_slot_start=start, token_slot_stop=stop, mac_service_ck=L.QUERIES,
                active_queries=L.QUERIES, bank_feature_lanes=16,
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
    w.phase("qk_gemv_release", lambda: w.ranks("PIM_GEMV_RELEASE", 0, 0),
            kind="gemv_release")


def generate(path, case="chain", schedule="double", rank_staging=False, staging_banks=1, rank_descriptors=False, qkv_fusion=False, whole_srf=False):
    if case not in OPERATORS or schedule not in ("double", "serial"):
        raise ValueError((case, schedule))
    w = Writer(path, rank_descriptors=rank_descriptors)
    try:
        if case in ("chain", "projection"):
            projection(w, schedule, rank_staging=rank_staging, staging_banks=staging_banks, qkv_fusion=qkv_fusion, whole_srf=whole_srf)
        if case in ("chain", "rope"):
            rope(w, schedule)
        if case in ("chain", "qk"):
            q_exchange(w)
            qk(w)
    finally:
        w.close()
    counts = dict(w.counts)
    expanded = w.expanded_counts
    dq = dict(x_gpu_to_pim=(w.counts["ST"] if rank_staging else w.counts["PIM_WR_GB"]) * L.BURST_BYTES,
              q_pim_to_gpu=expanded["PIM_Q_EXPORT"] * L.BURST_BYTES,
              q_gpu_to_remote_buffer=expanded["PIM_Q_REMOTE"] * L.BURST_BYTES,
              score_pim_to_gpu=w.counts["PIM_RD_BG"] * L.BURST_BYTES)
    dq["total"] = sum(dq.values())
    buffers = dict(L.manifest()["buffers"])
    # Native implementation uses two bounded output entries to allow the next
    # EXEC while the preceding completed output waits for its local STORE.
    buffers["rope_output_per_bank"] = 2 * L.BURST_BYTES
    buffers["rope_output_entries_per_bank"] = 2
    result = dict(case=case, variant=f"H={L.QUERIES} bank-local RoPE", schedule=schedule,
                action_horizon=L.QUERIES, attention_heads=L.HEADS,
                counts=counts, expanded_counts=dict(expanded), rank_batches=w.rank_batches,
                rank_descriptors=rank_descriptors, segments=w.segments, input_fragments=w.input_fragments,
                staging_inputs=getattr(w, "staging_inputs", []), rank_staging=rank_staging,
                staging_banks=staging_banks if rank_staging else 0,
                exports=w.exports, local_writes=w.local_writes,
                expected_dq_bytes=dq, buffers=buffers,
                expected_local_bytes=dict(
                    projection_spill=expanded["PIM_SPILL"] * L.BURST_BYTES,
                    rope_coeff_read=expanded["PIM_ROPE_COEF"] * L.BURST_BYTES,
                    rope_input_read=expanded["PIM_ROPE_LOAD"] * L.BURST_BYTES,
                    rope_output_write=expanded["PIM_ROPE_STORE"] * L.BURST_BYTES,
                    qk_k_read=w.counts["PIM_MAC_QK"] * L.BANKS_PER_RANK * L.BURST_BYTES),
                qk_query_outer_rescans=False, q_remote_dram_writes=0,
                physical_active_rank=L.PIM_RANK, score_padding_exported=True,
                score_tile_tokens=SCORE_TILE_TOKENS,
                required_backend=dict(pim_rank_rope=True, pim_input_broadcast=False,
                                      pim_mac_scope="rank", pim_horizon=L.QUERIES,
                                      nCCDAB=L.LOCAL_QUERIES, nCCDQK=L.QUERIES),
                initialization_excluded=L.manifest()["initialization_bytes"],
                standalone_preconditions={"projection": "X available; weights resident",
                    "rope": "BF16 Q/K local spill and coefficient row already resident; not an integrated chain",
                    "qk": "post-RoPE local Q and all K already resident; not an integrated chain",
                    "chain": "X available at controller; weights, coefficient tables, prefix K and zero padding resident"}[case],
                unknown_latency_us=dict(GPU_softmax=None, GPU_forward_dispatch=None,
                                        GPU_source_X_read=None),
                numeric_compute=False,
                numeric_scope="Trace validates ownership and timing dependencies, not bitwise model inference")
    if qkv_fusion:
        from qkv_token_layout import mapping_contract
        from pv_token_layout import cache_contract, trace_ranges
        result.update(qkv_fusion=True, v_token_sharding=True, pv_token_sharding=True,
            qkv_mapping=mapping_contract(result),
            pv_token_mapping=dict(schema_version=1,enabled=True,operator="qkv",local_cache=cache_contract(True),
                trace_ranges=trace_ranges(result),numerical_compute=False))
        result["required_backend"].update(pim_qkv_fusion=True)
        result["buffers"]["projection_accumulator_per_bank"]=1536
        result["initialization_excluded"].pop("weights_qk")
        result["initialization_excluded"]["weights_qkv"]=256*98304
        result["initialization_excluded"].update(prefix_v=8*968*256*2,padding_v=8*8*256*2)
        result["expected_local_bytes"].update(v_cache_rmw_read=16*16*16*32,
            v_cache_rmw_write=16*16*16*32)
        result["expected_local_bytes"]["projection_spill"]-=16*16*16*32
    return result

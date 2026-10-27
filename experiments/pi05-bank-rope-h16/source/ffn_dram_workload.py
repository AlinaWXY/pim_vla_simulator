"""FFN movement service with DRAM-resident Z and bounded nonlinear staging.

No32KiB rank G/U gather buffer and no8KiB rank Z SRAM are assumed. Native
PIM_Z_STORE accounts for per-column gather/return and local DRAM write;
nonlinear arithmetic and final GPU partial reduction remain unmeasured.
"""

import ffn_dram_layout as L
import ffn_workload as F

OPERATORS = ("gate_up_z_store", "down_tiled_dram", "combined_timing_chain")
MAC_SERVICE_CK = F.MAC_SERVICE_CK
MOVEMENT_TAIL_CK = F.MOVEMENT_TAIL_CK


def _columns(w, command, fragment, buffer=None):
    for bank in range(L.BANKS_PER_RANK):
        for rank in range(L.TOTAL_RANKS):
            w.emit(command, L.physical_address(*L.z_column_location(rank, fragment, bank)), buffer)


def projection(w):
    F.pipeline(w, name="input_and_compute", base=L.GU_BASE, fragments=64,
               slots=32, tile=None, source="gpu_wrgb")
    # The old whole-G/U gather WAIT is deliberately absent. Each native STORE
    # moves only one feature's128B G/U into bounded nonlinear staging.
    def stores():
        for fragment in range(L.Z_COLUMNS_PER_BANK):
            _columns(w, "PIM_Z_STORE", fragment)
    w.phase("z_stream_store", stores, kind="native_z_store",
            native_per_column_GU_gather_bytes=128, native_per_column_Z_return_bytes=32,
            z_store_bytes_system=L.TOTAL_BANKS*L.Z_BYTES_PER_BANK,
            rank_full_Z_sram_bytes=0, rank_full_GU_staging_bytes=0,
            nonlinear_latency_ck=None,
            nonlinear_boundary="Per-column128B FP32 G/U input to32B BF16 Z output; arithmetic excluded")


def down(w):
    for tile in range(4):
        name = f"tile_{tile}_input_and_compute"
        base = L.DOWN_BASE + tile*16*16*L.BURST_BYTES

        def compute(tile=tile, name=name, base=base):
            for fragment in range(16):
                buffer, first = fragment % 2, w.line_no+1
                _columns(w, "PIM_Z_LOAD", fragment, buffer)
                last, mac_first = w.line_no, w.line_no+1
                for slot in range(16):
                    w.ranks("PIM_MAC_AB", base+(fragment*16+slot)*L.BURST_BYTES, buffer)
                w.input_fragments.append(dict(phase=name, tile=tile, fragment=fragment, buffer=buffer,
                    input_source="bank_dram_z", input_command="PIM_Z_LOAD",
                    line_start=first, line_end=last, input_line_first=first, input_line_last=last,
                    mac_line_first=mac_first, mac_line_last=w.line_no,
                    command_count=L.TOTAL_RANKS*L.BANKS_PER_RANK,
                    bursts_per_rank=L.BANKS_PER_RANK,
                    bytes_per_rank=L.BANKS_PER_RANK*L.BURST_BYTES,
                    consumer_mac_count=16*L.TOTAL_RANKS, slots_per_fragment=16,
                    source_feature_column=fragment, z_dram_external_dq_bytes=0,
                    transpose="input[token][bank_lane] <- source_bank_column[token]"))
            w.drain_mac()
        w.phase(name, compute, kind="pipeline", tile=tile, fragment_count=16,
                slots_per_fragment=16, input_source="bank_dram_z", M=L.HORIZON,
                weight_base=base, weight_bursts_per_channel=256)
        F.export_results(w, name=f"tile_{tile}_result_export", slots=16,
                         tile=tile, output_start=tile*256)


def generate(path, case="combined_timing_chain", local_ck=2):
    if case not in OPERATORS:
        raise ValueError(case)
    if not isinstance(local_ck, int) or local_ck <= 0:
        raise ValueError("local_ck must be a positive memory-clock count")
    w = F.Writer(path)
    try:
        if case in ("gate_up_z_store", "combined_timing_chain"):
            projection(w)
        if case in ("down_tiled_dram", "combined_timing_chain"):
            down(w)
    finally:
        w.close()
    counts = dict(w.counts)
    dq = dict(pim_input_write=w.counts["PIM_WR_GB"]*L.BURST_BYTES,
              pim_result_read=w.counts["PIM_RD_RES"]*L.BURST_BYTES,
              dram_read=0, dram_write=0)
    dq["total"] = sum(dq.values())
    z_store_count, z_load_count = w.counts["PIM_Z_STORE"], w.counts["PIM_Z_LOAD"]
    internal = dict(GU_gather_to_nonlinear=z_store_count*128,
                    Z_return_to_producer_bank=z_store_count*32,
                    Z_dram_write=z_store_count*32, Z_dram_read=z_load_count*32,
                    Z_transpose_to_rank_input=z_load_count*32)
    matrices = {name: F.matrix_for(name) for name in ("gate_up", "down_tiled")}
    matrices["down_tiled"]["rank_local_z_sram_bytes"] = 0
    matrix = (matrices["gate_up"] if case == "gate_up_z_store" else
              matrices["down_tiled"] if case == "down_tiled_dram" else None)
    return dict(case=case, horizon=L.HORIZON, action_horizon=L.HORIZON,
        mode="native_ffn_dram_service", counts=counts, segments=w.segments,
        matrix=matrix, matrices=matrices, input_fragments=w.input_fragments, exports=w.exports,
        expected_dq_bytes=dq, expected_internal_bytes=internal,
        physical_active_rank=L.PIM_RANK, local_ck=local_ck,
        pim_local_transfer_cycles=local_ck, mac_service_ck=MAC_SERVICE_CK,
        z_base=L.Z_BASE, z_bytes_per_bank=L.Z_BYTES_PER_BANK,
        z_reserved_bytes_per_bank=L.Z_RESERVED_BYTES_PER_BANK,
        z_dram_layout="[feature_slot16][token16] BF16; rank input is[token16][feature_lane16]",
        z_read_passes=4 if case != "gate_up_z_store" else 0,
        rank_local_z_sram_bytes=0, rank_bulk_GU_staging_bytes=0,
        buffers=L.build_layout()["buffers"],
        local_gather_included=case != "down_tiled_dram",
        local_gather_model="Native PIM_Z_STORE per-column128B gather+32B return on rank local path",
        analytical_gather_WAIT_ck=0,
        nonlinear_latency_ck=None,
        nonlinear_status="Unmeasured per-column GELU/multiply/cast; native trace models movement only",
        gpu_partial_reduction_latency_ck=None,
        gpu_partial_count=0 if case == "gate_up_z_store" else L.TOTAL_RANKS,
        required_backend=dict(pim_ffn_dram=True, pim_rank_rope=False,
            pim_input_broadcast=False, pim_mac_scope="rank", nCCDAB=16),
        preconditions=("Z columns already resident in bank DRAM; weights resident"
                       if case == "down_tiled_dram" else
                       "Weights resident; X ready at input; nonlinear producer's arithmetic/availability stalls unmodeled"),
        timing_chain=case == "combined_timing_chain", integrated_chain=False, numeric_compute=False,
        unknown_latency_us=dict(nonlinear_GELU_multiply_cast=None,
                               GPU_final_partial_reduction=None, GPU_source_and_arrival=None),
        exclusions=["Nonlinear arithmetic and producer stalls", "GPU final partial reduction",
                    "GPU source/arrival overhead", "Norm/residual/conditioning arithmetic"],
        chain_semantics="Native movement dependency chain with an explicit unmeasured nonlinear boundary; not complete FFN inference")

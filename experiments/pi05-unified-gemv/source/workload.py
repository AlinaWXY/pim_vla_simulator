"""H16 workload API for one shared pair of bank GEMV slots.

The address generators preserve the agreed weight/cache ownership. The native
backend, not this adapter, owns fill completion, multicast and slot lifetimes.
"""
import horizon_attention as attention
import horizon_value as value
import horizon_ffn as ffn
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
from rank_staging import reservation, check_staging_banks

HORIZON = 16
PRIMARY = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")
CASES = (*PRIMARY, "combined_timing_chain", "serial_layer_control")


def unified_buffers(case, rank_staging=True, staging_banks=1, v_token_sharding=False,
                    pv_token_sharding=False,qkv_fusion=False):
    buffers = dict(gemv_slots_per_bank=2, gemv_slot_bytes=512,
                gemv_total_bytes_per_bank=1024, gemv_layout="[token16][lane16] BF16",
                separate_rank_MAC_input_bytes=0, separate_Q_buffer_bytes=0,
                rank_external_ingress_staging_bytes=32,
                bank_Z_column_staging_bytes=0,
                input_broadcast_extra_cycles=0,
                input_rows=8 if case == "chain" or case == "v" and v_token_sharding else 16,
                QKT_rows=16 if case == "chain" else 0)
    if rank_staging:
        buffers.update(input_staging_dram=reservation(staging_banks), input_staging_dram_bytes_per_rank=131072,
            input_fill_policy="Entire operator GPU input -> source-bank DRAM -> source-bank SRF -> same-rank SRF broadcast",
            staging_banks=staging_banks,
            column_read_includes_bank_to_srf_and_broadcast=True,
            input_prefetch_slots=2,
            input_compute_order="Fill both SRF slots before computing the two fragments")
    if case == "chain":
        buffers.update(projection_accumulator_per_bank=1024,
            rope_input_double_per_bank=64, rope_coefficient_per_bank=32,
            rope_output_per_bank=64, score_tile_per_rank=1024)
    elif case in ("v", "pv"):
        buffers.update(accumulator_per_bank=512)
    else:
        buffers.update(GU_accumulator_per_bank=2048, Down_accumulator_per_bank=256,
            rank_nonlinear_input_staging=0, rank_nonlinear_output_staging=0,
            rank_full_Z=0, rank_full_GU=0)
    if pv_token_sharding or qkv_fusion:
        rows=buffers["input_rows"]
        views=1024//(rows*32)
        buffers.update(input_srf_bytes_per_bank=1024,output_pool_bytes_per_bank=2048,
            gemv_slots_per_bank=views,gemv_slot_bytes=rows*32,gemv_views_per_bank=views,
            gemv_view_bytes=rows*32,gemv_layout="one 1KiB SRF partitioned into fragment views",
            input_prefetch_slots=views,whole_srf_batch=True,
            input_compute_order="Fill all valid views; consume the entire SRF batch before any overwrite")
        if case=="chain":
            buffers["projection_accumulator_per_bank"]=1536 if qkv_fusion else 1024
        elif case=="pv":
            buffers["accumulator_per_bank"]=1024
    return buffers


def generate(path, case, rank_staging=True, staging_banks=1, rank_descriptors=False, channel_independent_dispatch=False,
             v_token_sharding=False,pv_token_sharding=False,qkv_fusion=False):
    if pv_token_sharding and not v_token_sharding:
        raise ValueError("PV token sharding requires token-sharded V")
    if qkv_fusion and not pv_token_sharding:
        raise ValueError("QKV fusion requires token-parity V/PV cache mode")
    rank_descriptors = bool(rank_descriptors and rank_staging)
    channel_independent_dispatch = bool(channel_independent_dispatch and rank_staging)
    check_staging_banks(staging_banks)
    if case not in CASES:
        raise ValueError(case)
    if case == "serial_layer_control":
        return generate_serial(path, rank_staging=rank_staging, staging_banks=staging_banks,
            rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
            v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
    if case == "chain":
        result = attention.generate(path, horizon=16, case="chain", schedule="double", rank_staging=rank_staging, staging_banks=staging_banks, rank_descriptors=rank_descriptors,
            qkv_fusion=qkv_fusion,whole_srf=pv_token_sharding)
    elif case in ("v", "pv"):
        result = value.generate(path, 16, case, rank_staging=rank_staging, staging_banks=staging_banks,
                                v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,rank_descriptors=rank_descriptors)
    else:
        result = ffn.generate(path, 16, case, local_ck=2, rank_staging=rank_staging, staging_banks=staging_banks)
    result["buffers"] = unified_buffers(case, rank_staging=rank_staging, staging_banks=staging_banks,
                                         v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
    if "layout" in result:
        result["layout"]["buffers"] = unified_buffers(case, rank_staging=rank_staging, staging_banks=staging_banks,
                                                     v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
    result.setdefault("required_backend", {}).update(pim_unified_gemv=True,
        gemv_port_cycles=2, pim_horizon=16,
        pim_rank_staging=rank_staging, pim_rank_controller=rank_staging,
        pim_staging_banks=staging_banks if rank_staging else 1,
        pim_rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch)
    result["rank_descriptors"] = rank_descriptors
    result["channel_independent_dispatch"] = channel_independent_dispatch
    result["v_token_sharding"] = bool(v_token_sharding and
        (case=="v" or case=="pv" and pv_token_sharding or case=="chain" and qkv_fusion))
    if pv_token_sharding or qkv_fusion:
        token_cache=case in ("v","pv") or case=="chain" and qkv_fusion
        result.update(pv_token_sharding=bool(pv_token_sharding and token_cache),qkv_fusion=bool(qkv_fusion and case=="chain"),whole_srf=True)
        result["required_backend"].update(pim_whole_srf=True,pim_token_pv=bool(token_cache),
            pim_qkv_fusion=bool(qkv_fusion and case=="chain"),v_cache_preinstalled=case=="pv")
        result["input_resource"]="one whole 1KiB SRF in each physical PIM bank"
        if case=="v" and qkv_fusion:
            result.update(diagnostic_only=True,included_in_primary_stage_sum=False,fused_producer_case="chain")
    result.setdefault("expanded_counts", result["counts"].copy())
    result.setdefault("rank_batches", [])
    result.setdefault("input_resource", "same two 512B slots in each physical PIM bank")
    result["no_separate_rank_input_SRAM"] = True
    result["complete_ae_latency"] = False
    return result


def generate_serial(path, rank_staging=True, staging_banks=1, rank_descriptors=False, channel_independent_dispatch=False,
                    v_token_sharding=False,pv_token_sharding=False,qkv_fusion=False):
    """One controller lifetime across all modeled stages; no invented GPU wait."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines, segments, parts, boundary = [], [], [], 0
    staging_inputs, rank_batches, v_mapping = [], [], None
    pv_mappings,qkv_mapping={},None
    expanded_counts = Counter()
    dq = {}
    with tempfile.TemporaryDirectory(prefix="serial-parts-", dir=path.parent) as scratch:
        for case in PRIMARY:
            if case=="v" and qkv_fusion:
                continue
            part_path = Path(scratch) / f"{case}.trace"
            manifest = generate(part_path, case, rank_staging=rank_staging, staging_banks=staging_banks,
                rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
                v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
            expanded_counts.update(manifest["expanded_counts"])
            source = part_path.read_text(encoding="ascii").splitlines()
            skip = bool(lines)
            line_shift = len(lines)-int(skip)
            expanded_counts["BARRIER"] -= int(skip)
            for number, line in enumerate(source):
                if skip and number == 0:
                    if line != "BARRIER 0":
                        raise AssertionError("Stage must begin with its completion boundary0")
                    continue
                if line.startswith("BARRIER "):
                    line = f"BARRIER {int(line.split()[1])+boundary}"
                lines.append(line)
            for original in manifest["segments"]:
                segment = deepcopy(original)
                segment.update(name=f"{case}/{segment['name']}", stage=case,
                    start=segment["start"]+boundary, end=segment["end"]+boundary,
                    line_start=segment["line_start"]+line_shift,
                    line_end=segment["line_end"]+line_shift)
                segments.append(segment)
            for original in manifest.get("staging_inputs", []):
                staged = deepcopy(original)
                staged["stage"] = case
                for key in ("first_line", "store_first_line", "store_last_line", "last_line"):
                    staged[key] += line_shift
                for store in staged.get("store_sequence", []):
                    store["first_line"] += line_shift
                    store["last_line"] += line_shift
                staging_inputs.append(staged)
            for original in manifest.get("rank_batches", []):
                batch = deepcopy(original)
                batch.update(stage=case, line_start=batch["line_start"]+line_shift,
                    line_end=batch["line_end"]+line_shift)
                rank_batches.append(batch)
            if manifest.get("v_token_mapping"):
                from v_token_layout import shift_trace_lines
                v_mapping = shift_trace_lines(manifest["v_token_mapping"], line_shift)
                v_mapping["stage"] = case
            if manifest.get("pv_token_mapping"):
                from pv_token_layout import shift_trace_lines
                pv_mappings[case]=shift_trace_lines(manifest["pv_token_mapping"],line_shift)
                pv_mappings[case]["stage"]=case
            if manifest.get("qkv_mapping"):
                from pv_token_layout import shift_trace_lines
                qkv_mapping=shift_trace_lines(manifest["qkv_mapping"],line_shift)
                qkv_mapping["stage"]=case
            parts.append(dict(stage=case, start_marker=boundary,
                end_marker=segments[-1]["end"], first_line=2+line_shift, last_line=len(lines)))
            if manifest.get("v_token_sharding"):
                parts[-1]["v_token_sharding"] = True
            if manifest.get("pv_token_sharding"):
                parts[-1]["pv_token_sharding"]=True
            if manifest.get("qkv_fusion"):
                parts[-1]["qkv_fusion"]=True
            boundary = segments[-1]["end"]
            dq[case] = manifest["expected_dq_bytes"]["total"]
    path.write_text("\n".join(lines)+"\n", encoding="ascii")
    buffers = unified_buffers("combined_timing_chain", rank_staging=rank_staging, staging_banks=staging_banks,
        v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
    buffers.update(projection_accumulator_per_bank=1536 if qkv_fusion else 1024, value_accumulator_per_bank=512,
        rope_input_double_per_bank=64, rope_coefficient_per_bank=32, rope_output_per_bank=64,
        score_tile_per_rank=1024, QKT_rows=16,
        input_rows_by_operator=dict(QK_projection=8, V=8 if v_token_sharding else 16, PV=16, GU=16, Down=16))
    buffers.pop("input_rows", None)
    result = dict(case="serial_layer_control", horizon=16, action_horizon=16,
        segments=segments, counts=dict(Counter(line.split()[0] for line in lines)), parts=parts,
        expanded_counts=dict(expanded_counts), rank_batches=rank_batches,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
        v_token_sharding=bool(v_token_sharding),
        staging_inputs=staging_inputs, rank_staging=rank_staging,
        staging_banks=staging_banks if rank_staging else 0,
        expected_dq_bytes=dict(total=sum(dq.values()), by_stage=dq), buffers=buffers,
        input_resource="same two512B slots per bank throughout one native simulation",
        required_backend=dict(pim_unified_gemv=True, gemv_port_cycles=2, pim_horizon=16,
            pim_rank_staging=rank_staging, pim_rank_controller=rank_staging,
            pim_staging_banks=staging_banks if rank_staging else 1,
            pim_rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
            pim_rank_rope=True, pim_ffn_dram=False, ffn_z_preinstalled=False,
            pim_mac_scope="rank", pim_input_broadcast=False, nCCDAB=8, nCCDQK=16),
        continuous_refresh_state=True, no_separate_rank_input_SRAM=True,
        additional_control=True, included_in_primary_stage_sum=False,
        down_rank_partial_reduction=False, input_broadcast_extra_cycles=0,
        numerical_compute=False, complete_ae_latency=False,
        unknown_boundary_latency_us=dict(GPU_V_cast_pack=None, GPU_softmax_and_score_scaling=None,
            GPU_O_projection_norm_residual_FFN_input=None, GELU_multiply_cast=None,
            GPU_Q_forward_dispatch=None),
        excluded_boundary_policy="Consumer values are assumed ready at each excluded GPU/nonlinear boundary. No artificial zero-latency kernel or delay is modeled.",
        interpretation="Continuous native memory/resource schedule for modeled work only; not full numerical or end-to-end AE inference")
    if v_mapping is not None:
        result["v_token_mapping"] = v_mapping
    if pv_token_sharding:
        from pv_token_layout import cache_contract
        result.update(pv_token_sharding=True,qkv_fusion=bool(qkv_fusion),
            whole_srf=True,
            pv_token_mapping=dict(schema_version=1,enabled=True,operator="serial_layer_control",
                local_cache=cache_contract(qkv_fusion),stages=pv_mappings,trace_ranges={}),
            input_resource="one whole 1KiB SRF per bank throughout one native simulation")
        result["buffers"].update(pv_accumulator_per_bank=1024,
            gemv_views_by_operator=dict(QKV=4,V=4,PV=2,GU=2,Down=2),
            gemv_view_bytes_by_operator=dict(QKV=256,V=256,PV=512,GU=512,Down=512),
            input_prefetch_policy="Operator-dependent views of one whole 1KiB SRF; no overlapping batches")
        for key in ("gemv_slots_per_bank","gemv_slot_bytes","gemv_views_per_bank","gemv_view_bytes","input_prefetch_slots"):
            result["buffers"].pop(key,None)
        result["unknown_boundary_latency_us"].pop("GPU_V_cast_pack")
        result["unknown_boundary_latency_us"]["GPU_PV_partial_sum"]=None
        result["required_backend"].update(pim_token_pv=True,pim_whole_srf=True,v_cache_preinstalled=False)
    if qkv_mapping is not None:
        result["qkv_mapping"]=qkv_mapping
        result["required_backend"]["pim_qkv_fusion"]=True
    return result


def layouts(rank_staging=True, staging_banks=1, v_token_sharding=False,pv_token_sharding=False,qkv_fusion=False):
    result = dict(attention=attention.layout_for(16,qkv_fusion=qkv_fusion), value=value.layout(16, v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding),
                  ffn=ffn.layout(16, rank_staging=rank_staging, staging_banks=staging_banks))
    for name, case in (("attention", "chain"), ("value", "v"), ("ffn", "combined_timing_chain")):
        result[name]["buffers"] = unified_buffers(case, rank_staging=rank_staging, staging_banks=staging_banks,
                                                   v_token_sharding=v_token_sharding,pv_token_sharding=pv_token_sharding,qkv_fusion=qkv_fusion)
        if rank_staging:
            result[name]["input_staging"] = reservation(staging_banks)
        for stale in ("input_double_buffer_bytes_per_rank", "rank_input_layout"):
            result[name].pop(stale, None)
    return result

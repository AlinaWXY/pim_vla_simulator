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

HORIZON = 16
PRIMARY = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")
CASES = (*PRIMARY, "combined_timing_chain", "serial_layer_control")


def unified_buffers(case):
    buffers = dict(gemv_slots_per_bank=2, gemv_slot_bytes=512,
                gemv_total_bytes_per_bank=1024, gemv_layout="[token16][lane16] BF16",
                separate_rank_MAC_input_bytes=0, separate_Q_buffer_bytes=0,
                rank_external_ingress_staging_bytes=32,
                bank_Z_column_staging_bytes=0,
                input_broadcast_extra_cycles=0,
                input_rows=8 if case == "chain" else 16,
                QKT_rows=16 if case == "chain" else 0)
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
    return buffers


def generate(path, case):
    if case not in CASES:
        raise ValueError(case)
    if case == "serial_layer_control":
        return generate_serial(path)
    if case == "chain":
        result = attention.generate(path, horizon=16, case="chain", schedule="double")
    elif case in ("v", "pv"):
        result = value.generate(path, 16, case)
    else:
        result = ffn.generate(path, 16, case, local_ck=2)
    result["buffers"] = unified_buffers(case)
    if "layout" in result:
        result["layout"]["buffers"] = unified_buffers(case)
    result.setdefault("required_backend", {}).update(pim_unified_gemv=True,
        gemv_port_cycles=2, pim_horizon=16)
    result["input_resource"] = "same two 512B slots in each physical PIM bank"
    result["no_separate_rank_input_SRAM"] = True
    result["complete_ae_latency"] = False
    return result


def generate_serial(path):
    """One controller lifetime across all modeled stages; no invented GPU wait."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines, segments, parts, boundary = [], [], [], 0
    dq = {}
    with tempfile.TemporaryDirectory(prefix="serial-parts-", dir=path.parent) as scratch:
        for case in PRIMARY:
            part_path = Path(scratch) / f"{case}.trace"
            manifest = generate(part_path, case)
            source = part_path.read_text(encoding="ascii").splitlines()
            skip = bool(lines)
            line_shift = len(lines)-int(skip)
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
            parts.append(dict(stage=case, start_marker=boundary,
                end_marker=segments[-1]["end"], first_line=2+line_shift, last_line=len(lines)))
            boundary = segments[-1]["end"]
            dq[case] = manifest["expected_dq_bytes"]["total"]
    path.write_text("\n".join(lines)+"\n", encoding="ascii")
    buffers = unified_buffers("combined_timing_chain")
    buffers.update(projection_accumulator_per_bank=1024, value_accumulator_per_bank=512,
        rope_input_double_per_bank=64, rope_coefficient_per_bank=32, rope_output_per_bank=64,
        score_tile_per_rank=1024, QKT_rows=16,
        input_rows_by_operator=dict(QK_projection=8, V=16, PV=16, GU=16, Down=16))
    buffers.pop("input_rows", None)
    return dict(case="serial_layer_control", horizon=16, action_horizon=16,
        segments=segments, counts=dict(Counter(line.split()[0] for line in lines)), parts=parts,
        expected_dq_bytes=dict(total=sum(dq.values()), by_stage=dq), buffers=buffers,
        input_resource="same two512B slots per bank throughout one native simulation",
        required_backend=dict(pim_unified_gemv=True, gemv_port_cycles=2, pim_horizon=16,
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


def layouts():
    result = dict(attention=attention.layout_for(16), value=value.layout(16), ffn=ffn.layout(16))
    for name, case in (("attention", "chain"), ("value", "v"), ("ffn", "combined_timing_chain")):
        result[name]["buffers"] = unified_buffers(case)
        for stale in ("input_double_buffer_bytes_per_rank", "rank_input_layout"):
            result[name].pop(stale, None)
    return result

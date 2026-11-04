"""Fresh 2x2 score-spill / local O-projection experiment, common BF16 outputs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from reproduce import HERE, FUSED_PRIMARY, digest, dump, execute, run_case, setup

VARIANTS = {"neither": (False, False), "score_only": (True, False),
            "oproj_only": (False, True), "both": (True, True)}


def run_variant(output, variant, cases):
    score, proj = VARIANTS[variant]
    output.mkdir()
    runs = {case: run_case(output, case, rank_staging=True, staging_banks=2,
        rank_descriptors=True, channel_independent_dispatch=True,
        v_token_sharding=True, pv_token_sharding=True, qkv_fusion=True,
        score_dram_spill=score, o_projection=proj, pv_bf16_output=True) for case in cases}
    complete = all(case in runs for case in FUSED_PRIMARY)
    summary = dict(status="passed", variant=variant, score_dram_spill=score,
        o_projection=proj, pv_bf16_output=True, qkv_fusion=True,
        primary_cases=list(FUSED_PRIMARY), runs=runs,
        precision=dict(operands="BF16", weights="BF16", accumulator="FP32",
            PV_result="BF16", O_projection_result="BF16", scores="FP32"),
        measured_stage_sum_us_per_layer_step=sum(runs[c]["latency_us"] for c in FUSED_PRIMARY) if complete else None,
        external_DQ_bytes_per_layer_step=sum(runs[c]["external_DQ_bytes"] for c in FUSED_PRIMARY) if complete else None,
        serial_layer_memory_control_us=runs.get("serial_layer_control", {}).get("latency_us"),
        complete_AE_end_to_end_latency_us=None, numerical_compute=False,
        unknown_latency_us={"GPU_softmax_and_score_scaling": None,
            "GPU_PV_partial_sum" if not proj else "GPU_O_partial_sum": None,
            **({"GPU_O_projection": None} if not proj else {}),
            "GPU_norm_residual_and_FFN_nonlinearity": None, "GPU_dispatch_and_contention": None},
        interpretation="Native modeled memory/PIM work only. O projection is included only when enabled; excluded GPU arithmetic is unknown, not zero.")
    dump(output / "summary.json", summary)
    if complete:
        from channel_analysis import write_report
        write_report(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--cases", nargs="+", choices=(*FUSED_PRIMARY, "serial_layer_control"),
                        default=[*FUSED_PRIMARY, "serial_layer_control"])
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    library = backend / "ramulator2/libramulator.so"
    if not library.is_file():
        parser.error("Build the native backend first")
    if output.exists():
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True)
    env = setup(backend)
    files = sorted(set(p for folder in (HERE, backend / "simulator", backend / "ramulator2/src", backend / "ramulator2/python")
        for p in folder.rglob("*") if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc", ".sh")
        and "__pycache__" not in p.parts and "results" not in p.relative_to(folder).parts))
    library_hash = digest(library)
    dump(output / "provenance.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, source_sha256={str(p.relative_to(HERE.parent.parent)): digest(p) for p in files},
        native_library_sha256=library_hash, variants=args.variants, cases=args.cases,
        fresh_native_runs=True, common_BF16_PV_output=True, tests_executed=not args.skip_tests))
    if not args.skip_tests:
        execute("unit-tests", [sys.executable, "-m", "unittest", "discover", "-s", HERE / "validation", "-p", "test_*.py", "-v"], output, env)
        execute("legacy-native-guards", [sys.executable, HERE / "validation/native_guards.py", "--backend", backend,
            "--output", output / "legacy-native-guards"], output, env)
        for label, filename in (("score-native-guards", "new_score_native_guards.py"),
                                ("o-native-guards", "new_o_native_guards.py")):
            execute(label, [sys.executable, HERE / "validation" / filename, "--backend", backend,
                "--output", output / label], output, env)
    summaries = {v: run_variant(output / v, v, args.cases) for v in args.variants}
    if digest(library) != library_hash:
        raise AssertionError("Native library changed during experiment")
    dump(output / "summary.json", dict(status="passed", variants=summaries,
        tests_executed=not args.skip_tests, complete_end_to_end_latency_us=None))
    if set(summaries) == set(VARIANTS) and all(s["measured_stage_sum_us_per_layer_step"] is not None for s in summaries.values()):
        from compare_o_projection import compare
        compare(output)


if __name__ == "__main__":
    main()

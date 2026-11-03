"""Run fresh H16 native simulations with physically shared bank GEMV slots.

No historical latency is an oracle. Five primary stages retain the prior
separate-stage aggregation; combined FFN includes the GPU input/output traffic.
GPU GeLU and multiply arithmetic is an explicitly unmeasured boundary.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PRIMARY = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")
CASES = (*PRIMARY, "combined_timing_chain", "serial_layer_control")
EXTERNAL = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}
UNKNOWN = ("GPU_V_cast_pack", "GPU_softmax_and_score_scaling", "GPU_O_projection",
           "GELU_multiply_cast", "normalization_residual",
           "action_input_output_and_step_update", "GPU_Q_forward_kernel_and_dispatch",
           "background_VLM_contention")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(backend):
    paths = [HERE / "source", HERE / "validation", backend, backend / "ramulator2/python"]
    sys.path[:0] = [str(p) for p in paths]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(map(str, paths)))
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2")
    return env


def config_for(case, refresh=True, max_cycles=1_000_000, rank_staging=True, staging_banks=1,
               rank_descriptors=False, channel_independent_dispatch=False):
    from simulator.config import MemoryConfig
    chain = case in ("chain", "serial_layer_control")
    timings = dict(nCCDAB=8 if chain else 16, nLOCALWR=4)
    if chain:
        timings.update(nCCDK=2, nCCDQK=16)
    return MemoryConfig(model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=refresh, pim=True, pim_input_broadcast=False, pim_mac_scope="rank",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=max_cycles,
        pim_horizon=16, pim_rank_rope=chain, pim_ffn_dram=False,
        pim_unified_gemv=True, gemv_port_cycles=2,
        pim_rank_staging=rank_staging, pim_rank_controller=rank_staging,
        pim_staging_banks=staging_banks if rank_staging else 1,
        pim_rank_descriptors=bool(rank_descriptors and rank_staging),
        channel_independent_dispatch=bool(channel_independent_dispatch and rank_staging),
        ffn_z_preinstalled=False,
        rope_ii_cycles=1, rope_latency_cycles=4, pim_cast_cycles=1, rank_reduce_cycles=4,
        timing_overrides=timings, pim_local_transfer_cycles=2)


def run_case(output, case, rank_staging=True, staging_banks=1, rank_descriptors=False, channel_independent_dispatch=False):
    import workload
    from simulator.backend import run_trace
    from audit_native import audit_run
    directory = output / case
    directory.mkdir()
    manifest = workload.generate(directory / "workload.trace", case, rank_staging=rank_staging, staging_banks=staging_banks,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch)
    dump(directory / "manifest.json", manifest)
    config = config_for(case, rank_staging=rank_staging, staging_banks=staging_banks,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch)
    native = run_trace(config, directory / "workload.trace", directory / "simulation")
    with (directory / "simulation/barriers.csv").open(newline="") as stream:
        markers = {int(r["marker"]): int(r["frontend_tick"]) for r in csv.DictReader(stream)}
    phases = {}
    for segment in manifest["segments"]:
        name = segment["name"]
        phases[name] = phases.get(name, 0) + (markers[segment["end"]]-markers[segment["start"]]) * native["tCK_ns"] / (1000*config.frontend_ratio)
    dq = sum(native["issued_commands"].get(c, 0)*32 for c in EXTERNAL)
    if dq != manifest["expected_dq_bytes"]["total"]:
        raise AssertionError((case, "external DQ differs", dq, manifest["expected_dq_bytes"]))
    summary = dict(case=case, horizon=16, fresh_native_run=True, unified_bank_gemv=True,
        rank_staging=rank_staging, rank_controller=rank_staging,
        rank_descriptors=config.pim_rank_descriptors, channel_independent_dispatch=config.channel_independent_dispatch,
        staging_banks=staging_banks if rank_staging else 0,
        latency_us=native["latency_ns"]/1000, memory_cycles=native["memory_cycles"],
        tCK_ns=native["tCK_ns"], phase_us=phases,
        phase_boundary_us=native["latency_ns"]/1000-sum(phases.values()),
        external_DQ_bytes=dq, traffic_bytes=native["traffic_bytes"],
        trace_sha256=digest(directory / "workload.trace"), numerical_compute=False,
        complete_end_to_end_latency_us=None)
    dump(directory / "summary.json", summary)
    audit = audit_run(directory)
    dump(directory / "native-audit.json", audit)
    summary["audit_status"] = "passed"
    dump(directory / "summary.json", summary)
    print(json.dumps(dict(case=case, latency_us=summary["latency_us"],
                         external_DQ_bytes=dq, audit="passed")), flush=True)
    return summary


def execute(label, argv, output, env):
    logs = output / "runner-logs"
    logs.mkdir(exist_ok=True)
    result = subprocess.run(list(map(str, argv)), cwd=HERE, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (logs / f"{label}.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {logs / (label+'.log')}\n{result.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--direct-input-baseline", action="store_true", help="Reproduce xinyao_1101 direct WRGB input behavior")
    parser.add_argument("--staging-banks", type=int, choices=(1, 2), default=2,
                        help="Number of DRAM input staging banks per PIM rank (default: 2)")
    parser.add_argument("--legacy-rank-dispatch", action="store_true",
                        help="Reproduce the original 1103 serial frontend and per-bank PIM requests")
    parser.add_argument("--no-rank-descriptors", action="store_true",
                        help="Keep channel-independent dispatch but use per-bank PIM requests for ablation")
    parser.add_argument("--skip-tests", action="store_true", help="For diagnostic runs only")
    args = parser.parse_args()
    channel_independent_dispatch = not (args.direct_input_baseline or args.legacy_rank_dispatch)
    rank_descriptors = channel_independent_dispatch and not args.no_rank_descriptors
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    library = backend / "ramulator2/libramulator.so"
    if not library.is_file():
        parser.error("Build the native backend first")
    if output.exists():
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True)
    env = setup(backend)
    source_files = []
    for folder in (HERE, backend / "simulator", backend / "ramulator2/src", backend / "ramulator2/python"):
        source_files += [p for p in folder.rglob("*") if p.is_file() and
            p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc", ".sh") and
            "results" not in p.relative_to(folder).parts]
    library_hash = digest(library)
    dump(output / "provenance.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        horizon=16, python=sys.executable, backend=str(backend),
        native_library_sha256=library_hash,
        rank_staging=not args.direct_input_baseline,
        staging_banks=args.staging_banks if not args.direct_input_baseline else 0,
        requested_staging_banks=args.staging_banks,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
        source_sha256={str(p): digest(p) for p in sorted(set(source_files))},
        fresh_native_simulation=True, historical_latency_used_as_oracle=False,
        policy=f"{args.staging_banks}-bank whole-input DRAM staging; rank_descriptors={rank_descriptors}; channel_independent_dispatch={channel_independent_dispatch}" if not args.direct_input_baseline else "Legacy direct WRGB bank GEMV fill"))
    if not args.skip_tests:
        execute("unit-tests", [sys.executable, "-m", "unittest", "discover", "-s",
                HERE / "validation", "-p", "test_*.py", "-v"], output, env)
        execute("native-guards", [sys.executable, HERE / "validation/native_guards.py",
                "--backend", backend, "--output", output / "native-guards"], output, env)
    runs = {case: run_case(output, case, rank_staging=not args.direct_input_baseline, staging_banks=args.staging_banks,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch) for case in args.cases}
    import workload
    complete = all(case in runs for case in PRIMARY)
    summary = dict(status="passed", horizon=16, attention_heads=8,
        unified_bank_gemv=True, primary_cases=list(PRIMARY), runs=runs,
        rank_staging=not args.direct_input_baseline, rank_controller=not args.direct_input_baseline,
        rank_descriptors=rank_descriptors, channel_independent_dispatch=channel_independent_dispatch,
        staging_banks=args.staging_banks if not args.direct_input_baseline else 0,
        down_rank_partial_reduction=False, input_broadcast_extra_cycles=0,
        measured_stage_sum_us_per_layer_step=sum(runs[c]["latency_us"] for c in PRIMARY) if complete else None,
        external_DQ_bytes_per_layer_step=sum(runs[c]["external_DQ_bytes"] for c in PRIMARY) if complete else None,
        FFN_combined_control_us=runs.get("combined_timing_chain", {}).get("latency_us"),
        serial_layer_memory_control_us=runs.get("serial_layer_control", {}).get("latency_us"),
        layouts=workload.layouts(rank_staging=not args.direct_input_baseline, staging_banks=args.staging_banks), complete_AE_end_to_end_latency_ms=None,
        actual_GPU_usable_time_us=None, unknown_latency_us={name: None for name in UNKNOWN},
        tests_executed=not args.skip_tests,
        aggregation="Five independently timed stages; combined FFN and serial-layer memory schedule are additional controls, excluded from the sum")
    summary["measured_stage_sum_ms_18x10"] = summary["measured_stage_sum_us_per_layer_step"]*.18 if complete else None
    if digest(library) != library_hash:
        raise AssertionError("Native library changed while running")
    dump(output / "summary.json", summary)
    if complete:
        from channel_analysis import write_report
        channels = write_report(output)
        summary["complete_zero_external_phase_us_all_channels"] = channels["complete_zero_external_phase_us_all_channels"]
        summary["tail_clear_complete_zero_external_phase_us_all_channels"] = channels["tail_clear_complete_zero_external_phase_us_all_channels"]
        summary["channel_analysis_file"] = "channel-analysis.json"
        dump(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "measured_stage_sum_us_per_layer_step", "external_DQ_bytes_per_layer_step")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

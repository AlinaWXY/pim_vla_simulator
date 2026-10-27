"""Fresh whole-horizon AE simulation with buffers expanded for H.

H is action-token count; attention heads remain8. No token microbatching or
scaled historical timings are used. Unknown GPU/NL boundaries remain unknown.
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
BASELINE = HERE.parent / "pi05-bank-rope-h16"
PRIMARY = ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram")
CASES = (*PRIMARY, "combined_timing_chain")
EXTERNAL = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}
GOLDEN = dict(chain=34.886096, v=10.282356, pv=9.189586,
              gate_up_z_store=36.195544, down_tiled_dram=33.920894,
              combined_timing_chain=70.797426)
UNKNOWN = ["GPU_V_cast_pack", "GPU_softmax_and_score_scaling", "GPU_O_projection",
           "GELU_multiply_cast", "GPU_FFN_partial_reduction", "normalization_residual",
           "action_input_output_and_step_update", "GPU_Q_forward_kernel_and_dispatch",
           "background_VLM_contention"]


def dump(path, value):
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(label, argv, output, env):
    print(f"[{label}]", flush=True)
    done = subprocess.run([str(x) for x in argv], cwd=HERE, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logs = output / "runner-logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{label}.log").write_bytes((json.dumps(dict(argv=[str(x) for x in argv],
        returncode=done.returncode)) + "\n" + done.stdout).encode())
    if done.returncode:
        print(done.stdout, file=sys.stderr)
        raise RuntimeError(f"{label} failed")


def run_case(root, horizon, case):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import horizon_attention
    import horizon_value
    import horizon_ffn
    import audit_horizon_attention
    import audit_horizon_value
    import audit_horizon_ffn

    directory = root / case
    directory.mkdir()
    if case == "chain":
        horizon_attention.configure(horizon)
        manifest = horizon_attention.generate(directory / "workload.trace", case="chain", schedule="double")
        auditor = audit_horizon_attention.audit_run
    elif case in ("v", "pv"):
        manifest = horizon_value.generate(directory / "workload.trace", horizon, case)
        auditor = audit_horizon_value.audit_run
    else:
        manifest = horizon_ffn.generate(directory / "workload.trace", horizon, case, local_ck=2)
        auditor = audit_horizon_ffn.audit_run
    dump(directory / "manifest.json", manifest)
    is_chain, is_ffn = case == "chain", case not in ("chain", "v", "pv")
    timings = dict(nCCDAB=horizon // 2 if is_chain else horizon, nLOCALWR=4)
    if is_chain:
        timings.update(nCCDK=2, nCCDQK=horizon)
    config = MemoryConfig(model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=True, pim=True, pim_input_broadcast=False, pim_mac_scope="rank",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=1_000_000,
        pim_horizon=horizon, pim_rank_rope=is_chain, pim_ffn_dram=is_ffn,
        ffn_z_preinstalled=(case == "down_tiled_dram"),
        rope_ii_cycles=1, rope_latency_cycles=4, pim_cast_cycles=1, rank_reduce_cycles=4,
        timing_overrides=timings, pim_local_transfer_cycles=2)
    native = run_trace(config, directory / "workload.trace", directory / "simulation")
    with (directory / "simulation/barriers.csv").open(newline="") as stream:
        markers = {int(r["marker"]): int(r["frontend_tick"]) for r in csv.DictReader(stream)}
    phases = {}
    for segment in manifest["segments"]:
        name = segment["name"]
        phases[name] = phases.get(name, 0) + (markers[segment["end"]] - markers[segment["start"]]) * native["tCK_ns"] / (1000 * config.frontend_ratio)
    dq = sum(native["issued_commands"].get(c, 0) * 32 for c in EXTERNAL)
    assert dq == manifest["expected_dq_bytes"]["total"]
    summary = dict(case=case, horizon=horizon, fresh_native_run=True,
        latency_us=native["latency_ns"] / 1000, memory_cycles=native["memory_cycles"],
        tCK_ns=native["tCK_ns"], phase_us=phases,
        phase_boundary_us=native["latency_ns"] / 1000 - sum(phases.values()),
        external_DQ_bytes=dq, traffic_bytes=native["traffic_bytes"],
        trace_sha256=digest(directory / "workload.trace"), numerical_compute=False,
        complete_end_to_end_latency_us=None)
    dump(directory / "summary.json", summary)
    audit = auditor(directory)
    dump(directory / "native-audit.json", audit)
    summary["audit_status"] = "passed"
    if horizon == 16:
        assert abs(summary["latency_us"] - GOLDEN[case]) < 1e-9, (case, summary["latency_us"], GOLDEN[case])
        summary["exact_H16_regression"] = True
    dump(directory / "summary.json", summary)
    print(json.dumps(dict(horizon=horizon, case=case, latency_us=summary["latency_us"],
                         external_DQ_bytes=dq, audit="passed")), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(16, 20, 30, 40), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    args = parser.parse_args()
    backend, output, h = args.backend.resolve(strict=True), args.output.resolve(), args.horizon
    if output.exists():
        parser.error("Use a fresh output directory")
    library = backend / "ramulator2/libramulator.so"
    if not library.is_file():
        parser.error("Build the backend first")
    output.mkdir(parents=True)
    sys.path[:0] = [str(HERE / "source"), str(HERE / "validation"), str(BASELINE / "validation"),
                   str(backend), str(backend / "ramulator2/python")]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "source"), str(HERE / "validation"),
        str(backend / "ramulator2/python"), str(backend)])
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2")
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.sbatch"))
    for folder in (HERE / "source", HERE / "validation", BASELINE / "source", BASELINE / "validation",
                   backend / "simulator", backend / "ramulator2/src", backend / "ramulator2/python"):
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc"))
    files.extend([BASELINE / "validation/analyze_channel_windows.py", BASELINE / "build.sh"])
    library_hash = digest(library)
    dump(output / "provenance.json", dict(horizon=h, started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, backend=str(backend), native_library_sha256=library_hash,
        source_sha256={str(p): digest(p) for p in sorted(set(files))},
        baseline_commit="41199cee64c40c55cafb330ed838d7e5c13fc0b8", baseline_tag="xinyao_1027_h16",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), fresh_native_simulation=True,
        policy="Whole H; scale token buffers and MAC service, preserve weight ownership and old GPU boundaries"))
    if h == 16:
        for label, folder in (("source-tests", HERE / "source"), ("audit-tests", HERE / "validation")):
            execute(label, [sys.executable, "-m", "unittest", "discover", "-s", folder, "-p", "test_*.py", "-v"], output, env)
        execute("legacy-source-tests", [sys.executable, "-m", "unittest", "discover", "-s", BASELINE / "source", "-p", "test_*.py", "-v"], output, env)
        execute("native-ffn-guards", [sys.executable, BASELINE / "validation/test_ffn_dram_native_rejections.py", "--backend", backend, "--output", output / "native-ffn-guards"], output, env)
        execute("native-rope-guards", [sys.executable, BASELINE / "validation/test_native_rejections.py", "--backend", backend, "--output", output / "native-rope-guards"], output, env)
        execute("native-horizon-guards", [sys.executable, HERE / "validation/test_horizon_native_guards.py", "--backend", backend, "--output", output / "native-horizon-guards"], output, env)
    runs = {case: run_case(output, h, case) for case in CASES}
    import horizon_attention
    import horizon_value
    import horizon_ffn
    summary = dict(status="passed", horizon=h, attention_heads=8, token_microbatches=1,
        primary_cases=list(PRIMARY), runs=runs,
        measured_stage_sum_us_per_layer_step=sum(runs[c]["latency_us"] for c in PRIMARY),
        external_DQ_bytes_per_layer_step=sum(runs[c]["external_DQ_bytes"] for c in PRIMARY),
        FFN_combined_control_us=runs["combined_timing_chain"]["latency_us"],
        layouts=dict(attention=horizon_attention.layout_for(h), value=horizon_value.layout(h), ffn=horizon_ffn.layout(h)),
        complete_AE_end_to_end_latency_ms=None, actual_GPU_usable_time_us=None,
        unknown_latency_us={name: None for name in UNKNOWN},
        aggregation="Sum of five separately timed primary stages; combined FFN is a dependency control, excluded from sum")
    summary["measured_stage_sum_ms_18x10"] = summary["measured_stage_sum_us_per_layer_step"] * .18
    dump(output / "summary.json", summary)
    from analyze_channel_windows import analyze
    channel = analyze(output)
    dump(output / "channel-window-summary.json", channel)
    from verify_channel_windows import verify_run
    independent = verify_run(output)
    dump(output / "channel-window-independent-audit.json", independent)
    summary["zero_external_command_phase_us"] = channel["zero_external_command_phase_us"]
    summary["zero_external_command_phase_fraction"] = channel["zero_external_command_phase_us"] / channel["stage_sum_us"]
    summary["channel_payload_equivalent_spare_us"] = channel["payload_equivalent_spare_us"]
    assert digest(library) == library_hash
    dump(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("horizon", "status", "measured_stage_sum_us_per_layer_step", "zero_external_command_phase_us", "external_DQ_bytes_per_layer_step")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

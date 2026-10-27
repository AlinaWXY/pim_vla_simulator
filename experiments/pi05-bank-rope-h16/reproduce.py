"""Build-independent, fresh native simulation of the H16 bank-local RoPE design.

Run build.sh first, with the same Python used here.  No historical latency is
substituted for a native run.  Pipeline delays are explicit design assumptions.
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


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(label, argv, output, env):
    print(f"[{label}]", flush=True)
    result = subprocess.run([str(a) for a in argv], cwd=HERE, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logs = output / "runner-logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{label}.log").write_text(
        json.dumps({"argv": [str(a) for a in argv], "returncode": result.returncode})
        + "\n" + result.stdout, encoding="utf-8")
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"{label} failed: see {logs / (label + '.log')}")
    return result.stdout


def run_chain(output, label, rope_latency, rope_ii, reduce_latency):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import attention_workload as workload

    case = output / label
    case.mkdir()
    manifest = workload.generate(case / "workload.trace", case="chain", schedule="double")
    dump(case / "manifest.json", manifest)
    config = MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=True, pim=True, pim_input_broadcast=False, pim_mac_scope="rank",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=100_000_000,
        pim_rank_rope=True, rope_ii_cycles=rope_ii, rope_latency_cycles=rope_latency,
        pim_cast_cycles=1, rank_reduce_cycles=reduce_latency,
        timing_overrides={"nCCDAB": 8, "nCCDK": 2, "nCCDQK": 16, "nLOCALWR": 4},
        pim_local_transfer_cycles=2,
    )
    result = run_trace(config, case / "workload.trace", case / "simulation")
    assert result["requests"]["sha256"] == digest(case / "workload.trace")
    markers = {int(row["marker"]): int(row["frontend_tick"]) for row in
               csv.DictReader((case / "simulation/barriers.csv").open())}
    phases = {}
    for segment in manifest.get("segments", []):
        cycles = (markers[segment["end"]] - markers[segment["start"]]) / config.frontend_ratio
        phases[segment["name"]] = phases.get(segment["name"], 0.0) + cycles * result["tCK_ns"] / 1000
    summary = {
        "case": label, "fresh_native_run": True,
        "latency_us": result["latency_ns"] / 1000,
        "memory_cycles": result["memory_cycles"], "phase_us": phases,
        "phase_boundary_us": result["latency_ns"] / 1000 - sum(phases.values()),
        "tCK_ns": result["tCK_ns"], "traffic_bytes": result["traffic_bytes"],
        "assumptions": {"rope_ii_ck": rope_ii, "rope_pipeline_latency_ck": rope_latency,
                        "rank_reduce_latency_ck": reduce_latency, "projection_cast_ck_per_col": 1,
                        "gpu_forward_compute_latency": "not modeled; DQ transfers are modeled",
                        "background_vlm_traffic": "not supplied; rank0 reserved",
                        "weights_coefficients_prefix_K": "preinstalled, excluded from steady-state timing"},
        "numerical_compute": False, "trace_sha256": digest(case / "workload.trace"),
    }
    stages = {}
    for name, value in phases.items():
        group = ("qkt_compute_and_reduce" if name.startswith("qk_tile_") and name.endswith("_compute")
                 else "score_export" if name.startswith("qk_tile_") and name.endswith("_export") else name)
        stages[group] = stages.get(group, 0.0) + value
    summary["stage_us"] = stages
    dump(case / "summary.json", summary)
    print(json.dumps({"case": label, "latency_us": summary["latency_us"], "stage_us": stages}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rope-latency-ck", type=int, default=4)
    parser.add_argument("--rope-ii-ck", type=int, default=1)
    parser.add_argument("--rank-reduce-ck", type=int, default=4)
    parser.add_argument("--sensitivity", nargs="*", type=int, default=[])
    parser.add_argument("--check-baseline", action="store_true")
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if not (backend / "ramulator2/libramulator.so").is_file():
        parser.error("Build the native backend first with build.sh")
    if output.exists():
        parser.error("Use a fresh output directory; existing results are never overwritten")
    output.mkdir(parents=True)
    sys.path[:0] = [str(HERE / "source"), str(backend), str(backend / "ramulator2/python")]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(backend / "ramulator2/python"), str(backend)])
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2")
    files = list((HERE / "source").glob("*.py")) + list((HERE / "validation").glob("*.py")) + [Path(__file__), HERE / "build.sh"]
    for part in ("simulator", "ramulator2/src", "ramulator2/python"):
        files.extend(p for p in (backend / part).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc"))
    files.append(backend / "ramulator2/libramulator.so")
    dump(output / "provenance.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "backend": str(backend), "source_sha256": {str(p): digest(p) for p in sorted(set(files))},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "fresh_native_simulation": True,
        "reference_results_used_as_measurement": False,
    })
    execute("workload-tests", [sys.executable, "-m", "unittest", "discover", "-s", HERE / "source", "-p", "test_*.py", "-v"], output, env)
    execute("audit-unit-tests", [sys.executable, "-m", "unittest", "discover", "-s", HERE / "validation", "-p", "test_audit.py", "-v"], output, env)
    runs = []
    values = [args.rope_latency_ck] + [v for v in args.sensitivity if v != args.rope_latency_ck]
    for index, latency in enumerate(dict.fromkeys(values)):
        label = "chain" if index == 0 else f"chain-rope-latency-{latency}"
        runs.append(run_chain(output, label, latency, args.rope_ii_ck, args.rank_reduce_ck))
        execute(f"{label}-audit", [sys.executable, HERE / "validation/audit_run.py", output / label], output, env)
    execute("audit-fault-detection", [sys.executable, HERE / "validation/test_audit_detection.py", output / "chain"], output, env)
    execute("native-rejections", [sys.executable, HERE / "validation/test_native_rejections.py",
            "--backend", backend, "--output", output / "native-rejections"], output, env)
    baseline = None
    if args.check_baseline:
        execute("legacy-baseline", [sys.executable, HERE.parent / "pi05-bgk-gpuq-h16/source/run_attention.py",
                "--backend", backend, "--output", output / "legacy-baseline", "--cases", "qkv", "qk"], output, env)
        baseline = {}
        for case, expected in (("qkv", 16.751742), ("qk", 14.305438)):
            value = json.loads((output / "legacy-baseline" / case / "summary.json").read_text())["latency_us"]
            if abs(value - expected) > 1e-9:
                raise AssertionError(f"Legacy {case} changed: {value} vs {expected}")
            baseline[case] = {"latency_us": value, "reference_us": expected, "passed": True}
    dump(output / "summary.json", {"scope": "Q/K projection, bank-local RoPE, GPU-boundary Q exchange, rank-local QKT and score export",
          "runs": runs, "legacy_regression": baseline, "status": "passed",
          "baseline_not_a_speedup_comparator": "Old baseline has a different scope and rank allocation"})
    lines = ["# H16 bank-local RoPE native simulation", "", "Fresh native timing; pipeline latencies are configurable architectural assumptions.", "",
             "| Run | RoPE latency / CK | Total / us |", "|---|---:|---:|"]
    for run in runs:
        lines.append(f"| {run['case']} | {run['assumptions']['rope_pipeline_latency_ck']} | {run['latency_us']:.6f} |")
    lines += ["", "| Main-run stage | Time / us |", "|---|---:|"]
    lines += [f"| {name} | {value:.6f} |" for name, value in runs[0]["stage_us"].items()]
    audit = json.loads((output / "chain/native-audit.json").read_text())
    lines += ["", "Native external DQ payload:", "", "```json", json.dumps(audit["external_payload_bytes"], indent=2), "```"]
    lines += ["", "The scope excludes weight/coefficients/prefix installation, V/PV/softmax/O, GPU forwarding kernel execution, and background VLM traffic.",
              "", "See each case's native-audit.json, simulation/result.json, command timeline and bank-local event trace."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[complete] {output}", flush=True)


if __name__ == "__main__":
    main()

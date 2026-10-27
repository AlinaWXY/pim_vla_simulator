"""Rerun every previously modeled AE stage on one PIM rank/channel.

Q/K use the bank-local RoPE chain. V/PV and FFN retain the previous H16
operator schedule, remapped to physical rank1 with full matrix work preserved.
GPU/nonlinear boundaries without a timing model remain explicit unknowns.
"""
import argparse
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import sys

from reproduce import digest, dump, execute, run_chain

HERE = Path(__file__).resolve().parent
CASES = ("chain", "v", "pv", "gate_up", "down_tiled")
LABELS = {
    "chain": "Q/K projection + local RoPE + Q exchange + QKT/score export",
    "v": "V projection + export + suffix cache update",
    "pv": "PV + output export",
    "gate_up": "FFN Gate/Up + assumed local gather",
    "down_tiled": "FFN Down + 16-rank partial export",
}
UNKNOWN = {
    "GPU_V_cast_pack": None,
    "GPU_softmax_and_score_scaling": None,
    "GPU_O_projection": None,
    "GELU_multiply_cast": None,
    "GPU_FFN_partial_reduction": None,
    "normalization_residual": None,
    "action_input_output_and_step_update": None,
    "GPU_Q_forward_kernel_and_dispatch": None,
    "background_VLM_contention": None,
}


def run_rank_case(output, case):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import audit_rank_stages
    import value_workload
    import ffn_workload

    directory = output / case
    directory.mkdir()
    if case in ("v", "pv"):
        manifest = value_workload.generate(directory / "workload.trace", case=case)
    else:
        manifest = ffn_workload.generate(directory / "workload.trace", case, local_ck=2)
    dump(directory / "manifest.json", manifest)
    config = MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=True, pim=True, pim_input_broadcast=False, pim_mac_scope="rank",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=100_000_000,
        pim_rank_rope=False, timing_overrides={"nCCDAB": 16},
        pim_local_transfer_cycles=2,
    )
    result = run_trace(config, directory / "workload.trace", directory / "simulation")
    markers = {int(row["marker"]): int(row["frontend_tick"]) for row in
               csv.DictReader((directory / "simulation/barriers.csv").open())}
    phases = {}
    for segment in manifest["segments"]:
        duration = ((markers[segment["end"]] - markers[segment["start"]])
                    * result["tCK_ns"] / (1000 * config.frontend_ratio))
        phases[segment["name"]] = phases.get(segment["name"], 0) + duration
    summary = {
        "case": case, "fresh_native_run": True,
        "latency_us": result["latency_ns"] / 1000,
        "memory_cycles": result["memory_cycles"], "tCK_ns": result["tCK_ns"],
        "phase_us": phases,
        "phase_boundary_us": result["latency_ns"] / 1000 - sum(phases.values()),
        "traffic_bytes": result["traffic_bytes"],
        "trace_sha256": digest(directory / "workload.trace"),
        "numerical_compute": False,
        "active_PIM_ranks": 16, "active_PIM_banks": 256,
        "reserved_GPU_rank": 0, "physical_PIM_rank": 1,
        "legacy_completion": "MAC issue retirement + explicit WAIT16; result tail WAIT4",
    }
    dump(directory / "summary.json", summary)
    audit = audit_rank_stages.audit_run(directory)
    dump(directory / "native-audit.json", audit)
    summary["audit_status"] = "passed"
    summary["external_DQ_bytes"] = sum(result["issued_commands"].get(cmd, 0) * 32
        for cmd in ("RD", "RDA", "WR", "WRA", "WRGB", "RDRES"))
    dump(directory / "summary.json", summary)
    print(json.dumps({"case": case, "latency_us": summary["latency_us"],
                      "external_DQ_bytes": summary["external_DQ_bytes"], "audit": "passed"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rope-latency-ck", type=int, default=4)
    parser.add_argument("--rope-ii-ck", type=int, default=1)
    parser.add_argument("--rank-reduce-ck", type=int, default=4)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.layers <= 0 or args.steps <= 0:
        parser.error("layers and steps must be positive")
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if not (backend / "ramulator2/libramulator.so").is_file():
        parser.error("Build native backend using build.sh first")
    if output.exists():
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True)
    sys.path[:0] = [str(HERE / "source"), str(HERE / "validation"),
                   str(backend), str(backend / "ramulator2/python")]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(backend / "ramulator2/python"), str(backend)])
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2")
    files = [HERE / "reproduce.py", Path(__file__), HERE / "build.sh"]
    files += list((HERE / "source").glob("*.py")) + list((HERE / "validation").glob("*.py"))
    for part in ("simulator", "ramulator2/src", "ramulator2/python"):
        files += [p for p in (backend / part).rglob("*")
                  if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc")]
    library = backend / "ramulator2/libramulator.so"
    files.append(library)
    library_hash = digest(library)
    dump(output / "provenance.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "backend": str(backend), "source_sha256": {str(p): digest(p) for p in sorted(set(files))},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "fresh_native_simulation": True,
        "reference_results_used_as_measurement": False,
        "policy": "Preserve prior GPU/nonlinear boundaries; migrate every modeled AE stage to physical rank1",
    })
    execute("workload-tests", [sys.executable, "-m", "unittest", "discover", "-s", HERE / "source", "-p", "test_*.py", "-v"], output, env)
    for pattern in ("test_audit.py", "test_rank_stages.py"):
        execute(pattern.removesuffix(".py"), [sys.executable, "-m", "unittest", "discover",
                "-s", HERE / "validation", "-p", pattern, "-v"], output, env)
    runs = {"chain": run_chain(output, "chain", args.rope_latency_ck, args.rope_ii_ck, args.rank_reduce_ck)}
    execute("chain-audit", [sys.executable, HERE / "validation/audit_run.py", output / "chain"], output, env)
    execute("chain-fault-detection", [sys.executable, HERE / "validation/test_audit_detection.py", output / "chain"], output, env)
    chain_audit = json.loads((output / "chain/native-audit.json").read_text())
    runs["chain"]["external_DQ_bytes"] = chain_audit["external_payload_bytes"]["total"]
    runs["chain"]["audit_status"] = "passed"
    for case in CASES[1:]:
        runs[case] = run_rank_case(output, case)
    if digest(library) != library_hash:
        raise AssertionError("Native library changed during the run")
    total_us = sum(run["latency_us"] for run in runs.values())
    total_bytes = sum(run["external_DQ_bytes"] for run in runs.values())
    if total_bytes != 3872 * 1024:
        raise AssertionError(f"Independent full-stage DQ contract differs: {total_bytes}")
    summary = {
        "status": "passed", "horizon": 16, "attention_heads": 8,
        "physical_ranks_per_channel": 2, "active_PIM_ranks_per_channel": 1,
        "active_PIM_ranks": 16, "active_PIM_banks": 256,
        "runs": runs, "measured_stage_sum_us_per_layer_step": total_us,
        "layers": args.layers, "denoising_steps": args.steps,
        "measured_stage_sum_ms": total_us * args.layers * args.steps / 1000,
        "external_DQ_bytes_per_layer_step": total_bytes,
        "external_DQ_bytes_across_layers_steps": total_bytes * args.layers * args.steps,
        "complete_AE_end_to_end_latency_ms": None,
        "unknown_latency_us": UNKNOWN,
        "aggregation": "Separately timed native stages, following the previous AE accounting; no cross-stage overlap or continuous refresh-state claim",
        "V_projection": "Separate full16-query pass; extra X transfer is charged, not silently fused into token-sharded Q/K",
        "FFN_local_gather": "Inherited analytical 32B/2CK port model: 2048CK/rank; nonlinear execution is unknown",
        "initialization": "Resident weights, prefix KV, RoPE coefficients and padding are excluded from steady-state timing",
    }
    dump(output / "summary.json", summary)
    dump(output / "ae-stage-coverage.json", {
        "modeled": {case: LABELS[case] for case in CASES}, "unmeasured": UNKNOWN,
        "dependency_order": ["Q/K and V projection", "Q/K local RoPE", "Q exchange and QKT",
            "GPU softmax/scaling (unknown)", "PV", "GPU O/norm/residual (unknown)",
            "Gate/Up", "GELU/multiply/cast (unknown)", "Down",
            "GPU rank partial reduction/norm/residual (unknown)"],
        "coverage_policy": "All stages that had native timing in the previous H16 AE code are included",
    })
    lines = ["# H16 AE: all existing native stages on one PIM rank/channel", "",
             "Fresh Ramulator runs. Physical rank0 is reserved for GPU; only rank1 executes PIM.", "",
             "| Stage | us / layer / step | External DQ / KiB |", "|---|---:|---:|"]
    for case in CASES:
        lines.append(f"| {LABELS[case]} | {runs[case]['latency_us']:.6f} | {runs[case]['external_DQ_bytes']/1024:g} |")
    lines += [f"| **Measured stage sum** | **{total_us:.6f}** | **{total_bytes/1024:g}** |", "",
              f"The measured stage sum over {args.layers} layers and {args.steps} denoising steps is **{summary['measured_stage_sum_ms']:.6f} ms**.", "",
              "These are separately timed stages, as in the previous code. The sum does not include unknown GPU/nonlinear execution or background VLM contention and is not a complete end-to-end AE latency.", "",
              "## Mapping and assumptions", "",
              "- Q/K projection, bank-local RoPE, Q forwarding and QKT retain the agreed layout and native dependencies.",
              "- V/PV keep feature sharding, eight outputs/bank, with every request mapped to rank1. V uses a separate X pass; its input traffic is included.",
              "- Gate/Up keeps one copy of each weight: 256 intermediate features/rank and 2KiB accumulators/bank. Down keeps four output tiles, 256 local reduction terms and 16 exported rank partials. Z needs 8KiB/rank.",
              "- V/PV and FFN preserve the legacy 16CK MAC service and explicit completion waits. The Q/K projection computes eight local tokens per rank with 8CK service.",
              "- FFN gather retains the old analytical local-port model, now 2048CK/rank. GELU/multiply/cast and GPU partial reduction are not modeled.",
              "- RoPE uses one pair/CK and configurable pipeline latency; current latency is " + str(args.rope_latency_ck) + "CK.",
              "- Weights, prefix KV and coefficients are preinstalled; no tensor numerical calculation is performed.", "",
              "## Unmeasured stages retained from the prior mapping", ""]
    lines += [f"- {name}" for name in UNKNOWN]
    lines += ["", "Each case includes manifest.json, workload.trace, native-audit.json and simulation/ timelines. Source and compiled-library hashes are in provenance.json."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output),
                      "measured_us_per_layer_step": total_us,
                      "measured_ms_18x10": summary["measured_stage_sum_ms"],
                      "external_DQ_KiB": total_bytes / 1024}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Fresh AE timing with FFN Z resident in bank DRAM, not an 8KiB rank SRAM.

The previous SRAM workload is retained as a regression control. Unknown
nonlinear arithmetic remains an explicit boundary; all Z movement is timed.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from reproduce import digest, dump, execute, run_chain
from reproduce_ae import run_rank_case, UNKNOWN

HERE = Path(__file__).resolve().parent
FFN_CASES = ("gate_up_z_store", "down_tiled_dram", "combined_timing_chain")


def run_ffn_dram(output, case):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import ffn_dram_workload
    import audit_ffn_dram

    directory = output / case
    directory.mkdir()
    manifest = ffn_dram_workload.generate(directory / "workload.trace", case=case, local_ck=2)
    dump(directory / "manifest.json", manifest)
    config = MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=True, pim=True, pim_input_broadcast=False, pim_mac_scope="rank",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=1_000_000,
        pim_rank_rope=False, pim_ffn_dram=True,
        ffn_z_preinstalled=(case == "down_tiled_dram"),
        timing_overrides={"nCCDAB": 16, "nLOCALWR": 4}, pim_local_transfer_cycles=2,
    )
    result = run_trace(config, directory / "workload.trace", directory / "simulation")
    with (directory / "simulation/barriers.csv").open() as stream:
        markers = {int(r["marker"]): int(r["frontend_tick"]) for r in csv.DictReader(stream)}
    phases = {}
    for segment in manifest["segments"]:
        duration = ((markers[segment["end"]] - markers[segment["start"]])
                    * result["tCK_ns"] / (1000 * config.frontend_ratio))
        phases[segment["name"]] = phases.get(segment["name"], 0) + duration
    commands = result["issued_commands"]
    dq = sum(commands.get(cmd, 0) * 32 for cmd in ("RD", "RDA", "WR", "WRA", "WRGB", "RDRES"))
    summary = dict(case=case, fresh_native_run=True, latency_us=result["latency_ns"] / 1000,
        phase_us=phases, phase_boundary_us=result["latency_ns"] / 1000 - sum(phases.values()),
        memory_cycles=result["memory_cycles"], tCK_ns=result["tCK_ns"], external_DQ_bytes=dq,
        Z_local_DRAM_read_bytes=commands.get("LOCALRD", 0) * 32,
        Z_local_DRAM_write_bytes=commands.get("LOCALWR", 0) * 32,
        traffic_bytes=result["traffic_bytes"], numerical_compute=False,
        full_batch_Z_SRAM_bytes_per_rank=0, DRAM_Z_bytes_per_rank=8192,
        rank_MAC_input_double_buffer_bytes=1024, bank_column_staging_bytes=32,
        streaming_rank_nonlinear_input_bytes=128, streaming_rank_nonlinear_output_bytes=32,
        nonlinear_latency_us=None, GPU_partial_reduction_latency_us=None,
        Z_preinstalled=config.ffn_z_preinstalled,
        trace_sha256=digest(directory / "workload.trace"))
    dump(directory / "summary.json", summary)
    audit = audit_ffn_dram.audit_run(directory)
    dump(directory / "native-audit.json", audit)
    summary["audit_status"] = "passed"
    dump(directory / "summary.json", summary)
    print(json.dumps({"case": case, "latency_us": summary["latency_us"],
        "external_DQ_bytes": dq, "Z_read_bytes": summary["Z_local_DRAM_read_bytes"],
        "Z_write_bytes": summary["Z_local_DRAM_write_bytes"], "audit": "passed"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    library = backend / "ramulator2/libramulator.so"
    if not library.is_file():
        parser.error("Build the native backend first")
    if output.exists():
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True)
    sys.path[:0] = [str(HERE / "source"), str(HERE / "validation"),
                   str(backend), str(backend / "ramulator2/python")]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(backend / "ramulator2/python"), str(backend)])
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2")
    sources = list(HERE.glob("reproduce*.py")) + [HERE / "build.sh"]
    sources += list((HERE / "source").glob("*.py")) + list((HERE / "validation").glob("*.py"))
    for part in ("simulator", "ramulator2/src", "ramulator2/python"):
        sources += [p for p in (backend / part).rglob("*")
                    if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".inc")]
    library_hash = digest(library)
    dump(output / "provenance.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, backend=str(backend), backend_library_sha256=library_hash,
        source_sha256={str(p): digest(p) for p in sorted(set(sources))},
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), fresh_native_simulation=True,
        reference_results_used_as_measurement=False))
    execute("workload-tests", [sys.executable, "-m", "unittest", "discover", "-s", HERE / "source", "-p", "test_*.py", "-v"], output, env)
    for pattern in ("test_audit.py", "test_rank_stages.py", "test_ffn_dram_audit.py"):
        execute(pattern.removesuffix(".py"), [sys.executable, "-m", "unittest", "discover",
                "-s", HERE / "validation", "-p", pattern, "-v"], output, env)
    execute("ffn-native-rejections", [sys.executable, HERE / "validation/test_ffn_dram_native_rejections.py",
            "--backend", backend, "--output", output / "native-rejections"], output, env)
    runs = {"chain": run_chain(output, "chain", 4, 1, 4)}
    execute("chain-audit", [sys.executable, HERE / "validation/audit_run.py", output / "chain"], output, env)
    runs["chain"]["external_DQ_bytes"] = json.loads((output / "chain/native-audit.json").read_text())["external_payload_bytes"]["total"]
    for case in ("v", "pv"):
        runs[case] = run_rank_case(output, case)
    for case in FFN_CASES:
        runs[case] = run_ffn_dram(output, case)
    controls = output / "sram-regression"
    controls.mkdir()
    old = {case: run_rank_case(controls, case) for case in ("gate_up", "down_tiled")}
    regression = {}
    for case, measured, expected in (
        ("chain", runs["chain"]["latency_us"], 34.886096),
        ("v", runs["v"]["latency_us"], 10.282356),
        ("pv", runs["pv"]["latency_us"], 9.189586),
        ("gate_up", old["gate_up"]["latency_us"], 35.675892),
        ("down_tiled", old["down_tiled"]["latency_us"], 24.640322)):
        if abs(measured - expected) > 1e-9:
            raise AssertionError(f"Legacy regression: {case} {measured} != {expected}")
        regression[case] = dict(latency_us=measured, expected_us=expected, passed=True)
    dump(output / "legacy-regression.json", regression)
    if digest(library) != library_hash:
        raise AssertionError("Native library changed during simulation")
    primary = ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram")
    total_us = sum(runs[c]["latency_us"] for c in primary)
    dq = sum(runs[c]["external_DQ_bytes"] for c in primary)
    if dq != 3872 * 1024:
        raise AssertionError(f"External DQ budget changed unexpectedly: {dq}")
    ffn_us = runs["gate_up_z_store"]["latency_us"] + runs["down_tiled_dram"]["latency_us"]
    old_ffn = old["gate_up"]["latency_us"] + old["down_tiled"]["latency_us"]
    summary = dict(status="passed", horizon=16, attention_heads=8, runs=runs,
        primary_cases=list(primary), legacy_regression=regression,
        measured_stage_sum_us_per_layer_step=total_us, measured_stage_sum_ms_18x10=total_us * .18,
        FFN_separate_stage_sum_us=ffn_us, FFN_previous_SRAM_stage_sum_us=old_ffn,
        FFN_delta_us=ffn_us - old_ffn,
        FFN_combined_transfer_chain_us=runs["combined_timing_chain"]["latency_us"],
        external_DQ_bytes_per_layer_step=dq, full_batch_Z_SRAM_bytes_per_rank=0,
        Z_DRAM_payload_per_bank=512, Z_DRAM_reserved_per_bank=2048,
        Z_DRAM_write_bytes_per_layer_step=128*1024, Z_DRAM_read_bytes_per_layer_step=512*1024,
        complete_AE_end_to_end_latency_ms=None, unknown_latency_us=UNKNOWN,
        aggregation="Same separate-stage sum as prior AE report. Combined FFN transfer chain is an additional dependency control, not added again.",
        nonlinear_boundary="128B G/U gather and32B Z return are timed per column; unknown nonlinear arithmetic is excluded. No full-batch GU/Z SRAM is assumed.",
        initialization="Resident weights and prefix KV; standalone Down explicitly declares resident Z; combined FFN requires earlier native stores.")
    dump(output / "summary.json", summary)
    lines = ["# H16 AE with FFN Z in bank DRAM", "",
        "All timings are fresh native simulations. This is a sum of separately timed modeled stages, not complete end-to-end AE latency.", "",
        "| Stage | us/layer/step | External DQ/KiB |", "|---|---:|---:|"]
    for c in primary:
        lines.append(f"| {c} | {runs[c]['latency_us']:.6f} | {runs[c]['external_DQ_bytes']/1024:g} |")
    lines += [f"| **Measured sum** | **{total_us:.6f}** | **{dq/1024:g}** |", "",
        f"Measured stage sum x18 layers x10 steps: **{total_us*.18:.6f} ms**.", "",
        f"FFN separate-stage subtotal: **{ffn_us:.6f} us**, previously **{old_ffn:.6f} us** with SRAM. Delta: **{ffn_us-old_ffn:+.6f} us**.",
        f"Combined FFN transfer chain: **{runs['combined_timing_chain']['latency_us']:.6f} us**; checked as a dependency control and excluded from the above sum.", "",
        "## Storage and dataflow", "",
        "- Weight ownership is unchanged: all16 tokens/rank,256 intermediate features/rank,16 Down partials returned to GPU. No weight replication is added.",
        "- Z occupies512B/bank in row176 (byte base360448), columns0..15. Each column holds one feature across16 tokens. One2KiB DRAM row/bank is reserved.",
        "- Each store streams128B G/U from the bank accumulator through a rank-local nonlinear boundary, returns32B Z to that bank, then performs a real local DRAM write. Unknown nonlinear compute is not assigned a hardware latency.",
        "- Down reads one column from each of16 banks for each input fragment. A bounded rank-local path transposes these into the512B MAC input slice. All16 source banks must be ready before MAC.",
        "- The1KiB input double buffer remains. There is no8KiB full-Z SRAM. Temporary storage is32B/bank plus128B nonlinear input and32B output staging/rank.",
        "- These buffer sizes do not include the existing accumulators: Gate/Up retains2KiB/bank and tiled Down retains1KiB/bank.",
        "- Native ACT/PRE, LOCALRD/LOCALWR latency and write recovery, row conflicts, shared local ports and buffer ownership are modeled. Local-path throughput remains a configurable32B/2CK assumption.",
        "- Array preparation follows trace order to prevent conflicting Z/weight activations from livelocking. Read completion and local-port delivery may overlap another input buffer's MAC. This is the measured current schedule, not the minimum latency obtainable with reordered prefetch.",
        "- Z adds128KiB local array writes and512KiB local array reads per layer/step; these stay in the producing rank and do not traverse external DQ. Rank-internal links are an architectural assumption.", "",
        "## Unmeasured boundaries", "",
        *[f"- {name}" for name in UNKNOWN], "",
        "## Evidence", "",
        "Each case includes the source trace, manifest, per-channel native timelines, local-transfer events and independent audit. Old SRAM cases and unchanged attention stages pass exact timing regression. Native invalid-sequence tests are in native-rejections/."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "AE_measured_us": total_us,
        "FFN_measured_us": ffn_us, "FFN_delta_us": ffn_us-old_ffn, "DQ_KiB": dq/1024}, indent=2), flush=True)


if __name__ == "__main__":
    main()

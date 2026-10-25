"""Rerun frozen FFN schedules on the new native backend, then relocate weights.

The old traces are regression controls.  Primary cases reserve 44 KiB/bank
for the new attention mapping.  No old result is reused as a new measurement.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relocate(reserved):
    import ffn_layout as layout
    assert reserved >= 44 * 1024 and reserved % layout.ROW_BYTES == 0
    layout.ATTENTION_BYTES_PER_BANK = reserved
    layout.GU_BASE = reserved
    layout.DOWN_BASE = layout.GU_BASE + layout.GU_BYTES_PER_BANK
    layout.LAYER_STRIDE_BYTES_PER_BANK = layout.DOWN_BASE + layout.DOWN_BYTES_PER_BANK
    updated = layout.build_layout()
    # ffn_workload imports this dictionary by reference.
    layout.REGIONS.clear()
    layout.REGIONS.update(updated["regions"])
    layout.LAYOUT = updated
    return updated


def run_case(root, name, op):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import ffn_workload
    import audit_ffn
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = ffn_workload.generate(directory / "workload.trace", op, local_ck=2)
    dump(directory / "manifest.json", manifest)
    config = MemoryConfig(model="lpddr5x-8533", channels=16, ranks=2, density_gbit=8,
        refresh=True, pim=True, pim_input_broadcast=True, pim_mac_scope="channel",
        pim_data_path="attacc", frontend_ratio=32, max_cycles=100_000_000,
        timing_overrides={"nCCDAB": 20}, pim_local_transfer_cycles=2)
    result = run_trace(config, directory / "workload.trace", directory / "simulation")
    barriers = {int(r["marker"]): int(r["frontend_tick"]) for r in
                csv.DictReader((directory / "simulation/barriers.csv").open())}
    phases = Counter()
    for segment in manifest["segments"]:
        phases[segment["name"]] += ((barriers[segment["end"]] - barriers[segment["start"]])
                                   * result["tCK_ns"] / 32000)
    summary = dict(name=name, operator=op, latency_us=result["latency_ns"] / 1000,
        phase_us=dict(phases), boundary_us=result["latency_ns"] / 1000 - sum(phases.values()),
        matrix=manifest["matrix"], counts=manifest["counts"], traffic_bytes=result["traffic_bytes"],
        memory_cycles=result["memory_cycles"], tCK_ns=result["tCK_ns"],
        timing=result["resolved_timings_ck"], trace_sha256=sha(directory / "workload.trace"),
        fresh_native_run=True, local_transfer_cycles=2,
        scope="Native FFN service; nonlinear/GPU reduction are not measured.")
    summary["audit"] = audit_ffn.audit_run(directory, manifest, result)
    summary["audit_status"] = "passed"
    dump(directory / "summary.json", summary)
    row, channels, resources, traffic, evidence = audit_ffn.inspect_operator(root, name)
    dump(directory / "breakdown.json", dict(row=row, channels=channels,
        resources=resources, traffic=traffic, evidence=evidence))
    print(json.dumps(dict(name=name, latency_us=summary["latency_us"], audit="passed")), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--attention-bytes-per-bank", type=int, default=44*1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.backend), str(args.backend / "ramulator2/python")]
    import simulator.backend
    import ramulator
    assert Path(simulator.backend.__file__).is_relative_to(args.backend)
    assert Path(ramulator.__file__).is_relative_to(args.backend)
    library = args.backend / "ramulator2/libramulator.so"
    provenance = dict(backend_root=str(args.backend), backend_library_sha256=sha(library),
        sources={str(p.relative_to(HERE)): sha(p) for p in HERE.rglob("*.py")},
        old_run=str(args.old_run), old_results_used_only_as_regression_reference=True)
    dump(args.output / "ffn-provenance.json", provenance)
    controls = {op: run_case(args.output, op + "_legacy_control", op)
                for op in ("gate_up", "down_tiled")}
    parity = {}
    for op, new in controls.items():
        old = json.loads((args.old_run / op / "simulation/result.json").read_text())
        new_result = json.loads((args.output / (op + "_legacy_control") / "simulation/result.json").read_text())
        parity[op] = dict(same_trace_hash=new["trace_sha256"] == old["requests"]["sha256"],
            same_latency=new_result["latency_ns"] == old["latency_ns"],
            old_latency_us=old["latency_ns"]/1000,
            new_latency_us=new_result["latency_ns"]/1000,
            latency_delta_us=(new_result["latency_ns"]-old["latency_ns"])/1000,
            same_issued_commands={k:v for k,v in new_result["issued_commands"].items() if v}
                                == {k:v for k,v in old["issued_commands"].items() if v},
            same_work_commands=all(new_result["issued_commands"].get(k,0)==old["issued_commands"].get(k,0)
                                   for k in ("MACAB","WRGB","RDRES","RD","WR")),
            note="Legacy WRGB now completes after nBL_min. Timing and refresh counts may change; payload and MAC work must match.")
        assert parity[op]["same_trace_hash"] and parity[op]["same_work_commands"], (op, parity[op])
    dump(args.output / "ffn-regression.json", parity)
    dump(args.output / "ffn-layout.json", relocate(args.attention_bytes_per_bank))
    primary = {op: run_case(args.output, op, op) for op in ("gate_up", "down_tiled")}
    for op, row in primary.items():
        row["relocation_preserves_latency"] = row["latency_us"] == controls[op]["latency_us"]
        row["relocation_latency_delta_us"] = row["latency_us"] - controls[op]["latency_us"]
    assert sha(library) == provenance["backend_library_sha256"], "Backend changed during FFN run"
    dump(args.output / "ffn-summary.json", dict(primary=primary, controls=controls,
        fresh_native_run=True, regression=parity,
        nonlinear_latency_us=None, gpu_partial_reduction_latency_us=None,
        note="Primary FFN addresses follow new attention reservation; GU/down remain separately timed stages."))


if __name__ == "__main__":
    main()

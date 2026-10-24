"""Execute the accepted GPU-Q/BG-K layout on an isolated native backend."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend_hashes(root):
    files = []
    for part in ("simulator", "ramulator2/src", "ramulator2/include", "ramulator2/python"):
        files.extend(p for p in (root / part).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".so"))
    files.append(root / "ramulator2/libramulator.so")
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(files))}


def run_case(root, name, op, schedule="double", local_write_ck=4):
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig
    import attention_workload as workload
    directory = root / name
    directory.mkdir()
    manifest = workload.generate(directory / "workload.trace", op,
                                 mode="complete", schedule=schedule)
    dump(directory / "manifest.json", manifest)
    config = MemoryConfig(model="lpddr5x-8533", channels=16, ranks=2,
        density_gbit=8, refresh=True, pim=True, pim_input_broadcast=True,
        pim_mac_scope="channel", pim_data_path="attacc", frontend_ratio=32,
        max_cycles=100_000_000, timing_overrides={"nCCDAB": 20,
            "nCCDK": 2, "nCCDQK": 20, "nLOCALWR": local_write_ck},
        pim_local_transfer_cycles=2)
    result = run_trace(config, directory / "workload.trace", directory / "simulation")
    assert result["requests"]["sha256"] == digest(directory / "workload.trace")
    for native, requests in workload.NATIVE_REQUEST_GROUPS.items():
        expected = sum(manifest["counts"].get(request, 0) for request in requests)
        assert expected == result["issued_commands"].get(native, 0), (name, native, expected)
    assert result["requests"]["pim_local_wr"] == manifest["counts"].get("PIM_LOCAL_WR", 0)
    barriers = {int(r["marker"]): int(r["frontend_tick"]) for r in
                csv.DictReader((directory / "simulation/barriers.csv").open())}
    phases = Counter()
    ratio = result["config"]["frontend_ratio"]
    for segment in manifest["segments"]:
        phases[segment["name"]] += ((barriers[segment["end"]] - barriers[segment["start"]])
                                   * result["tCK_ns"] / (ratio * 1000))
    total = result["latency_ns"] / 1000
    boundary = total - sum(phases.values())
    assert -1e-8 <= boundary <= 2 * result["tCK_ns"] / 1000, (name, boundary)
    summary = dict(name=name, operator=op, latency_us=total,
        phase_us=dict(phases), boundary_us=boundary, counts=manifest["counts"],
        traffic_bytes=result["traffic_bytes"], memory_cycles=result["memory_cycles"],
        tCK_ns=result["tCK_ns"], timing=result["resolved_timings_ck"],
        matrix=manifest.get("matrix"), trace_sha256=digest(directory / "workload.trace"),
        schedule=schedule, local_write_ck=local_write_ck, fresh_native_run=True,
        scope="Native PIM service and memory movement; GPU compute and local RoPE/cast are unmeasured.",
        audit_status="pending")
    dump(directory / "summary.json", summary)
    print(json.dumps(dict(name=name, latency_us=total, phase_us=dict(phases))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-backend", type=Path,
                        help="Optional separate backend to verify remains unchanged")
    parser.add_argument("--cases", nargs="+", default=["k", "v", "qk", "pv", "chain_k_qk",
                                                      "k_local8", "k_serial", "qk_serial"])
    args = parser.parse_args()
    args.backend = args.backend.resolve(strict=True)
    args.output = args.output.resolve()
    if args.original_backend is not None:
        args.original_backend = args.original_backend.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in (".py", ".sbatch"):
            shutil.copy2(p, snapshot / p.name)
    sys.path[:0] = [str(args.backend), str(args.backend / "ramulator2/python")]
    import simulator.backend
    import ramulator
    assert Path(simulator.backend.__file__).is_relative_to(args.backend)
    assert Path(ramulator.__file__).is_relative_to(args.backend)
    before = backend_hashes(args.backend)
    original_before = (backend_hashes(args.original_backend)
                       if args.original_backend is not None else None)
    provenance = dict(backend_root=str(args.backend), backend_hashes=before,
        original_backend_root=(str(args.original_backend) if args.original_backend is not None else None),
        original_backend_hashes=original_before,
        original_backend_check_performed=args.original_backend is not None,
        source_hashes={p.name: digest(p) for p in snapshot.iterdir()},
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), fresh_native_simulation=True)
    dump(args.output / "provenance.json", provenance)
    rows = []
    for name in args.cases:
        if name == "k_local8":
            rows.append(run_case(args.output, name, "k", local_write_ck=8))
        elif name.endswith("_serial"):
            rows.append(run_case(args.output, name, name.removesuffix("_serial"), schedule="serial"))
        else:
            rows.append(run_case(args.output, name, name))
    assert backend_hashes(args.backend) == before, "Native backend changed during attention run"
    if args.original_backend is not None:
        assert backend_hashes(args.original_backend) == original_before, "Original backend changed"
    dump(args.output / "attention-summary.json", dict(runs=rows,
        primary_cases=["k", "v", "qk", "pv"],
        unknown_latency_us=dict(GPU_Q=None, GPU_O=None, GPU_softmax=None,
            local_K_RoPE_cast=None, GPU_V_cast_pack=None, GPU_arrival_processing=None),
        original_backend_check_performed=args.original_backend is not None,
        original_backend_unchanged=(True if args.original_backend is not None else None),
        private_backend_unchanged_during_run=True))


if __name__ == "__main__":
    main()

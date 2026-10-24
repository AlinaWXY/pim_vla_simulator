"""Compare identical half-bank compute/read work with and without overlap."""

import argparse
import csv
import json
from pathlib import Path

from simulator import MemoryConfig, run_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compute = [f"PIM_MAC_HB {i % 64 * 32}" for i in range(128)]
    reads = [f"LD {4096 + i % 64 * 32}" for i in range(192)]
    interleaved = []
    for i in range(64):
        interleaved.extend(compute[2 * i:2 * i + 2])
        interleaved.extend(reads[3 * i:3 * i + 3])
    traces = {
        "concurrent": interleaved,
        "serial": compute + ["BARRIER 0"] + reads,
    }
    for name, lines in traces.items():
        (args.output / f"{name}.trace").write_text("\n".join(lines) + "\n")
    comparisons = []
    for model in ("lpddr5-6400", "lpddr5x-8533"):
        results = {
            name: run_trace(MemoryConfig(model=model, pim=True, refresh=False),
                            args.output / f"{name}.trace", args.output / model / name)
            for name in traces
        }
        parallel, serial = results["concurrent"], results["serial"]
        assert parallel["traffic_bytes"] == serial["traffic_bytes"]
        assert parallel["completed_requests"] == serial["completed_requests"] == 320
        assert parallel["memory_cycles"] < serial["memory_cycles"]
        with (args.output / model / "concurrent" / "timeline.csv.ch0").open() as stream:
            events = list(csv.DictReader(stream))
        macs = [int(e["clock"]) for e in events if e["command"] == "MACHB"]
        rds = [int(e["clock"]) for e in events if e["command"] == "RD"]
        t = parallel["resolved_timings_ck"]
        assert min(rds) < max(macs)
        assert min(macs) < max(rds) + t["nCL"] + t["nBL_min"]
        comparisons.append({
            "model": model,
            "compute_requests": len(compute), "gpu_reads": len(reads),
            "serial_cycles": serial["memory_cycles"],
            "concurrent_cycles": parallel["memory_cycles"],
            "serial_ns": serial["latency_ns"], "concurrent_ns": parallel["latency_ns"],
            "speedup": serial["memory_cycles"] / parallel["memory_cycles"],
            "read_latency_ck": t["nCL"] + t["nBL_min"],
            "traffic_bytes": parallel["traffic_bytes"],
        })
    report = {
        "assumptions": "one channel/rank; 8 compute banks + 8 read banks; refresh disabled; "
                       "identical work; calibrated RD timing; no transfer latency scaling",
        "comparisons": comparisons,
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

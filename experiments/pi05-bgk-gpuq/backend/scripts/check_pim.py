"""Run a reproducible LPDDR5/PIM trace suite and check hand-calculated cases."""

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
    repo = Path(__file__).resolve().parents[1]
    cases = [
        ("normal-one-read", False, False, ["LD 0"], 36),
        ("pim-one-mac", True, False, ["PIM_MAC_AB 0"], 17),
        (
            "pim-eight-row-hits",
            True,
            False,
            (repo / "traces/pim-row-hit.trace").read_text().splitlines(),
            45,
        ),
        (
            "pim-row-conflict",
            True,
            False,
            (repo / "traces/pim-row-conflict.trace").read_text().splitlines(),
            68,
        ),
        ("mixed", True, True, (repo / "traces/pim-mixed.trace").read_text().splitlines(), None),
        ("pim-refresh", True, True, ["PIM_MAC_AB 0"] * 1000, None),
    ]
    summary = []
    for name, pim, refresh, lines, expected in cases:
        trace = args.output / f"{name}.trace"
        trace.write_text("\n".join(lines) + "\n")
        result = run_trace(MemoryConfig(pim=pim, refresh=refresh), trace, args.output / name)
        assert result["completed_requests"] == len(lines), name
        if expected is not None:
            assert result["memory_cycles"] == expected, (name, result["memory_cycles"], expected)
        with (args.output / name / "timeline.csv.ch0").open() as stream:
            events = list(csv.DictReader(stream))
        macs = [int(e["clock"]) for e in events if e["command"] == "MACAB"]
        assert all(b - a >= 4 for a, b in zip(macs, macs[1:])), name
        if macs:
            assert result["memory_cycles"] >= macs[-1], name
        if name == "pim-refresh":
            assert result["issued_commands"]["REFab"] > 0
        row = {
            "case": name,
            "requests": len(lines),
            "cycles": result["memory_cycles"],
            "latency_ns": result["latency_ns"],
            "expected_cycles": expected,
            "commands": {c: n for c, n in result["issued_commands"].items() if n},
            "energy_pj": result["energy_pj"],
        }
        summary.append(row)
        print(f"{name}: {row['cycles']} CK, {row['latency_ns']:.3f} ns")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

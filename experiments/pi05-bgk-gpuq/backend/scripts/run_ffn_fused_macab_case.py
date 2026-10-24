"""Run one H=10 full-shape fused-MACAB Ramulator case."""

import argparse
import json
from pathlib import Path

from simulator.ffn import FFNConfig, run_ffn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", required=True, type=int, choices=range(2, 11))
    parser.add_argument("--nccdab", required=True, type=int, choices=range(8, 13))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = FFNConfig(
        horizon=10,
        reuse_tokens=args.reuse,
        macab_tokens_per_command=args.reuse,
        macab_issue_interval_ck=args.nccdab,
    )
    result = run_ffn(cfg, args.output)
    assert result["logical_mac_operations"] == 125_829_120
    assert result["pim_issue_interval_ck"] == args.nccdab
    expected_scans = (10 + args.reuse - 1) // args.reuse
    assert result["issued_commands"]["MACAB"] == 49_152 * expected_scans
    print(json.dumps({
        "reuse": args.reuse,
        "nCCDAB": args.nccdab,
        "latency_us": result["latency_ns"] / 1000,
        "memory_cycles": result["memory_cycles"],
        "MACAB": result["issued_commands"]["MACAB"],
        "ACTAB": result["issued_commands"]["ACTAB"],
        "REFab": result["issued_commands"]["REFab"],
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

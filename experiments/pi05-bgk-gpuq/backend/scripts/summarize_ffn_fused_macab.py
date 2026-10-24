"""Validate and summarize the H=10 fused-MACAB Ramulator sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PENALTY_TO_CK = {20: 8, 30: 8, 40: 9, 50: 9, 60: 10,
                 70: 11, 80: 11, 90: 12, 100: 12}
BASELINE_US = 241.61942000000002


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-reuse", type=int, default=6)
    parser.add_argument("--max-reuse", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.min_reuse <= args.max_reuse <= 10:
        parser.error("reuse range must satisfy 2 <= min <= max <= 10")
    reuse_values = range(args.min_reuse, args.max_reuse + 1)

    unique = {}
    for reuse in reuse_values:
        for nccdab in range(8, 13):
            case = args.output / "cases" / f"r{reuse}-ck{nccdab}" / "result.json"
            result = json.loads(case.read_text())
            assert result["config"]["horizon"] == 10
            assert result["config"]["reuse_tokens"] == reuse
            assert result["config"]["macab_tokens_per_command"] == reuse
            assert result["pim_issue_interval_ck"] == nccdab
            assert result["logical_mac_operations"] == 125_829_120
            assert result["issued_commands"]["MACAB"] == 49_152 * ((10 + reuse - 1) // reuse)
            unique[reuse, nccdab] = result

    rows = []
    for reuse in reuse_values:
        for penalty, nccdab in PENALTY_TO_CK.items():
            result = unique[reuse, nccdab]
            rows.append({
                "horizon": 10,
                "reuse_tokens": reuse,
                "macab_penalty_pct": penalty,
                "resolved_nCCDAB_ck": nccdab,
                "weight_scans": (10 + reuse - 1) // reuse,
                "memory_cycles": result["memory_cycles"],
                "latency_us": result["latency_ns"] / 1000,
                "speedup_vs_legacy": BASELINE_US / (result["latency_ns"] / 1000),
                "MACAB": result["issued_commands"]["MACAB"],
                "ACTAB": result["issued_commands"]["ACTAB"],
                "PREab": result["issued_commands"]["PREab"],
                "REFab": result["issued_commands"]["REFab"],
                "weight_internal_mib": result["traffic_bytes"]["pim_internal"] / 1024**2,
                "result_kind": "ramulator_fused_macab_timing_simulation",
            })

    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    penalties = list(PENALTY_TO_CK)
    matrix = np.array([
        [next(row["latency_us"] for row in rows
              if row["reuse_tokens"] == reuse and row["macab_penalty_pct"] == penalty)
         for penalty in penalties]
        for reuse in reuse_values
    ])
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(penalties)), [f"+{p}%" for p in penalties])
    ax.set_yticks(range(len(reuse_values)), [f"R={r}" for r in reuse_values])
    ax.set_xlabel("Requested MACAB issue-interval increase")
    ax.set_ylabel("K reuse tokens")
    ax.set_title(
        f"H=10 fused-MACAB Ramulator latency, R={args.min_reuse}-{args.max_reuse} (us)",
        loc="left",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.mean() else "black")
    fig.colorbar(image, ax=ax, label="Latency (us)")
    fig.savefig(args.output / "latency-heatmap.png", dpi=180)
    fig.savefig(args.output / "latency-heatmap.pdf")
    plt.close(fig)
    print(json.dumps({"cases": len(unique), "rows": len(rows),
                      "summary": str(args.output / "summary.csv")}, indent=2))


if __name__ == "__main__":
    main()

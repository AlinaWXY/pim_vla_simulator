"""Plot an offline read_bottleneck report and its largest half-bank row transition."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_budget(report, output):
    cases = report["cases"]
    labels = [c["label"].replace("experiment/", "").replace("refresh-on/", "refresh: ")
              for c in cases]
    channels = [c["per_channel"][0] for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    y = list(range(len(cases)))
    colors = ["#b37b41" if c["config"]["refresh"] else "#357b9d" for c in cases]
    bars = axes[0].barh(y, [c["aggregate_bandwidth_GBps"] for c in cases], color=colors)
    axes[0].bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    axes[0].axvline(51.2, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 59), xlabel="Aggregate read bandwidth (GB/s)",
                yticks=y, yticklabels=labels, title="Bandwidth | 4 x LPDDR5-6400")
    axes[0].invert_yaxis()
    # Normalize accounting components by each channel's own window. Half-LPDDR
    # has only two active GPU channels, so its per-channel DQ stays nearly full.
    left = [0.0] * len(channels)
    for key, label, color in (("dq_busy_ck", "DQ bursts", "#357b9d"),
                              ("rd_gap_ck", "RD gaps", "#d88459"),
                              ("boundary_ck", "Window boundary", "#555555")):
        values = [100 * c[key] / c["window_ck"] for c in channels]
        axes[1].barh(y, values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    axes[1].set(xlim=(0, 100), xlabel="Share of active read window (%)", yticks=y,
                yticklabels=[], title="Exact cycle accounting | one active channel")
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=3, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Read bandwidth audit | saved simulations from 2026-10-02", fontsize=14)
    fig.savefig(output / "bandwidth_budget.png", dpi=180)
    fig.savefig(output / "bandwidth_budget.svg")
    plt.close(fig)


def plot_row_transition(report, output):
    case = next(c for c in report["cases"] if c["label"].endswith("/half-bank")
                and not c["config"]["refresh"])
    channel = case["per_channel"][0]
    gap = next(g for g in channel["largest_gaps"] if g["next_rd_changes_bank_row"])
    with (Path(case["source"]) / f"timeline.csv.ch{channel['channel']}").open() as stream:
        events = list(csv.DictReader(stream))
    timing = case["timings"]
    pre, act, rd = (gap[k] for k in ("next_bank_pre_ck", "next_bank_act2_ck", "next_rd_ck"))
    lo = min(pre, gap["previous_rd_ck"]) - 4
    hi = rd + timing["nCL"] + timing["nBL_min"] + 4
    view = [e for e in events if lo <= int(e["clock"]) <= hi]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.2, 1]})
    colors = {"RD": "#357b9d", "PREpb": "#b37b41", "ACT1": "#78935b",
              "ACT2": "#45967a", "MACHB": "#9466a0"}
    for y, (command, color) in enumerate(colors.items()):
        times = [int(e["clock"]) for e in view if e["command"] == command]
        axes[0].scatter(times, [y] * len(times), color=color, marker="|", s=180, linewidths=2)
    axes[0].set(yticks=list(range(5)), yticklabels=list(colors), ylim=(-0.7, 4.7),
                title="Issued commands (all banks on the channel)")
    axes[0].axvspan(gap["gap_start_ck"], rd, color="#d88459", alpha=0.12)
    axes[0].text((gap["gap_start_ck"] + rd) / 2, -0.55,
                 f"{gap['gap_cycles']} CK beyond normal RD spacing", ha="center", fontsize=9)
    axes[0].invert_yaxis()
    for time, label, color in ((pre, "PRE", colors["PREpb"]),
                               (act, "ACT2", colors["ACT2"]), (rd, "RD", colors["RD"])):
        axes[1].scatter([time], [1], color=color, s=60, zorder=3)
        axes[1].text(time, 1.15, f"{label}\nCK{time}", ha="center", fontsize=9)
    for start, end, label in ((pre, act, f"{act-pre} CK >= nRP={timing['nRP']}"),
                              (act, rd, f"{rd-act} CK = nRCD")):
        axes[1].annotate("", (start, 0.78), (end, 0.78),
                         arrowprops={"arrowstyle": "<->", "color": "#555555"})
        axes[1].text((start + end) / 2, 0.55, label, ha="center", fontsize=9)
    burst = timing["nBL_min"]
    intervals = [(int(e["clock"]) + timing["nCL"], burst) for e in events
                 if e["command"] == "RD"
                 and lo - burst <= int(e["clock"]) + timing["nCL"] <= hi]
    axes[1].broken_barh(intervals, (-0.12, 0.24), facecolors=colors["RD"])
    idle_start = gap["gap_start_ck"] + timing["nCL"]
    idle_end = rd + timing["nCL"]
    axes[1].broken_barh([(idle_start, idle_end - idle_start)], (-0.12, 0.24),
                        facecolors="#d88459", alpha=0.3)
    axes[1].text((idle_start + idle_end) / 2, -0.38,
                 f"DQ idle: CK{idle_start} to CK{idle_end} ({idle_end-idle_start} CK)",
                 ha="center", fontsize=9)
    axes[1].set(xlim=(lo, hi), ylim=(-0.65, 1.65), yticks=[0, 1],
                yticklabels=["DQ bus", f"BG{gap['next_bg']} / Bank{gap['next_bank']}"],
                xlabel="Memory clock (CK; 1 CK = 1.25 ns)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(f"Half-bank row transition | CH{channel['channel']} | "
                 f"at least {gap['min_admitted_unissued_reads']} admitted reads waiting "
                 "in this gap",
                 fontsize=13)
    fig.savefig(output / "row_transition.png", dpi=180)
    fig.savefig(output / "row_transition.svg")
    plt.close(fig)
    with (output / "row_transition_commands.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis.parent
    for name in ("bandwidth_budget.png", "bandwidth_budget.svg", "row_transition.png",
                 "row_transition.svg", "row_transition_commands.csv"):
        if (output / name).exists():
            parser.error(f"refusing to overwrite {output / name}")
    report = json.loads(args.analysis.read_text())
    plot_budget(report, output)
    plot_row_transition(report, output)


if __name__ == "__main__":
    main()

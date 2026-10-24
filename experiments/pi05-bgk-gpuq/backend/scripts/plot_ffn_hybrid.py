"""Plot completed coarse screening results without extrapolating untested splits."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    if not report["complete"]:
        raise ValueError("Plot requires every planned case to finish")
    rows = report["rows"]
    horizons = sorted({r["horizon"] for r in rows})
    fig, axes = plt.subplots(1, len(horizons), figsize=(17, 4.5), sharey=True)
    lines = [
        "# GPU/PIM coarse sweep results", "",
        "All 230 ordinary-MACAB simulations completed. GPU values are weight-bandwidth",
        "bounds, not Thor measurements. PIM nonlinear compute is excluded.", "",
        "| H | Best hybrid PIM packages | PIM width | PIM branch (us) | GPU weight bound (us) | Delivery + add (us) | Hybrid scenario (us) | GPU-only bound (us) | PIM-only scenario (us) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ax, h in zip(axes, horizons):
        group = [r for r in rows if r["horizon"] == h]
        gpu = next(r for r in group if r["pim_packages"] == 0)
        pim = next(r for r in group if r["pim_packages"] == 8)
        hybrid = min((r for r in group if 0 < r["pim_packages"] < 8),
                     key=lambda r: r["scenario_ns"])
        for packages in (1, 2, 4):
            curve = sorted((r for r in group if r["pim_packages"] == packages),
                           key=lambda r: r["pim_fraction"])
            ax.plot([r["pim_fraction"] * 100 for r in curve],
                    [r["scenario_ns"] / 1000 for r in curve],
                    marker=".", label=f"{packages} PIM packages")
        ax.axhline(gpu["scenario_ns"] / 1000, color="black", ls="--",
                   label="GPU-only weight bound")
        ax.scatter([hybrid["pim_fraction"] * 100], [hybrid["scenario_ns"] / 1000],
                   marker="*", s=130, color="red", zorder=5)
        ax.set(title=f"H = {h}", xlabel="PIM intermediate width (%)", yscale="log")
        ax.grid(alpha=0.25)
        lines.append(
            f"| {h} | {hybrid['pim_packages']} | {hybrid['pim_width']} | "
            f"{hybrid['pim_ns']/1000:.3f} | {hybrid['gpu_weight_bound_ns']/1000:.3f} | "
            f"{(hybrid['delivery_write_ns']+hybrid['add_traffic_ns'])/1000:.3f} | "
            f"{hybrid['scenario_ns']/1000:.3f} | {gpu['scenario_ns']/1000:.3f} | "
            f"{pim['scenario_ns']/1000:.3f} |"
        )
    axes[0].set_ylabel("Idealized latency scenario (us, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("FFN GPU/PIM coarse screening: ordinary MACAB; GPU bandwidth model")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(args.root / f"latency-curves.{suffix}", dpi=180)
    lines.extend(["", "Only PIM package counts 1/2/4 and width multiples of 256 were",
                  "screened. These results do not establish a global optimum or a real",
                  "Thor speedup. Details and cost assumptions are in plan.json and",
                  "docs/FFN-GPU-PIM-sweep.md.", ""])
    (args.root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

"""Compare the four fresh variants using issued-command DQ intervals."""
import csv
import json
from pathlib import Path

NAMES = ("neither", "score_only", "oproj_only", "both")
LABELS = ("Neither", "Score DRAM", "PIM O", "Both")
STAGES = ("chain", "pv", "gate_up_gpu", "down_gpu_input")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(root):
    root = Path(root)
    summaries = {n: read(root / n / "summary.json") for n in NAMES}
    channels = {n: read(root / n / "channel-analysis.json") for n in NAMES}
    rows, stages, all_channels, phases = [], [], [], []
    for name in NAMES:
        summary, report = summaries[name], channels[name]
        if not summary["pv_bf16_output"]:
            raise AssertionError("All four variants require common BF16 PV output")
        for c in report["channels"]:
            row = dict(variant=name, channel=c["channel"], latency_us=c["stage_us"],
                dq_busy_us=c["modelled_dq_occupied_us"], dq_idle_us=c["modelled_dq_idle_us"],
                dq_utilization_percent=100*c["modelled_dq_occupied_us"]/c["stage_us"],
                whole_work_without_dq_us=c["tail_clear_complete_zero_external_phase_us"],
                other_dq_idle_us=c["modelled_dq_idle_outside_tail_clear_work_us"],
                dq_and_global_ca_idle_us=c["command_bus"]["dq_idle_global_ca_free_us"],
                external_dq_KiB_channel=c["external_DQ_bytes"]/1024)
            if abs(row["latency_us"]-row["dq_busy_us"]-row["whole_work_without_dq_us"]-row["other_dq_idle_us"]) > 1e-8:
                raise AssertionError("DQ breakdown does not close")
            all_channels.append(row)
            if c["channel"] == 0:
                rows.append(dict(row, external_dq_KiB_all_channels=summary["external_DQ_bytes_per_layer_step"]/1024,
                    serial_control_us=summary["serial_layer_memory_control_us"]))
        for stage in STAGES:
            s = report["stages"][stage]
            c = s["channels"][0]
            stages.append(dict(variant=name, stage=stage, latency_us=s["stage_us"],
                dq_busy_us=c["modelled_dq_occupied_us"], dq_idle_us=c["modelled_dq_idle_us"],
                dq_utilization_percent=100*c["modelled_dq_occupied_us"]/s["stage_us"],
                whole_work_without_dq_us=c["tail_clear_complete_zero_external_phase_us"],
                other_dq_idle_us=c["modelled_dq_idle_outside_tail_clear_work_us"],
                external_dq_KiB_all_channels=summary["runs"][stage]["external_DQ_bytes"]/1024))
            for p in s["phases"]:
                for c in p["channels"]:
                    phases.append(dict(variant=name, stage=stage, phase=p["phase"], kind=p["kind"], channel=c["channel"],
                        duration_us=p["duration_us"], dq_busy_us=c["modelled_dq_occupied_us"],
                        dq_idle_us=c["modelled_dq_idle_us"], whole_work_without_dq_us=c["tail_clear_window_us"]))
    base = rows[0]["latency_us"]
    for row in rows:
        row["latency_delta_us"] = row["latency_us"]-base
        row["latency_delta_percent"] = 100*(row["latency_us"]/base-1)
    interaction = rows[3]["latency_us"]-rows[1]["latency_us"]-rows[2]["latency_us"]+base
    serial = [r["serial_control_us"] for r in rows]
    serial_interaction = serial[3]-serial[1]-serial[2]+serial[0] if all(v is not None for v in serial) else None
    result = dict(precision="PV/O: BF16 operands/weights/results, FP32 accumulation. QKT scores and FFN exports: FP32.", rows=rows,
        latency_interaction_us=interaction, independent_stage_sum_interaction_us=interaction,
        serial_control_interaction_us=serial_interaction,
        interaction_definition="Both - score_only - oproj_only + neither. Independent-stage interaction is zero by construction when each switch changes only its own stage; it does not establish absence of interactions in continuous execution. Serial-control interaction uses the separate continuous-memory runs.",
        complete_end_to_end_latency_us=None,
        interpretation="Sum of independently timed QKV/attention, PV or PV+O, Gate/Up and Down stages. GPU O projection is unmeasured in non-O variants; GPU reductions, softmax, nonlinearities and contention are unmeasured in all variants.",
        dq_definition="Occupied modeled issued-command burst intervals / native stage wall time, per channel. No concurrent GPU traffic or validated electrical DQ waveform is simulated.",
        whole_work_without_dq_definition="Sum of complete semantic work phases with no external DQ commands after removing incoming burst tails. Includes local-memory waits and refresh; it is neither MAC active time nor one contiguous GPU-available window.")
    (root / "comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    write_csv(root / "comparison.csv", rows)
    write_csv(root / "stage-comparison.csv", stages)
    write_csv(root / "channel-comparison.csv", all_channels)
    write_csv(root / "phase-comparison.csv", phases)
    lines = ["# QKT DRAM spill / PIM O projection: 2×2 native experiment", "",
        "All values are per layer and denoising step. PV and O operands/weights/results are BF16; accumulators remain FP32. Scores remain FP32.", "",
        "The latency column sums modeled memory/PIM stages. GPU O projection is unmeasured in non-O variants, so these differences are not end-to-end speedups.", "",
        "| Variant | Modeled latency (µs) | Δ (µs) | DQ busy/ch0 (µs) | DQ utilization/ch0 | All-channel external DQ (KiB) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['latency_us']:.6f} | {row['latency_delta_us']:+.6f} | {row['dq_busy_us']:.6f} | {row['dq_utilization_percent']:.3f}% | {row['external_dq_KiB_all_channels']:.0f} |")
    lines += ["", "| Variant | Whole work without DQ (µs/ch0) | Other DQ idle (µs/ch0) | DQ and global CA idle (µs/ch0) | Serial control (µs) |",
        "|---|---:|---:|---:|---:|"]
    for r in rows:
        serial = f"{r['serial_control_us']:.6f}" if r['serial_control_us'] is not None else "not run"
        lines.append(f"| {r['variant']} | {r['whole_work_without_dq_us']:.6f} | {r['other_dq_idle_us']:.6f} | {r['dq_and_global_ca_idle_us']:.6f} | {serial} |")
    lines += ["", "The DQ breakdown closes as latency = DQ busy + whole work without DQ + other DQ idle. A lower utilization can result from longer computation or local-memory work; it does not by itself imply less data transfer or greater performance.", "",
        result["whole_work_without_dq_definition"], "",
        f"Independent-stage interaction: {interaction:+.6f} µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: {serial_interaction:+.6f} µs." if serial_interaction is not None else
        f"Independent-stage interaction: {interaction:+.6f} µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: not run.", "",
        result["interpretation"], "", "These are timing and data-mapping simulations, not numerical network execution or full end-to-end latency. BF16 partial rounding and the changed reduction order require separate numerical validation.", "",
        "Detailed evidence: [stages](stage-comparison.csv), [all channels](channel-comparison.csv), [all phases](phase-comparison.csv), [source provenance](provenance.json).", ""]
    (root / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    plot(root, rows, channels)
    return result


def plot(root, rows, channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(4)
    bottom = np.zeros(4)
    for stage, label, color in zip(STAGES, ("QKV / RoPE / QKT", "PV or PV + O", "Gate / Up", "Down"),
                                  ("#4979ad", "#44a89b", "#d9a441", "#ad739b")):
        values = [channels[n]["stages"][stage]["stage_us"] for n in NAMES]
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    for i, r in enumerate(rows):
        axes[0].text(i, r["latency_us"]+1, f"{r['latency_us']:.3f}", ha="center", fontsize=9)
    bottom = np.zeros(4)
    for key, label, color in (("dq_busy_us", "DQ occupied", "#d26a45"),
            ("whole_work_without_dq_us", "Whole work without DQ", "#4979ad"),
            ("other_dq_idle_us", "Other DQ idle", "#cbd3db")):
        values = [r[key] for r in rows]
        axes[1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    for i, r in enumerate(rows):
        axes[1].text(i, r["latency_us"]+1, f"DQ {r['dq_utilization_percent']:.2f}%", ha="center", fontsize=9)
    for ax, title in zip(axes, ("Modeled stage latency", "Channel 0 DQ breakdown")):
        ax.set_xticks(x, LABELS)
        ax.set_ylabel("µs / layer / step")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, max(r["latency_us"] for r in rows)*1.28)
    fig.suptitle("Score buffering × PIM output projection — common BF16 PV boundary", fontsize=14)
    fig.supxlabel("Modeled stage sums; GPU O projection is unmeasured in non-O variants.\n"
                  "Not an end-to-end speedup; DQ idle does not establish GPU usable bandwidth.", fontsize=9)
    fig.savefig(root / "latency-dq-breakdown.png", dpi=180)
    fig.savefig(root / "latency-dq-breakdown.pdf")
    plt.close(fig)
    from channel_analysis import dq_intervals, read_csv
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, name, label in zip(axes, NAMES, LABELS):
        case = root / name / "chain"
        report = channels[name]["stages"]["chain"]
        result = read(case / "simulation/result.json")
        ratio = result["config"]["frontend_ratio"]
        scale = result["tCK_ns"]/(ratio*1000)
        qphases = [p for p in report["phases"] if p["phase"].startswith("qk_")]
        start, end = min(p["begin_tick"] for p in qphases), max(p["end_tick"] for p in qphases)
        width = max(1, round(.1/scale))
        edges = np.arange(start, end+width, width)
        occupancy = np.zeros(len(edges)-1)
        intervals = dq_intervals(read_csv(case / "simulation/timeline.csv.ch0"), result["resolved_timings_ck"], ratio, "attacc")
        for e in intervals:
            if e["end"] <= start or e["begin"] >= end:
                continue
            lo, hi = max(0, (e["begin"]-start)//width), min(len(occupancy)-1, (e["end"]-start)//width)
            for b in range(lo, hi+1):
                occupancy[b] += max(0, min(e["end"], edges[b+1])-max(e["begin"], edges[b]))/width
        ax.stairs(100*occupancy, (edges-start)*scale, fill=True, color="#4979ad")
        ax.set_ylim(0, 105)
        ax.set_ylabel(label+"\nDQ %")
    axes[-1].set_xlabel("µs from first QKT tile (0.1 µs display bins)")
    fig.suptitle("QKT channel 0: tile exports versus final DRAM readout")
    fig.savefig(root / "qkt-dq-timeline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    compare(p.parse_args().root)

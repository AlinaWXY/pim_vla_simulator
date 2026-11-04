"""Compare the four fresh variants using issued-command DQ intervals."""
import csv
import hashlib
import json
from pathlib import Path

NAMES = ("neither", "score_only", "oproj_only", "both")
LABELS = ("Neither\nGPU O unmeasured", "Score DRAM\nGPU O unmeasured", "PIM O\nO included", "Both\nO included")
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


def compare(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root
    output.mkdir(parents=True, exist_ok=True)
    from continuous_dq import write_report
    continuous = write_report(root, output / "continuous-dq", allow_nested_output=output.resolve()==root.resolve())
    def window(name, case, scope="stage"):
        return next(r for r in continuous["records"] if
            (r["variant"],r["case"],r["scope"],r["channel"])==(name,case,scope,0))
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
        name = row["variant"]
        phases_us = summaries[name]["runs"]["pv"]["phase_us"]
        o_us = sum(value for phase,value in phases_us.items() if phase.startswith("o_projection_"))
        row.update(o_projection_included=summaries[name]["o_projection"],
            pim_o_subflow_us=o_us if summaries[name]["o_projection"] else None,
            o_timing_scope="PIM O input read/broadcast/compute, control and BF16 export" if summaries[name]["o_projection"] else "GPU O projection unmeasured",
            pv_path_without_o_us=summaries[name]["runs"]["pv"]["latency_us"]-o_us,
            qkt_longest_complete_work_window_us=window(name,"chain","qkt")["longest_complete_work_window_us"],
            qkt_complete_work_window_count=window(name,"chain","qkt")["complete_work_window_count"],
            qkt_longest_dq_idle_us=window(name,"chain","qkt")["longest_duration_us"],
            pv_longest_complete_work_window_us=window(name,"pv")["longest_complete_work_window_us"])
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
        whole_work_without_dq_definition="Historical cumulative diagnostic only: sum of separate complete semantic work phases after incoming burst tails. Superseded for uninterrupted-window comparison; never interpret this sum as one window.",
        continuous_window_report="continuous-dq/continuous-dq.json",
        primary_idle_metric="Longest single complete DQ-free work window, merging adjacent work across DQ-free control bridges; never sum windows across exports or independent stages.")
    result["source_sha256"]={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root/"provenance.json"]+[root/n/f for n in NAMES for f in ("summary.json","channel-analysis.json")]}
    result["continuous_report_sha256"]=hashlib.sha256((output/"continuous-dq/continuous-dq.json").read_bytes()).hexdigest()
    result["derivation_code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),Path(__import__("continuous_dq").__file__),Path(__import__("channel_analysis").__file__))}
    (output / "comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "stage-comparison.csv", stages)
    write_csv(output / "channel-comparison.csv", all_channels)
    write_csv(output / "phase-comparison.csv", phases)
    lines = ["# QKT DRAM spill / PIM O projection: 2×2 native experiment", "",
        "All values are per layer and denoising step. PV and O operands/weights/results are BF16; accumulators remain FP32. Scores remain FP32.", "",
        "The latency column sums modeled memory/PIM stages. GPU O projection is unmeasured in non-O variants, so these differences are not end-to-end speedups.", "",
        "| Variant | Modeled latency (µs) | Timed O subflow (µs) | Longest complete QKT DQ-free work window (µs/ch0) | QKT work-window count | DQ utilization/ch0 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        o_label=f"{row['pim_o_subflow_us']:.6f} (included)" if row['o_projection_included'] else "GPU O unmeasured"
        lines.append(f"| {row['variant']} | {row['latency_us']:.6f} | {o_label} | {row['qkt_longest_complete_work_window_us']:.6f} | {row['qkt_complete_work_window_count']} | {row['dq_utilization_percent']:.3f}% |")
    lines += ["", "Timed O includes its input read/broadcast/compute, release/rewind and BF16 export. It is not the net latency increase, because the baseline PV export is replaced by local PV spill.", "",
        "| Variant | Longest physical QKT DQ-idle window (µs/ch0) | Longest complete PV/PV+O DQ-free work window (µs/ch0) | Serial control (µs) |",
        "|---|---:|---:|---:|"]
    for r in rows:
        serial = f"{r['serial_control_us']:.6f}" if r['serial_control_us'] is not None else "not run"
        lines.append(f"| {r['variant']} | {r['qkt_longest_dq_idle_us']:.6f} | {r['pv_longest_complete_work_window_us']:.6f} | {serial} |")
    lines += ["", "Continuous work windows stop at a score/result transfer phase. DQ-free control bridges between work phases are allowed; idle fragments inside transfer phases are excluded. Physical DQ-idle windows are reported separately and can include such gaps. Neither metric is MAC-active time or measured GPU-usable time.", "",
        result["whole_work_without_dq_definition"], "",
        f"Independent-stage interaction: {interaction:+.6f} µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: {serial_interaction:+.6f} µs." if serial_interaction is not None else
        f"Independent-stage interaction: {interaction:+.6f} µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: not run.", "",
        result["interpretation"], "", "These are timing and data-mapping simulations, not numerical network execution or full end-to-end latency. BF16 partial rounding and the changed reduction order require separate numerical validation.", "",
        "Detailed evidence: [continuous windows](continuous-dq/continuous-dq.json), [all complete windows](continuous-dq/continuous-dq-complete-work-windows.csv), [stages](stage-comparison.csv), [all channels](channel-comparison.csv), [all phases](phase-comparison.csv), [source provenance](provenance.json).", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    plot(root, output, rows, channels, continuous)
    return result


def plot(root, output, rows, channels, continuous):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    x = np.arange(4)
    bottom = np.zeros(4)
    segments = [
        ("QKV / RoPE / QKT", "#4979ad", [channels[n]["stages"]["chain"]["stage_us"] for n in NAMES]),
        ("PV + result handling", "#44a89b", [r["pv_path_without_o_us"] for r in rows]),
        ("PIM O + BF16 export (included in exp. 2)", "#d66543", [r["pim_o_subflow_us"] or 0 for r in rows]),
        ("Gate / Up", "#d9a441", [channels[n]["stages"]["gate_up_gpu"]["stage_us"] for n in NAMES]),
        ("Down", "#ad739b", [channels[n]["stages"]["down_gpu_input"]["stage_us"] for n in NAMES])]
    for label, color, values in segments:
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        if label.startswith("PIM O"):
            for i,value in enumerate(values):
                if value:
                    axes[0].text(i,bottom[i]+value/2,f"O included\n{value:.3f} µs",ha="center",va="center",fontsize=8,color="white")
        bottom += values
    for i, r in enumerate(rows):
        axes[0].text(i, r["latency_us"]+1, f"{r['latency_us']:.3f}", ha="center", fontsize=9)
    specs=(("chain","qkt"),("pv","stage"),("gate_up_gpu","stage"),("down_gpu_input","stage"))
    colors=("#4979ad","#d9a441","#44a89b","#ad739b")
    for index,(name,color) in enumerate(zip(NAMES,colors)):
        values=[next(r["longest_complete_work_window_us"] for r in continuous["records"]
            if (r["variant"],r["case"],r["scope"],r["channel"])==(name,case,scope,0)) for case,scope in specs]
        positions=x+(index-1.5)*.20
        axes[1].bar(positions,values,width=.19,label=("Neither","Score DRAM","PIM O","Both")[index],color=color)
        for pos,value in zip(positions,values):
            axes[1].text(pos,value+.6,f"{value:.2f}",ha="center",fontsize=7,rotation=90)
    axes[0].set_xticks(x,LABELS,fontsize=8)
    axes[0].set_ylabel("µs / layer / step")
    axes[0].set_title("Modeled stage latency: experiment 2 includes O")
    axes[0].set_ylim(0,max(r["latency_us"] for r in rows)*1.44)
    axes[1].set_xticks(x,("QKT only","PV / PV+O","Gate / Up","Down"))
    axes[1].set_ylabel("Longest single complete work window (µs)")
    axes[1].set_title("Uninterrupted DQ-free work — channel 0")
    axes[1].set_ylim(0,65)
    for ax in axes:ax.legend(loc="upper left",fontsize=8)
    fig.suptitle("Score buffering × PIM output projection — common BF16 PV boundary", fontsize=14)
    fig.supxlabel("GPU O projection is unmeasured in non-O variants; this is not an end-to-end speedup comparison.\n"
                  "Right: one uninterrupted window, never a sum across transfers; includes local/control waits, not measured GPU service.", fontsize=9)
    fig.savefig(output / "latency-dq-breakdown.png", dpi=180)
    fig.savefig(output / "latency-dq-breakdown.pdf")
    plt.close(fig)
    from channel_analysis import dq_intervals, read_csv
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    max_end=0
    for ax, name, label in zip(axes, NAMES, ("Neither","Score DRAM","PIM O","Both")):
        case = root / name / "chain"
        report = channels[name]["stages"]["chain"]
        result = read(case / "simulation/result.json")
        ratio = result["config"]["frontend_ratio"]
        scale = result["tCK_ns"]/(ratio*1000)
        qphases = [p for p in report["phases"] if p["phase"].startswith("qk_")]
        start, end = min(p["begin_tick"] for p in qphases), max(p["end_tick"] for p in qphases)
        max_end=max(max_end,(end-start)*scale)
        rec=next(r for r in continuous["records"] if
            (r["variant"],r["case"],r["scope"],r["channel"])==(name,"chain","qkt",0))
        for w in continuous["complete_work_windows"]:
            if (w["variant"],w["case"],w["scope"],w["channel"])==(name,"chain","qkt",0):
                ax.axvspan((w["begin_tick"]-start)*scale,(w["end_tick"]-start)*scale,color="#44a89b",alpha=.20)
        intervals = dq_intervals(read_csv(case / "simulation/timeline.csv.ch0"), result["resolved_timings_ck"], ratio, "attacc")
        # Exact bursts avoid visually smearing a transfer into an adjacent
        # complete DQ-free window, as a coarse utilization bin would do.
        bursts=[((max(e["begin"],start)-start)*scale,
                 (min(e["end"],end)-max(e["begin"],start))*scale)
                for e in intervals if e["end"]>start and e["begin"]<end]
        ax.broken_barh(bursts,(0,1),facecolors="#4979ad",linewidth=0)
        ax.axvline((end-start)*scale,color="#555555",linestyle="--",linewidth=.8)
        noun="window" if rec['complete_work_window_count']==1 else "windows"
        ax.text(.01,.91,f"{rec['complete_work_window_count']} complete {noun}; longest {rec['longest_complete_work_window_us']:.6f} µs",
            transform=ax.transAxes,fontsize=9,va="top")
        ax.set_ylim(0,1.05)
        ax.set_yticks((0,1),("idle","busy"))
        ax.set_ylabel(label)
    axes[-1].set_xlabel("µs from first QKT tile (exact modeled DQ burst intervals)")
    axes[-1].set_xlim(0,max_end+.3)
    fig.suptitle("QKT channel 0: shaded regions are separate complete DQ-free work windows")
    fig.savefig(output / "qkt-dq-timeline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, help="Write derived reports outside an existing frozen native run")
    args=p.parse_args()
    compare(args.root,args.output)

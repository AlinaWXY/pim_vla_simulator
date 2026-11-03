"""Compare fresh legacy, independent-ingress, and rank-descriptor native runs."""
import argparse
import csv
import json
from pathlib import Path
import re

NAMES = dict(chain="Q/K + RoPE + QKT", v="V", pv="PV", gate_up_gpu="Gate/Up", down_gpu_input="Down")
FIELDS = ("stage_us", "modelled_dq_occupied_us", "modelled_dq_idle_us",
          "tail_clear_complete_zero_external_phase_us", "modelled_dq_idle_outside_tail_clear_work_us")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def phase_groups(run):
    groups = {}
    for phase, value in run["phase_us"].items():
        match = re.fullmatch(r"qk_tile_\d+_(compute|export)", phase)
        name = "qkt_" + match[1] if match else phase
        groups[name] = groups.get(name, 0.0) + value
    groups["phase_boundary_overhead"] = run["phase_boundary_us"]
    assert abs(sum(groups.values()) - run["latency_us"]) < 1e-8
    return groups


def compare(after, before, independent):
    roots = {"before": before, "independent_only": independent, "after": after}
    summaries = {mode: read(root / "summary.json") for mode, root in roots.items()}
    analyses = {mode: read(root / "channel-analysis.json") for mode, root in roots.items()}
    for mode in roots:
        assert summaries[mode]["status"] == analyses[mode]["status"] == "passed", mode
        assert summaries[mode]["staging_banks"] == 2, mode
    assert not summaries["before"]["channel_independent_dispatch"]
    assert not summaries["before"]["rank_descriptors"]
    assert all(set(s["runs"]) == set(summaries["before"]["runs"]) for s in summaries.values())
    assert summaries["independent_only"]["channel_independent_dispatch"]
    assert not summaries["independent_only"]["rank_descriptors"]
    assert summaries["after"]["channel_independent_dispatch"] and summaries["after"]["rank_descriptors"]
    result = dict(status="passed", before_commit="786842a", units="microseconds per layer/denoising step; DQ per channel",
                  modes={mode: dict(stage_sum_us=analysis["stage_sum_us"], channels=analysis["channels"])
                         for mode, analysis in analyses.items()}, cases={}, phase_breakdown={})
    rows = []
    for case in summaries["after"]["runs"]:
        runs = {mode: summary["runs"][case] for mode, summary in summaries.items()}
        natives = {mode: read(root / case / "simulation/result.json") for mode, root in roots.items()}
        for mode in ("independent_only", "after"):
            assert runs[mode]["external_DQ_bytes"] == runs["before"]["external_DQ_bytes"], (case, mode, "bytes")
            assert natives[mode]["resolved_timings_ck"] == natives["before"]["resolved_timings_ck"], (case, mode, "DRAM timings")
            for command, count in natives["before"]["issued_commands"].items():
                if not command.startswith(("ACT", "PRE", "REF")):
                    assert natives[mode]["issued_commands"].get(command, 0) == count, (case, mode, command)
        result["cases"][case] = {mode: dict(latency_us=run["latency_us"], memory_cycles=run["memory_cycles"],
              external_DQ_bytes=run["external_DQ_bytes"]) for mode, run in runs.items()}
        result["cases"][case]["saved_us"] = runs["before"]["latency_us"] - runs["after"]["latency_us"]
        result["phase_breakdown"][case] = {mode: phase_groups(run) for mode, run in runs.items()}
        assert all(set(p) == set(result["phase_breakdown"][case]["before"])
                   for p in result["phase_breakdown"][case].values()), (case, "phase keys")
        if case not in NAMES:
            continue
        for channel in range(16):
            parts = {mode: analysis["stages"][case]["channels"][channel] for mode, analysis in analyses.items()}
            for mode, part in parts.items():
                assert part["channel"] == channel
                assert abs(part["stage_us"] - part["modelled_dq_occupied_us"] - part["modelled_dq_idle_us"]) < 1e-8
                assert abs(part["modelled_dq_idle_us"] - part["tail_clear_complete_zero_external_phase_us"]
                           - part["modelled_dq_idle_outside_tail_clear_work_us"]) < 1e-8
                assert abs(part["modelled_dq_occupied_us"] - parts["before"]["modelled_dq_occupied_us"]) < 1e-8
                assert abs(part["command_bus"]["dq_idle_global_ca_busy_us"]
                           + part["command_bus"]["dq_idle_global_ca_free_us"] - part["modelled_dq_idle_us"]) < 1e-8
            metrics = {key: {mode: part[key] for mode, part in parts.items()} for key in FIELDS}
            for resource in ("global", "rank0", "rank1"):
                metrics[resource + "_CA_busy_us"] = {mode: part["command_bus"][resource]["occupied_us"] for mode, part in parts.items()}
            for key in ("dq_idle_global_ca_busy_us", "dq_idle_global_ca_free_us"):
                metrics[key] = {mode: part["command_bus"][key] for mode, part in parts.items()}
            for key, values in metrics.items():
                rows.append(dict(case=case, channel=channel, metric=key, **values,
                                 after_minus_before=values["after"] - values["before"]))
    old, new = analyses["before"]["stage_sum_us"], analyses["after"]["stage_sum_us"]
    result.update(saved_us=old-new, saved_percent=100*(old-new)/old,
                  preserved="External bytes, non-maintenance native command counts, DRAM timings; all 16-channel DQ partitions close",
                  interpretation="Mode differences include refresh/row-state phase changes; no concurrent GPU workload is simulated")
    (after / "rank-dispatch-comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    with (after / "rank-dispatch-channel-comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    phase_rows = []
    for case, modes in result["phase_breakdown"].items():
        for phase in modes["before"]:
            values = {mode: phases[phase] for mode, phases in modes.items()}
            phase_rows.append(dict(case=case, phase=phase, **values,
                                   saved_us=values["before"]-values["after"]))
    with (after / "rank-dispatch-phase-comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0])); writer.writeheader(); writer.writerows(phase_rows)
    lines = ["# Independent channel dispatch and rank descriptors", "",
             "Fresh native runs; latency in µs per layer/step. Baseline: xinyao_1103 commit 786842a.", "",
             "| Stage | Before | Independent dispatch only | Both fixes | Saved |", "|---|---:|---:|---:|---:|"]
    for case, name in NAMES.items():
        c = result["cases"][case]
        lines.append(f"| {name} | {c['before']['latency_us']:.6f} | {c['independent_only']['latency_us']:.6f} | {c['after']['latency_us']:.6f} | {c['saved_us']:.6f} |")
    lines += [f"| Total | {old:.6f} | {analyses['independent_only']['stage_sum_us']:.6f} | {new:.6f} | {old-new:.6f} |", "",
              "DQ table shows channel 0; the CSV audits all 16 channels. Full no-transfer windows exclude boundary data tails; other idle includes gaps within transfer phases.", "",
              "| Mode | DQ busy | DQ idle | Full no-transfer windows | Other DQ idle |", "|---|---:|---:|---:|---:|"]
    for mode, analysis in analyses.items():
        c = analysis["channels"][0]
        lines.append(f"| {mode} | {c['modelled_dq_occupied_us']:.6f} | {c['modelled_dq_idle_us']:.6f} | {c['tail_clear_complete_zero_external_phase_us']:.6f} | {c['modelled_dq_idle_outside_tail_clear_work_us']:.6f} |")
    lines += ["", "Per-phase latency: [CSV](rank-dispatch-phase-comparison.csv). Per-channel DQ and CA comparison: [CSV](rank-dispatch-channel-comparison.csv).", "",
              "Retained constraints: finite 32-entry per-channel ingress FIFO, one dispatch opportunity per channel/CK, actual external/rank CA arbitration, DRAM timing, SRF ownership, peer-export completion, and global barriers.", "",
              "These are modeled memory/PIM stages, not complete AE end-to-end latency. DQ availability is an upper bound; no competing GPU requests are injected."]
    (after / "RANK-DISPATCH-COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result, analyses


def plot(root, result, analyses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    colors = ("#4077b8", "#4f9c78", "#d6ab56")
    for i, (mode, label) in enumerate((("before", "Before"), ("after", "After"))):
        x = [j + (i-.5)*.34 for j in range(5)]
        parts = [analyses[mode]["stages"][case]["channels"][0] for case in NAMES]
        bottom = [0.0]*5
        for key, title, color in zip((FIELDS[1], FIELDS[3], FIELDS[4]),
                                    ("DQ busy", "Full no-transfer windows", "Other DQ idle"), colors):
            values = [p[key] for p in parts]
            axes[0].bar(x, values, .31, bottom=bottom, color=color, label=title if i==0 else None,
                        hatch="//" if i==0 else None, edgecolor="white", linewidth=.5)
            bottom = [a+b for a,b in zip(bottom,values)]
        for xpos, height in zip(x,bottom): axes[0].text(xpos,height+.5,f"{height:.1f}",ha="center",fontsize=8)
    axes[0].set_xticks(range(5), ["Q/K+RoPE+QKT", "V", "PV", "Gate/Up", "Down"])
    axes[0].set_title("Stage latency and channel-0 DQ partition\nHatched: before; solid: after")
    axes[0].set_ylabel("Microseconds / layer / step")
    axes[0].legend(frameon=False, fontsize=8)
    modes = ("before", "independent_only", "after")
    values = [analyses[m]["stage_sum_us"] for m in modes]
    axes[1].bar(range(3),values,color=["#8b96a5", "#5589bc", "#4f9c78"])
    axes[1].set_xticks(range(3),["Before", "Independent\nchannels", "+ Rank\ndescriptors"])
    axes[1].set_title(f"Five-stage sum: {result['saved_percent']:.2f}% reduction")
    for i,value in enumerate(values): axes[1].text(i,value+2,f"{value:.3f}",ha="center")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    fig.tight_layout()
    fig.savefig(root / "rank-dispatch-comparison.png",dpi=200)
    fig.savefig(root / "rank-dispatch-comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path); parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--dispatch-only",type=Path,required=True); parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    result,analyses=compare(args.run,args.before,args.dispatch_only)
    if args.plot: plot(args.run,result,analyses)
    print(json.dumps(dict(status=result["status"],saved_us=result["saved_us"],saved_percent=result["saved_percent"]),indent=2))

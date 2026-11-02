"""Channel time accounting from issued commands and semantic phase barriers.

The native experiment uses AttAcc movement timing. These are intervals in that
model, not a validated LPDDR electrical waveform or measured GPU availability.
Only complete compute/local-work phases enter the semantic-window statistic;
idle fragments inside transfer phases and release-only control phases do not.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

EXTERNAL = frozenset(("RD", "RDA", "WR", "WRA", "WRGB", "RDRES"))
CONTROL = frozenset(("BARRIER", "WAIT", "PIM_GEMV_RELEASE"))
PRIMARY = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def issue_tick(clock, ratio):
    """A memory edge m occurs at frontend tick (m-1)*ratio+1."""
    return (int(clock)-1)*ratio+1


def dq_intervals(rows, timings, ratio, data_path):
    """Return disjoint half-open modelled DQ intervals in frontend ticks.

    WRGB/RDRES have no CL/CWL in AttAcc. Ordinary RD/WR retain their LPDDR
    latency. A collision means this cannot describe even a serial model bus;
    reject it instead of silently taking a union and hiding the conflict.
    """
    require(data_path in ("attacc", "lpddr"), "Unknown PIM movement timing")
    burst = int(timings["nBL_min"])*ratio
    require(burst > 0 and ratio > 0, "Invalid clock/burst geometry")
    intervals = []
    for row in rows:
        command = row["command"]
        if command not in EXTERNAL:
            continue
        issued = issue_tick(row["clock"], ratio)
        if command in ("RD", "RDA") or command == "RDRES" and data_path == "lpddr":
            delay = int(timings["nCL"])*ratio
        elif command in ("WR", "WRA") or command == "WRGB" and data_path == "lpddr":
            delay = int(timings["nCWL"])*ratio
        else:
            delay = 0
        intervals.append(dict(begin=issued+delay, end=issued+delay+burst,
                              issue=issued, command=command, clock=int(row["clock"])))
    intervals.sort(key=lambda e: (e["begin"], e["end"]))
    for a, b in zip(intervals, intervals[1:]):
        require(a["end"] <= b["begin"],
                f"Modelled DQ overlap: {a['command']}@CK{a['clock']} "
                f"[{a['begin']},{a['end']}) and {b['command']}@CK{b['clock']} "
                f"[{b['begin']},{b['end']}) frontend ticks")
    return intervals


def interval_accounting(intervals, begin, end):
    """Occupied duration and maximal idle intervals, clipped to a window."""
    require(begin <= end, "Reversed interval window")
    cursor, occupied, idle = begin, 0, []
    for item in intervals:
        left, right = max(begin, item["begin"]), min(end, item["end"])
        if right <= left:
            continue
        require(left >= cursor, "Overlapping occupied intervals")
        if cursor < left:
            idle.append((cursor, left))
        occupied += right-left
        cursor = right
    if cursor < end:
        idle.append((cursor, end))
    require(occupied+sum(b-a for a, b in idle) == end-begin, "DQ partition does not close")
    return occupied, idle


def tail_clear_window(intervals, begin, end):
    """Conservative suffix after every incoming data tail has completed.

    Caller must first establish that the entire semantic phase has no
    external issue on this channel. A delayed preceding burst can start
    inside the phase; the pre-burst gap is deliberately not counted.
    """
    tail = max([begin]+[e["end"] for e in intervals
                        if e["issue"] < begin and e["end"] > begin and e["begin"] < end])
    return min(tail, end), end


def _has_work(segment):
    return segment.get("kind") != "gemv_release" and any(
        count > 0 and operation not in CONTROL for operation, count in segment.get("counts", {}).items())


def analyze_stage(directory, case):
    directory = Path(directory)
    manifest_path, result_path, barrier_path = (directory/"manifest.json",
        directory/"simulation/result.json", directory/"simulation/barriers.csv")
    manifest, result = read_json(manifest_path), read_json(result_path)
    config, timings = result["config"], result["resolved_timings_ck"]
    ratio, count = int(config["frontend_ratio"]), int(config["channels"])
    require(ratio > 0 and count > 0, "Invalid channel/clock geometry")
    tick_us = result["tCK_ns"]/(ratio*1000)
    wall = int(result["memory_cycles"])*ratio
    require(abs(wall*tick_us-result["latency_ns"]/1000) < 1e-8, "Native wall-time mismatch")
    require(result["transaction_bytes"] == 32, "Expected 32B external transactions")
    marker_rows = read_csv(barrier_path)
    markers = {int(r["marker"]): int(r["frontend_tick"]) for r in marker_rows}
    require(len(markers) == len(marker_rows), "Duplicate barrier marker")
    paths = [manifest_path, result_path, barrier_path]
    timelines, intervals, issued_ticks, external_counts = [], [], [], []
    for channel in range(count):
        path = directory/f"simulation/timeline.csv.ch{channel}"
        paths.append(path)
        rows = read_csv(path)
        require(all(int(e["Channel"]) == channel for e in rows), "Timeline channel mismatch")
        require(all(int(a["clock"]) <= int(b["clock"]) for a, b in zip(rows, rows[1:])),
                "Nonmonotonic issued commands")
        timelines.append(rows)
        try:
            intervals.append(dq_intervals(rows, timings, ratio, config["pim_data_path"]))
        except ValueError as error:
            raise ValueError(f"{case} channel {channel}: {error}") from error
        issued_ticks.append(sorted(issue_tick(e["clock"], ratio) for e in rows if e["command"] in EXTERNAL))
        external_counts.append(Counter(e["command"] for e in rows if e["command"] in EXTERNAL))
    actual = sum(external_counts, Counter())
    require(actual == Counter({c: n for c, n in result["issued_commands"].items() if c in EXTERNAL and n}),
            "Timeline differs from native external counters")
    require(sum(actual.values())*32 == manifest["expected_dq_bytes"]["total"], "External byte mismatch")
    phases, classified = [], [0]*count
    phase_ticks, previous_end = 0, None
    qualified_ticks, clear_ticks, windows = [0]*count, [0]*count, [[] for _ in range(count)]
    common_qualified, common_clear = 0, 0
    for segment in manifest["segments"]:
        begin, end = markers[segment["start"]], markers[segment["end"]]
        require(0 <= begin <= end <= wall, "Phase outside native wall time")
        require(previous_end is None or begin == previous_end, "Semantic phases must be contiguous")
        previous_end = end
        phase_ticks += end-begin
        counts = [bisect_left(ticks, end)-bisect_left(ticks, begin) for ticks in issued_ticks]
        classified = [a+b for a, b in zip(classified, counts)]
        meaningful = _has_work(segment)
        zero_all = meaningful and not any(counts)
        channel_rows = []
        for channel in range(count):
            qualified = meaningful and counts[channel] == 0
            clear_begin = end
            if qualified:
                qualified_ticks[channel] += end-begin
                clear_begin, _ = tail_clear_window(intervals[channel], begin, end)
                clear_ticks[channel] += end-clear_begin
                windows[channel].append(dict(phase=segment["name"], begin_tick=clear_begin, end_tick=end,
                    begin_us=clear_begin*tick_us, end_us=end*tick_us, duration_us=(end-clear_begin)*tick_us))
            occupied, _ = interval_accounting(intervals[channel], begin, end)
            channel_rows.append(dict(channel=channel, external_commands=counts[channel],
                complete_zero_external_phase=qualified, modelled_dq_occupied_us=occupied*tick_us,
                modelled_dq_idle_us=(end-begin-occupied)*tick_us,
                tail_clear_begin_tick=clear_begin if qualified else None,
                tail_clear_window_us=(end-clear_begin)*tick_us if qualified else 0.0))
        if zero_all:
            common_qualified += end-begin
            common_clear += end-max(r["tail_clear_begin_tick"] for r in channel_rows)
        phases.append(dict(case=case, phase=segment["name"], kind=segment.get("kind"),
            begin_tick=begin, end_tick=end, duration_us=(end-begin)*tick_us,
            meaningful_work=meaningful, complete_zero_external_phase_all_channels=zero_all,
            excluded_reason=None if meaningful else "control/release/WAIT-only phase",
            channels=channel_rows))
    require(classified == [len(t) for t in issued_ticks], "External command outside semantic phases")
    require(phase_ticks <= wall, "Phase accounting exceeds stage")
    channels = []
    for channel in range(count):
        occupied, idle = interval_accounting(intervals[channel], 0, wall)
        tail = sum(max(0, e["end"]-max(wall, e["begin"])) for e in intervals[channel])
        payload_ticks = len(intervals[channel])*int(timings["nBL_min"])*ratio
        require(occupied+tail == payload_ticks, "Payload/burst accounting does not close")
        channels.append(dict(channel=channel, stage_us=wall*tick_us,
            external_commands=dict(external_counts[channel]), external_DQ_bytes=len(intervals[channel])*32,
            modelled_dq_occupied_us=occupied*tick_us, modelled_dq_idle_us=(wall-occupied)*tick_us,
            modelled_dq_idle_outside_tail_clear_work_us=(wall-occupied-clear_ticks[channel])*tick_us,
            modelled_dq_tail_after_stage_us=tail*tick_us,
            modelled_idle_interval_count=len(idle),
            longest_modelled_idle_interval_us=max([0]+[b-a for a, b in idle])*tick_us,
            complete_zero_external_phase_us=qualified_ticks[channel]*tick_us,
            tail_clear_complete_zero_external_phase_us=clear_ticks[channel]*tick_us,
            removed_boundary_data_tail_us=(qualified_ticks[channel]-clear_ticks[channel])*tick_us,
            other_semantic_phase_us=(phase_ticks-qualified_ticks[channel])*tick_us,
            boundary_overhead_us=(wall-phase_ticks)*tick_us,
            tail_clear_windows=windows[channel]))
        require(abs(channels[-1]["stage_us"]-sum(channels[-1][k] for k in
                ("complete_zero_external_phase_us", "other_semantic_phase_us", "boundary_overhead_us"))) < 1e-8,
                "Semantic partition does not close")
        require(wall-occupied >= clear_ticks[channel], "Tail-clear work windows exceed modelled DQ idle")
    return dict(case=case, stage_us=wall*tick_us, frontend_ratio=ratio, tCK_ns=result["tCK_ns"],
        pim_data_path=config["pim_data_path"], channels=channels, phases=phases,
        complete_zero_external_phase_us_all_channels=common_qualified*tick_us,
        tail_clear_complete_zero_external_phase_us_all_channels=common_clear*tick_us,
        source_sha256={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def analyze(root):
    root = Path(root)
    summary = read_json(root/"summary.json")
    require(tuple(summary["primary_cases"]) == PRIMARY, "Aggregate exactly the five primary cases")
    stages = {case: analyze_stage(root/case, case) for case in PRIMARY}
    first = stages[PRIMARY[0]]
    count = len(first["channels"])
    require(all(len(s["channels"]) == count for s in stages.values()), "Channel counts differ by stage")
    fields = ("stage_us", "external_DQ_bytes", "modelled_dq_occupied_us", "modelled_dq_idle_us",
              "modelled_dq_idle_outside_tail_clear_work_us",
              "modelled_dq_tail_after_stage_us", "complete_zero_external_phase_us",
              "tail_clear_complete_zero_external_phase_us", "removed_boundary_data_tail_us",
              "other_semantic_phase_us", "boundary_overhead_us")
    channels = [dict(channel=ch, **{key: sum(s["channels"][ch][key] for s in stages.values())
                                  for key in fields}) for ch in range(count)]
    wall = sum(s["stage_us"] for s in stages.values())
    if summary.get("measured_stage_sum_us_per_layer_step") is not None:
        require(abs(summary["measured_stage_sum_us_per_layer_step"]-wall) < 1e-8, "Stage sum differs from runner")
    return dict(status="passed", units="microseconds per channel, per layer, per denoising step",
        stage_sum_us=wall, stage_sum_ms_18x10=wall*.18, primary_cases=list(PRIMARY),
        channels=channels, stages=stages,
        complete_zero_external_phase_us_all_channels=sum(s["complete_zero_external_phase_us_all_channels"] for s in stages.values()),
        tail_clear_complete_zero_external_phase_us_all_channels=sum(s["tail_clear_complete_zero_external_phase_us_all_channels"] for s in stages.values()),
        actual_GPU_usable_time_us=None, validated_physical_DQ_waveform=False,
        dq_interval_model="AttAcc WRGB/RDRES: issue+[0,nBL_min); ordinary RD/WR: issue+[nCL/nCWL,nCL/nCWL+nBL_min). All intervals checked disjoint.",
        clock_conversion="memory edge m -> frontend tick (m-1)*frontend_ratio+1; all intervals half-open",
        partitions=["stage = modelled DQ occupied + modelled DQ idle",
            "stage = modelled DQ occupied + tail-clear whole-work windows + modelled DQ idle outside those windows",
            "stage = complete zero-external-work phases + other semantic phases + boundary overhead",
            "complete zero-external-work phases = tail-clear suffix windows + removed incoming data tails"],
        limitations=["AttAcc movement timing is an abstraction; no validated LPDDR electrical DQ waveform",
            "No rank0 GPU workload, CA/rank-turnaround admission or GPU dispatch simulation; GPU usable time is unknown",
            "Five independent native stages; no invented time between stages or double-counted combined FFN",
            "Transfer-phase idle fragments and release/WAIT-only phases are excluded from whole-phase candidate windows",
            "GPU/nonlinear arithmetic and background VLM contention remain unmeasured"])


def write_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root
    output.mkdir(parents=True, exist_ok=True)
    report = analyze(root)
    (output/"channel-analysis.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    fields = ("case", "channel", "stage_us", "modelled_dq_occupied_us", "modelled_dq_idle_us",
              "modelled_dq_idle_outside_tail_clear_work_us",
              "complete_zero_external_phase_us", "tail_clear_complete_zero_external_phase_us",
              "removed_boundary_data_tail_us", "other_semantic_phase_us", "boundary_overhead_us",
              "modelled_dq_tail_after_stage_us", "external_DQ_bytes")
    with (output/"channel-partitions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case, stage in report["stages"].items():
            writer.writerows(dict(case=case, **row) for row in stage["channels"])
    with (output/"channel-phase-windows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("case", "phase", "kind", "channel", "begin_tick", "end_tick", "duration_us",
                  "external_commands", "complete_zero_external_phase", "tail_clear_begin_tick", "tail_clear_window_us",
                  "modelled_dq_occupied_us", "modelled_dq_idle_us")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for stage in report["stages"].values():
            for phase in stage["phases"]:
                writer.writerows(dict(phase, **row) for row in phase["channels"])
    lines = ["# Native H16 channel accounting", "",
        "All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.", "",
        "| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |",
        "|---|---:|---:|---:|---:|---:|"]
    for case, stage in report["stages"].items():
        values = lambda key: [r[key] for r in stage["channels"]]
        span = lambda key: f"{min(values(key)):.6f}–{max(values(key)):.6f}"
        lines.append(f"| {case} | {stage['stage_us']:.6f} | {span('modelled_dq_occupied_us')} | {span('modelled_dq_idle_us')} | {stage['complete_zero_external_phase_us_all_channels']:.6f} | {stage['tail_clear_complete_zero_external_phase_us_all_channels']:.6f} |")
    lines += ["", f"Five-stage sum: {report['stage_sum_us']:.6f} us; x18x10: {report['stage_sum_ms_18x10']:.6f} ms. This is a sum of measured stages, not complete AE end-to-end latency.", "",
        "Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. These windows still require GPU command/address, rank-turnaround and row-admission modelling before they can be called usable GPU time.", "",
        "The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.", ""]
    (output/"CHANNEL-ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = write_report(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("status", "stage_sum_us", "complete_zero_external_phase_us_all_channels",
        "tail_clear_complete_zero_external_phase_us_all_channels", "actual_GPU_usable_time_us")}, indent=2))

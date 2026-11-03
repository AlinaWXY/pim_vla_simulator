"""Compare ACT1/ACT2 deadline runs using raw native DQ timelines.

Example:
    python compare_act2_deadline.py --before results/validated-1902756 \
        --after results/validated-NEW

Requires complete raw result directories, not the compact reference snapshot.
The existing channel analyzer is loaded read-only; no simulation is rerun.
Seven cases are reported separately. Only the five primary stages are summed.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


PRIMARY = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")
CONTROLS = ("combined_timing_chain", "serial_layer_control")
CASES = PRIMARY + CONTROLS
SUM_NAME = "primary_five_stage_sum"
TOLERANCE_US = 1e-8
CHANNEL_FIELDS = (
    "stage_us", "external_DQ_bytes", "modelled_dq_occupied_us",
    "modelled_dq_idle_us", "tail_clear_complete_zero_external_phase_us",
    "modelled_dq_idle_outside_tail_clear_work_us", "modelled_dq_tail_after_stage_us",
    "complete_zero_external_phase_us", "removed_boundary_data_tail_us",
    "other_semantic_phase_us", "boundary_overhead_us",
)
BUS_FIELDS = (
    "global_CA_busy_us", "rank0_CA_busy_us", "rank1_CA_busy_us",
    "dq_idle_global_ca_busy_us", "dq_idle_global_ca_free_us",
)
KIND_FIELDS = ("wall_us", "dq_occupied_us", "dq_idle_us", "tail_clear_whole_work_us")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= TOLERANCE_US,
            f"{label}: {actual!r} != {expected!r}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_analyzer():
    path = Path(__file__).resolve().parent / "experiments/pi05-unified-gemv/channel_analysis.py"
    require(path.is_file(), f"Missing repository channel analyzer: {path}")
    spec = importlib.util.spec_from_file_location("act2_deadline_channel_analysis", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, path


def channel_metrics(part):
    metrics = {key: part[key] for key in CHANNEL_FIELDS}
    if "command_bus" in part:
        bus = part["command_bus"]
        metrics.update({f"{resource}_CA_busy_us": bus[resource]["occupied_us"]
                        for resource in ("global", "rank0", "rank1")})
        metrics.update({key: bus[key] for key in BUS_FIELDS[-2:]})
    return metrics


def validate_partition(metrics, label):
    for key, value in metrics.items():
        require(math.isfinite(value) and value >= -TOLERANCE_US,
                f"{label}: invalid {key}={value}")
    wall = metrics["stage_us"]
    busy, idle = metrics["modelled_dq_occupied_us"], metrics["modelled_dq_idle_us"]
    whole = metrics["tail_clear_complete_zero_external_phase_us"]
    close(wall, busy + idle, f"{label}: wall = DQ busy + idle")
    close(idle, whole + metrics["modelled_dq_idle_outside_tail_clear_work_us"],
          f"{label}: DQ idle = whole-work windows + other idle")
    close(metrics["complete_zero_external_phase_us"],
          whole + metrics["removed_boundary_data_tail_us"], f"{label}: incoming tails")
    close(wall, metrics["complete_zero_external_phase_us"]
          + metrics["other_semantic_phase_us"] + metrics["boundary_overhead_us"],
          f"{label}: semantic partition")
    if "dq_idle_global_ca_busy_us" in metrics:
        close(idle, metrics["dq_idle_global_ca_busy_us"]
              + metrics["dq_idle_global_ca_free_us"], f"{label}: DQ/CA intersection")
        for resource in ("global", "rank0", "rank1"):
            require(metrics[f"{resource}_CA_busy_us"] <= wall + TOLERANCE_US,
                    f"{label}: {resource} CA exceeds wall")


def load_run(root, analyzer):
    root = Path(root).resolve()
    summary = read_json(root / "summary.json")
    require(summary.get("status") == "passed", f"{root}: run did not pass")
    require(tuple(summary["primary_cases"]) == PRIMARY, f"{root}: primary case set changed")
    require(set(summary["runs"]) == set(CASES), f"{root}: expected exactly seven cases")
    stages, natives, traces = {}, {}, {}
    for case in CASES:
        print(f"Checking {root.name}/{case} from raw timelines", file=sys.stderr, flush=True)
        run = summary["runs"][case]
        native = read_json(root / case / "simulation/result.json")
        stage = analyzer.analyze_stage(root / case, case)
        require(run.get("audit_status") == "passed", f"{root}/{case}: native audit did not pass")
        require(run["memory_cycles"] == native["memory_cycles"], f"{case}: cycle mismatch")
        close(run["latency_us"], native["latency_ns"] / 1000, f"{case}: native latency")
        close(run["latency_us"], stage["stage_us"], f"{case}: timeline latency")
        close(run["tCK_ns"], native["tCK_ns"], f"{case}: clock period")
        close(sum(run["phase_us"].values()) + run["phase_boundary_us"], run["latency_us"],
              f"{case}: summary phase closure")
        phases = {phase["phase"]: phase["duration_us"] for phase in stage["phases"]}
        require(set(phases) == set(run["phase_us"]), f"{case}: raw/summary phase names differ")
        for name, value in phases.items():
            close(value, run["phase_us"][name], f"{case}/{name}: phase duration")
        require(run["traffic_bytes"] == native["traffic_bytes"], f"{case}: summary/native traffic differs")
        require(run["external_DQ_bytes"] == sum(c["external_DQ_bytes"] for c in stage["channels"]),
                f"{case}: per-channel bytes do not sum to summary")
        trace_hash = digest(root / case / "workload.trace")
        require(trace_hash == run["trace_sha256"], f"{case}: trace hash differs from summary")
        require(len(stage["channels"]) == 16, f"{case}: expected all 16 channels")
        for channel, part in enumerate(stage["channels"]):
            require(part["channel"] == channel, f"{case}: channel index mismatch")
            metrics = channel_metrics(part)
            validate_partition(metrics, f"{root.name}/{case}/ch{channel}")
            close(sum(p["wall_us"] for p in part["phase_kind_breakdown"].values())
                  + part["boundary_overhead_us"], stage["stage_us"], f"{case}: phase-kind closure")
            for kind, values in part["phase_kind_breakdown"].items():
                close(values["wall_us"], values["dq_occupied_us"] + values["dq_idle_us"],
                      f"{case}/ch{channel}/{kind}: DQ partition")
        stages[case], natives[case], traces[case] = stage, native, trace_hash
    primary_wall = sum(summary["runs"][case]["latency_us"] for case in PRIMARY)
    close(primary_wall, summary["measured_stage_sum_us_per_layer_step"], "Primary stage sum")
    require(sum(summary["runs"][case]["external_DQ_bytes"] for case in PRIMARY)
            == summary["external_DQ_bytes_per_layer_step"], "Primary external byte sum differs")
    return dict(root=root, summary=summary, stages=stages, natives=natives, traces=traces)


def pair(before, after):
    return dict(before=before, after=after, after_minus_before=after - before)


def nonmaintenance_counts(native):
    return {command: count for command, count in native["issued_commands"].items()
            if count and not command.startswith(("ACT", "PRE", "REF"))}


def compare_runs(before, after, analyzer_path):
    result = dict(
        status="passed", units="microseconds per layer/denoising step; DQ/CA time per channel; bytes",
        primary_cases=list(PRIMARY), control_cases=list(CONTROLS),
        aggregation="Only primary_five_stage_sum is a total. The two control cases are never added to it.",
        delta_convention="after_minus_before: negative means shorter or smaller",
        sources={mode: dict(directory=str(data["root"]), summary_sha256=digest(data["root"] / "summary.json"),
                            trace_sha256=data["traces"],
                            accounting_source_sha256={case: stage["source_sha256"] for case, stage in data["stages"].items()})
                 for mode, data in (("before", before), ("after", after))},
        analyzer=dict(path=str(analyzer_path), sha256=digest(analyzer_path), recomputed_from_raw_logs=True),
        checks=dict(trace_bytes_unchanged=True, all_traffic_bytes_unchanged=True,
                    nonmaintenance_command_counts_unchanged=True, dram_timing_parameters_unchanged=True,
                    all_16_channel_partitions_close=True, all_16_channel_external_commands_unchanged=True),
        limitations=[
            "DQ intervals are the existing AttAcc/LPDDR movement model, not a validated electrical waveform.",
            "DQ/CA idle is modeled resource availability; concurrent GPU service is not simulated.",
            "Phase deltas include refresh and row-state effects and are not a pure ACT-stall attribution.",
            "All seven cases remain separate; primary totals sum five independently measured stages only.",
        ], cases={}, primary_five_stage_sum={})
    for case in CASES:
        old, new = before["summary"]["runs"][case], after["summary"]["runs"][case]
        old_native, new_native = before["natives"][case], after["natives"][case]
        require(before["traces"][case] == after["traces"][case], f"{case}: workload trace changed")
        require(old["traffic_bytes"] == new["traffic_bytes"], f"{case}: data volume changed")
        require(old["external_DQ_bytes"] == new["external_DQ_bytes"], f"{case}: external bytes changed")
        require(old_native["resolved_timings_ck"] == new_native["resolved_timings_ck"],
                f"{case}: DRAM timing parameters changed")
        require(old_native["tCK_ns"] == new_native["tCK_ns"], f"{case}: tCK changed")
        require(nonmaintenance_counts(old_native) == nonmaintenance_counts(new_native),
                f"{case}: non-maintenance native command counts changed")
        for key in ("frontend_ratio", "memory_ratio", "channels", "ranks", "pim_data_path",
                    "pim_staging_banks", "pim_rank_descriptors", "channel_independent_dispatch"):
            require(old_native["config"].get(key) == new_native["config"].get(key),
                    f"{case}: comparison geometry/control mode changed: {key}")
        require(set(old["phase_us"]) == set(new["phase_us"]), f"{case}: phase names changed")
        phases = {name: pair(old["phase_us"][name], new["phase_us"][name]) for name in old["phase_us"]}
        phases["phase_boundary_overhead"] = pair(old["phase_boundary_us"], new["phase_boundary_us"])
        latency = pair(old["latency_us"], new["latency_us"])
        close(sum(p["after_minus_before"] for p in phases.values()), latency["after_minus_before"],
              f"{case}: phase deltas close")
        entry = dict(scope="primary_stage" if case in PRIMARY else "control_only", latency_us=latency,
                     memory_cycles=pair(old["memory_cycles"], new["memory_cycles"]),
                     external_DQ_bytes=pair(old["external_DQ_bytes"], new["external_DQ_bytes"]),
                     saved_us=old["latency_us"] - new["latency_us"],
                     saved_percent=100 * (old["latency_us"] - new["latency_us"]) / old["latency_us"],
                     phases=phases, channels=[], phase_kinds=[],
                     native_command_counts={command: pair(old_native["issued_commands"].get(command, 0),
                                                          new_native["issued_commands"].get(command, 0))
                         for command in sorted(set(old_native["issued_commands"]) | set(new_native["issued_commands"]))})
        for channel in range(16):
            left, right = (data["stages"][case]["channels"][channel] for data in (before, after))
            require(left["external_commands"] == right["external_commands"],
                    f"{case}/ch{channel}: external command counts changed")
            require(left["external_DQ_bytes"] == right["external_DQ_bytes"],
                    f"{case}/ch{channel}: external bytes changed")
            close(left["modelled_dq_occupied_us"] + left["modelled_dq_tail_after_stage_us"],
                  right["modelled_dq_occupied_us"] + right["modelled_dq_tail_after_stage_us"],
                  f"{case}/ch{channel}: total DQ burst duration changed")
            old_metrics, new_metrics = channel_metrics(left), channel_metrics(right)
            require(set(old_metrics) == set(new_metrics), f"{case}: DQ/CA metric set changed")
            entry["channels"].append(dict(channel=channel, metrics={
                key: pair(old_metrics[key], new_metrics[key]) for key in old_metrics}))
            old_kinds, new_kinds = left["phase_kind_breakdown"], right["phase_kind_breakdown"]
            require(set(old_kinds) == set(new_kinds), f"{case}: phase kind set changed")
            for kind in old_kinds:
                entry["phase_kinds"].append(dict(channel=channel, kind=kind, metrics={
                    key: pair(old_kinds[kind][key], new_kinds[kind][key]) for key in KIND_FIELDS}))
        result["cases"][case] = entry
    total = result["primary_five_stage_sum"]
    total["scope"], total["included_cases"] = "primary_sum", list(PRIMARY)
    for metric in ("latency_us", "external_DQ_bytes"):
        total[metric] = pair(*(sum(result["cases"][case][metric][mode] for case in PRIMARY)
                               for mode in ("before", "after")))
    total["saved_us"] = -total["latency_us"]["after_minus_before"]
    total["saved_percent"] = 100 * total["saved_us"] / total["latency_us"]["before"]
    total["channels"] = []
    for channel in range(16):
        fields = result["cases"][PRIMARY[0]]["channels"][channel]["metrics"]
        metrics = {key: pair(*(sum(result["cases"][case]["channels"][channel]["metrics"][key][mode]
                                  for case in PRIMARY) for mode in ("before", "after"))) for key in fields}
        for mode in ("before", "after"):
            validate_partition({key: value[mode] for key, value in metrics.items()}, f"primary/ch{channel}/{mode}")
        total["channels"].append(dict(channel=channel, metrics=metrics))
    return result


def write_csv(path, rows):
    require(bool(rows), f"No output rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = dict(result["cases"], **{SUM_NAME: result["primary_five_stage_sum"]})
    latency_rows, channel_rows, phase_rows, kind_rows, command_rows = [], [], [], [], []
    for case, entry in cases.items():
        row = dict(case=case, scope=entry["scope"])
        for metric in ("latency_us", "external_DQ_bytes"):
            row.update({f"{mode}_{metric}": value for mode, value in entry[metric].items()})
        row.update(saved_us=entry["saved_us"], saved_percent=entry["saved_percent"])
        latency_rows.append(row)
        for part in entry["channels"]:
            channel_rows.extend(dict(case=case, scope=entry["scope"], channel=part["channel"], metric=metric, **value)
                                for metric, value in part["metrics"].items())
        phase_rows.extend(dict(case=case, scope=entry["scope"], phase=phase, **value)
                          for phase, value in entry.get("phases", {}).items())
        for part in entry.get("phase_kinds", []):
            kind_rows.extend(dict(case=case, scope=entry["scope"], channel=part["channel"], kind=part["kind"],
                                  metric=metric, **value) for metric, value in part["metrics"].items())
        command_rows.extend(dict(case=case, scope=entry["scope"], command=command, **value)
                            for command, value in entry.get("native_command_counts", {}).items())
    for name, rows in (("case-latency", latency_rows), ("channel-dq-comparison", channel_rows),
                       ("phase-latency", phase_rows), ("phase-kind-dq-comparison", kind_rows),
                       ("native-command-comparison", command_rows)):
        write_csv(output / f"{name}.csv", rows)
    (output / "act2-deadline-comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Complete raw run before the timing fix")
    parser.add_argument("--after", type=Path, required=True, help="Complete raw run after the timing fix")
    parser.add_argument("--output", type=Path, help="Output directory (default: AFTER/act2-deadline-comparison)")
    parser.add_argument("--check-only", action="store_true", help="Validate and print totals without writing files")
    args = parser.parse_args()
    try:
        analyzer, analyzer_path = load_analyzer()
        before = load_run(args.before, analyzer)
        after = before if args.before.resolve() == args.after.resolve() else load_run(args.after, analyzer)
        result = compare_runs(before, after, analyzer_path)
        output = args.output or args.after / "act2-deadline-comparison"
        if not args.check_only:
            write_report(result, output)
        print(json.dumps(dict(status=result["status"], case_count=len(result["cases"]),
                              primary_cases=list(PRIMARY),
                              primary_latency_us=result["primary_five_stage_sum"]["latency_us"],
                              saved_us=result["primary_five_stage_sum"]["saved_us"],
                              saved_percent=result["primary_five_stage_sum"]["saved_percent"],
                              output=None if args.check_only else str(output.resolve())), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"ACT2 deadline comparison failed: {error}\n")


if __name__ == "__main__":
    main()

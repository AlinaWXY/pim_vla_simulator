"""Audit saved resource_partition traces; no simulation or timing changes.

The exact accounting identity is window = RD_count * nBL_min + RD gaps + nCL.
Gap annotations describe the issued sequence, not scheduler stall attribution.
"""

import argparse
import csv
import hashlib
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def analyze_channel(events, completions, timing, transaction_bytes):
    burst = timing["nBL_min"]
    latency = timing["nCL"] + burst
    if any(e["command"] in ("WR", "WRA", "RDA", "RD_L", "RDA_L") for e in events):
        raise ValueError("Only fixed-burst, read-only PartitionTrace runs are supported")
    clocks = [int(e["clock"]) for e in events]
    if clocks != sorted(clocks) or len(set(clocks)) != len(clocks):
        raise ValueError("Expected strictly increasing per-channel command clocks")
    reads = [e for e in events if e["command"] == "RD"]
    if not reads:
        if completions:
            raise ValueError("Read completions without RD commands")
        return None
    issues = [int(e["clock"]) for e in reads]
    done = sorted(completions, key=lambda e: int(e["depart"]))
    if [int(e["depart"]) for e in done] != [t + latency for t in issues]:
        raise ValueError("RD/completion mismatch or nonconstant read completion latency")
    admissions = sorted(int(e["arrive"]) for e in done)
    waits = [issue - int(e["arrive"]) for issue, e in zip(issues, done)]
    if min(waits) < 1:
        raise ValueError("Expected admission before the controller issue tick")
    ranks = {e["Rank"] for e in reads}
    if len(ranks) != 1:
        raise ValueError("This audit supports one GPU rank per channel")

    last_bg, last_row, last_act2, last_pre = {}, {}, {}, {}
    gaps, histogram = [], Counter()
    previous = None
    for event in events:
        clock = int(event["clock"])
        bank = tuple(event[k] for k in ("Rank", "BankGroup", "Bank"))
        group = bank[:2]
        command = event["command"]
        if command == "ACT2":
            last_act2[bank] = clock
        elif command == "PREpb":
            last_pre[bank] = clock
        if command != "RD":
            continue
        if previous is not None:
            prev = int(previous["clock"])
            delta = clock - prev
            histogram[delta] += 1
            if delta < burst:
                raise ValueError("Overlapping DQ bursts")
            # Lower bound conditional on the observed RD order. An alternative
            # ready bank could change this bound; this is not a lost-cycle cause.
            column_ready = max(prev + burst, prev + timing["nCCDS"],
                               last_bg.get(group, -timing["nCCDL"]) + timing["nCCDL"])
            if clock < column_ready:
                raise ValueError("Observed RD sequence violates column timing")
            if delta > burst:
                start = prev + burst
                # PartitionTrace injects before the controller increments m_clk:
                # arrive=t is visible to scheduling at t+1. All admitted reads
                # minus already issued RDs includes read/active/activating buffers.
                available = [bisect_left(admissions, t) - bisect_left(issues, t)
                             for t in range(start, clock)]
                if min(available) < 0:
                    raise ValueError("Inconsistent admission/issue history")
                act = last_act2.get(bank)
                gaps.append({
                    "previous_rd_ck": prev, "next_rd_ck": clock,
                    "gap_start_ck": start, "gap_cycles": delta - burst,
                    "next_bg": int(event["BankGroup"]), "next_bank": int(event["Bank"]),
                    "next_row": int(event["Row"]),
                    "next_rd_changes_bank_row": bank in last_row
                    and last_row[bank] != event["Row"],
                    "next_bank_pre_ck": last_pre.get(bank), "next_bank_act2_ck": act,
                    "next_bank_rcd_ready_ck": act + timing["nRCD"] if act is not None else None,
                    "next_rd_at_rcd_boundary": act is not None
                    and clock == act + timing["nRCD"],
                    "observed_order_column_gap_ck": column_ready - start,
                    "remaining_gap_ck": clock - column_ready,
                    "min_admitted_unissued_reads": min(available),
                    "no_admitted_read_cycles": available.count(0),
                })
        last_bg[group] = clock
        last_row[bank] = event["Row"]
        previous = event

    window = int(done[-1]["depart"]) - issues[0]
    gap_cycles = sum(g["gap_cycles"] for g in gaps)
    busy = len(reads) * burst
    if window != busy + gap_cycles + timing["nCL"]:
        raise ValueError("Cycle accounting failed")
    wait_sorted = sorted(waits)
    return {
        "rd_count": len(reads), "first_rd_ck": issues[0], "last_rd_ck": issues[-1],
        "last_completion_ck": int(done[-1]["depart"]), "window_ck": window,
        "dq_busy_ck": busy, "rd_gap_ck": gap_cycles, "boundary_ck": timing["nCL"],
        "dq_utilization": busy / window,
        "bandwidth_GBps": len(reads) * transaction_bytes / (window * timing["tCK_ps"] / 1000),
        "rd_interval_histogram": dict(sorted(histogram.items())),
        "observed_order_column_gap_ck": sum(g["observed_order_column_gap_ck"] for g in gaps),
        "remaining_gap_ck": sum(g["remaining_gap_ck"] for g in gaps),
        "no_admitted_read_gap_ck": sum(g["no_admitted_read_cycles"] for g in gaps),
        "min_admitted_unissued_reads_in_gaps": min(
            (g["min_admitted_unissued_reads"] for g in gaps), default=None),
        "gaps_ending_on_changed_row_ck": sum(
            g["gap_cycles"] for g in gaps if g["next_rd_changes_bank_row"]),
        "admission_to_issue_mean_ck": sum(waits) / len(waits),
        "admission_to_issue_p99_ck": wait_sorted[(99 * len(waits) + 99) // 100 - 1],
        "issue_to_completion_ck": latency,
        "commands": dict(Counter(e["command"] for e in events)),
        "largest_gaps": sorted(gaps, key=lambda g: -g["gap_cycles"])[:8],
    }


def analyze_case(path):
    result_path = path / "result.json"
    result = json.loads(result_path.read_text())
    config_path = path / "ramulator_config.json"
    config = json.loads(config_path.read_text())
    if config["frontend"]["impl"] != "PartitionTrace":
        raise ValueError("Expected a PartitionTrace result")
    if (config["frontend"]["clock_ratio"], config["memory_system"]["clock_ratio"]) != (1, 1):
        raise ValueError("Expected one frontend tick per memory tick")
    completion_path = path / "completions.csv"
    completions = read_csv(completion_path)
    sources = [result_path, config_path, completion_path]
    channels = []
    for original in result["per_channel"]:
        channel = original["channel"]
        timeline = path / f"timeline.csv.ch{channel}"
        sources.append(timeline)
        reads = [e for e in completions if int(e["type"]) == 0 and int(e["channel"]) == channel]
        audit = analyze_channel(read_csv(timeline), reads, result["timings"],
                                result["transaction_bytes"])
        if audit is None:
            if original["rd_bursts"]:
                raise ValueError("Missing expected channel reads")
            continue
        if (audit["rd_count"], audit["window_ck"]) != (
                original["rd_bursts"], original["gpu_active_cycles"]):
            raise ValueError("Audit disagrees with original summary")
        channels.append({"channel": channel, **audit,
                         "read_queue_mean": original["read_queue_gpu_window_avg"]})
    if not channels:
        raise ValueError("No GPU reads")
    global_window = max(c["last_completion_ck"] for c in channels) - min(
        c["first_rd_ck"] for c in channels)
    bandwidth = sum(c["rd_count"] for c in channels) * result["transaction_bytes"] / (
        global_window * result["timings"]["tCK_ps"] / 1000)
    if abs(bandwidth - result["aggregate_bandwidth_GBps"]) > 1e-9:
        raise ValueError("Aggregate bandwidth disagrees with original summary")
    return {
        "label": f"{path.parent.name}/{path.name}", "source": str(path.resolve()),
        "config": result["config"], "timings": result["timings"],
        "aggregate_bandwidth_GBps": bandwidth,
        "peak_GBps": result["active_channel_peak_GBps"],
        "per_channel": channels,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }


def write_report(output, cases):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "analysis": "offline analysis of previously saved simulations",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definitions": {
            "cycle_accounting": "last completion - first RD = N*nBL_min + sum(RD intervals "
            "- nBL_min) + nCL; nCL is a window-boundary term, not an internal stall",
            "column_gap": "minimum extra spacing for the observed RD order from nCCDS/nCCDL; "
            "not a causal split of scheduler stalls",
            "remaining_gap": "gap after observed-order column constraints; may include row "
            "timings, refresh, CA arbitration and request availability",
            "request_supply": "admitted but not issued = admissions with arrive<t minus "
            "RDs issued before t; checks every CK in every RD gap",
            "latency": "admission-to-issue includes queueing, prerequisites and scheduling; "
            "excludes frontend backpressure; completion follows RD by nCL+nBL_min",
            "limitations": "issued-command traces cannot prove which ready alternative the "
            "scheduler rejected; row-change annotations and command overlap are correlations",
        }, "cases": cases,
    }
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    gaps = []
    for case in cases:
        for channel in case["per_channel"]:
            rows.append({"case": case["label"], "aggregate_GBps": case["aggregate_bandwidth_GBps"],
                         **{k: v for k, v in channel.items() if not isinstance(v, (dict, list))}})
            gaps.extend({"case": case["label"], "channel": channel["channel"], **gap}
                        for gap in channel["largest_gaps"])
    for name, data in (("per_channel.csv", rows), ("largest_gaps.csv", gaps)):
        if data:
            with (output / name).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="saved resource_partition suite directory; repeat for refresh runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    cases = []
    for root in args.input:
        suite = json.loads((root / "results.json").read_text())
        for entry in suite["results"]:
            label = entry["label"]
            if Path(label).name != label or label in (".", ".."):
                raise ValueError("Expected a direct child case directory")
            case = analyze_case(root / label)
            cases.append(case)
            c = case["per_channel"][0]
            print(f"{case['label']}: {case['aggregate_bandwidth_GBps']:.4f} GB/s; "
                  f"CH{c['channel']} {c['window_ck']} CK = {c['dq_busy_ck']} DQ + "
                  f"{c['rd_gap_ck']} gaps + {c['boundary_ck']} boundary; "
                  f"no-demand gap CK={c['no_admitted_read_gap_ck']}")
    write_report(args.output, cases)


if __name__ == "__main__":
    main()

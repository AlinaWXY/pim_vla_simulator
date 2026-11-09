"""Independently audit and summarize one frozen full-VLM calibration.

Usage: python analyze_calibration.py CAMPAIGN --output REPORT_DIRECTORY
Copied campaigns are supported: the recorded raw_path is relocated into the
supplied campaign. No simulation, input modification, or runner imports occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=0.002):
    need(math.isfinite(actual) and math.isfinite(expected)
         and abs(actual - expected) <= tolerance,
         f"{message}: actual={actual}, expected={expected}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if lo != hi else values[lo]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(campaign, output, top):
    campaign, output = campaign.resolve(), output.resolve()
    calibration_path = campaign / "calibration/calibration.json"
    calibration = read(calibration_path)
    recorded_raw = calibration["raw_path"].replace("\\", "/")
    marker = "/calibration/"
    need(marker in recorded_raw, "raw_path is not inside a calibration directory")
    raw_path = (campaign / "calibration" / recorded_raw.rsplit(marker, 1)[1]).resolve()
    need(raw_path.is_relative_to(campaign / "calibration"), "raw_path escapes calibration directory")
    candidate = calibration["selected_candidate"]
    need(raw_path == campaign / "calibration" / f"candidate-{candidate:02d}" / "raw.json",
         "raw_path differs from selected candidate")
    vlm_path = campaign / "frozen/source/input-data/vlm.json"
    need(digest(raw_path) == calibration["raw_sha256"], "selected raw SHA-256 mismatch")
    need(digest(vlm_path) == calibration["vlm_sha256"], "VLM input SHA-256 mismatch")
    raw, vlm = read(raw_path), read(vlm_path)
    params, metrics = calibration["parameters"], raw["metrics"]
    alpha = params["alpha"]
    need(0.05 <= alpha <= 1, "alpha outside declared bounds")
    units, observed = vlm["units"], raw["kernels"]
    need(len(units) == len(observed) == 2357, "expected 2357 complete VLM work units")
    need(Counter(u["kind"] for u in units) == {"kernel": 2353, "memory": 4},
         "expected 2353 kernels and four memory copies")
    need([u["ordinal"] for u in units] == list(range(2357)), "input ordinals differ")
    need([u["ordinal"] for u in observed] == list(range(2357)), "native ordinals differ or repeat")
    need(not raw["stages"] and metrics["pim_read_bytes"] == metrics["pim_write_bytes"] == 0,
         "selected raw is not a standalone VLM run")
    close(metrics["AE_elapsed_us"], 0, "standalone AE time")

    rows, groups = [], defaultdict(list)
    previous_end = 0.0
    expected_read = expected_write = 0
    channel_bytes = [dict(read=0, write=0) for _ in range(16)]
    for source, actual in zip(units, observed):
        ordinal = source["ordinal"]
        close(actual["start_us"], previous_end, f"kernel {ordinal} timeline gap")
        duration = actual["end_us"] - actual["start_us"]
        close(duration, actual["duration_us"], f"kernel {ordinal} duration", 0.000002)
        need(duration + 0.000001 >= source["us"], f"kernel {ordinal} below measured floor")
        reads = source["read_sectors"]
        writes = math.ceil(alpha * source["write_sectors"]) if source["write_is_proxy"] else source["write_sectors"]
        need(actual["read_bytes"] == reads * 32 and actual["write_bytes"] == writes * 32,
             f"kernel {ordinal} byte conservation failed")
        expected_read += reads * 32
        expected_write += writes * 32
        for channel in range(16):
            channel_bytes[channel]["read"] += ((reads + 15 - channel) // 16) * 32
            channel_bytes[channel]["write"] += ((writes + 15 - channel) // 16) * 32
        excess = duration - source["us"]
        row = dict(ordinal=ordinal, kind=source["kind"], name=source["name"],
                   measured_us=source["us"], native_us=duration,
                   excess_over_measured_us=excess,
                   relative_error=excess / source["us"] if source["us"] else None,
                   read_bytes=reads * 32, write_bytes=writes * 32,
                   start_us=actual["start_us"], end_us=actual["end_us"])
        rows.append(row)
        if row["kind"] == "kernel":
            groups[row["name"]].append(row)
        previous_end = actual["end_us"]
    close(previous_end, metrics["total_us"], "final completion timestamp")
    close(sum(r["native_us"] for r in rows), metrics["total_us"], "serial kernel duration sum")
    close(sum(r["measured_us"] for r in rows), vlm["measured_duration_sum_us"], "measured duration sum")
    need((metrics["gpu_read_bytes"], metrics["gpu_write_bytes"]) == (expected_read, expected_write),
         "global GPU byte conservation failed")
    need(metrics["GPU_reads"] * 32 == expected_read and metrics["GPU_writes"] * 32 == expected_write,
         "GPU command/byte conservation failed")
    need(metrics["GPU_row_hits"] + metrics["GPU_row_misses"] == metrics["GPU_reads"] + metrics["GPU_writes"],
         "GPU row hit/miss conservation failed")

    spec = calibration["spec"]
    close(raw["clock_ps"], spec["clock_ps"], "spec/raw CK duration", 0.000001)
    total_cycles = round(metrics["total_us"] / (raw["clock_ps"] / 1_000_000))
    burst_read = spec["metadata"]["commands"]["RD"]["burst"]
    burst_write = spec["metadata"]["commands"]["WR"]["burst"]
    buses = sorted(raw["bus"], key=lambda row: row["channel"])
    need(len(buses) == 16 and [b["channel"] for b in buses] == list(range(16)), "bus channel coverage differs")
    bus_rows = []
    for bus in buses:
        channel = bus["channel"]
        need(bus["stage_id"] == -1 and bus["duration_cycles"] == total_cycles,
             f"channel {channel} standalone duration/stage differs")
        need(0 <= bus["ca_cycles"] <= total_cycles and 0 <= bus["dq_cycles"] <= total_cycles,
             f"channel {channel} bus utilization exceeds physical bound")
        for direction in ("read", "write"):
            need(bus[f"gpu_{direction}_bytes"] == channel_bytes[channel][direction],
                 f"channel {channel} GPU {direction} bytes differ")
            need(bus[f"pim_{direction}_bytes"] == 0 and bus[f"{direction}_bytes"] == channel_bytes[channel][direction],
                 f"channel {channel} combined {direction} bytes differ")
        expected_dq = (channel_bytes[channel]["read"] // 32 * burst_read
                       + channel_bytes[channel]["write"] // 32 * burst_write)
        need(bus["dq_cycles"] == expected_dq, f"channel {channel} DQ occupancy conservation failed")
        bus_rows.append(dict(bus, ca_utilization=bus["ca_cycles"] / total_cycles,
                             dq_utilization=bus["dq_cycles"] / total_cycles))

    kernel_rows = [r for r in rows if r["kind"] == "kernel"]
    errors = [r["relative_error"] for r in kernel_rows if r["relative_error"] is not None]
    total, target = metrics["total_us"], vlm["target_us"]
    validation = dict(total_us=total, target_us=target, relative_error=(total - target) / target,
                      kernel_relative_error_median=percentile(errors, .5),
                      kernel_relative_error_p90=percentile(errors, .9),
                      work_units=2357, kernel_count=2353, passed=abs((total - target) / target) <= .05)
    for name, value in validation.items():
        close(value, calibration["validation"][name], f"frozen assessment {name}", 0.00000001)
    need(validation["passed"] and calibration["no_latency_offsets"] is True, "calibration gate/offset check failed")
    group_rows = []
    for name, members in groups.items():
        measured = sum(r["measured_us"] for r in members)
        native = sum(r["native_us"] for r in members)
        group_rows.append(dict(name=name, count=len(members), measured_us=measured, native_us=native,
                               excess_over_measured_us=native - measured,
                               weighted_relative_error=(native - measured) / measured if measured else None,
                               read_bytes=sum(r["read_bytes"] for r in members),
                               write_bytes=sum(r["write_bytes"] for r in members)))
    group_rows.sort(key=lambda r: r["excess_over_measured_us"], reverse=True)
    ranked = sorted(kernel_rows, key=lambda r: r["excess_over_measured_us"], reverse=True)
    counters = {key: metrics[key] for key in ("GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab",
                                              "GPU_reads", "GPU_writes", "GPU_row_hits", "GPU_row_misses")}
    summary = dict(schema_version=1, passed=True, campaign=str(campaign), selected_candidate=candidate,
                   parameters=params, validation=validation,
                   measured_duration_sum_us=vlm["measured_duration_sum_us"],
                   excess_over_measured_sum_us=total - vlm["measured_duration_sum_us"],
                   gate_upper_us=target * 1.05, source_commit=calibration["source_commit"],
                   source_hashes=dict(calibration=digest(calibration_path), raw=digest(raw_path), vlm=digest(vlm_path)),
                   binary_sha256=calibration["binary_sha256"], runtime_libraries=calibration["runtime_libraries"],
                   gpu_read_bytes=expected_read, gpu_write_bytes=expected_write,
                   kernel_write_proxy_bytes=vlm["write_proxy_bytes"],
                   exact_copy_write_bytes=vlm["exact_copy_write_bytes_rounded_to_sectors"],
                   counters=counters,
                   row_hit_fraction=metrics["GPU_row_hits"] / (metrics["GPU_reads"] + metrics["GPU_writes"]),
                   bus=dict(channels=16, duration_cycles_per_channel=total_cycles,
                            aggregate_ca_cycles=sum(b["ca_cycles"] for b in buses),
                            aggregate_dq_cycles=sum(b["dq_cycles"] for b in buses),
                            mean_ca_utilization=sum(b["ca_cycles"] for b in buses) / (16 * total_cycles),
                            mean_dq_utilization=sum(b["dq_cycles"] for b in buses) / (16 * total_cycles),
                            ca_scope="All external CA commands, including refresh on both ranks"),
                   top_kernel_groups=group_rows[:top], top_kernel_ordinals=ranked[:top],
                   checks=["2357 ordered unique units: 2353 kernels + 4 copies", "measured duration floor and contiguous standalone timeline",
                           "per-unit, global and per-channel GPU byte conservation", "RD/WR and row hit/miss command conservation",
                           "16-channel DQ cycles match transfer bytes and spec bursts", "CA/DQ utilization within bounds",
                           "selected raw/input hashes and recomputed frozen calibration assessment"],
                   interpretation="Native simulation bytes, not additional hardware DRAM measurements; alpha scales kernel write proxies only.",
                   error_definition="Native standalone duration minus unchanged measured duration; not the concurrent VLM_memory_stall metric.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output / "kernel-errors.csv", rows)
    write_csv(output / "kernel-groups.csv", group_rows)
    write_csv(output / "bus.csv", bus_rows)
    lines = ["# VLM calibration audit", "",
             f"Passed: {total / 1000:.6f} ms versus {target / 1000:.6f} ms; error {validation['relative_error']:.3%}.",
             f"Candidate {candidate}: `{json.dumps(params, sort_keys=True)}`.", "",
             f"Kernel error median {validation['kernel_relative_error_median']:.3%}; p90 {validation['kernel_relative_error_p90']:.3%}.",
             f"Unchanged measured duration sum {vlm['measured_duration_sum_us'] / 1000:.6f} ms; native excess {(total - vlm['measured_duration_sum_us']) / 1000:.6f} ms.",
             f"GPU read/write bytes: {expected_read:,} / {expected_write:,}; row hit fraction {summary['row_hit_fraction']:.3%}.",
             f"Mean external CA/DQ utilization: {summary['bus']['mean_ca_utilization']:.3%} / {summary['bus']['mean_dq_utilization']:.3%}.", "",
             "All 2357 units, bytes, DQ cycles, timestamps and frozen assessment checks passed.",
             "CA includes refresh on both ranks. Native traffic is simulated; alpha is a fitted writeback fraction, not a measured DRAM counter.",
             "Total error is the acceptance gate. Kernel median/p90 and the following rankings are diagnostics.", "",
             "| Kernel name | Count | Measured ms | Native ms | Excess ms |", "|---|---:|---:|---:|---:|"]
    for row in group_rows[:top]:
        name = row["name"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {row['count']} | {row['measured_us']/1000:.6f} | {row['native_us']/1000:.6f} | {row['excess_over_measured_us']/1000:.6f} |")
    lines += ["", "Full per-ordinal errors, grouped errors and channel details are in the adjacent CSV files.", ""]
    (output / "calibration-report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    need(args.top > 0, "--top must be positive")
    summary = analyze(args.campaign, args.output, args.top)
    print(json.dumps({"passed": summary["passed"], "selected_candidate": summary["selected_candidate"],
                      "parameters": summary["parameters"], "validation": summary["validation"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

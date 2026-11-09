"""Independently audit native concurrency outputs and write a CSV/JSON table.

No simulator/runner imports, no remote access, and no result rewriting.
GPU byte ratios describe command-issue stage attribution, not overlap time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

CASES = ("b", "c", "d", "e", "f", "g", "e-frfcfs", "e-lookahead", "e-gap")
VARIANTS = dict(zip(CASES, (
    "feature_gpu_before", "feature_gpu_before", "token_gpu_before",
    "token_local_before", "token_gpu_after", "token_local_after",
    "token_local_before", "token_local_before", "token_local_before")))
PAIR_SET = {(step, layer) for step in range(10) for layer in range(18)}
TOLERANCE_US = 0.002


def require(value, message):
    if not value:
        raise ValueError(message)


def near(a, b, message, tolerance=TOLERANCE_US):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance,
            f"{message}: {a} vs {b}")


def load(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def metric(raw, name):
    value = float(raw["metrics"][name])
    require(math.isfinite(value), f"nonfinite metric {name}")
    return value


def refresh(raw, label):
    require(raw["refresh_policy"] == {"rank0": "REFpb", "rank1": "REFab"},
            f"{label}: wrong declared refresh policy")
    counts = raw["refresh_commands"]
    for rank, wanted, forbidden in (("rank0", "REFpb", "REFab"), ("rank1", "REFab", "REFpb")):
        require(counts[rank][wanted] > 0, f"{label}: no actual {rank} {wanted}")
        require(counts[rank][forbidden] == 0, f"{label}: forbidden {rank} {forbidden}")
    return counts


def stages(raw, label):
    rows = raw["stages"]
    require(rows, f"{label}: empty PIM stages")
    require([r["stage_id"] for r in rows] == list(range(len(rows))),
            f"{label}: duplicate, missing, or reordered stage IDs")
    result, by_name = {}, {}
    ae_end = metric(raw, "AE_elapsed_us")
    for row in rows:
        key = (row["stage_id"], row["name"], row["step"], row["layer"])
        require(key not in result, f"{label}: duplicate stage {key}")
        start, end = float(row["start_us"]), float(row["end_us"])
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= ae_end + 1e-8,
                f"{label}: invalid stage interval {key}")
        span = end - start
        if "duration_us" in row:
            near(span, float(row["duration_us"]), f"{label}: stage duration {key}", 0.000002)
        result[key] = span
        by_name.setdefault(row["name"], Counter())[(row["step"], row["layer"])] += 1
    for name, coverage in by_name.items():
        require(set(coverage) == PAIR_SET and set(coverage.values()) == {1},
                f"{label}: {name} does not cover 18 layers x 10 steps exactly once")
    return result, {name: sum(counts.values()) for name, counts in by_name.items()}


def native_checks(raw, label):
    counts = refresh(raw, label)
    stage_map, coverage = stages(raw, label)
    total, ae_end = metric(raw, "total_us"), metric(raw, "AE_elapsed_us")
    tail, wait = metric(raw, "VLM_remaining_us"), metric(raw, "AE_GPU_wait_us")
    require(total >= ae_end >= 0 and tail >= 0 and wait >= 0, f"{label}: invalid completion ordering")
    near(total, ae_end + tail, f"{label}: total != AE_end + VLM_tail")
    for row in raw["kernels"]:
        start, end = float(row["start_us"]), float(row["end_us"])
        require(0 <= start <= end <= total + 1e-8, f"{label}: invalid VLM interval")
    expected_ids = set(range(2357)) if raw["kernels"] else set()
    require({r["ordinal"] for r in raw["kernels"]} == expected_ids
            and len(raw["kernels"]) == len(expected_ids), f"{label}: VLM coverage differs")
    bus = raw["bus"]
    require(bus and len({(r["channel"], r["stage_id"]) for r in bus}) == len(bus),
            f"{label}: missing or duplicate bus rows")
    for row in bus:
        require(0 <= row["channel"] < 16 and (row["stage_id"] == -1 or 0 <= row["stage_id"] < len(stage_map)),
                f"{label}: invalid bus channel/stage")
        require(0 <= row["ca_cycles"] <= row["duration_cycles"] and
                0 <= row["dq_cycles"] <= row["duration_cycles"], f"{label}: impossible bus occupancy")
    for field in ("gpu_read_bytes", "gpu_write_bytes", "pim_read_bytes", "pim_write_bytes"):
        require(sum(r[field] for r in bus) == raw["metrics"][field], f"{label}: {field} bus conservation")
    return stage_map, coverage, counts


def case_path(campaign, case, repeat):
    choices = (campaign / "cases" / case / "native" / case / f"repeat-{repeat}" / "raw.json",
               campaign / "native" / case / f"repeat-{repeat}" / "raw.json")
    existing = [p for p in choices if p.is_file()]
    require(len(existing) == 1, f"expected exactly one {case} repeat-{repeat} raw.json: {existing}")
    return existing[0]


def summarize(campaign, selected):
    rows, details, solos = [], {}, {}
    for case in selected:
        variant = VARIANTS[case]
        if variant not in solos:
            path = campaign / "solos" / "ae-solo" / variant / "raw.json"
            raw, digest = load(path)
            stage_map, coverage, counts = native_checks(raw, f"solo/{variant}")
            require(not raw["kernels"], f"solo/{variant}: unexpectedly contains VLM")
            near(metric(raw, "AE_GPU_wait_us"), 0, f"solo/{variant}: has GPU wait")
            solos[variant] = dict(raw=raw, sha256=digest, stages=stage_map, coverage=coverage, refresh=counts)
        solo = solos[variant]
        repeats = []
        for repeat in range(2):
            path = case_path(campaign, case, repeat)
            raw, digest = load(path)
            stage_map, coverage, counts = native_checks(raw, f"{case}/repeat-{repeat}")
            require(len(raw["kernels"]) == 2357, f"{case}: incomplete full VLM capture")
            require(stage_map.keys() == solo["stages"].keys(), f"{case}: stage coverage differs from AE solo")
            repeats.append(dict(path=str(path.relative_to(campaign)), raw=raw, sha256=digest,
                                stages=stage_map, coverage=coverage, refresh=counts))
        require(repeats[0]["sha256"] == repeats[1]["sha256"], f"{case}: raw repetitions differ byte-for-byte")
        first, raw = repeats[0], repeats[0]["raw"]
        total, ae_end = metric(raw, "total_us"), metric(raw, "AE_elapsed_us")
        wait, tail = metric(raw, "AE_GPU_wait_us"), metric(raw, "VLM_remaining_us")
        ae_solo = metric(solo["raw"], "total_us")
        interference = math.fsum(first["stages"][key] - solo["stages"][key] for key in first["stages"])
        ae_residual = ae_end - ae_solo - interference - wait
        residual = total - ae_solo - interference - wait - tail
        near(ae_residual, 0, f"{case}: AE accounting residual")
        near(residual, 0, f"{case}: e2e accounting residual")
        gpu_bytes = raw["metrics"]["gpu_read_bytes"] + raw["metrics"]["gpu_write_bytes"]
        in_stage = sum(r["gpu_read_bytes"] + r["gpu_write_bytes"] for r in raw["bus"] if r["stage_id"] >= 0)
        require(gpu_bytes > 0 and 0 <= in_stage <= gpu_bytes, f"{case}: invalid GPU byte ratio")
        require(case != "b" or in_stage == 0, "serial b issued GPU traffic during a PIM stage")
        row = dict(case=case, variant=variant, total_ms=total / 1000, AE_solo_ms=ae_solo / 1000,
                   AE_elapsed_ms=ae_end / 1000, PIM_interference_ms=interference / 1000,
                   AE_GPU_wait_ms=wait / 1000, VLM_tail_ms=tail / 1000,
                   algebra_residual_us=residual, AE_residual_us=ae_residual,
                   GPU_DQ_bytes_issued_during_PIM_stages=in_stage, GPU_DQ_bytes_total=gpu_bytes,
                   GPU_DQ_bytes_issued_during_PIM_stages_pct=100 * in_stage / gpu_bytes,
                   rank0_REFpb=first["refresh"]["rank0"]["REFpb"], rank0_REFab=first["refresh"]["rank0"]["REFab"],
                   rank1_REFab=first["refresh"]["rank1"]["REFab"], rank1_REFpb=first["refresh"]["rank1"]["REFpb"],
                   raw_repeats_byte_identical=True, PIM_stage_count=len(first["stages"]), VLM_units=len(raw["kernels"]))
        rows.append(row)
        details[case] = dict(variant=variant, raw_sha256=[r["sha256"] for r in repeats],
                             raw_paths=[r["path"] for r in repeats], AE_solo_raw_sha256=solo["sha256"],
                             stage_coverage_each_repeat=[r["coverage"] for r in repeats],
                             refresh_each_repeat=[r["refresh"] for r in repeats],
                             AE_solo_refresh=solo["refresh"])
    calibration_path = campaign / "calibration" / "calibration.json"
    calibration = None
    if calibration_path.is_file():
        data, digest = load(calibration_path)
        calibration = dict(sha256=digest, parameters=data["parameters"], validation=data["validation"])
        require(calibration["validation"]["passed"], "calibration did not pass")
    return dict(schema_version=1, passed=True, campaign=str(campaign), cases=selected,
                scope="18 layers x 10 denoising steps; complete 2357-unit VLM; two raw repetitions per case",
                tolerance_us=TOLERANCE_US, calibration=calibration, evaluation=rows, evidence=details,
                notes=["All latency terms come directly from native timestamps; no offsets or overlap correction.",
                       "PIM interference is the sum of per-stage concurrent duration minus the matching AE solo duration.",
                       "GPU DQ byte share is attributed by command-issue stage. It is not a time-overlap ratio; DQ returns can cross stage boundaries.",
                       "This compact audit supplements, rather than replaces, full source/binary/input provenance acceptance."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, help="output filename stem (default: campaign/independent-concurrency-summary)")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    require(len(set(args.cases)) == len(args.cases), "duplicate requested cases")
    report = summarize(campaign, args.cases)
    output = args.output or campaign / "independent-concurrency-summary"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output.with_suffix(".json"), output.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["evaluation"][0]))
        writer.writeheader()
        writer.writerows(report["evaluation"])
    print("case          e2e_ms   AE_solo   PIM_delta   GPU_wait   VLM_tail   GPU_bytes_in_PIM_pct")
    for r in report["evaluation"]:
        print(f"{r['case']:<12} {r['total_ms']:9.6f} {r['AE_solo_ms']:9.6f} {r['PIM_interference_ms']:10.6f} "
              f"{r['AE_GPU_wait_ms']:10.6f} {r['VLM_tail_ms']:10.6f} {r['GPU_DQ_bytes_issued_during_PIM_stages_pct']:19.6f}")
    print(f"PASS: {len(args.cases)} cases; output: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()

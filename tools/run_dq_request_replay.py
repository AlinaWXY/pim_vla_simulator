"""Replay b–g on the measured AE DQ windows with whole-kernel GPU handoff."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256, write_csv, write_json
from tools.capture_gap_traces import COMPARISON_SLICES, REPEATS
from tools.estimate_dq_gaps import (
    FUSED_COMPUTE_STAGES, FUSED_SHARED_IO, MATRIX, N, PROFILE, edge_costs, load_trace, units_for,
)

CASES = "bcdefg"


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


def usable_windows(gaps, blocked, begin, end):
    """Clip candidates without discarding the actual neighboring burst."""
    blocked_ends = [stop for _, stop in blocked]
    for a, b, left, right in gaps:
        lo, hi = max(a, begin), min(b, end)
        i = bisect_right(blocked_ends, lo)
        while i < len(blocked) and blocked[i][0] < hi:
            start, stop = blocked[i]
            if start - lo >= 2:
                yield a - begin, b - begin, lo - begin, start - begin, left, right
            lo = max(lo, stop)
            i += 1
        if hi - lo >= 2:
            yield a - begin, b - begin, lo - begin, hi - begin, left, right


def fused_epochs(path):
    epochs = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            epochs[int(row["stage_id"])].append((int(row["start_ck"]), int(row["end_ck"])))
    return {key: union(values) for key, values in epochs.items()}


def prepare(case, step, layer, traces, output, units, edges, spec_digest, clock_ps):
    name = f"{case}-s{step}-l{layer}"
    path = traces / name
    rules = FUSED_SHARED_IO.get(case)
    timeline, geometry, gaps, verification = load_trace(path, step, layer, shared_io_rules=rules)
    if verification["spec_sha256"] != spec_digest or timeline["clock_ps"] != clock_ps:
        raise ValueError(f"source spec differs for {name}")
    rows = {row["event_index"]: row for row in geometry}
    epochs = fused_epochs(path / "on/dq-epochs.csv") if rules else {}
    stages, shared_ck = [], {}
    for index, event in enumerate(timeline["events"]):
        start, stop = event["start_ck"], event["end_ck"]
        duration = stop - start
        pim = event["kind"] == "pim"
        if not pim:
            stages.append((event["name"], duration, None))
            continue
        begin, end = 0, duration
        shared = rows[index].get("shared_tensor_io")
        if shared:
            a, b = shared["start_ck"], shared["end_ck"]
            if a == 0:
                begin = b
            elif b == duration:
                end = a
            else:
                raise ValueError(f"shared I/O is inside the stage: {name}/{event['name']}")
            shared_ck[event["name"]] = b - a
        if begin == end:
            continue
        blocked = [(a-start, b-start) for a, b in epochs.get(event["stage_id"], ())]
        if rules and event["name"] not in FUSED_COMPUTE_STAGES:
            blocked = [(0, duration)]
        channels = [list(usable_windows(c, blocked, begin, end)) for c in gaps[index]]
        if event["name"].startswith("q_exchange") and case in "fg" and any(channels):
            raise ValueError("fused Q exchange must reserve DQ")
        stages.append((event["name"], end - begin, channels))
    ae_ck = sum(duration for _, duration, _ in stages)
    if ae_ck + sum(shared_ck.values()) != timeline["ae_end_ck"]:
        raise ValueError(f"AE shared-I/O accounting does not reconcile for {name}")

    dest = output / f"{name}.txt"
    with dest.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(" ".join(str(edges[a + b]) for a in "RWO" for b in "RWO") + "\n")
        stream.write(f"{len(units)} {len(stages)}\n")
        stream.writelines(f"{r} {w} {duration:.17g}\n" for r, w, duration in units)
        for stage, duration, channels in stages:
            stream.write(f"{json.dumps(stage)} {duration} {int(channels is not None)}\n")
            for channel in channels or ():
                stream.write(f"{len(channel)}\n")
                stream.writelines(" ".join(map(str, gap)) + "\n" for gap in channel)
    return dest, ae_ck, shared_ck


def run(engine, case, step, layer, traces, output, units, edges, spec_digest, clock_ps, prior):
    source, ae_ck, shared = prepare(case, step, layer, traces, output / "inputs", units, edges,
                                   spec_digest, clock_ps)
    raw = json.loads(subprocess.check_output([str(engine), str(source), str(REPEATS), str(int(case == "b"))],
                                            text=True, encoding="utf-8"))
    if raw["ae_solo_ck"] != ae_ck * REPEATS:
        raise ValueError(f"unexpected AE time for {source.stem}")
    total = [sum(unit[i] for unit in units) for i in range(2)]
    if raw["total_requests"] != total or [
        sum(raw[k][i] for k in ("pim_requests", "join_requests", "tail_requests")) for i in range(2)
    ] != total:
        raise ValueError(f"request counts do not reconcile for {source.stem}")
    if [sum(row["requests"][i] for row in raw["stages"]) for i in range(2)] != raw["pim_requests"]:
        raise ValueError(f"PIM request counts do not reconcile for {source.stem}")
    ck_ms = clock_ps / 1e9
    old = prior[case, step, layer]
    if not math.isclose(ae_ck * REPEATS * ck_ms, float(old["accounted_ae_ms"]), abs_tol=1e-8):
        raise ValueError(f"shared I/O differs from previous accounting for {source.stem}")
    write_json(output / f"{source.stem}.json", dict(raw, shared_io_ck_per_slice=shared,
                                                    input_sha256=sha256(source)))
    summary = dict(case=case, step=step, layer=layer, e2e_ms=raw["total_ck"] * ck_ms,
                   prior_e2e_ms=float(old["e2e_ms"]), ae_solo_ms=ae_ck * REPEATS * ck_ms,
                   ae_gpu_wait_ms=raw["join_ck"] * ck_ms, ae_elapsed_ms=raw["ae_ck"] * ck_ms,
                   vlm_tail_ms=raw["tail_ck"] * ck_ms, vlm_solo_ms=raw["vlm_solo_ck"] * ck_ms,
                   vlm_done_at_ms=raw["vlm_end_ck"] * ck_ms, pim_vlm_overlap_ms=raw["overlap_ck"] * ck_ms,
                   shared_io_profile_ms=sum(shared.values()) * REPEATS * ck_ms,
                   gpu_requests_in_pim=sum(raw["pim_requests"]),
                   gpu_requests_during_handoff=sum(raw["join_requests"]),
                   gpu_requests_in_tail=sum(raw["tail_requests"]),
                   gpu_requests_total=sum(total), units_at_ae_end=raw["units_at_ae_end"])
    print(f"{source.stem}: E2E {summary['e2e_ms']:.6f} ms", flush=True)
    stages = [dict(case=case, step=step, layer=layer, stage=row["name"],
                   gpu_reads=row["requests"][0], gpu_writes=row["requests"][1],
                   vlm_overlap_ms=row["overlap_ck"] * ck_ms) for row in raw["stages"]]
    return summary, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--slice", nargs=2, type=int, metavar=("STEP", "LAYER"),
                        help="replay just one captured slice")
    args = parser.parse_args()
    slices = (tuple(args.slice),) if args.slice else COMPARISON_SLICES
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs").mkdir(exist_ok=True)
    spec_path = args.traces / "spec-k2.json"
    clock_ps = read_json(spec_path)["clock_ps"]
    source_units = units_for(1.0, clock_ps)
    units = [(round(r * N), round(w * N), t) for r, w, t in source_units]
    edges = edge_costs(MATRIX)
    spec_digest = sha256(spec_path)
    with args.prior_summary.open(newline="", encoding="utf-8") as stream:
        prior = {(r["case"], int(r["step"]), int(r["layer"])): r for r in csv.DictReader(stream)}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, args.engine.resolve(), case, step, layer, args.traces, args.output,
                               units, edges, spec_digest, clock_ps, prior)
                   for step, layer in slices for case in CASES]
        results = [future.result() for future in futures]
    rows = [row for row, _ in results]
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "stage-summary.csv", [stage for _, stages in results for stage in stages])
    write_json(args.output / "summary.json", dict(
        clock_ps=clock_ps, repeats=REPEATS, alpha=1.0, burst_ck=2, scheduling="same_first",
        profile_ms=math.fsum(t for _, _, t in units) * clock_ps / 1e9,
        spec_sha256=spec_digest, engine_sha256=sha256(args.engine), profile_sha256=sha256(PROFILE),
        turnaround_ck=edges, rows=rows))


if __name__ == "__main__":
    main()

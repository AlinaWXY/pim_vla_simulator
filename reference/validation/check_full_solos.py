"""Read-only audit of the five full AE solos in an immutable campaign."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

VARIANTS = ("feature_gpu_before", "token_gpu_before", "token_local_before", "token_gpu_after", "token_local_after")
GPU_FIELDS = ("GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab", "GPU_reads", "GPU_writes", "GPU_row_hits", "GPU_row_misses")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def walk(operations):
    for op in operations:
        yield op
        yield from walk(op.get("children", []))


def slurm_records(job):
    text = subprocess.check_output(["sacct", "-j", str(job), "-n", "-P", "--format=JobID,State,ElapsedRaw,TotalCPU,MaxRSS,ExitCode"], text=True)
    records = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) >= 6:
            records[parts[0]] = dict(state=parts[1], elapsed_seconds=int(parts[2] or 0),
                                     total_cpu=parts[3], max_rss=parts[4], exit_code=parts[5])
    return records


def check_one(run, variant, index, job, jobs, binary_hash, library_hash, source_commit):
    directory = run / "solos/ae-solo" / variant
    raw_path = directory / "raw.json"
    raw, cfg, provenance = (read(directory / name) for name in ("raw.json", "input-config.json", "provenance.json"))
    job_id = f"{job}_{index}"
    record = jobs[job_id]
    require(record["state"] == "COMPLETED" and record["exit_code"] == "0:0", f"{job_id}: Slurm is not successfully complete: {record}")
    require(provenance["binary_sha256"] == binary_hash, "binary SHA mismatch")
    library = provenance["runtime_libraries"]["libramulator.so"]
    require(library["sha256"] == library_hash, "library SHA mismatch")
    require(Path(library["path"]).resolve() == (run / "frozen/lib/libramulator.so").resolve(), "run loaded a library outside frozen/lib")
    require(provenance["source_commit"] == source_commit, "source commit mismatch")
    require(provenance["raw_sha256"] == sha(raw_path), "raw output SHA mismatch")
    require(provenance["config_sha256"] == sha(directory / "input-config.json"), "configuration SHA mismatch")
    require(cfg["run_kind"] == "ae_solo", "run is not an AE solo")
    for key in ("program_path", "ae_schedule_path", "vlm_path"):
        path = Path(cfg[key]).resolve()
        require(path.is_relative_to(run / "frozen/source"), f"input {key} is outside frozen/source")
        require(provenance["inputs"][key]["sha256"] == sha(path), f"input SHA mismatch: {key}")
    program, schedule = read(cfg["program_path"]), read(cfg["ae_schedule_path"])
    require(program["variant"] == schedule["variant"] == variant, "variant mismatch")
    m, stages = raw["metrics"], raw["stages"]
    tick = raw["clock_ps"] / 1_000_000
    require(raw["clock_ps"] == 938, "unexpected clock")
    wanted = [(e["stage"], e["step"], e["layer"]) for e in schedule["events"] if e["kind"] == "pim"]
    observed = [(e["name"], e["step"], e["layer"]) for e in stages]
    require(observed == wanted, "PIM stage order differs from the full measured schedule")
    names = {s["name"] for s in program["stages"]}
    require(Counter(s["name"] for s in stages) == Counter({name: 180 for name in names}), "stage count is not 180 per template stage")
    require({(s["step"], s["layer"]) for s in stages} == {(step, layer) for step in range(10) for layer in range(18)}, "18 layers x 10 steps coverage failed")
    require([s["stage_id"] for s in stages] == list(range(len(stages))), "noncontiguous stage IDs")
    require(not raw["kernels"], "AE solo executed VLM work")
    require(m["gpu_read_bytes"] == m["gpu_write_bytes"] == 0, "AE solo has rank0 memory traffic")
    require(all(m[key] == 0 for key in GPU_FIELDS), "AE solo has rank0 command/row counts")
    require(m["GPU_row_hits"] + m["GPU_row_misses"] == m["GPU_reads"] + m["GPU_writes"], "GPU row count identity failed")
    require(m["AE_GPU_wait_us"] == m["VLM_remaining_us"] == 0, "AE solo reports GPU wait or VLM tail")
    require(m["total_us"] == m["AE_elapsed_us"] > 0, "AE solo completion timestamp mismatch")
    gpu_events = [e for e in schedule["events"] if e["kind"] == "gpu"]
    gpu_us = sum(e["us"] for e in gpu_events)
    rounded_gpu_us = sum(math.ceil(e["us"] / tick - 1e-10) for e in gpu_events) * tick
    pim_us = sum(s["duration_us"] for s in stages)
    require(abs(pim_us + rounded_gpu_us - m["total_us"]) < 0.002, "AE total != PIM stages + integer-CK measured GPU blocks")
    for i, stage in enumerate(stages):
        require(stage["duration_us"] >= 0 and stage["gpu_wait_us"] == 0, "invalid solo stage duration/wait")
        require(abs(stage["end_us"] - stage["start_us"] - stage["duration_us"]) < 1e-6, "stage timestamp duration mismatch")
        require(i == 0 or stage["start_us"] + 1e-8 >= stages[i - 1]["end_us"], "overlapping sequential PIM stages")
    expected_stage = {}
    expected_cycles = 0
    for stage in program["stages"]:
        for channel, operations in enumerate(stage["channels"]):
            counts = Counter(op["op"] for op in walk(operations))
            expected_stage[stage["name"], channel] = ((counts["RD"] + counts["RDRES"]) * 32,
                                                     (counts["WR"] + counts["WRGB"]) * 32)
            expected_cycles += sum(counts[name] * cfg["spec"]["metadata"]["commands"][name]["burst"]
                                   for name in ("RD", "RDRES", "WR", "WRGB")) * 180
    expected_read = sum(value[0] for value in expected_stage.values()) * 180
    expected_write = sum(value[1] for value in expected_stage.values()) * 180
    require(expected_read + expected_write == program["expected_dq_bytes_per_layer"] * 180, "program DQ metadata disagrees with actual operation walk")
    require((m["pim_read_bytes"], m["pim_write_bytes"]) == (expected_read, expected_write), "PIM read/write conservation failed")
    bus_bytes = defaultdict(lambda: [0, 0])
    channel_duration = Counter()
    for row in raw["bus"]:
        channel, sid = row["channel"], row["stage_id"]
        require(0 <= channel < 16 and -1 <= sid < len(stages), "invalid bus channel/stage")
        require(0 <= row["ca_cycles"] <= row["duration_cycles"] and 0 <= row["dq_cycles"] <= row["duration_cycles"], "bus occupancy exceeds wall interval")
        require(row["gpu_read_bytes"] == row["gpu_write_bytes"] == 0, "solo bus has rank0 bytes")
        require(row["read_bytes"] == row["pim_read_bytes"] and row["write_bytes"] == row["pim_write_bytes"], "bus directional totals mismatch")
        bus_bytes[sid, channel][0] += row["read_bytes"]
        bus_bytes[sid, channel][1] += row["write_bytes"]
        channel_duration[channel] += row["duration_cycles"]
    total_ck = round(m["total_us"] / tick)
    require(set(channel_duration) == set(range(16)) and all(t == total_ck for t in channel_duration.values()), "bus time intervals do not cover all 16 channels")
    require(all(value == [0, 0] for (sid, ch), value in bus_bytes.items() if sid == -1), "background interval contains PIM data bytes")
    for stage in stages:
        for channel in range(16):
            require(tuple(bus_bytes[stage["stage_id"], channel]) == expected_stage[stage["name"], channel], "per-stage/channel DQ conservation failed")
    require(sum(row["read_bytes"] for row in raw["bus"]) == expected_read and sum(row["write_bytes"] for row in raw["bus"]) == expected_write, "total bus DQ bytes differ from metrics")
    require(sum(row["dq_cycles"] for row in raw["bus"]) == expected_cycles, "DQ occupied cycles differ from exported burst widths")
    require(m["checked_commands"] >= (expected_read + expected_write) // 32, "checker did not see all external requests")
    log = (directory / "native.log").read_text(encoding="utf-8")
    require("completed ae_solo" in log and not re.search(r"ERROR:|Assertion.*failed|terminate called|SIGSEGV", log), "native log did not complete cleanly")
    return dict(variant=variant, status="passed", job_id=job_id, slurm=record,
                batch=jobs.get(job_id + ".batch"), total_us=m["total_us"], AE_elapsed_us=m["AE_elapsed_us"],
                PIM_stage_sum_us=pim_us, GPU_measured_sum_us=gpu_us, GPU_rounded_sum_us=rounded_gpu_us,
                gpu_block_count=len(gpu_events), stages=len(stages), stage_occurrences=dict(Counter(s["name"] for s in stages)),
                pim_read_bytes=expected_read, pim_write_bytes=expected_write,
                checked_commands=m["checked_commands"], dq_cycles=expected_cycles,
                gpu_command_counts={key: m[key] for key in GPU_FIELDS},
                raw_path=str(raw_path), raw_sha256=sha(raw_path), provenance_sha256=sha(directory / "provenance.json"),
                program_sha256=sha(cfg["program_path"]), ae_schedule_sha256=sha(cfg["ae_schedule_path"]),
                binary_sha256=binary_hash, library_sha256=library_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="wait with 30-second read-only polls until the solo array finishes")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    require(not output.is_relative_to(run), "acceptance output must be outside the immutable campaign")
    source_commit = (run / "frozen/SOURCE_COMMIT").read_text().strip()
    binary_hash, library_hash = sha(run / "frozen/bin/pim_vla_sim"), sha(run / "frozen/lib/libramulator.so")
    jobs = slurm_records(args.job)
    while args.wait and any(jobs.get(f"{args.job}_{i}", {}).get("state", "PENDING") in ("PENDING", "RUNNING", "COMPLETING") for i in range(5)):
        time.sleep(30)
        jobs = slurm_records(args.job)
    results = []
    for index, variant in enumerate(VARIANTS):
        directory = run / "solos/ae-solo" / variant
        job = jobs.get(f"{args.job}_{index}", {})
        if job.get("state", "PENDING") in ("RUNNING", "PENDING", "COMPLETING"):
            results.append(dict(variant=variant, status="pending", slurm=job))
            continue
        try:
            results.append(check_one(run, variant, index, args.job, jobs, binary_hash, library_hash, source_commit))
        except Exception as error:
            results.append(dict(variant=variant, status="failed", error=str(error)))
    completed = all(result["status"] == "passed" for result in results)
    if completed:
        for cwd, manifest in ((run, "frozen/runtime-SHA256SUMS"), (run, "frozen/config-SHA256SUMS"),
                              (run / "frozen/source", "../source-SHA256SUMS")):
            subprocess.run(["sha256sum", "-c", manifest, "--quiet"], cwd=cwd, check=True)
    report = dict(campaign=str(run), source_commit=source_commit, solo_array=args.job, passed=completed,
                  immutable_manifests_verified=completed, results=results,
                  scope="AE standalone only; zero GPU memory requests make GPU row-hit identity trivial here")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)
    if not completed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent, read-only verification of an extracted single-layer archive.

Usage: python audit-rank-refresh.py ARCHIVE_DIRECTORY SOURCE_COMMIT \
    --frozen-dir /path/to/run/frozen --source-root /path/to/repo

No simulator or repository Python code is imported/executed. Only the audit JSON
is written. With manifest-only archives, unavailable program/binary/source bytes
are explicitly reported and the exit code is 2 (incomplete), never a full pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

VARIANTS = ("feature_gpu_before", "token_gpu_before", "token_local_before",
            "token_gpu_after", "token_local_after")
POLICY = {"rank0": "REFpb", "rank1": "REFab"}
GPU_COUNTERS = ("gpu_read_bytes", "gpu_write_bytes", "GPU_act1", "GPU_act2",
                "GPU_prepb", "GPU_preab", "GPU_reads", "GPU_writes",
                "GPU_row_hits", "GPU_row_misses", "AE_GPU_wait_us", "VLM_remaining_us")
SOURCE_FILES = {"programs/gen_pim_program.py", "programs/descriptors.py", "workloads/layout.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, f"{message}: {actual!r} != {expected!r}")


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-6),
            f"{message}: {actual!r} != {expected!r}")


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_record(actual, expected, label, *, csv_strings=False):
    for key, value in expected.items():
        require(key in actual, f"{label}: missing {key}")
        if isinstance(value, bool):
            same(actual[key], str(value) if csv_strings else value, f"{label}.{key}")
        elif isinstance(value, (int, float)):
            close(actual[key], value, f"{label}.{key}")
        else:
            same(actual[key], value, f"{label}.{key}")


def refresh(raw):
    same(raw.get("refresh_policy"), POLICY, "fixed rank refresh policy")
    commands = raw.get("refresh_commands")
    require(isinstance(commands, dict) and set(commands) == set(POLICY), "both rank refresh counters required")
    result = {}
    for rank, kind in POLICY.items():
        require(isinstance(commands[rank], dict) and set(commands[rank]) == {"REFpb", "REFab"},
                f"{rank}: both refresh command types required")
        for name, count in commands[rank].items():
            require(type(count) is int and count >= 0, f"{rank}.{name}: invalid count")
            require(count > 0 if name == kind else count == 0, f"{rank}.{name}: wrong refresh activity")
            result[f"{rank}_{name}"] = count
    return result


def walk(ops):
    for op in ops:
        yield op
        yield from walk(op.get("children", []))


def program_contract(program, variant, spec):
    same(program["variant"], variant, "program variant")
    same(program["channels"], 16, "program channels")
    result = {}
    for stage in program["stages"]:
        same(len(stage["channels"]), 16, "program stage channels")
        require(stage["name"] not in result, "duplicate program stage")
        channels = []
        for ops in stage["channels"]:
            counts = Counter(op["op"] for op in walk(ops))
            reads = 32 * (counts["RD"] + counts["RDRES"])
            writes = 32 * (counts["WR"] + counts["WRGB"])
            dq = sum(counts[name] * spec["commands"][name]["dq_cycles"]
                     for name in ("RD", "RDRES", "WR", "WRGB"))
            channels.append((reads, writes, dq))
        result[stage["name"]] = channels
        if "expected_dq_bytes" in stage:
            same(sum(r + w for r, w, _ in channels), stage["expected_dq_bytes"], "program stage byte declaration")
    same(sum(r + w for channels in result.values() for r, w, _ in channels),
         program["expected_dq_bytes_per_layer"], "program total byte declaration")
    return result


def schedule_check(schedule, variant, measured=None):
    same(schedule["variant"], variant, "schedule variant")
    for key, value in (("layers", 1), ("denoising_steps", 1), ("selected_layer", 0), ("selected_step", 0)):
        same(schedule[key], value, f"schedule.{key}")
    events = schedule["events"]
    require(all(e["step"] == e["layer"] == 0 for e in events), "non-single-layer event")
    expected_stages = ["qkv"] + (["rope_export", "rope_import"] if "_gpu_" in variant else ["rope"])
    expected_stages += ["q_exchange_export", "q_exchange_import", "qkt", "pv", "o", "gate_up", "down"]
    same([e["stage"] for e in events if e["kind"] == "pim"], expected_stages, "complete PIM sequence")
    same(schedule["stage_occurrences"], dict.fromkeys(expected_stages, 1), "stage occurrences")
    natural = [e for e in events if e["kind"] == "gpu" and e["source_ordinal"] >= 0]
    ordinals = list(range(30, 41)) + (list(range(44, 54)) if "_gpu_" in variant else [])
    ordinals += list(range(57, 61)) + list(range(64, 77)) + [78, 80, 83, 84]
    same([e["source_ordinal"] for e in natural], ordinals, "complete retained natural GPU kernels")
    same(schedule["boundary"]["source_start"], 30, "layer start")
    same(schedule["boundary"]["source_end_exclusive"], 85, "layer end")
    same(schedule["boundary"]["retained_natural_gpu_ordinals"], ordinals, "boundary GPU ordinals")
    transpose = [i for i, e in enumerate(events) if e.get("name") == "pv_to_o_transpose"]
    same(len(transpose), int(variant.endswith("before")), "streaming transpose count")
    if transpose:
        index = transpose[0]
        same(events[index - 1].get("stage"), "pv", "transpose follows PV")
        same(events[index + 1].get("stage"), "o", "transpose precedes O")
    helper_names = ["common_O_rank_sum_cast", "common_GateUp_fp32_to_bf16", "common_Down_fp32_to_bf16"]
    if variant.startswith("feature_"):
        helper_names.insert(0, "feature_score_rank_sum")
    extras = [e["name"] for e in events if e["kind"] == "gpu" and e["source_ordinal"] < 0]
    same(Counter(extras), Counter(helper_names + (["pv_to_o_transpose"] if transpose else [])), "measured helpers")
    if measured:
        ledger, helpers, transposes = measured
        for e in events:
            if e["kind"] != "gpu":
                continue
            if e["source_ordinal"] >= 0:
                source = ledger[e["source_ordinal"]]
                same(e["name"], source["category"], "GPU category source")
                expected = source["step_mean_us"][0]
            elif e["name"] == "pv_to_o_transpose":
                expected = transposes[f"{variant.split('_')[0]}_direct_PV_to_O_pack"]["cold32MiB"]["device_kernel_median_us"]
            else:
                expected = helpers[e["name"]]["cold32MiB"]["device_kernel_median_us"]
            close(e["us"], expected, f"GPU measured source {e['name']}")
    close(sum(e["us"] for e in events if e["kind"] == "gpu"), schedule["gpu_us"], "schedule GPU total")


def native_check(raw, schedule, spec, contract, variant):
    result = refresh(raw)
    same(raw["clock_ps"], spec["clock_ps"], "native clock")
    tick = spec["clock_ps"] / 1e6
    metrics, stages = raw["metrics"], raw["stages"]
    same(raw["kernels"], [], "no VLM kernels")
    for key in GPU_COUNTERS:
        same(metrics[key], 0, f"no GPU/VLM work: {key}")
    close(metrics["total_us"], metrics["AE_elapsed_us"], "AE equals total")
    names = [e["stage"] for e in schedule["events"] if e["kind"] == "pim"]
    same([s["name"] for s in stages], names, "native stage coverage")
    same([s["stage_id"] for s in stages], list(range(len(stages))), "native stage IDs")
    for s in stages:
        same((s["step"], s["layer"], s["gpu_wait_us"]), (0, 0, 0), "stage step/layer/wait")
        close(s["duration_us"], s["end_us"] - s["start_us"], "stage timestamp duration")
        require(s["duration_us"] >= 0, "negative duration")
    buses = {}
    for row in raw["bus"]:
        key = (row["stage_id"], row["channel"])
        require(key not in buses, f"duplicate bus interval {key}")
        buses[key] = row
        for field in ("duration_cycles", "ca_cycles", "dq_cycles", "read_bytes", "write_bytes", "gpu_read_bytes",
                      "gpu_write_bytes", "pim_read_bytes", "pim_write_bytes"):
            require(type(row[field]) is int and row[field] >= 0, f"invalid bus field {field}")
        require(row["ca_cycles"] <= row["duration_cycles"] and row["dq_cycles"] <= row["duration_cycles"], "bus utilization exceeds 1")
        for direction in ("read", "write"):
            same(row[f"gpu_{direction}_bytes"], 0, "no rank0 bus requests")
            same(row[f"{direction}_bytes"], row[f"pim_{direction}_bytes"], "PIM-only byte identity")
            same(row[f"{direction}_bytes"] % 32, 0, "32-byte packet alignment")
    same(set(buses), {(sid, ch) for sid in range(-1, len(stages)) for ch in range(16)}, "complete bus intervals")
    stage_rows = []
    for s in stages:
        totals = dict.fromkeys(("read_bytes", "write_bytes", "ca_cycles", "dq_cycles"), 0)
        for ch in range(16):
            row = buses[s["stage_id"], ch]
            close(row["duration_cycles"] * tick, s["duration_us"], "stage bus duration")
            if contract is not None:
                same((row["read_bytes"], row["write_bytes"], row["dq_cycles"]), contract[s["name"]][ch],
                     f"{variant}/{s['name']}/ch{ch} descriptor traffic")
            for field in totals:
                totals[field] += row[field]
        stage_rows.append(dict(variant=variant, **s, **totals))
    for ch in range(16):
        background = buses[-1, ch]
        same((background["read_bytes"], background["write_bytes"], background["dq_cycles"]), (0, 0, 0), "no traffic outside stages")
        close(sum(row["duration_cycles"] for (sid, c), row in buses.items() if c == ch) * tick,
              metrics["total_us"], "channel time partition")
    read_bytes = sum(r["read_bytes"] for r in raw["bus"])
    write_bytes = sum(r["write_bytes"] for r in raw["bus"])
    same((read_bytes, write_bytes), (metrics["pim_read_bytes"], metrics["pim_write_bytes"]), "bus total bytes")
    # All four external commands have the same burst length; verify, then count
    # occupied cycles from packet bytes independently of the per-stage contract.
    bursts = {spec["commands"][name]["dq_cycles"] for name in ("RD", "RDRES", "WR", "WRGB")}
    same(len(bursts), 1, "equal external burst widths")
    same(sum(r["dq_cycles"] for r in raw["bus"]), (read_bytes + write_bytes) // 32 * next(iter(bursts)), "DQ occupied cycles")
    require(metrics["checked_commands"] >= (read_bytes + write_bytes) // 32 + sum(result.values()), "checker command count too small")
    now, stage_index, gpu_rows = 0.0, 0, []
    for i, e in enumerate(schedule["events"]):
        if e["kind"] == "pim":
            s = stages[stage_index]
            close(s["start_us"], now, "PIM/GPU event timeline")
            now, stage_index = s["end_us"], stage_index + 1
        else:
            duration = math.ceil(e["us"] / tick - 1e-10) * tick
            gpu_rows.append(dict(variant=variant, event=i, name=e["name"], source_ordinal=e["source_ordinal"],
                                 measured_us=e["us"], duration_us=duration, start_us=now, end_us=now + duration))
            now += duration
    pim = sum(s["duration_us"] for s in stages)
    gpu = sum(r["duration_us"] for r in gpu_rows)
    close(now, metrics["total_us"], "complete layer timeline")
    close(pim + gpu, metrics["total_us"], "PIM + rounded GPU = total")
    result.update(variant=variant, total_us=metrics["total_us"], PIM_us=pim, GPU_us=gpu,
                  GPU_measured_us=sum(r["measured_us"] for r in gpu_rows),
                  transpose_measured_us=sum(r["measured_us"] for r in gpu_rows if r["name"] == "pv_to_o_transpose"),
                  pim_read_bytes=read_bytes, pim_write_bytes=write_bytes, checked_commands=metrics["checked_commands"],
                  PIM_stages=len(stages), GPU_blocks=len(gpu_rows),
                  PIM_out_of_order_issues=metrics["PIM_out_of_order_issues"],
                  PIM_prefetched_prerequisites=metrics["PIM_prefetched_prerequisites"])
    return result, stage_rows, gpu_rows


def git_bytes(root, commit, relative):
    relative = relative.replace("\\", "/")
    require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts, "invalid source path")
    return subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{relative}"], stderr=subprocess.PIPE)


def audit(args, report):
    root = args.archive_directory.resolve()
    experiment = root / "experiment"
    manifest, provenance = read(root / "frozen/manifest.json"), read(experiment / "provenance.json")
    summary, spec = read(experiment / "summary.json"), read(root / "frozen/spec.json")
    same(manifest["source_commit"], args.source_commit, "manifest source")
    same(provenance["source_commit"], args.source_commit, "experiment source")
    require(summary["no_vlm"] and summary["no_full_campaign"] and summary["checks_passed"], "single-layer summary flags")
    for key in ("calibration_executed", "full_matrix_executed", "document_published"):
        same(provenance[key], False, key)
    same(object_sha(spec), provenance["spec_sha256"], "spec canonical hash")
    files = manifest["files"]
    same(sha(root / "frozen/spec.json"), files["spec.json"]["sha256"], "spec file hash")
    same(provenance["binary_sha256"], files["bin/pim_vla_sim"]["sha256"], "binary manifest binding")
    same(provenance["runtime_libraries"]["libramulator.so"]["sha256"], files["lib/libramulator.so"]["sha256"], "library manifest binding")
    frozen = args.frozen_dir.resolve() if args.frozen_dir else root / "frozen"
    missing = report["unavailable_checks"]
    verified_files = {}
    for name, record in files.items():
        path = frozen / name
        if path.is_file():
            same(sha(path), record["sha256"], f"actual frozen {name} hash")
            same(path.stat().st_size, record["bytes"], f"actual frozen {name} size")
            verified_files[name] = record["sha256"]
        else:
            missing.append(f"actual frozen artifact bytes: {name}")
    report["verified_frozen_artifacts"] = verified_files
    source_root, measured, source_hashes = args.source_root, None, {}
    if source_root:
        source_root = source_root.resolve()
        head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        same(head, args.source_commit, "source repository HEAD")
        for name, expected in provenance["measured_sources"].items():
            relative = "input-data/measured/" + name
            actual = sha(source_root / relative)
            same(actual, expected, f"measured source {name}")
            same(hashlib.sha256(git_bytes(source_root, args.source_commit, relative)).hexdigest(), expected, f"committed measured source {name}")
            source_hashes[relative] = expected
        directory = source_root / "input-data/measured"
        measured = (read(directory / "natural-ae-kernel-ledger.json")["units"],
                    read(directory / "arithmetic-helper-measurements.json")["operators"],
                    read(directory / "direct-pv-handoff.json")["operators"])
    else:
        missing.append("source HEAD, generator/layout git blobs and actual measured inputs")
    result_table = table(experiment / "single-layer-latency.csv")
    stage_table = table(experiment / "stage-latencies.csv")
    gpu_table = table(experiment / "gpu-blocks.csv")
    same([r["variant"] for r in summary["results"]], list(VARIANTS), "summary variants")
    same([r["variant"] for r in result_table], list(VARIANTS), "CSV variants")
    same(set(provenance["variants"]), set(VARIANTS), "provenance variants")
    expected_paths = {f"{v}/repeat-{r}/raw.json" for v in VARIANTS for r in range(2)}
    same({p.relative_to(experiment / "native").as_posix() for p in (experiment / "native").glob("*/repeat-*/raw.json")}, expected_paths, "exactly ten native results")
    generation = {r["variant"]: r for r in (json.loads(line) for line in (root / "validation/program-generation.jsonl").read_text().splitlines() if line.strip())}
    all_stage_rows, all_gpu_rows = [], []
    for variant in VARIANTS:
        print(f"Auditing {variant}", flush=True)
        binding = provenance["variants"][variant]
        same(read(experiment / f"{variant}-checks.json"), binding, "variant provenance copy")
        program_path = frozen / "programs" / f"{variant}.json"
        program_hash = files[f"programs/{variant}.json"]["sha256"]
        same(binding["program_sha256"], program_hash, "program manifest binding")
        same(generation[variant]["sha256"], program_hash, "generation program hash")
        contract = None
        if program_path.is_file():
            program = read(program_path)
            contract = program_contract(program, variant, spec)
            bindings = {k.replace("\\", "/"): v for k, v in program["source_sha256"].items()}
            same(set(bindings), SOURCE_FILES, "meaningful program source bindings")
            if source_root:
                for name, expected in bindings.items():
                    same(sha(source_root / name), expected, f"program source file {name}")
                    same(hashlib.sha256(git_bytes(source_root, args.source_commit, name)).hexdigest(), expected, f"program source git blob {name}")
                    source_hashes[name] = expected
            del program
            gc.collect()
        else:
            missing.append(f"descriptor-level per-stage/channel traffic: {variant}")
        schedule_path = experiment / "inputs" / f"ae-{variant}-step0-layer0.json"
        schedule = read(schedule_path)
        same(sha(schedule_path), binding["schedule_sha256"], "schedule binding")
        same(schedule["boundary"], binding["boundary"], "provenance boundary")
        schedule_check(schedule, variant, measured)
        if source_root:
            relative = f"input-data/ae-{variant}.json"
            full_hash = schedule["source_full_schedule_sha256"]
            same(sha(source_root / relative), full_hash, "full schedule source hash")
            same(hashlib.sha256(git_bytes(source_root, args.source_commit, relative)).hexdigest(), full_hash, "committed full schedule source")
            source_hashes[relative] = full_hash
        cfg_reference = read(experiment / "configs" / f"{variant}.json")
        hashes, first_result = [], None
        for repetition in range(2):
            folder = experiment / "native" / variant / f"repeat-{repetition}"
            raw, cfg, prov = read(folder / "raw.json"), read(folder / "input-config.json"), read(folder / "provenance.json")
            same(cfg, cfg_reference, "native config matches prepared config")
            same((cfg["mode"], cfg["run_kind"]), ("serial", "ae_solo"), "AE-only configuration")
            require({"vlm_path", "calibration", "arbitration", "refresh_allbank"}.isdisjoint(cfg), "forbidden single-layer configuration")
            same(f"{cfg['layout']}_{cfg['rope']}_{'after' if cfg['dataflow'] == 'fused' else 'before'}", variant, "config design")
            same(cfg["spec"], spec, "config embedded spec")
            same(prov["spec"], spec, "native provenance spec")
            same(prov["spec_sha256"], object_sha(spec), "native spec hash")
            same(prov["calibration"], None, "no calibration provenance")
            same(prov["source_commit"], args.source_commit, "native source")
            same(prov["binary_sha256"], provenance["binary_sha256"], "native binary hash")
            same(prov["runtime_libraries"], provenance["runtime_libraries"], "native loaded runtime library")
            expected_library_path = manifest["server_run_directory"].rstrip("/") + "/frozen/lib/libramulator.so"
            same(prov["runtime_libraries"]["libramulator.so"]["path"], expected_library_path, "actual loaded frozen library path")
            same(prov["config_sha256"], sha(folder / "input-config.json"), "config file hash")
            same(set(prov["inputs"]), {"ae_schedule_path", "program_path"}, "only AE input provenance")
            for key, expected_hash, expected_name in (("program_path", program_hash, f"{variant}.json"),
                                                      ("ae_schedule_path", sha(schedule_path), schedule_path.name)):
                same(prov["inputs"][key], {"name": expected_name, "sha256": expected_hash}, f"native {key} hash")
            raw_hash = sha(folder / "raw.json")
            same(raw_hash, prov["raw_sha256"], "native raw hash")
            same(raw_hash, binding["raw_sha256"][repetition], "experiment raw hash")
            hashes.append(raw_hash)
            log = (folder / "native.log").read_text(encoding="utf-8", errors="replace")
            require(not re.search(r"assertion.*failed|terminate called|segmentation fault|bus checker.*(error|fail)", log, re.I), "native failure diagnostic")
            result, stage_rows, gpu_rows = native_check(raw, schedule, spec, contract, variant)
            same(read_bytes := result["pim_read_bytes"] + result["pim_write_bytes"], generation[variant]["dq_bytes"], "generated total DQ bytes")
            check = binding["checks"][repetition]
            compare_record(check, dict(repetition=repetition, checks_passed=True, checked_commands=result["checked_commands"]), "repeat check")
            if first_result is not None:
                same(result, first_result, "repeat numerical result")
            first_result = result
        same(hashes[0], hashes[1], "repeat raw byte identity")
        result.update(repeats=2, repeat_byte_identical=True, wall_seconds=sum(binding["wall_seconds"]))
        published = next(row for row in summary["results"] if row["variant"] == variant)
        compare_record(published, result, "summary result")
        compare_record(next(row for row in result_table if row["variant"] == variant), result, "latency CSV", csv_strings=True)
        all_stage_rows.extend(stage_rows)
        all_gpu_rows.extend(gpu_rows)
        report["results"].append(dict(result, raw_sha256=hashes[0], program_sha256=program_hash,
                                      stages={s["name"]: s["duration_us"] for s in stage_rows}))
    for actual_rows, expected_rows, label in ((stage_table, all_stage_rows, "stage CSV"), (gpu_table, all_gpu_rows, "GPU CSV")):
        same(len(actual_rows), len(expected_rows), f"{label} row count")
        for actual, expected in zip(actual_rows, expected_rows):
            compare_record(actual, expected, label, csv_strings=True)
    report.update(job=manifest.get("slurm_job_id"), binary_sha256=provenance["binary_sha256"],
                  library_sha256=provenance["runtime_libraries"]["libramulator.so"]["sha256"],
                  spec_sha256=object_sha(spec), requirements_sha256=provenance["requirements_sha256"],
                  verified_source_bindings=source_hashes,
                  checks=dict(ten_raw_config_provenance=True, fixed_rank_refresh=True, repeats_byte_identical=True,
                              step0_layer0_only=True, no_gpu_or_vlm_requests=True, timeline_and_total=True,
                              per_channel_time_and_byte_conservation=True, dq_occupancy_exact=True,
                              csv_and_summary_match=True, manifest_hash_bindings=True,
                              descriptor_traffic_verified=all(f"programs/{v}.json" in verified_files for v in VARIANTS),
                              actual_artifacts_and_source_verified=not missing))
    report["status"] = "incomplete" if missing else "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive_directory", type=Path, help="extracted archive root containing experiment/ and frozen/")
    parser.add_argument("source_commit", help="expected full 40-character source commit")
    parser.add_argument("--frozen-dir", type=Path, help="actual bin/lib/program artifacts; read only")
    parser.add_argument("--source-root", type=Path, help="repository at expected HEAD; read only git/file checks")
    parser.add_argument("--output", type=Path, help="default: ARCHIVE_DIRECTORY/independent-audit.json")
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.source_commit), "expected source must be a full commit SHA")
    report = dict(schema_version=1, status="failed", source_commit=args.source_commit,
                  audit_script_sha256=sha(Path(__file__)), refresh_policy=POLICY,
                  results=[], unavailable_checks=[], simulator_executed=False, calibration_executed=False)
    try:
        audit(args, report)
    except (ValueError, KeyError, OSError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    output = args.output or args.archive_directory / "independent-audit.json"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "error": report.get("error"),
                      "results": [{k: r[k] for k in ("variant", "total_us", "PIM_us", "GPU_us", "rank0_REFpb", "rank0_REFab", "rank1_REFpb", "rank1_REFab")}
                                  for r in report["results"]], "unavailable_checks": report["unavailable_checks"]}, indent=2), flush=True)
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())

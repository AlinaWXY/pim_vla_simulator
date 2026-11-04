"""Retry one corrected guard assertion without modifying frozen experiment code."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CASE = "O_missing_rewind"
BEFORE = "Whole SRF batch order mismatch"
AFTER = "duplicate staging load"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def import_frozen(root):
    source = root / "experiments/pi05-unified-gemv/validation/new_o_native_guards.py"
    spec = importlib.util.spec_from_file_location("frozen_o_native_guards", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.CASES[CASE] != BEFORE:
        raise AssertionError("Frozen guard expectation does not match original assertion")
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--original-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    frozen = args.frozen_root.resolve(strict=True)
    original = args.original_suite.resolve(strict=True)
    output = args.output.resolve()
    backend = frozen / "experiments/pi05-bgk-gpuq/backend"
    if args.worker:
        module = import_frozen(frozen)
        module.CASES[CASE] = AFTER  # The only frozen-module runtime override.
        return module.worker(backend, output, CASE)

    output.mkdir(parents=True, exist_ok=False)
    source = frozen / "experiments/pi05-unified-gemv/validation/new_o_native_guards.py"
    library = backend / "ramulator2/libramulator.so"
    original_summary_path = original / "o-native-guards/summary.json"
    original_result_path = original / "o-native-guards" / CASE / "case-result.json"
    original_trace = original / "o-native-guards" / CASE / "guard.trace"
    original_log = original / "o-native-guards" / (CASE + ".log")
    provenance_path = original / "provenance.json"
    provenance = read(provenance_path)
    original_summary = read(original_summary_path)
    original_case = next(r for r in original_summary["native_cases"] if r["case"] == CASE)
    if original_case["passed"] or original_case["expected_message"] != BEFORE or original_case["message"] != AFTER:
        raise AssertionError("Original evidence is not the expected assertion-only mismatch")
    if not all(r["passed"] for r in original_summary["native_cases"] if r["case"] != CASE):
        raise AssertionError("Other original guard failures must not be masked")
    source_key = str(source.relative_to(frozen))
    library_hash, source_hash = sha(library), sha(source)
    if provenance["native_library_sha256"] != library_hash or provenance["source_sha256"][source_key] != source_hash:
        raise AssertionError("Frozen guard source or binary differs from original full-test provenance")
    preserved = output / "original"
    preserved.mkdir()
    copies = {"summary.json": original_summary_path, "case-result.json": original_result_path,
        "worker.log": original_log, "guard.trace": original_trace}
    for name, path in copies.items():
        shutil.copyfile(path, preserved / name)
    shutil.copyfile(provenance_path, output / "provenance.json")
    original_full_log = original / "runner-logs/o-native-guards.log"
    if original_full_log.is_file():
        shutil.copyfile(original_full_log, preserved / "suite-runner.log")
    protected_paths = [source, library, provenance_path, original_summary_path, original_result_path, original_log, original_trace]
    protected_hashes = {str(path): sha(path) for path in protected_paths}
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    retry_dir = output / "retry-case"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--frozen-root", str(frozen),
        "--original-suite", str(original), "--output", str(retry_dir)]
    run = subprocess.run(command, cwd=frozen, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=600)
    retry_log = output / "retry-worker.log"
    retry_log.write_text(run.stdout, encoding="utf-8")
    retry_result_path = retry_dir / "case-result.json"
    retry = read(retry_result_path) if retry_result_path.is_file() else dict(case=CASE, passed=False,
        message="Retry worker did not produce case evidence")
    retry["worker_returncode"] = run.returncode
    retry["passed"] = bool(retry["passed"]) and run.returncode == 0
    if any(sha(Path(path)) != value for path, value in protected_hashes.items()):
        raise AssertionError("Frozen source, binary or original evidence changed during isolated retry")
    retry_trace = retry_dir / "guard.trace"
    trace_identical = retry_trace.is_file() and sha(retry_trace) == sha(original_trace)
    artifacts = [retry_log, output / "provenance.json"] + [p for p in retry_dir.rglob("*") if p.is_file()]
    correction = dict(schema_version=1, case=CASE, created_utc=datetime.now(timezone.utc).isoformat(),
        reason="The missing rewind is rejected by the earlier duplicate-load guard before the whole-SRF batch-order guard.",
        scope="Expected error substring only; original native binary, workload generator and mutant trace unchanged.",
        original_suite=str(original), original_summary_sha256=sha(original_summary_path),
        original_case=original_case, original_actual_error=original_case["message"],
        expectation_before=BEFORE, expectation_after=AFTER,
        original_provenance_sha256=sha(provenance_path), native_library_sha256=library_hash,
        frozen_guard_source_sha256=source_hash, retry_driver_sha256=sha(Path(__file__).resolve()),
        original_artifact_sha256={name: sha(path) for name, path in copies.items()},
        retry_artifact_sha256={str(path.relative_to(output)): sha(path) for path in artifacts},
        original_and_retry_trace_identical=trace_identical, frozen_and_original_artifacts_unchanged=True,
        retry_result=retry, separate_retry_execution=True, all_tests_rerun=False)
    dump(output / "assertion-correction.json", correction)
    aggregate = deepcopy(original_summary)
    aggregate["native_cases"] = [retry if r["case"] == CASE else r for r in aggregate["native_cases"]]
    aggregate["passed"] = trace_identical and all(r["passed"] for r in aggregate["native_cases"])
    aggregate["correction"] = dict(case=CASE, metadata="../assertion-correction.json",
        metadata_sha256=sha(output / "assertion-correction.json"), original_summary="../original/summary.json",
        original_case="../original/case-result.json", retry_case="../retry-case/case-result.json",
        separate_retry_execution=True, replaced_case_count=1, original_passed_case_count=12)
    aggregate["provenance_policy"] = "Original full-suite provenance preserved verbatim; this aggregate substitutes one separately retried assertion result."
    aggregate_dir = output / "o-native-guards-corrected"
    aggregate_dir.mkdir()
    dump(aggregate_dir / "summary.json", aggregate)
    print(json.dumps(dict(passed=aggregate["passed"], case_count=aggregate["case_count"],
        retry_result=retry, trace_identical=trace_identical,
        aggregate=str(aggregate_dir / "summary.json"), correction=str(output / "assertion-correction.json"))), flush=True)
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

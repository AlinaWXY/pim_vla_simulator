"""Rerun H=16 fused QKV / GPU RoPE transfers / DRAM Q relay native timing.

Build the native backend first with build.sh, then run with that build's Python.
This runner always simulates fresh traces; reference files are comparison data.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PRIMARY_CASES = ("qkv", "qk", "pv", "gate_up", "down_tiled")
SCHEMA_VERSION = "xinyao_1025_h16_qkv_dram_q"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_leaves(value, prefix=()):
    """Index every numeric field, including per-channel resource partitions."""
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            found.update(numeric_leaves(child, prefix + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(numeric_leaves(child, prefix + (str(index),)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found[prefix] = value
    return found


def same_number(actual, expected):
    # Integer clocks/counts must match exactly. The tolerance only permits
    # floating-point presentation differences, far below a simulator clock.
    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def compare_reference(run, reference):
    expected_json = reference / "all-linear-breakdown/breakdown.json"
    actual_json = run / "all-linear-breakdown/breakdown.json"
    expected_csv = reference / "all-linear-breakdown/all-linear-breakdown.csv"
    actual_csv = run / "all-linear-breakdown/all-linear-breakdown.csv"
    expected, actual = load(expected_json), load(actual_json)
    for label, data in (("reference", expected), ("fresh run", actual)):
        if data.get("schema_version") != SCHEMA_VERSION or data.get("horizon") != 16:
            raise AssertionError(f"{label} is not this H=16 schema; H=20 results are not an exact reproduction baseline")
    expected_numbers, actual_numbers = numeric_leaves(expected), numeric_leaves(actual)
    if expected_numbers.keys() != actual_numbers.keys():
        missing = sorted(expected_numbers.keys() - actual_numbers.keys())[:10]
        extra = sorted(actual_numbers.keys() - expected_numbers.keys())[:10]
        raise AssertionError(f"Breakdown numeric schema differs: missing={missing}, extra={extra}")
    mismatches = [("/".join(key), actual_numbers[key], value)
                  for key, value in expected_numbers.items()
                  if not same_number(actual_numbers[key], value)]
    if mismatches:
        raise AssertionError(f"Breakdown numeric mismatch (actual, expected): {mismatches[:10]}")
    with expected_csv.open(newline="", encoding="utf-8") as stream:
        expected_rows = list(csv.DictReader(stream))
    with actual_csv.open(newline="", encoding="utf-8") as stream:
        actual_rows = list(csv.DictReader(stream))
    if len(expected_rows) != len(actual_rows):
        raise AssertionError("CSV row count differs")
    csv_fields_checked = 0
    for old, new in zip(expected_rows, actual_rows):
        if old.keys() != new.keys() or old["operator"] != new["operator"]:
            raise AssertionError("CSV schema/operator order differs")
        for key in old.keys() - {"operator"}:
            if not same_number(float(new[key]), float(old[key])):
                raise AssertionError(f"CSV mismatch for {old['operator']}/{key}: {new[key]} != {old[key]}")
            csv_fields_checked += 1
    expected_primary = {row["operator"]: row for row in expected["primary_rows"]}
    actual_primary = {row["operator"]: row for row in actual["primary_rows"]}
    if set(expected_primary) != set(PRIMARY_CASES) or set(actual_primary) != set(PRIMARY_CASES):
        raise AssertionError("Expected five primary PIM cases")
    latencies = {}
    for case in PRIMARY_CASES:
        case_dir = run / case if case in PRIMARY_CASES[:3] else run / "ffn" / case
        measured = load(case_dir / "summary.json")["latency_us"]
        expected_us = expected_primary[case]["total_us"]
        if not same_number(measured, expected_us):
            raise AssertionError(f"Primary latency differs for {case}: {measured} != {expected_us}")
        latencies[case] = measured
    return dict(status="passed", comparison_mode="exact_same_schema", reference_run=str(reference),
                numeric_breakdown_fields_checked=len(expected_numbers),
                numeric_csv_fields_checked=csv_fields_checked,
                primary_latency_us=latencies,
                comparison=dict(integer_fields="exact", relative_tolerance=1e-12,
                                absolute_tolerance=1e-12),
                reference_sha256={"breakdown.json": sha256(expected_json),
                                  "all-linear-breakdown.csv": sha256(expected_csv)},
                measured_pim_subset_us_per_layer_step=actual["measured_pim_subset_us_per_layer_step"],
                measured_pim_subset_ms_18layers_10steps=actual["measured_pim_subset_ms_18layers_10steps"],
                complete_ae_latency_ms=None)


def candidate_without_reference(run):
    """Validate a first H=16 run without calling its H=20 predecessor a baseline."""
    actual = load(run / "all-linear-breakdown/breakdown.json")
    if actual.get("schema_version") != SCHEMA_VERSION or actual.get("horizon") != 16:
        raise AssertionError("Fresh candidate is not an H=16 breakdown")
    primary = {row["operator"]: row for row in actual["primary_rows"]}
    if set(primary) != set(PRIMARY_CASES):
        raise AssertionError("Expected five primary H=16 PIM cases")
    latencies = {}
    for case in PRIMARY_CASES:
        directory = run / case if case in PRIMARY_CASES[:3] else run / "ffn" / case
        value = load(directory / "summary.json")["latency_us"]
        if not same_number(value, primary[case]["total_us"]):
            raise AssertionError(f"Fresh summary/breakdown mismatch for {case}")
        latencies[case] = value
    return dict(status="passed", horizon=16, attention_heads=8,
                comparison_mode="no_reference_first_H16_run", reference_run=None,
                numeric_breakdown_fields_checked=0, numeric_csv_fields_checked=0,
                primary_latency_us=latencies,
                measured_pim_subset_us_per_layer_step=actual["measured_pim_subset_us_per_layer_step"],
                measured_pim_subset_ms_18layers_10steps=actual["measured_pim_subset_ms_18layers_10steps"],
                complete_ae_latency_ms=None,
                note="Fresh native simulation and independent audits passed. Freeze this candidate under reference/H16, then rerun for exact same-version reproduction; no H=20 timing is reused.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=HERE.parent / "pi05-bgk-gpuq/backend")
    parser.add_argument("--output", type=Path,
                        help="New output directory (default: results/reproduce-<timestamp>)")
    parser.add_argument("--original-backend", type=Path,
                        help="Optional separate backend to check for unintended changes")
    parser.add_argument("--reference", type=Path,
                        help="Same-version H=16 reference for exact numeric reproduction (default: reference/H16)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Explicitly use a previously built backend (the runner never builds)")
    args = parser.parse_args()
    backend = args.backend.resolve(strict=True)
    run = (args.output or HERE / "results" /
           ("reproduce-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    if run.exists():
        parser.error(f"Output already exists; choose a fresh directory: {run}")
    if not (backend / "ramulator2/libramulator.so").is_file():
        parser.error(f"Backend is not built: {backend / 'ramulator2/libramulator.so'}; run build.sh first")
    reference = args.reference or HERE / "reference/H16"
    required = [reference / "all-linear-breakdown/breakdown.json",
                reference / "all-linear-breakdown/all-linear-breakdown.csv"]
    reference_available = all(path.is_file() for path in required)
    if args.reference is not None or reference.exists():
        for path in required:
            if not path.is_file():
                parser.error(f"Missing H=16 reference: {path}")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(backend / "ramulator2/python"), str(backend)])
    env["LD_LIBRARY_PATH"] = str(backend / "ramulator2") + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    logs = []

    def execute(label, arguments):
        print(f"[{label}]", flush=True)
        command = [sys.executable, *map(str, arguments)]
        result = subprocess.run(command, cwd=HERE, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append((label, command, result.returncode, result.stdout))
        if run.exists():
            log_dir = run / "runner-logs"
            log_dir.mkdir(exist_ok=True)
            for index, (name, argv, code, output) in enumerate(logs, 1):
                (log_dir / f"{index:02d}-{name}.log").write_text(
                    json.dumps(dict(command=argv, returncode=code)) + "\n" + output,
                    encoding="utf-8")
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    execute("workload-tests", ["-m", "unittest", "discover", "-s", HERE / "source", "-p", "test_*.py", "-v"])
    execute("analysis-tests", ["-m", "unittest", "discover", "-s", HERE / "analysis", "-p", "test_*.py", "-v"])
    execute("reference-tests", ["-m", "unittest", "discover", "-s", HERE / "validation", "-p", "test_reference.py", "-v"])
    attention = [HERE / "source/run_attention.py", "--backend", backend, "--output", run]
    if args.original_backend is not None:
        attention += ["--original-backend", args.original_backend.resolve(strict=True)]
    execute("attention", attention)
    execute("shared-backend-tests", [backend / "test_bgk_native.py", "--output", run / "native-tests"])
    execute("ffn", [HERE / "analysis/run_ffn.py", "--backend", backend,
                    "--output", run / "ffn"])
    execute("native-audit", [HERE / "validation/audit_native.py", run])
    execute("audit-fault-detection", [HERE / "validation/test_audit_detection.py", run])
    execute("breakdown", [HERE / "analysis/analyze_run.py", run])
    execute("plot", [HERE / "analysis/plot_breakdown.py", run / "all-linear-breakdown/breakdown.json"])
    execute("diagnostics", [HERE / "validation/explain_native.py", run])
    comparison = compare_reference(run, reference) if reference_available else candidate_without_reference(run)
    comparison.update(horizon=16, attention_heads=8)
    comparison["runner_sha256"] = sha256(Path(__file__))
    dump(run / "reference-comparison.json", comparison)
    dump(run / "reference-candidate.json", dict(schema_version=SCHEMA_VERSION, horizon=16, attention_heads=8,
        primary_latency_us=comparison['primary_latency_us'],
        comparison_mode=comparison['comparison_mode'],
        breakdown_sha256=sha256(run/'all-linear-breakdown/breakdown.json'),
        csv_sha256=sha256(run/'all-linear-breakdown/all-linear-breakdown.csv'),
        source_of_truth='all-linear-breakdown/breakdown.json; no historical result reused as a new measurement'))
    print(f"[reference comparison] {comparison['comparison_mode']}; "
          f"{comparison['numeric_breakdown_fields_checked']} exact JSON numeric fields checked", flush=True)
    execute("finalize", [HERE / "validation/finalize_run.py", run])
    # Finalize hashes all outputs. Its own runner log is necessarily written
    # after it returns, so refresh the manifest once without changing results.
    files = sorted(p for p in run.rglob("*") if p.is_file() and p.name != "artifact-sha256.json")
    dump(run / "artifact-sha256.json", {str(p.relative_to(run)): sha256(p) for p in files})
    print(json.dumps(dict(status="passed", horizon=16, attention_heads=8, output=str(run),
        measured_pim_subset_us=comparison["measured_pim_subset_us_per_layer_step"],
        measured_pim_subset_ms_18x10=comparison["measured_pim_subset_ms_18layers_10steps"],
        complete_ae_latency_ms=None), indent=2))


if __name__ == "__main__":
    main()

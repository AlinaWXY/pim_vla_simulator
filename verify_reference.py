"""Compare a fresh native run with the published H16 timing evidence."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    reference_root = Path(__file__).resolve().parent / "reference"
    expected = read(reference_root / "expected.json")
    expected_guards = read(reference_root / "validation.json")["native_guards"]
    summary, channels = read(root / "summary.json"), read(root / "channel-analysis.json")
    if summary["status"] != "passed" or channels["status"] != "passed":
        raise AssertionError("Fresh native run or channel audit did not pass")
    if expected.get("v_token_sharding") and summary.get("v_token_sharding") is not True:
        raise AssertionError("Expected token-sharded V projection")
    if not summary["tests_executed"]:
        raise AssertionError("Use the default runner, including native guards")
    guards = read(root / "native-guards/summary.json")
    expected_names = {case["case"] for case in expected_guards["native_cases"]}
    actual_names = {case["case"] for case in guards["native_cases"]}
    if (not guards["passed"] or guards["case_count"] != len(expected_names)
            or actual_names != expected_names or not all(case["passed"] for case in guards["native_cases"])):
        raise AssertionError("Expected every published native guard to pass")
    for case, reference in expected["cases"].items():
        actual = summary["runs"][case]
        for key in ("memory_cycles", "external_DQ_bytes", "trace_sha256"):
            if actual[key] != reference[key]:
                raise AssertionError((case, key, actual[key], reference[key]))
        if abs(actual["latency_us"] - reference["latency_us"]) > 1e-9:
            raise AssertionError((case, "latency_us"))
        audit = read(root / case / "native-audit.json")
        if audit["status"] != "passed" or audit["native_sha256"] != reference["native_sha256"]:
            raise AssertionError((case, "native command/GEMV CSV audit or hash mismatch"))
        if expected.get("v_token_sharding") and case in ("v", "serial_layer_control"):
            contract = audit.get("v_token_projection", {})
            if (not contract.get("legacy_cache_layout_preserved")
                    or contract.get("logical_output_elements_checked") != 32768):
                raise AssertionError((case, "V token-to-feature cache mapping audit did not pass"))
        if expected.get("full_split_activation_deadline_enforced"):
            if audit.get("split_activation", {}).get("status") != "passed":
                raise AssertionError((case, "complete ACT1/ACT2 protocol audit did not pass"))
        elif expected.get("staging_split_activation_deadline_enforced"):
            if audit.get("staging_split_activation", {}).get("status") != "passed":
                raise AssertionError((case, "staging ACT1/ACT2 protocol audit did not pass"))
    fields = ("stage_us", "modelled_dq_occupied_us", "modelled_dq_idle_us",
              "tail_clear_complete_zero_external_phase_us",
              "modelled_dq_idle_outside_tail_clear_work_us",
              "complete_zero_external_phase_us", "removed_boundary_data_tail_us",
              "other_semantic_phase_us", "boundary_overhead_us", "external_DQ_bytes")
    for case, rows in expected["stage_channel_partitions"].items():
        actual = channels["stages"][case]["channels"]
        if len(actual) != len(rows):
            raise AssertionError((case, "channel count"))
        for reference, result in zip(rows, actual):
            if reference["channel"] != result["channel"]:
                raise AssertionError((case, "channel order"))
            for field in fields:
                if abs(reference[field] - result[field]) > 1e-9:
                    raise AssertionError((case, result["channel"], field))
            if reference["tail_clear_windows"] != result["tail_clear_windows"]:
                raise AssertionError((case, result["channel"], "window boundaries"))
    if abs(channels["stage_sum_us"] - expected["stage_sum_us"]) > 1e-9:
        raise AssertionError("Five-stage sum differs")
    return dict(status="passed", native_cases=len(expected["cases"]), native_guards=len(expected_names),
                channels=16, exact_cycles_bytes_traces_and_native_CSVs=True,
                channel_time_partitions_and_windows_match=True,
                stage_sum_us=channels["stage_sum_us"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run.resolve(strict=True)), indent=2))

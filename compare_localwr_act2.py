"""Compare the LOCALWR ACT1/ACT2 correction against baseline a987650.

Usage: python compare_localwr_act2.py --before results/corrected-1902838 \
           --after results/localwr-NEW

Requires complete raw result directories. Reuses the existing seven-case
latency/DQ comparison, sums only the five primary stages, and independently
audits the complete ACT protocol. Historical spill failures remain visible;
every corrected case must pass the full protocol audit before reports are saved.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


BASELINE_REVISION = "a987650"
TITLE = "LOCALWR ACT1/ACT2 correction (baseline a987650)"
HISTORICAL_SPILL_CASES = frozenset(("chain", "serial_layer_control"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def audit_protocol(data, analyzer, auditor, cases, *, strict):
    reports = {}
    for case in cases:
        directory = data["root"] / case / "simulation"
        # Read all channels, commands and request types. Do not filter to LOCALWR
        # or staging: an unrelated newly exposed ACT violation must also fail.
        native = [event for channel in range(16)
                  for event in analyzer.read_csv(directory / f"timeline.csv.ch{channel}")]
        staging = [event for channel in range(16)
                   for event in analyzer.read_csv(directory / f"staging_events.csv.ch{channel}")]
        reports[case] = auditor.audit_act2(native, data["natives"][case]["resolved_timings_ck"],
                                          staging_events=staging, strict=strict)
    return reports


def check_historical_protocol(reports, require):
    failures = {case for case, report in reports.items() if report["status"] != "passed"}
    require(failures == HISTORICAL_SPILL_CASES,
            f"Expected a987650 spill failures only in chain/serial_layer_control; got {sorted(failures)}")
    for case, report in reports.items():
        staging = report["classifications"].get("staging", {})
        require(staging.get("paired_ACT2", 0) > 0 and not staging.get("late_ACT2", 0)
                and not staging.get("cancelled", 0) and not staging.get("pending_at_end", 0),
                f"{case}: baseline is not the staging-corrected a987650 mode")
        if case in HISTORICAL_SPILL_CASES:
            require(set(report["violations"]) == {"late_ACT2", "ACT1_while_channel_ACT2_pending"},
                    f"{case}: unexpected historical ACT violations")
            internal = report["classifications"].get("other_internal", {})
            require(internal.get("late_ACT2", 0) == report["violations"]["late_ACT2"],
                    f"{case}: historical late ACT2 is not exclusively non-staging internal traffic")


def write_report(common, result, output):
    # Keep the established CSV schema. The dedicated directory, renamed JSON,
    # metadata and Markdown title distinguish this from the staging experiment.
    common.write_report(result, output)
    (output / "act2-deadline-comparison.json").replace(output / "localwr-act2-comparison.json")
    protocol = dict(title=TITLE, baseline_revision=BASELINE_REVISION,
                    after_full_protocol_passed=True, **result["act_protocol"])
    (output / "localwr-act2-protocol-audit.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"# {TITLE}", "", "Times are microseconds per layer/denoising step.", "",
             "| Case | Scope | Before | After | After minus before |", "|---|---|---:|---:|---:|"]
    entries = dict(result["cases"], **{common.SUM_NAME: result["primary_five_stage_sum"]})
    for case, entry in entries.items():
        value = entry["latency_us"]
        lines.append(f"| {case} | {entry['scope']} | {value['before']:.6f} | {value['after']:.6f} | "
                     f"{value['after_minus_before']:+.6f} |")
    lines += ["", "Only the five primary stages are summed. The combined FFN and serial-layer cases are separate controls.",
              "", "All seven corrected cases pass the complete ACT protocol audit across all 16 channels. "
              "The historical a987650 spill failures are retained in the before audit and are not treated as passing.",
              "", "[Full ACT protocol audit](localwr-act2-protocol-audit.json), "
              "[comparison JSON](localwr-act2-comparison.json), [case latency](case-latency.csv), "
              "[all semantic phases](phase-latency.csv), [all-channel DQ/CA](channel-dq-comparison.csv), "
              "[phase-kind DQ](phase-kind-dq-comparison.csv), [native commands](native-command-comparison.csv).",
              "", "DQ/CA idle is modeled resource availability, not measured concurrent GPU service. "
              "Phase differences include row-state and refresh effects; they are not pure ACT-stall measurements.", ""]
    (output / "LOCALWR-ACT2-COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Raw a987650 baseline run")
    parser.add_argument("--after", type=Path, required=True, help="Raw LOCALWR-corrected run")
    parser.add_argument("--output", type=Path, help="Default: AFTER/localwr-act2-comparison")
    parser.add_argument("--check-only", action="store_true", help="Validate without writing reports")
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent
        common_path = root / "compare_act2_deadline.py"
        audit_path = root / "experiments/pi05-unified-gemv/validation/act2_audit.py"
        common = load_module("localwr_comparison_common", common_path)
        auditor = load_module("localwr_protocol_auditor", audit_path)
        analyzer, analyzer_path = common.load_analyzer()
        before = common.load_run(args.before, analyzer)
        after = before if args.before.resolve() == args.after.resolve() else common.load_run(args.after, analyzer)
        result = common.compare_runs(before, after, analyzer_path)
        old_protocol = audit_protocol(before, analyzer, auditor, common.CASES, strict=False)
        check_historical_protocol(old_protocol, common.require)
        # strict=True is deliberately mandatory for new-run acceptance.
        new_protocol = audit_protocol(after, analyzer, auditor, common.CASES, strict=True)
        common.require(all(report["status"] == "passed" for report in new_protocol.values()),
                       "Corrected run did not pass the complete ACT protocol")
        result.update(title=TITLE, baseline_revision=BASELINE_REVISION,
                      fix_scope="LOCALWR internal split-activation scheduling; staging fix retained",
                      act_protocol=dict(before=old_protocol, after=new_protocol),
                      protocol_auditor=dict(path=str(audit_path), sha256=common.digest(audit_path)),
                      comparison_helper=dict(path=str(common_path), sha256=common.digest(common_path)))
        result["checks"].update(before_matches_expected_spill_failure_scope=True,
                                after_full_ACT_protocol_passed=True)
        output = args.output or args.after / "localwr-act2-comparison"
        if not args.check_only:
            write_report(common, result, output)
        print(json.dumps(dict(status="passed", title=TITLE, baseline_revision=BASELINE_REVISION,
                              case_count=len(result["cases"]), after_full_ACT_protocol_passed=True,
                              primary_latency_us=result["primary_five_stage_sum"]["latency_us"],
                              saved_us=result["primary_five_stage_sum"]["saved_us"],
                              output=None if args.check_only else str(output.resolve())), indent=2))
    except (AssertionError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"LOCALWR ACT2 comparison failed: {error}\n")


if __name__ == "__main__":
    main()

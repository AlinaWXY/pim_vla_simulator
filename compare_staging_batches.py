"""Compare a fresh two-slot native run with the retained single-slot run."""
import argparse
import csv
import json
from pathlib import Path


def compare(run, previous):
    old = json.loads(previous.read_text(encoding="utf-8"))
    new = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if new.get("channel_independent_dispatch") or new.get("rank_descriptors") or new.get("staging_banks", 1) != 1:
        raise AssertionError("Historical slot-only comparison requires --staging-banks 1 --legacy-rank-dispatch")
    primary = new["primary_cases"]
    if new["status"] != "passed" or not new["rank_staging"]:
        raise AssertionError("Requires a successful native staging run")
    cases = {}
    for case, result in new["runs"].items():
        before = old["cases"][case]
        if result["external_DQ_bytes"] != before["external_DQ_bytes"]:
            raise AssertionError((case, "external traffic changed"))
        changes = []
        for channel in range(16):
            path = run / case / "simulation" / f"commands.ch{channel}.csv"
            with path.open(newline="") as stream:
                counts = {name.strip(): int(count) for name, count in csv.reader(stream)}
            prior = before["issued_commands_by_channel"][channel]
            # Only row preparation and refresh may change. Payload, compute,
            # local reads/writes and result movements must remain identical.
            row_commands = {"ACT1", "ACT2", "ACTAB", "ACTHB", "PREpb", "PREab", "PREHB", "REFab", "REFpb"}
            for name in set(counts) | set(prior):
                if name not in row_commands and counts.get(name, 0) != prior.get(name, 0):
                    raise AssertionError((case, channel, name, "work changed"))
            changes.append(dict(channel=channel, commands={name: dict(
                single_slot=prior.get(name, 0), two_slot=counts.get(name, 0))
                for name in sorted(row_commands | {"LOCALRD", "MACAB", "MACQK"})}))
        cases[case] = dict(single_slot_us=before["latency_us"], two_slot_us=result["latency_us"],
            saved_us=before["latency_us"]-result["latency_us"],
            saved_percent=100*(1-result["latency_us"]/before["latency_us"]),
            phase_us=result["phase_us"], command_counts_by_channel=changes)
    single = old["stage_sum_us"]
    total = sum(cases[c]["two_slot_us"] for c in primary)
    return dict(status="passed", single_slot_commit=old["git_commit"],
        single_slot_native_job=old["native_job"], primary_cases=primary,
        single_slot_sum_us=single, two_slot_sum_us=total, saved_us=single-total,
        saved_percent=100*(1-total/single), identical_payload_and_compute_all_16_channels=True,
        cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--previous", type=Path,
        default=Path(__file__).resolve().parent / "reference/previous-single-slot.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.run, args.previous)
    target = args.output or args.run / "staging-batch-comparison.json"
    target.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "single_slot_sum_us", "two_slot_sum_us", "saved_percent")}, indent=2))

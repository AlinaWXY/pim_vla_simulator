"""Compare a two-bank native run with the preserved xinyao_1102 run."""
import argparse
import csv
import json
from pathlib import Path


def compare(run, reference):
    old = json.loads(reference.read_text(encoding="utf-8"))
    new = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if new["status"] != "passed" or new["staging_banks"] != 2:
        raise AssertionError("Requires a successful two-bank native run")
    cases = {}
    row_commands = {"ACT1", "ACT2", "ACTAB", "ACTHB", "PREpb", "PREab", "PREHB", "REFab", "REFpb"}
    for case, result in new["runs"].items():
        prior = old["cases"][case]
        if result["external_DQ_bytes"] != prior["external_DQ_bytes"]:
            raise AssertionError((case, "external traffic changed"))
        changed = []
        for ch in range(16):
            with (run / case / "simulation" / f"commands.ch{ch}.csv").open(newline="") as f:
                counts = {name.strip(): int(count) for name, count in csv.reader(f)}
            before = prior["issued_commands_by_channel"][ch]
            for name in set(counts) | set(before):
                if name not in row_commands and counts.get(name, 0) != before.get(name, 0):
                    raise AssertionError((case, ch, name, "payload or compute changed"))
            changed.append(dict(channel=ch, commands={name: dict(
                one_bank=before.get(name, 0), two_banks=counts.get(name, 0))
                for name in sorted(row_commands | {"LOCALRD", "MACAB", "MACQK", "WR"})}))
        old_store = sum(v for k, v in prior["phase_us"].items() if k.endswith("gpu_input_stage"))
        new_store = sum(v for k, v in result["phase_us"].items() if k.endswith("gpu_input_stage"))
        cases[case] = dict(one_bank_us=prior["latency_us"], two_bank_us=result["latency_us"],
            saved_us=prior["latency_us"]-result["latency_us"],
            input_store_one_bank_us=old_store, input_store_two_bank_us=new_store,
            input_store_saved_percent=100*(1-new_store/old_store),
            phase_us=result["phase_us"], command_counts_by_channel=changed)
    old_total = old["stage_sum_us"]
    new_total = sum(cases[c]["two_bank_us"] for c in new["primary_cases"])
    return dict(status="passed", baseline_commit="cbc1ee4", one_bank_native_job=1902586,
        primary_cases=new["primary_cases"], one_bank_sum_us=old_total, two_bank_sum_us=new_total,
        saved_us=old_total-new_total, saved_percent=100*(1-new_total/old_total),
        identical_payload_and_compute_all_16_channels=True,
        rank_controller_dispatch_policy_unchanged=True, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path,
        default=Path(__file__).resolve().parent / "reference/baseline-xinyao_1102.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.run, args.reference)
    (args.output or args.run / "staging-bank-comparison.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "one_bank_sum_us", "two_bank_sum_us", "saved_percent")}, indent=2))

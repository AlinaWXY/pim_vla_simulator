"""Read-only post-run check of rank Q_EXPORT callbacks against physical SRF fills.

This is a review utility outside the frozen experiment snapshot. It never
modifies evidence and imports neither the generator nor the native audit.
"""
import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            yield {key: int(value) if value.lstrip("-").isdigit() else value
                   for key, value in row.items()}


def check_case(case):
    simulation = case / "simulation"
    result = json.loads((simulation / "result.json").read_text(encoding="utf-8"))
    if not result["config"].get("pim_rank_descriptors", False):
        return None
    fills = defaultdict(list)
    for channel in range(16):
        for event in rows(simulation / f"unified_gemv.csv.ch{channel}"):
            if event["operation"] == "GEMV_Q_LOCAL":
                key = channel, event["bank_mask"], event["token_mask"]
                assert event["buffer"] == 0, (case, "Q local fill uses wrong slot")
                fills[key].append(event)
    fills = {key: deque(sorted(values, key=lambda e: e["clock"])) for key, values in fills.items()}
    callbacks = []
    for channel in range(16):
        callbacks.extend(event for event in rows(simulation / f"rank_descriptors.csv.ch{channel}")
                         if event["event"] == "CHILD_COMPLETE" and event["operation"] == "PIM_Q_EXPORT")
    minimum_slack = None
    for child in sorted(callbacks, key=lambda e: (e["clock"], e["Channel"], e["descriptor_id"])):
        bank = 4*child["BankGroup"]+child["Bank"]
        assert child["Rank"] == 1 and child["Row"] == 40 and 8 <= child["Column"] < 16
        query = 2*(child["Column"]-8)+child["Channel"]%2
        key = child["Channel"], 1 << bank, 1 << query
        assert key in fills and fills[key], (case, "Q_EXPORT child lacks physical local SRF fill", key)
        fill = fills[key].popleft()
        assert child["ready_at"] >= fill["ready_at"], (case, "Q_EXPORT child service ends before local SRF write", child, fill)
        assert child["clock"] >= fill["ready_at"], (case, "Q_EXPORT callback precedes local SRF write", child, fill)
        slack = child["clock"]-fill["ready_at"]
        minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
    assert all(not remaining for remaining in fills.values()), (case, "Unmatched physical Q local fills")
    return dict(case=case.name, Q_EXPORT_children_checked=len(callbacks),
                callback_covers_local_SRF_write=True, minimum_callback_slack_ck=minimum_slack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--guards", action="store_true")
    args = parser.parse_args()
    cases = sorted(path.parent.parent for path in args.run.glob("*/simulation/result.json"))
    if args.guards:
        cases += sorted(path.parent.parent for path in (args.run / "native-guards").glob("*/simulation/result.json"))
    results = [result for case in cases if (result := check_case(case)) is not None]
    assert results, "No new rank-descriptor evidence found"
    print(json.dumps(dict(status="passed", read_only=True, cases=results,
                         total_Q_EXPORT_children_checked=sum(r["Q_EXPORT_children_checked"] for r in results)), indent=2))


if __name__ == "__main__":
    main()

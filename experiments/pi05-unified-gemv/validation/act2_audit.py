"""Independently reconstruct LPDDR5 split-activation owners and deadlines.

ACT1/ACT2 ownership is per physical bank. The existing ordinary controller's
pending-activation protocol is channel-wide: no second ACT1 or refresh may be
issued until ACT2 completes. Accesses and precharges of other banks are legal.
No controller pending/deadline metadata is trusted by this audit.
"""
from collections import Counter, defaultdict


def _bank(event):
    return tuple(int(event[name]) for name in ("Channel", "Rank", "BankGroup", "Bank"))


def _targets(event, bank):
    channel, rank, group, local_bank = bank
    if int(event["Channel"]) != channel or int(event["Rank"]) not in (-1, rank):
        return False
    command = event["command"]
    if command in ("PREab", "REFab", "ACTAB", "MACAB", "MACK", "MACQK"):
        return True
    bg, b = int(event["BankGroup"]), int(event["Bank"])
    if command in ("PREHB", "ACTHB", "MACHB"):
        return bg == -1 or bg//2 == group//2
    return bg in (-1, group) and b in (-1, local_bank)


def audit_act2(native, timings, *, staging_events=(), strict=True, max_examples=8):
    """Check full native logs; return bounded diagnostics or raise on failure.

    ``native`` uses the ordinary timeline CSV fields. ``timings`` must contain
    nAAD. ``staging_events`` is optional: it only refines classification when
    native request ``type`` is absent. It cannot change any validity check.
    ``strict=False`` returns the SAME failed verdict and violations, permitting
    diagnosis of historical logs. New-run acceptance must use the default.
    """
    n_aad = int(timings["nAAD"])
    if n_aad < 1 or max_examples < 0:
        raise ValueError("nAAD must be positive and max_examples nonnegative")
    windows = defaultdict(list)
    stages = defaultdict(list)
    for event in staging_events:
        stages[int(event["Channel"]), int(event["stage_generation"])].append(event)
    for (channel, _), events in stages.items():
        seals = [int(e["clock"]) for e in events if e["operation"] == "SEAL"]
        ends = [int(e["ready_at"]) for e in events if e["operation"] == "BROADCAST"]
        if seals and ends:
            windows[channel].append((min(seals), max(ends)))

    def kind(event):
        type_id = int(event.get("type", -1))
        if type_id in (26, 27):  # Independent fixed request IDs: STAGE_LOAD/FRAGMENT.
            return "staging"
        if type_id in (0, 1):
            return "ordinary"
        if type_id >= 0:
            return "other_internal"
        if int(event["Rank"]) == 1 and int(event["Row"]) >= 256:
            if any(start <= int(event["clock"]) <= end for start, end in windows[int(event["Channel"])]):
                return "staging"
            return "staging_region_unclassified"
        return "unclassified"

    pending = {}
    pending_by_channel = defaultdict(set)
    violations, examples, gaps = Counter(), [], Counter()
    violations_by_kind = defaultdict(Counter)
    by_kind = defaultdict(lambda: dict(ACT1=0, paired_ACT2=0, late_ACT2=0, cancelled=0,
                                      pending_at_end=0, gap_histogram_ck=Counter()))
    counts = Counter()
    maximum_pending = defaultdict(int)

    def fail(code, event=None, owner=None, **extra):
        violations[code] += 1
        affected = {owner["kind"]} if owner is not None else set()
        if event is not None:
            affected.add(kind(event))
            if code in ("ACT1_while_channel_ACT2_pending", "refresh_while_channel_ACT2_pending"):
                affected.update(pending[b]["kind"] for b in pending_by_channel[int(event["Channel"])])
        for category in affected:
            violations_by_kind[category][code] += 1
        if len(examples) < max_examples:
            item = dict(violation=code, **extra)
            if event is not None:
                item.update(command=event["command"], clock=int(event["clock"]),
                            bank=list(_bank(event)), row=int(event["Row"]))
            if owner is not None:
                item.update(ACT1_clock=owner["clock"], ACT1_row=owner["row"],
                            ACT1_bank=list(owner["bank"]), request_kind=owner["kind"],
                            ACT2_deadline=owner["clock"]+n_aad)
            examples.append(item)

    def remove(bank):
        owner = pending.pop(bank)
        pending_by_channel[bank[0]].remove(bank)
        return owner

    # Preserve original file order for commands with the same channel/clock.
    # Simultaneous commands on different channels do not share this protocol.
    events = sorted(enumerate(native), key=lambda pair: (int(pair[1]["Channel"]), int(pair[1]["clock"]), pair[0]))
    for _, event in events:
        command, clock = event["command"], int(event["clock"])
        bank = _bank(event)
        channel = bank[0]
        if command in ("ACT1", "ACT2"):
            counts[command] += 1
            if any(value < 0 for value in bank) or int(event["Row"]) < 0:
                fail("nonphysical_ACT_address", event)
                continue
        if command == "ACT1":
            category = kind(event)
            by_kind[category]["ACT1"] += 1
            if pending_by_channel[channel]:
                prior_bank = min(pending_by_channel[channel])
                fail("ACT1_while_channel_ACT2_pending", event, pending[prior_bank])
            if bank in pending:
                old = remove(bank)
                fail("ACT1_overwrites_pending_owner", event, old)
                counts["cancelled"] += 1
                by_kind[old["kind"]]["cancelled"] += 1
            pending[bank] = dict(clock=clock, row=int(event["Row"]), bank=bank, kind=category)
            pending_by_channel[channel].add(bank)
            maximum_pending[channel] = max(maximum_pending[channel], len(pending_by_channel[channel]))
        elif command == "ACT2":
            if bank not in pending:
                fail("orphan_ACT2", event)
                continue
            owner = remove(bank)
            gap = clock-owner["clock"]
            counts["paired_ACT2"] += 1
            gaps[gap] += 1
            by_kind[owner["kind"]]["paired_ACT2"] += 1
            by_kind[owner["kind"]]["gap_histogram_ck"][gap] += 1
            if int(event["Row"]) != owner["row"]:
                fail("ACT2_row_mismatch", event, owner)
            if gap < 1:
                fail("ACT2_before_ACT1_command_finishes", event, owner, gap_ck=gap)
            if gap > n_aad:
                fail("late_ACT2", event, owner, gap_ck=gap)
                by_kind[owner["kind"]]["late_ACT2"] += 1
        else:
            if command.startswith("REF") and pending_by_channel[channel]:
                owner = pending[min(pending_by_channel[channel])]
                fail("refresh_while_channel_ACT2_pending", event, owner)
            closing = command.startswith(("PRE", "REF")) or command in ("RDA", "WRA", "ACTAB", "ACTHB")
            for target in sorted(pending_by_channel[channel]):
                if not _targets(event, target):
                    continue
                if closing:
                    owner = remove(target)
                    fail("pending_ACT1_cancelled", event, owner)
                    counts["cancelled"] += 1
                    by_kind[owner["kind"]]["cancelled"] += 1
                elif command in ("RD", "WR", "LOCALRD", "LOCALWR", "MACAB", "MACHB", "MACK", "MACQK"):
                    fail("target_bank_access_before_ACT2", event, pending[target])
    for owner in pending.values():
        fail("pending_ACT2_at_end", owner=owner)
        by_kind[owner["kind"]]["pending_at_end"] += 1
    result = dict(status="failed" if violations else "passed", nAAD_ck=n_aad,
        protocol_scope="channel; ACT1/ACT2 owner keyed by channel/rank/BG/bank",
        ACT1_commands=counts["ACT1"], ACT2_commands=counts["ACT2"], paired_ACT2=counts["paired_ACT2"],
        cancelled_ACT1=counts["cancelled"], pending_ACT2_at_end=len(pending),
        minimum_gap_ck=min(gaps, default=None), maximum_gap_ck=max(gaps, default=None),
        gap_histogram_ck=dict(sorted(gaps.items())),
        maximum_pending_by_channel=dict(sorted(maximum_pending.items())),
        classifications={name: {**stats, "gap_histogram_ck": dict(sorted(stats["gap_histogram_ck"].items()))}
                         for name, stats in sorted(by_kind.items())},
        violation_count=sum(violations.values()), violations=dict(sorted(violations.items())),
        violations_by_kind={name: dict(sorted(values.items())) for name, values in sorted(violations_by_kind.items())},
        examples=examples,
        classification="native request type; optional sealed-stage windows; unclassified source region otherwise")
    if strict and violations:
        raise AssertionError(f"ACT1/ACT2 protocol audit failed: {dict(violations)}; first examples={examples}")
    return result


def audit_staging_act2(native, timings, *, staging_events=(), strict=True, max_examples=8):
    """Enforce staging while retaining the separate full-protocol verdict.

    Reconstruct ALL commands and owners; never filter out a non-staging
    command that can interfere with a staging activation. The other paths'
    pre-existing violations remain failed in ``global_audit``.
    """
    report = audit_act2(native, timings, staging_events=staging_events,
                        strict=False, max_examples=max_examples)
    violations = report["violations_by_kind"].get("staging", {})
    result = dict(status="failed" if violations else "passed", acceptance_scope="staging only",
        violations=violations, violation_count=sum(violations.values()),
        staging=report["classifications"].get("staging", {}),
        global_audit=report,
        known_other_internal_violations=report["violations_by_kind"].get("other_internal", {}))
    if strict and violations:
        raise AssertionError(f"Staging ACT1/ACT2 protocol audit failed: {violations}")
    return result


def main():
    """Read-only CLI: accept either one case directory or a whole run directory."""
    import argparse
    import csv
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Print historical violations; still exit nonzero on failure")
    args = parser.parse_args()
    cases = [args.run] if (args.run/"simulation/result.json").is_file() else sorted(
        p.parent.parent for p in args.run.glob("*/simulation/result.json"))
    if not cases:
        parser.error("No simulation/result.json found")
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return [{key: int(value) if value.lstrip("-").isdigit() else value
                     for key, value in row.items()} for row in csv.DictReader(stream)]
    reports = {}
    for case in cases:
        simulation = case/"simulation"
        result = json.loads((simulation/"result.json").read_text(encoding="utf-8"))
        native = [e for path in sorted(simulation.glob("timeline.csv.ch*")) for e in read_csv(path)]
        staging = [e for path in sorted(simulation.glob("staging_events.csv.ch*")) for e in read_csv(path)]
        reports[case.name] = audit_act2(native, result["resolved_timings_ck"], staging_events=staging,
                                      strict=not args.report_only)
    passed = all(report["status"] == "passed" for report in reports.values())
    print(json.dumps(dict(status="passed" if passed else "failed", cases=reports), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Explain chain projection-spill scheduling from before/after raw logs.

--before and --after accept a run directory or its chain subdirectory.
--output is a JSON file; default: AFTER/localwr-act2-comparison/spill-scheduling.json.
No simulation is run. All clocks and differences are derived from the logs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys


SPILL = "projection_local_spill"
GEOMETRY = ("Rank", "BankGroup", "Bank")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: int(value) if value.lstrip("-").isdigit() else value
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def bank(event):
    return tuple(event[key] for key in GEOMETRY)


def address(value):
    return dict(zip(GEOMETRY, value))


def compact(event):
    return {key: event[key] for key in ("clock", "command", "Row", "Column")}


def continuous_runs(intervals):
    """Merge adjacent, already validated rank-CA reservations."""
    merged = []
    for begin, end, source_bank in sorted(intervals):
        require(end > begin, "Empty CA reservation")
        require(not merged or begin >= merged[-1]["end_ck_exclusive"], "Overlapping rank CA")
        if merged and begin == merged[-1]["end_ck_exclusive"]:
            item = merged[-1]
            item["end_ck_exclusive"] = end
            item["command_count"] += 1
            item["source_banks"].add(source_bank)
        else:
            item = dict(begin_ck=begin, end_ck_exclusive=end, command_count=1, source_banks={source_bank})
            merged.append(item)
        item["duration_ck"] = item["end_ck_exclusive"] - item["begin_ck"]
    for item in merged:
        item["source_banks"] = [address(value) for value in sorted(item["source_banks"])]
    return merged


def analyze(directory):
    directory = Path(directory).resolve()
    case = directory if (directory / "simulation/result.json").is_file() else directory / "chain"
    paths = {"summary": case / "summary.json", "manifest": case / "manifest.json",
             "native": case / "simulation/result.json", "barriers": case / "simulation/barriers.csv",
             "timeline": case / "simulation/timeline.csv.ch0", "rank_ca": case / "simulation/rank_ca_events.csv.ch0"}
    summary, manifest, native = (read_json(paths[name]) for name in ("summary", "manifest", "native"))
    ratio, tck = native["config"]["frontend_ratio"], native["tCK_ns"]
    require(ratio > 0 and tck > 0, "Invalid clock conversion")
    markers = {row["marker"]: row["frontend_tick"] for row in read_csv(paths["barriers"])}
    phases = {segment["name"]: dict(begin_tick=markers[segment["start"]], end_tick=markers[segment["end"]])
              for segment in manifest["segments"]}
    require(len(phases) == len(manifest["segments"]) and SPILL in phases, "Missing/duplicate chain spill phase")
    require(set(phases) == set(summary["phase_us"]), "Manifest/summary phase mismatch")
    for name, phase in phases.items():
        phase["duration_ck"] = (phase["end_tick"] - phase["begin_tick"]) / ratio
        require(phase["duration_ck"] >= 0, "Reversed phase")
        require(abs(phase["duration_ck"] * tck / 1000 - summary["phase_us"][name]) < 1e-8,
                f"{name}: barrier/summary time mismatch")
    wall = native["memory_cycles"]
    boundary = wall - sum(phase["duration_ck"] for phase in phases.values())
    require(abs(boundary * tck / 1000 - summary["phase_boundary_us"]) < 1e-8, "Boundary time mismatch")
    require(wall == summary["memory_cycles"], "Native/summary cycles mismatch")
    require(abs(wall * tck / 1000 - summary["latency_us"]) < 1e-8, "Native/summary latency mismatch")
    timeline, ca = read_csv(paths["timeline"]), read_csv(paths["rank_ca"])
    require(all(row["Channel"] == 0 for row in timeline + ca), "Expected channel 0 logs")
    window = phases[SPILL]
    spill = [row for row in timeline if window["begin_tick"] <= (row["clock"] - 1) * ratio + 1 < window["end_tick"]]
    require(spill, "No spill commands found")
    by_bank = defaultdict(list)
    for event in spill:
        if event["command"] in ("PREpb", "ACT1", "ACT2", "LOCALWR"):
            by_bank[bank(event)].append(event)
    require(len(by_bank) == 16, "Expected all 16 projection-spill banks on channel 0")
    rank_ca = [row for row in ca if row["scope"] == "rank" and row["Rank"] == 1]
    require(all(a["ready_at"] <= b["clock"] for a, b in zip(rank_ca, rank_ca[1:])), "Overlapping rank1 CA")
    ca_by_command = defaultdict(list)
    for row in rank_ca:
        ca_by_command[row["clock"], row["command"]].append(row)
    writes = []
    for row in spill:
        if row["command"] != "LOCALWR":
            continue
        reservations = ca_by_command[row["clock"], "LOCALWR"]
        require(row["Rank"] == 1 and len(reservations) == 1, "LOCALWR lacks unique rank1 CA reservation")
        writes.append((row["clock"], reservations[0]["ready_at"], bank(row)))
    banks, pairs, pending = [], [], {}
    for value, events in sorted(by_bank.items()):
        local = [event for event in events if event["command"] == "LOCALWR"]
        require(local, f"Missing LOCALWR for bank {value}")
        banks.append(dict(**address(value),
                          PREpb=[compact(e) for e in events if e["command"] == "PREpb"],
                          ACT1=[compact(e) for e in events if e["command"] == "ACT1"],
                          ACT2=[compact(e) for e in events if e["command"] == "ACT2"],
                          LOCALWR_count=len(local), first_LOCALWR=compact(local[0]), last_LOCALWR=compact(local[-1])))
    for event in spill:
        value, command = bank(event), event["command"]
        if command == "ACT1":
            require(value not in pending, "ACT1 overwrites an owner in spill analysis")
            pending[value] = event
        elif command == "ACT2":
            require(value in pending, "Unpaired spill ACT2")
            owner = pending.pop(value)
            require(owner["Row"] == event["Row"], "ACT pair row mismatch")
            begin, end = owner["clock"], event["clock"]
            # This is observed CA occupancy while ACT2 is pending, not a claim
            # that ACT2 was already timing-eligible throughout the interval.
            other = [(max(begin, left), min(end, right), source) for left, right, source in writes
                     if source != value and source[0] == value[0] and max(begin, left) < min(end, right)]
            runs = continuous_runs(other)
            longest = max(runs, key=lambda item: item["duration_ck"], default=None)
            pairs.append(dict(**address(value), Row=event["Row"], ACT1_ck=begin, ACT2_ck=end,
                              gap_ck=end-begin, deadline_ck=begin+native["resolved_timings_ck"]["nAAD"],
                              other_bank_LOCALWR_ca_ck=sum(right-left for left, right, _ in other),
                              longest_continuous_other_bank_LOCALWR_ca=longest))
    require(not pending, "Uncompleted spill ACT2 owner")
    maximum = max(pairs, key=lambda pair: (pair["longest_continuous_other_bank_LOCALWR_ca"] or {}).get("duration_ck", 0),
                  default=None)
    if maximum is not None and maximum["longest_continuous_other_bank_LOCALWR_ca"] is None:
        maximum = None
    return dict(case_directory=str(case), channel=0, frontend_ratio=ratio, tCK_ns=tck,
                timings_ck={key: native["resolved_timings_ck"][key] for key in ("nAAD", "nRP", "nRCDWR")},
                memory_cycles=wall, latency_us=summary["latency_us"], phase_order=list(phases), phases=phases,
                boundary_ck=boundary, spill_command_counts=dict(Counter(event["command"] for event in spill)),
                native_command_counts=native["issued_commands"], trace_sha256=summary["trace_sha256"],
                spill_banks=banks, activation_pairs=pairs,
                longest_pending_other_bank_LOCALWR_ca=maximum,
                source_sha256={str(path.relative_to(case)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths.values()})


def pair(before, after):
    return dict(before=before, after=after, after_minus_before=after-before)


def compare(before, after):
    require(before["tCK_ns"] == after["tCK_ns"], "tCK changed")
    require(before["frontend_ratio"] == after["frontend_ratio"], "Frontend ratio changed")
    require(before["timings_ck"] == after["timings_ck"], "ACT/PRE timing parameters changed")
    require(before["trace_sha256"] == after["trace_sha256"], "Workload trace changed")
    require(before["phase_order"] == after["phase_order"], "Phase order changed")
    phases = {name: pair(before["phases"][name]["duration_ck"], after["phases"][name]["duration_ck"])
              for name in before["phase_order"]}
    phases["phase_boundary_overhead"] = pair(before["boundary_ck"], after["boundary_ck"])
    total = pair(before["memory_cycles"], after["memory_cycles"])
    require(sum(value["after_minus_before"] for value in phases.values()) == total["after_minus_before"],
            "Phase CK differences do not close to total")
    names = before["phase_order"]
    index = names.index(SPILL)
    savings = dict(total_saved_ck=-total["after_minus_before"],
                   pre_spill_saved_ck=-sum(phases[name]["after_minus_before"] for name in names[:index]),
                   observed_spill_saved_ck=-phases[SPILL]["after_minus_before"],
                   downstream_saved_ck=-sum(phases[name]["after_minus_before"] for name in names[index+1:]),
                   boundary_saved_ck=-phases["phase_boundary_overhead"]["after_minus_before"])
    commands = {}
    for field in ("spill_command_counts", "native_command_counts"):
        commands[field] = {name: pair(before[field].get(name, 0), after[field].get(name, 0))
                           for name in sorted(set(before[field]) | set(after[field]))}
    first_old = min(row["first_LOCALWR"]["clock"] for row in before["spill_banks"])
    first_new = min(row["first_LOCALWR"]["clock"] for row in after["spill_banks"])
    last_old = max(row["last_LOCALWR"]["clock"] for row in before["spill_banks"])
    last_new = max(row["last_LOCALWR"]["clock"] for row in after["spill_banks"])
    return dict(status="passed", title="LOCALWR spill scheduling: baseline a987650 versus corrected run",
                units="DRAM CK; absolute command clocks; half-open CA intervals; phase boundaries retain fractional CK",
                before=before, after=after, latency_ck=total, phase_differences_ck=phases,
                changes_only_ck={name: value for name, value in phases.items() if value["after_minus_before"]},
                savings_breakdown_ck=savings, command_count_comparison=commands,
                first_spill_LOCALWR_ck=pair(first_old, first_new), last_spill_LOCALWR_ck=pair(last_old, last_new),
                evidence=dict(
                    observed_spill_change=f"Spill wall time changes by {phases[SPILL]['after_minus_before']:+g} CK; "
                                          f"its last channel-0 LOCALWR issue changes by {last_new-last_old:+g} CK.",
                    downstream_change=f"Downstream phases save {savings['downstream_saved_ck']:g} CK in total. "
                                      "This is an observed downstream scheduling change; refresh, row-state and CA contributions "
                                      "have not been independently attributed.",
                    occupancy_method="Each other-bank LOCALWR is matched to its rank1 CA reservation. "
                                     "Only exactly adjacent reservations within that bank's ACT1-to-ACT2 interval are merged.",
                    causal_limit="Continuous LOCALWR occupancy proves the shared rank CA was unavailable during those CKs. "
                                 "It does not prove ACT2 was timing-eligible for every CK or that all latency changes are ACT stalls."),
                command_count_scope=dict(spill_command_counts="channel 0 spill phase",
                                         native_command_counts="whole chain, all channels"),
                scope="Chain latency/phases; detailed spill command scheduling is channel 0 only. No simulation rerun.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="JSON output file")
    args = parser.parse_args()
    try:
        result = compare(analyze(args.before), analyze(args.after))
        after_root = args.after.parent if (args.after / "simulation/result.json").is_file() else args.after
        output = args.output or after_root / "localwr-act2-comparison/spill-scheduling.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(status="passed", output=str(output.resolve()),
                              savings_breakdown_ck=result["savings_breakdown_ck"]), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"LOCALWR scheduling analysis failed: {error}\n")


if __name__ == "__main__":
    main()

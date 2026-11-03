"""Prove two-slot prefetch batching without assuming extra bank input storage."""
from bisect import bisect_left, bisect_right
from collections import defaultdict

from staging_audit import (BASE, INPUTS, WEIGHTS, WEIGHT_INPUTS, offset, require,
                           source_bank, source_packet, source_location)


def address_offset(address):
    return (address >> 20)*2048+(address >> 9 & 63)*32


def audit_paired_trace(records, case, staging_banks=1):
    """The address multiset is unchanged; independently check the new ordering."""
    actual, stores = defaultdict(list), defaultdict(list)
    generation = defaultdict(int)
    for e in records:
        ch = e["address"] >> 5 & 15
        if e["operation"] == "PIM_STAGE_BEGIN":
            generation[ch] += 1
        elif e["operation"] == "ST" and address_offset(e["address"]) >= BASE:
            bank = (e["address"] >> 16 & 3)*4+(e["address"] >> 18 & 3)
            stores[ch, generation[ch]].append((bank, address_offset(e["address"])))
        elif e["operation"] in ("PIM_STAGE_FRAGMENT", "PIM_MAC_AB"):
            bank = (e["address"] >> 16 & 3)*4+(e["address"] >> 18 & 3)
            actual[ch, generation[ch]].append((e["operation"], bank, address_offset(e["address"]), e["buffer"]))
    require(set(actual) == {(ch, gen) for ch in range(16) for gen in range(1, len(INPUTS[case])+1)},
            "Paired trace lacks an operator/channel input schedule")
    batches = 0
    for (ch, gen), sequence in sorted(actual.items()):
        payload, rows = INPUTS[case][gen-1]
        base = WEIGHT_INPUTS[case][gen-1]
        columns = next(count for start, _, count in WEIGHTS if start == base)
        fragments = payload//(rows*32)
        require(fragments%2 == 0, "Paired schedule requires an even fragment count")
        expected = []
        for first in range(0, fragments, 2):
            expected.extend(("PIM_STAGE_FRAGMENT", *source_location(f, 0, rows, staging_banks), f%2)
                            for f in (first, first+1))
            expected.extend(("PIM_MAC_AB", 0, base+(f*columns+c)*32, f%2)
                            for f in (first, first+1) for c in range(columns))
            batches += 1
        require(sequence == expected,
                f"Paired trace must fill both slots before either fragment's MACs: channel{ch}, stage{gen}")
        order = ((f, t) for f in range(fragments) for t in range(rows)) if staging_banks == 1 else (
            (f, t) for first in range(0, fragments, 2) for t in range(rows) for f in (first, first+1))
        require(stores[ch, gen] == [source_location(f, t, rows, staging_banks) for f, t in order],
                "Staged write order must interleave each token across the two source bank groups")
    return batches


def audit_paired_staging(staging, native, gemv, case, complete_workload=True, staging_banks=1):
    """Reconstruct pair readiness, capacity, and source/weight row visits."""
    reads, broadcasts, macs = defaultdict(list), defaultdict(list), defaultdict(list)
    seals = {}
    for e in staging:
        gen = e["stage_generation"]
        if e["operation"] == "SEAL":
            seals[e["Channel"], gen] = e["clock"]
        if e["operation"] not in ("LOCALRD", "BROADCAST"):
            continue
        rows = INPUTS[case][gen-1][1]
        fragment = source_packet(e, rows, staging_banks)//rows
        (reads if e["operation"] == "LOCALRD" else broadcasts)[e["Channel"], gen, fragment].append(e)
    completions = {(e["Channel"], e["clock"]): e["ready_at"] for e in gemv if e["operation"] == "GEMV_MAC"}
    by_channel = defaultdict(list)
    weight_activation_ticks = defaultdict(list)
    refresh_ticks = defaultdict(list)
    for e in native:
        ch = e["Channel"]
        by_channel[ch].append(e)
        if e["command"] == "ACTAB" and e["Rank"] in (-1, 1):
            weight_activation_ticks[ch].append(e["clock"])
        if e["command"].startswith("REF") and e["Rank"] in (-1, 1):
            refresh_ticks[ch].append(e["clock"])
        if e["command"] != "MACAB":
            continue
        addr = offset(e)
        base, columns = next((base, columns) for base, end, columns in WEIGHTS if base <= addr < end)
        gen = WEIGHT_INPUTS[case].index(base)+1
        fragment = (addr-base)//(columns*32)
        require((ch, e["clock"]) in completions, "Paired schedule MAC lacks native completion evidence")
        macs[ch, gen, fragment].append(dict(e, ready_at=completions[ch, e["clock"]]))
    for ticks in (*weight_activation_ticks.values(), *refresh_ticks.values()):
        ticks.sort()
    expected = {(ch, gen, f) for ch in range(16) for gen, (payload, rows) in enumerate(INPUTS[case], 1)
                for f in range(payload//(rows*32))} if complete_workload else set(reads)
    require(set(reads) == set(broadcasts) == set(macs) == expected,
            "Paired schedule requires exact read/broadcast/MAC fragment coverage")
    pairs = sorted({(ch, gen, f//2) for ch, gen, f in expected})
    require(len(expected) == len(pairs)*2, "Paired schedule leaves an unpaired input fragment")
    batch_rows, stage_windows = [], {}
    for ch, gen, pair in pairs:
        rows = INPUTS[case][gen-1][1]
        base = WEIGHT_INPUTS[case][gen-1]
        columns = next(count for start, _, count in WEIGHTS if start == base)
        first, second = (ch, gen, pair*2), (ch, gen, pair*2+1)
        for key, slot in ((first, 0), (second, 1)):
            require(len(reads[key]) == len(broadcasts[key]) == rows, "Pair does not fill exactly one fragment per slot")
            require(all(e["buffer"] == slot for e in reads[key]+broadcasts[key]), "Pair aliases or exceeds its two physical slots")
            require(len(macs[key]) == columns, "Pair changes the number of weight MACs")
            require({source_bank(e) for e in reads[key]+broadcasts[key]} == {4*(key[2]%staging_banks)},
                    "Each fragment must remain in its independently assigned source bank")
        all_reads = reads[first]+reads[second]
        ready = max(e["ready_at"] for key in (first, second) for e in broadcasts[key])
        begin = min(e["clock"] for e in reads[first])
        last_read = max(e["clock"] for e in all_reads)
        first_mac = min(e["clock"] for e in macs[first])
        last_mac_done = max(e["ready_at"] for e in macs[second])
        require(max(e["clock"] for e in reads[first]) < min(e["clock"] for e in reads[second]),
                "Paired input fragments are not loaded consecutively")
        require(ready <= first_mac, "First MAC begins before both prefetched slots are ready")
        require(max(e["clock"] for e in macs[first]) < min(e["clock"] for e in macs[second]),
                "Paired schedule changes fragment reduction order")
        require(len({e["Row"] for e in all_reads}) == 1, "A paired input batch unexpectedly spans source rows")
        ticks = weight_activation_ticks[ch]
        require(bisect_right(ticks, last_read) == bisect_left(ticks, begin),
                "Source row switches to weights before both input slots are loaded")
        refresh = refresh_ticks[ch]
        interrupted = bisect_right(refresh, last_read)-bisect_left(refresh, begin)
        require(rows*32 <= 512 and 2*rows*32 <= 1024, "Paired inputs exceed physical SRF capacity")
        batch_rows.append(dict(channel=ch, stage_generation=gen, batch=pair, source_row=all_reads[0]["Row"],
            input_begin_ck=begin, last_input_issue_ck=last_read, both_slots_ready_ck=ready,
            first_MAC_ck=first_mac, last_MAC_complete_ck=last_mac_done,
            input_bytes_per_bank=2*rows*32, refresh_commands_during_load=interrupted))
        if staging_banks == 2:
            batch_rows[-1]["source_banks"] = sorted({source_bank(e) for e in all_reads})
        key = ch, gen
        stage_windows[key] = max(stage_windows.get(key, 0), last_mac_done)
    row_counts = []
    for (ch, gen), end in sorted(stage_windows.items()):
        begin = seals[ch, gen]
        selected = [e for e in by_channel[ch] if begin <= e["clock"] < end and e["Rank"] in (-1, 1)]
        row_counts.append(dict(channel=ch, stage_generation=gen,
            input_batches=sum(row["channel"] == ch and row["stage_generation"] == gen for row in batch_rows),
            source_ACT2=sum(e["command"] == "ACT2" and e["Row"] >= 256 for e in selected),
            weight_ACTAB=sum(e["command"] == "ACTAB" for e in selected),
            refresh_commands=sum(e["command"].startswith("REF") for e in selected)))
    return dict(input_batches_checked=len(pairs), slots_per_bank=2, slot_capacity_bytes=512,
        physical_input_capacity_bytes_per_bank=1024,
        maximum_prefetched_input_bytes_per_bank=max(row["input_bytes_per_bank"] for row in batch_rows),
        both_slots_ready_before_first_MAC=True, no_weight_row_switch_between_paired_fills=True,
        MAC_counts_and_reduction_order_preserved=True, row_command_counts=row_counts,
        first_two_batches_channel0=[row for row in batch_rows if row["channel"] == 0 and row["batch"] < 2])

"""Prove two-slot prefetch batching without assuming extra bank input storage."""
from bisect import bisect_left, bisect_right
from collections import defaultdict

from staging_audit import (BASE, INPUTS, WEIGHTS, WEIGHT_INPUTS, offset, require,
                           source_bank, source_packet, source_location, input_geometry, srf_views)


def address_offset(address):
    return (address >> 20)*2048+(address >> 9 & 63)*32


def audit_paired_trace(records, case, staging_banks=1, v_token_sharding=False,
                       pv_token_sharding=False, qkv_fusion=False, whole_srf=False):
    """The address multiset is unchanged; independently check the new ordering."""
    inputs, weights = input_geometry(case, v_token_sharding, pv_token_sharding, qkv_fusion)
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
    require(set(actual) == {(ch, gen) for ch in range(16) for gen in range(1, len(inputs)+1)},
            "Paired trace lacks an operator/channel input schedule")
    batches = 0
    for (ch, gen), sequence in sorted(actual.items()):
        payload, rows = inputs[gen-1]
        base = weights[gen-1]
        columns = next(count for start, _, count in WEIGHTS if start == base)
        fragments = payload//(rows*32)
        require(pv_token_sharding or fragments%2 == 0, "Paired schedule requires an even fragment count")
        views = srf_views(rows, whole_srf)
        expected = []
        for first in range(0, fragments, views):
            batch = range(first, min(first+views, fragments))
            expected.extend(("PIM_STAGE_FRAGMENT", *source_location(f, 0, rows, staging_banks), f%views)
                            for f in batch)
            expected.extend(("PIM_MAC_AB", 0, base+(f*columns+c)*32, f%views)
                            for f in batch for c in range(columns))
            batches += 1
        require(sequence == expected,
                f"Paired trace must fill both slots before either fragment's MACs: channel{ch}, stage{gen}")
        order = ((f, t) for f in range(fragments) for t in range(rows)) if staging_banks == 1 else (
            (f, t) for first in range(0, fragments, 2) for t in range(rows) for f in range(first,min(first+2,fragments)))
        require(stores[ch, gen] == [source_location(f, t, rows, staging_banks) for f, t in order],
                "Staged write order must interleave each token across the two source bank groups")
    return batches


def audit_paired_staging(staging, native, gemv, case, complete_workload=True, staging_banks=1, v_token_sharding=False,
                         pv_token_sharding=False, qkv_fusion=False, whole_srf=False):
    """Reconstruct pair readiness, capacity, and source/weight row visits."""
    inputs, weights = input_geometry(case, v_token_sharding, pv_token_sharding, qkv_fusion)
    reads, broadcasts, macs = defaultdict(list), defaultdict(list), defaultdict(list)
    seals = {}
    for e in staging:
        gen = e["stage_generation"]
        if e["operation"] == "SEAL":
            seals[e["Channel"], gen] = e["clock"]
        if e["operation"] not in ("LOCALRD", "BROADCAST"):
            continue
        rows = inputs[gen-1][1]
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
        require(base in weights, "Paired MAC uses the wrong operator weight region")
        gen = weights.index(base)+1
        fragment = (addr-base)//(columns*32)
        require((ch, e["clock"]) in completions, "Paired schedule MAC lacks native completion evidence")
        macs[ch, gen, fragment].append(dict(e, ready_at=completions[ch, e["clock"]]))
    for ticks in (*weight_activation_ticks.values(), *refresh_ticks.values()):
        ticks.sort()
    expected = {(ch, gen, f) for ch in range(16) for gen, (payload, rows) in enumerate(inputs, 1)
                for f in range(payload//(rows*32))} if complete_workload else set(reads)
    require(set(reads) == set(broadcasts) == set(macs) == expected,
            "Paired schedule requires exact read/broadcast/MAC fragment coverage")
    grouped = defaultdict(list)
    for ch, gen, f in expected:
        grouped[ch,gen,f//srf_views(inputs[gen-1][1], whole_srf)].append((ch,gen,f))
    pairs = sorted(grouped)
    if not whole_srf and not pv_token_sharding:
        require(len(expected) == len(pairs)*2, "Paired schedule leaves an unpaired input fragment")
    batch_rows, stage_windows, previous_batch_done = [], {}, {}
    for ch, gen, pair in pairs:
        rows = inputs[gen-1][1]
        base = weights[gen-1]
        columns = next(count for start, _, count in WEIGHTS if start == base)
        views = srf_views(rows, whole_srf)
        keys = sorted(grouped[ch,gen,pair])
        require([k[2]%views for k in keys] == list(range(len(keys))), "Input batch skips a physical SRF view")
        for key in keys:
            slot = key[2]%views
            require(len(reads[key]) == len(broadcasts[key]) == rows, "Pair does not fill exactly one fragment per slot")
            require(all(e["buffer"] == slot for e in reads[key]+broadcasts[key]), "Pair aliases or exceeds its two physical slots")
            require(len(macs[key]) == columns, "Pair changes the number of weight MACs")
            require({source_bank(e) for e in reads[key]+broadcasts[key]} == {4*(key[2]%staging_banks)},
                    "Each fragment must remain in its independently assigned source bank")
            require(len({e["Row"] for e in reads[key]}) == 1, "One input fragment spans multiple source rows")
        all_reads = [e for key in keys for e in reads[key]]
        ready = max(e["ready_at"] for key in keys for e in broadcasts[key])
        begin = min(e["clock"] for e in reads[keys[0]])
        last_read = max(e["clock"] for e in all_reads)
        first_mac = min(e["clock"] for e in macs[keys[0]])
        last_mac_done = max(e["ready_at"] for key in keys for e in macs[key])
        for first, second in zip(keys, keys[1:]):
            require(max(e["clock"] for e in reads[first]) < min(e["clock"] for e in reads[second]),
                    "Paired input fragments are not loaded consecutively")
            require(max(e["clock"] for e in macs[first]) < min(e["clock"] for e in macs[second]),
                    "Paired schedule changes fragment reduction order")
        require(ready <= first_mac, "First MAC begins before both prefetched slots are ready")
        if not whole_srf and not pv_token_sharding:
            require(len({e["Row"] for e in all_reads}) == 1, "A paired input batch unexpectedly spans source rows")
        if whole_srf:
            require(begin >= previous_batch_done.get((ch,gen),0), "SRF view overwritten before the whole previous batch completes")
            previous_batch_done[ch,gen] = last_mac_done
        ticks = weight_activation_ticks[ch]
        require(bisect_right(ticks, last_read) == bisect_left(ticks, begin),
                "Source row switches to weights before both input slots are loaded")
        refresh = refresh_ticks[ch]
        interrupted = bisect_right(refresh, last_read)-bisect_left(refresh, begin)
        require(rows*32 <= 512 and len(keys)*rows*32 <= 1024, "Paired inputs exceed physical SRF capacity")
        batch_rows.append(dict(channel=ch, stage_generation=gen, batch=pair, source_row=all_reads[0]["Row"],
            input_begin_ck=begin, last_input_issue_ck=last_read, both_slots_ready_ck=ready,
            first_MAC_ck=first_mac, last_MAC_complete_ck=last_mac_done,
            input_bytes_per_bank=len(keys)*rows*32, refresh_commands_during_load=interrupted))
        if whole_srf:
            batch_rows[-1].update(view_count=len(keys), view_bytes=rows*32,
                                  source_rows=sorted({e["Row"] for e in all_reads}))
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
    result = dict(input_batches_checked=len(pairs), slots_per_bank=2, slot_capacity_bytes=512,
        physical_input_capacity_bytes_per_bank=1024,
        maximum_prefetched_input_bytes_per_bank=max(row["input_bytes_per_bank"] for row in batch_rows),
        both_slots_ready_before_first_MAC=True, no_weight_row_switch_between_paired_fills=True,
        MAC_counts_and_reduction_order_preserved=True, row_command_counts=row_counts,
        first_two_batches_channel0=[row for row in batch_rows if row["channel"] == 0 and row["batch"] < 2])
    if whole_srf:
        result.pop("slots_per_bank")
        result.pop("slot_capacity_bytes")
        result.update(whole_SRF_bytes_per_bank=1024, view_bytes_by_rows={8:256,16:512},
            full_batch_consumption_before_any_view_overwrite=True,
            all_views_ready_before_first_MAC=True)
    return result

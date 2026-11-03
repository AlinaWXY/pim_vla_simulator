"""Exercise shared-slot hazards in isolated compiled-backend processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "source"), str(HERE / "validation")]

CASES = {
    "missing_token": "requires every token and lane",
    "duplicate_token": "overwrite before full fragment consumption",
    "overwrite_incomplete_fragment": "overwrite before full fragment consumption",
    "release_incomplete_fragment": "release of unconsumed input fragment",
    "operator_switch_without_release": "Unified GEMV MAC column order or operator switch before release",
    "Down_missing_GPU_token": "requires every token and lane",
    "Down_wrong_weight_column": "column order or operator switch",
    "Down_release_incomplete_fragment": "release of unconsumed input fragment",
    "Down_overwrite_incomplete_fragment": "overwrite before full fragment consumption",
    "Q_held_ordinary_write": "Q slot held until PIM_GEMV_RELEASE",
    "Q_fill_without_projection_release": "Q fill requires explicit release",
    "QKT_missing_query": "Unified QKT missing query/lane coverage",
    "Q_release_before_scan": "Q release requires completed 496-token K scan",
    "QKT_then_V_reuses_same_slot": None,
    "Down_GPU_input_complete": None,
    "Stage_missing_write_at_seal": "staging seal requires every declared input write",
    "Stage_load_before_seal": "staging load requires sealed input",
    "Stage_wrong_source_bank": "staging requires rank1 bank0",
    "Stage_duplicate_write": "duplicate staging write",
    "Stage_out_of_bounds_load": "staging load address outside payload",
    "Stage_rebegin_before_consume": "staging overwrite before full consumption",
    "Stage_complete_Down_fragment": None,
    "Stage_mixed_fragment_rows": "staging mixed source fragments",
    "Stage_wrong_weight_fragment": "staging source/weight fragment mismatch",
    "Stage_wrong_token_rows": "staging token rows do not match operator",
    "Stage_two_slot_Down_batch": None,
    "Stage_three_fragments_overflow": "staging overwrite before full consumption",
    "Stage_two_bank_Down_batch": None,
    "Stage_two_bank_same_BG_source": "staging requires rank1 BG0/BG1 bank0",
    "Stage_two_bank_unreserved_source": "staging requires rank1 BG0/BG1 bank0",
    "Stage_two_bank_missing_write": "staging seal requires every declared input write",
    "Stage_two_bank_wrong_weight_fragment": "staging source/weight fragment mismatch",
    "Stage_two_bank_bank_capacity": "staging source bank capacity exceeded",
    "Dispatch_queue_capacity_one": None,
    "Dispatch_late_peer": None,
    "Dispatch_repeated_peer_source": None,
    "Dispatch_no_peer_export": "Q_REMOTE has no preceding peer Q_EXPORT",
    "Descriptor_noncanonical_source": "rank descriptor requires canonical rank1 BG0 bank0",
    "Local_ACT2_spill_rope_refresh": None,
}


def address(offset=0, ch=0, bank=0):
    row, column = divmod(offset, 2048)
    return row << 20 | (bank % 4) << 18 | (bank // 4) << 16 | 1 << 15 | (column // 32) << 9 | ch << 5


def command(op, offset=0, slot=0, ch=0, bank=0):
    return f"{op} {address(offset, ch, bank)} {slot}"


def trace_for(path, name):
    if name == "Local_ACT2_spill_rope_refresh":
        # Control-path guard: spills use the initialized accumulator state.
        # Every token deliberately moves all banks off row40 before SPILL
        # and again before STORE; neither activation check can pass on row hits.
        lines = ["BARRIER 0"]
        marker = 0
        def fence():
            nonlocal marker
            marker += 1
            lines.append(f"BARRIER {marker}")
        for token in range(8):
            q, k = 82176+token*32, 82432+token*32
            lines += [f"ST {address(83968, bank=bank)}" for bank in range(16)]
            fence()
            lines += [f"PIM_RANK_SPILL {address(offset)}" for offset in (q, k)]
            fence()
            lines += [f"PIM_RANK_ROPE_COEF {address(81920+token*32)}",
                      command("PIM_RANK_ROPE_LOAD", q, 0),
                      command("PIM_RANK_ROPE_EXEC", q, 0),
                      command("PIM_RANK_ROPE_LOAD", k, 1),
                      command("PIM_RANK_ROPE_EXEC", k, 1)]
            fence()
            lines += [f"ST {address(83968, bank=bank)}" for bank in range(16)]
            fence()
            lines += [f"PIM_RANK_ROPE_STORE {address(offset)}" for offset in (q, k)]
            fence()
    elif name == "Dispatch_queue_capacity_one":
        lines = ["BARRIER 0"]+[f"LD {address(2048*(20+i))}" for i in range(192)]
        lines += [f"LD {address(2048*20+(i%64)*32, ch=1)}" for i in range(96)]
        lines += ["BARRIER 1", f"ST {address(409600, ch=1)}", "BARRIER 2",
                  f"LD {address(409600, ch=1)}", "BARRIER 3"]
    elif name == "Dispatch_no_peer_export":
        lines = [command("PIM_RANK_Q_REMOTE", 82176, 0)]
    elif name == "Descriptor_noncanonical_source":
        lines = [f"PIM_RANK_ROPE_COEF {address(81920, bank=1)}"]
    elif name in ("Dispatch_late_peer", "Dispatch_repeated_peer_source"):
        import workload
        manifest = workload.generate(path, "chain", staging_banks=2, rank_descriptors=True,
                                     channel_independent_dispatch=True)
        original = path.read_text(encoding="ascii").splitlines()
        boundary = next(s["line_end"] for s in manifest["segments"] if s["name"] == "q_export_to_gpu")
        delayed = [line for i, line in enumerate(original, 1) if i != boundary]
        first_export = next(i for i, line in enumerate(delayed) if line.startswith("PIM_RANK_Q_EXPORT "))
        # COEF cannot be reloaded until both Q and K consume it. Delay only
        # channel1 with complete, legal recomputation from existing spills.
        delay = []
        for token in range(8):
            q, k = 82176+token*32, 82432+token*32
            delay += [f"PIM_RANK_ROPE_COEF {address(81920+token*32, ch=1)}",
                      command("PIM_RANK_ROPE_LOAD", q, 0, ch=1),
                      command("PIM_RANK_ROPE_EXEC", q, 0, ch=1),
                      command("PIM_RANK_ROPE_LOAD", k, 1, ch=1),
                      command("PIM_RANK_ROPE_EXEC", k, 1, ch=1),
                      f"PIM_RANK_ROPE_STORE {address(q, ch=1)}",
                      f"PIM_RANK_ROPE_STORE {address(k, ch=1)}"]
        delayed[first_export:first_export] = delay
        lines = original+delayed if name == "Dispatch_repeated_peer_source" else delayed
        marker = 0
        for i, line in enumerate(lines):
            if line.startswith("BARRIER "):
                lines[i] = f"BARRIER {marker}"
                marker += 1
    elif name.startswith("Stage_two_bank_"):
        base = 524288
        payload = 131072 if name == "Stage_two_bank_bank_capacity" else 1024
        lines = [f"PIM_STAGE_BEGIN {address(base)} {payload} 16"]
        lines += [f"ST {address(base+32*t, bank=bank)}" for t in range(16) for bank in (0, 4)
                  if not (name == "Stage_two_bank_missing_write" and t == 15 and bank == 4)]
        if name == "Stage_two_bank_bank_capacity":
            lines.append(f"ST {address(base+65536)}")
        lines.append(f"PIM_STAGE_SEAL {address(base)}")
        if name in ("Stage_two_bank_same_BG_source", "Stage_two_bank_unreserved_source"):
            bank = 1 if name == "Stage_two_bank_same_BG_source" else 8
            lines.append(command("PIM_STAGE_FRAGMENT", base, 0, bank=bank))
        else:
            lines += [command("PIM_STAGE_FRAGMENT", base, slot, bank=4*slot) for slot in (0, 1)]
            if name == "Stage_two_bank_wrong_weight_fragment":
                lines.append(command("PIM_MAC_AB", 327680, 1))
            else:
                lines += [command("PIM_MAC_AB", 327680+(f*4+c)*32, f) for f in (0, 1) for c in range(4)]
                lines += [command("PIM_GEMV_RELEASE", slot=slot) for slot in (0, 1)]
    elif name.startswith("Stage_"):
        base = 524288
        payload = (1536 if name == "Stage_three_fragments_overflow" else
                   1024 if name in ("Stage_mixed_fragment_rows", "Stage_two_slot_Down_batch") else
                   256 if name == "Stage_wrong_token_rows" else 512)
        rows = 8 if name == "Stage_wrong_token_rows" else 16
        begin_bank = 1 if name == "Stage_wrong_source_bank" else 0
        lines = [f"PIM_STAGE_BEGIN {address(base, bank=begin_bank)} {payload} {rows}"]
        packets = 15 if name == "Stage_missing_write_at_seal" else payload//32
        lines += [f"ST {address(base+32*p)}" for p in range(packets)]
        if name == "Stage_duplicate_write":
            lines.append(f"ST {address(base)}")
        if name != "Stage_load_before_seal":
            lines.append(f"PIM_STAGE_SEAL {address(base)}")
        if name in ("Stage_two_slot_Down_batch", "Stage_three_fragments_overflow"):
            lines += [command("PIM_STAGE_FRAGMENT", base+512*f, f%2) for f in range(payload//512)]
            if name == "Stage_two_slot_Down_batch":
                lines += [command("PIM_MAC_AB", 327680+(f*4+c)*32, f%2) for f in range(2) for c in range(4)]
                lines += [command("PIM_GEMV_RELEASE", slot=slot) for slot in (0, 1)]
        elif name == "Stage_rebegin_before_consume":
            lines.append(f"PIM_STAGE_BEGIN {address(base)} {payload} {rows}")
        elif name == "Stage_mixed_fragment_rows":
            lines += [command("PIM_STAGE_LOAD", base), command("PIM_STAGE_LOAD", base+17*32)]
        else:
            packets = (16,) if name == "Stage_out_of_bounds_load" else range(16)
            if name in ("Stage_complete_Down_fragment", "Stage_wrong_weight_fragment", "Stage_wrong_token_rows"):
                lines.append(command("PIM_STAGE_FRAGMENT", base))
            else:
                lines += [command("PIM_STAGE_LOAD", base+32*p) for p in packets]
            weight_shift = 128 if name == "Stage_wrong_weight_fragment" else 0
            lines += [command("PIM_MAC_AB", 327680+weight_shift+32*p) for p in range(4)]
            lines.append(command("PIM_GEMV_RELEASE"))
    elif name.startswith("Q"):
        import workload
        manifest = workload.generate(path, "chain", rank_staging=False)
        lines = path.read_text(encoding="ascii").splitlines()
        if name in ("Q_held_ordinary_write", "Q_release_before_scan"):
            segment = next(s for s in manifest["segments"] if s["name"] == "q_remote_buffer_fill")
            lines = lines[:segment["line_end"]]
            lines.append(command("PIM_WR_GB" if name == "Q_held_ordinary_write" else "PIM_GEMV_RELEASE"))
        elif name == "Q_fill_without_projection_release":
            segment = next(s for s in manifest["segments"] if s["name"] == "projection_gemv_release")
            lines = [line for n, line in enumerate(lines, 1)
                     if not (segment["line_start"] <= n <= segment["line_end"] and line.startswith("PIM_GEMV_RELEASE"))]
        elif name == "QKT_missing_query":
            target = command("PIM_Q_REMOTE", 81920+8*32)
            if lines.count(target) != 1:
                raise AssertionError("Query deletion must identify exactly one bank/query")
            lines.remove(target)
        elif name == "QKT_then_V_reuses_same_slot":
            lines += [command("PIM_WR_GB", t*32) for t in range(16)]
            lines += [command("PIM_MAC_AB", 83968+s*32) for s in range(8)]
            lines.append(command("PIM_GEMV_RELEASE"))
        else:
            raise ValueError(name)
    elif name.startswith("Down_"):
        tokens = 15 if name == "Down_missing_GPU_token" else 16
        lines = [command("PIM_WR_GB", t*32) for t in range(tokens)]
        if name == "Down_wrong_weight_column":
            lines.append(command("PIM_MAC_AB", 327680+32))
        else:
            columns = 4 if name == "Down_GPU_input_complete" else 1
            lines += [command("PIM_MAC_AB", 327680+s*32) for s in range(columns)]
            if name in ("Down_GPU_input_complete", "Down_release_incomplete_fragment"):
                lines.append(command("PIM_GEMV_RELEASE"))
            elif name == "Down_overwrite_incomplete_fragment":
                lines.append(command("PIM_WR_GB"))
    else:
        tokens = 15 if name == "missing_token" else 16
        lines = [command("PIM_WR_GB", t*32) for t in range(tokens)]
        if name == "duplicate_token":
            lines.append(command("PIM_WR_GB"))
        else:
            lines.append(command("PIM_MAC_AB", 83968))
            if name == "overwrite_incomplete_fragment":
                lines.append(command("PIM_WR_GB"))
            elif name == "release_incomplete_fragment":
                lines.append(command("PIM_GEMV_RELEASE"))
            elif name == "operator_switch_without_release":
                lines.append(command("PIM_MAC_AB", 100352+32))
    path.write_text("\n".join(lines)+"\n", encoding="ascii")


def dispatch_positive_audit(directory, name):
    from audit_native import (read_csv, read_json, trace_records, audit_native_commands,
                              audit_slot_events, audit_external_fills)
    from dispatch_audit import audit_ingress, audit_rank_descriptors
    from staging_audit import audit_rank_ca, require
    simulation = directory / "simulation"
    read_channels = lambda name: [e for ch in range(16) for e in read_csv(simulation / f"{name}.csv.ch{ch}")]
    native, gemv, stage, ca, fine, descriptors = map(read_channels,
        ("timeline", "unified_gemv", "staging_events", "rank_ca_events", "bgk_events", "rank_descriptors"))
    records = trace_records(directory / "guard.trace")
    dispatch, ingress = read_csv(simulation / "frontend_dispatch.csv"), read_csv(simulation / "channel_ingress.csv")
    result = read_json(simulation / "result.json")
    config = result["config"]
    audit = dict(channel_ingress=audit_ingress(records, dispatch, ingress, read_csv(simulation / "barriers.csv"),
        config["channel_queue_capacity"]),
        native_data_commands_checked=audit_native_commands(records, native),
        rank_controller=audit_rank_ca(ca, native, stage, fine, records, gemv),
        rank_descriptors=audit_rank_descriptors(records, descriptors, fine, ca, dispatch, config["frontend_ratio"]))
    if name == "Dispatch_queue_capacity_one":
        require(audit["channel_ingress"]["queue_high_water_by_channel"][0] == 1,
                "Capacity-one guard never filled the blocked channel queue")
        enqueue = {e["request_id"]: e["clock"] for e in ingress if e["operation"] == "ENQUEUE"}
        stalls = [(enqueue[e["request_id"]], e["clock"]) for e in ingress
                  if e["operation"] == "DISPATCH" and e["channel"] == 0 and e["clock"] > enqueue[e["request_id"]]+1]
        progressed = [e for e in ingress if e["operation"] == "DISPATCH" and e["channel"] == 1 and
                      any(begin+1 < e["clock"] < end for begin, end in stalls)]
        require(progressed, "Other channel never dispatches while channel0 is backpressured")
        audit["other_channel_dispatches_during_channel0_backpressure"] = len(progressed)
        barriers = {e["marker"]: e for e in read_csv(simulation / "barriers.csv")}
        write = next(e for e in native if e["command"] == "WR")
        t = result["resolved_timings_ck"]
        ready = write["clock"]+t["nCWL"]+t["nBL_max"]+t["nWR"]
        require(barriers[2]["frontend_tick"] >= (ready-1)*config["frontend_ratio"],
                "Global barrier ignores native DRAM write recovery")
        audit["write_barrier_covers_DRAM_recovery"] = True
    else:
        audit["unified_gemv"] = audit_slot_events(gemv, "chain")
        audit["external_fill_checks"] = audit_external_fills(native, gemv, rank_staging=True)
        waits = [e for e in dispatch if e["operation"] == "PEER_WAIT" and e["channel"] == 0]
        require(waits, "Adverse peer control never waits for delayed channel1 export")
        completes = {e["trace_index"]: e["frontend_tick"] for e in dispatch if e["operation"] == "COMPLETE"}
        accepts = {e["trace_index"]: e["frontend_tick"] for e in dispatch if e["operation"] == "ACCEPT"}
        progressed = sum(any(wait["frontend_tick"] < e["frontend_tick"] < accepts[wait["trace_index"]] for wait in waits)
                         for e in dispatch if e["operation"] == "COMPLETE" and e["channel"] == 2)
        require(progressed > 0, "Peer wait blocks an unrelated channel")
        audit["unrelated_channel_completions_during_peer_wait"] = progressed
        if name == "Dispatch_repeated_peer_source":
            exports = [e for e in records if e["operation"] == "PIM_RANK_Q_EXPORT" and e["address"] == address(82176, ch=1)]
            remote = [e for e in records if e["operation"] == "PIM_RANK_Q_REMOTE" and e["address"] == address(82176)]
            require(len(exports) == len(remote) == 2 and accepts[remote[1]["line"]-1] >= completes[exports[1]["line"]-1],
                    "Repeated Q address reused the previous export completion")
            audit["repeated_address_waits_for_latest_export_instance"] = True
    return audit


def positive_audit(directory, name):
    if name == "Local_ACT2_spill_rope_refresh":
        return local_act2_positive_audit(directory)
    if name.startswith("Dispatch_"):
        return dispatch_positive_audit(directory, name)
    from audit_native import read_csv, read_json, audit_slot_events
    events = [e for ch in range(16)
              for e in read_csv(directory / "simulation" / f"unified_gemv.csv.ch{ch}")]
    if name in ("Stage_complete_Down_fragment", "Stage_two_slot_Down_batch", "Stage_two_bank_Down_batch"):
        result = audit_slot_events(events, "down_gpu_input")
        native = [e for ch in range(16)
                  for e in read_csv(directory / "simulation" / f"timeline.csv.ch{ch}")]
        staged = [e for ch in range(16)
                  for e in read_csv(directory / "simulation" / f"staging_events.csv.ch{ch}")]
        ca = [e for ch in range(16)
              for e in read_csv(directory / "simulation" / f"rank_ca_events.csv.ch{ch}")]
        from collections import Counter
        from staging_audit import audit_rank_ca, audit_write_timing, audit_source_identity, source_bank, offset
        from audit_native import trace_records
        two_bank = name == "Stage_two_bank_Down_batch"
        packets = 16 if name == "Stage_complete_Down_fragment" else 32
        if Counter(e["operation"] for e in staged) != Counter(BEGIN=1, WR=packets, SEAL=1, LOCALRD=packets, BROADCAST=packets):
            raise AssertionError("Positive staging control lacks complete store/read/broadcast evidence")
        if any(e["command"] == "WRGB" for e in native):
            raise AssertionError("Positive staging control unexpectedly issued direct input WRGB")
        seal = next(e for e in staged if e["operation"] == "SEAL")
        if seal["clock"] < max(e["ready_at"] for e in staged if e["operation"] == "WR"):
            raise AssertionError("Positive staging control sealed before complete DRAM write recovery")
        timings = read_json(directory / "simulation/result.json")["resolved_timings_ck"]
        reads = sorted((e for e in staged if e["operation"] == "LOCALRD"), key=lambda e: e["clock"])
        gaps = [b["clock"]-a["clock"] for a, b in zip(reads, reads[1:])]
        if not gaps or min(gaps) >= timings["nCL"]+timings["nBL_min"]:
            raise AssertionError("Staged token reads incorrectly serialize on full column completion")
        same_bank_gaps = [b["clock"]-a["clock"] for a, b in zip(reads, reads[1:]) if source_bank(a) == source_bank(b)]
        if min(same_bank_gaps) < timings["nCCDL"]:
            raise AssertionError("Pipelined staged reads violate same-bank column issue spacing")
        result["rank_controller"] = audit_rank_ca(ca, native, staged,
            records=trace_records(directory / "guard.trace"))
        result["minimum_staged_read_issue_gap_ck"] = min(gaps)
        result["staged_read_latency_ck"] = timings["nCL"]+timings["nBL_min"]
        result["different_tokens_pipeline_before_prior_read_completion"] = True
        result["external_input_bytes"] = packets*32
        result["internal_read_bytes"] = packets*32
        if name in ("Stage_two_slot_Down_batch", "Stage_two_bank_Down_batch"):
            from paired_audit import audit_paired_staging
            result["input_batching"] = audit_paired_staging(staged, native, events, "down_gpu_input",
                complete_workload=False, staging_banks=2 if two_bank else 1)
        if two_bank:
            expected = Counter((bank, 524288+32*t) for bank in (0, 4) for t in range(16))
            for operation in ("WR", "LOCALRD", "BROADCAST"):
                actual = Counter((source_bank(e), offset(e)) for e in staged if e["operation"] == operation)
                if actual != expected:
                    raise AssertionError("Two-bank control lacks independent exact source-bank coverage")
            result["writer"] = audit_write_timing([e for e in staged if e["operation"] == "WR"], timings, 2, True)
            result["source_fragment_to_MAC_weight_checks"] = audit_source_identity(staged, native, events,
                "down_gpu_input", staging_banks=2)
        return result
    if name == "Down_GPU_input_complete":
        result = audit_slot_events(events, "down_gpu_input")
        if result["operation_counts"].get("GEMV_MULTICAST") != 16 or result["operation_counts"].get("GEMV_MAC") != 4:
            raise AssertionError("Positive Down control did not consume sixteen GPU rows and four weight columns")
        from audit_native import audit_external_fills
        native = [e for ch in range(16)
                  for e in read_csv(directory / "simulation" / f"timeline.csv.ch{ch}")]
        result["external_fill_checks"] = audit_external_fills(native, events)
        return result
    # The full QK trace releases all banks; channel0 then writes one V fragment
    # into the exact same physical slot. Split only at its native Q release.
    q_releases, last_qk = {}, {}
    for event in events:
        if event["operation"] == "GEMV_QK":
            last_qk[event["Channel"]] = max(last_qk.get(event["Channel"], 0), event["ready_at"])
    for event in events:
        ch = event["Channel"]
        if event["operation"] == "GEMV_RELEASE" and event["buffer"] == 0 and event["clock"] >= last_qk.get(ch, float("inf")):
            q_releases[ch] = min(q_releases.get(ch, event["clock"]), event["clock"])
    if len(q_releases) != 16:
        raise AssertionError("Cross-operator control lacks Q ownership-release evidence")
    old = [e for e in events if e["clock"] <= q_releases[e["Channel"]]]
    new = [e for e in events if e["clock"] > q_releases[e["Channel"]]]
    before = audit_slot_events(old, "chain")
    after = audit_slot_events(new, "v")
    if set(e["buffer"] for e in new) != {0} or len([e for e in new if e["operation"] == "GEMV_MAC"]) != 8:
        raise AssertionError("V did not reuse Q's physical slot0")
    q_generation = {e["Channel"]: e["generation"] for e in old if e["operation"] == "GEMV_QK"}
    if any(e["generation"] != q_generation[e["Channel"]]+1 for e in new):
        raise AssertionError("V slot state was reset instead of continuing the shared Q generation")
    native = [e for ch in range(16)
              for e in read_csv(directory / "simulation" / f"timeline.csv.ch{ch}")]
    refresh = [e for e in native if e["command"].startswith("REF")]
    config = read_json(directory / "simulation/result.json")["config"]
    if not config["refresh"] or {e["Channel"] for e in refresh} != set(range(16)):
        raise AssertionError("QKT/V same-simulation control lacks enabled and observed native refresh")
    return dict(QKT=before, V=after, same_bank_slot_reused=True,
        generation_continues_without_reset=True, single_native_simulation=True,
        refresh_enabled=True, native_refresh_commands=len(refresh))


def local_act2_positive_audit(directory):
    """Require real split activations on both LOCALWR users under refresh."""
    from collections import Counter
    from act2_audit import audit_act2
    from audit_native import read_csv, read_json, trace_records, audit_native_commands
    from staging_audit import audit_rank_ca, require
    simulation = directory / "simulation"
    read_channels = lambda name: [e for ch in range(16) for e in read_csv(simulation / f"{name}.csv.ch{ch}")]
    native, ca, fine = map(read_channels, ("timeline", "rank_ca_events", "bgk_events"))
    result = read_json(simulation / "result.json")
    split = audit_act2(native, result["resolved_timings_ck"])
    activations = {}
    for type_id, operation in ((14, "PIM_SPILL"), (18, "PIM_ROPE_STORE")):
        writes = [e for e in native if e["command"] == "LOCALWR" and e["type"] == type_id]
        starts = [e for e in native if e["command"] == "ACT1" and e["type"] == type_id]
        expected = {(0, 1, bg, bank) for bg in range(4) for bank in range(4)}
        key = lambda e: tuple(e[k] for k in ("Channel", "Rank", "BankGroup", "Bank"))
        require(Counter(map(key, writes)) == Counter({bank: 16 for bank in expected}),
                f"{operation} guard lacks exact sixteen 32B writes per bank")
        counts = Counter(map(key, starts))
        require(set(counts) == expected and min(counts.values()) >= 8,
                f"{operation} guard failed to force activation for every token and bank")
        require(all(e["Row"] == 40 for e in starts+writes), f"{operation} used wrong working row")
        activations[operation] = dict(ACT1_commands=len(starts), LOCALWR_commands=len(writes),
                                      minimum_ACT1_per_bank=min(counts.values()))
    refresh = [e for e in native if e["command"].startswith("REF") and e["Channel"] == 0]
    internal = [e for e in native if e["type"] in (14, 18)]
    require(result["config"]["refresh"] and any(
        min(e["clock"] for e in internal) < r["clock"] < max(e["clock"] for e in internal)
        for r in refresh), "LOCALWR guard never encounters refresh during its active sequence")
    conflicting = [e for e in native if e["command"] in ("WR", "WRA") and e["Row"] == 41]
    require(len(conflicting) == 256, "LOCALWR guard omitted forced conflicting-row stores")
    return dict(split_activation=split, forced_localwrite_activations=activations,
        native_data_commands_checked=audit_native_commands(trace_records(directory / "guard.trace"), native),
        rank_controller=audit_rank_ca(ca, native, (), fine),
        native_refresh_commands=len(refresh), conflicting_row_writes=len(conflicting),
        localwrite_bytes=512*32, guard_only_nREFI_ck=result["resolved_timings_ck"]["nREFI"])


def worker(backend, output, name):
    from reproduce import setup, config_for, dump
    setup(backend)
    from simulator.backend import run_trace
    output.mkdir(parents=True)
    trace = output / "guard.trace"
    trace_for(trace, name)
    is_q = name.startswith("Q")
    is_down = name.startswith("Down_")
    independent = name.startswith(("Dispatch_", "Descriptor_", "Local_ACT2_"))
    staged = name.startswith("Stage_") or independent
    config = config_for("chain" if is_q or independent else "down_gpu_input" if is_down or staged else "v",
                        refresh=CASES[name] is None, max_cycles=500_000, rank_staging=staged,
                        staging_banks=2 if name.startswith("Stage_two_bank_") or independent else 1,
                        rank_descriptors=independent, channel_independent_dispatch=independent)
    if name == "Dispatch_queue_capacity_one":
        from dataclasses import replace
        config = replace(config, channel_queue_capacity=1)
    if name == "Local_ACT2_spill_rope_refresh":
        from dataclasses import replace
        config = replace(config, timing_overrides={**config.timing_overrides, "nREFI": 512})
    try:
        if name == "Descriptor_noncanonical_source":
            # Exercise the compiled controller rather than stopping in the
            # adapter's duplicate Python address precheck. Metadata comes
            # from an otherwise identical valid request; input.trace remains
            # the deliberately invalid original throughout the native run.
            from unittest.mock import patch
            import simulator.backend as adapter
            metadata_trace = output / "guard-metadata.trace"
            metadata_trace.write_text(f"PIM_RANK_ROPE_COEF {address(81920)}\n", encoding="ascii")
            inspect = adapter.inspect_trace
            with patch.object(adapter, "inspect_trace", lambda path, **kwargs: inspect(metadata_trace, **kwargs)):
                native = run_trace(config, trace, output / "simulation")
        else:
            native = run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = CASES[name] is not None and isinstance(error, RuntimeError) and CASES[name] in str(error)
        result = dict(case=name, passed=passed, compiled_backend_invoked=True,
            expected_message=CASES[name], exception_type=type(error).__name__, message=str(error))
        if name == "Descriptor_noncanonical_source":
            result["python_trace_precheck_bypassed"] = True
            result["native_input_retains_invalid_address"] = True
    else:
        if CASES[name] is not None:
            result = dict(case=name, passed=False, compiled_backend_invoked=True,
                          message="Invalid dependency trace was accepted")
        else:
            try:
                audit = positive_audit(output, name)
                result = dict(case=name, passed=True, compiled_backend_invoked=True,
                              memory_cycles=native["memory_cycles"], native_audit=audit)
            except Exception as error:
                result = dict(case=name, passed=False, compiled_backend_invoked=True,
                              exception_type=type(error).__name__, message=str(error))
    dump(output / "case-result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if args.case:
        return worker(backend, output, args.case)
    from reproduce import setup, dump
    env = setup(backend)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in CASES:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--backend", str(backend),
            "--output", str(output / name), "--case", name], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        (output / f"{name}.log").write_text(run.stdout, encoding="utf-8")
        evidence = output / name / "case-result.json"
        result = json.loads(evidence.read_text(encoding="utf-8")) if evidence.is_file() else dict(
            case=name, passed=False, message="Worker did not produce native evidence")
        result["worker_returncode"] = run.returncode
        result["passed"] = bool(result["passed"]) and run.returncode == 0
        results.append(result)
    summary = dict(passed=all(r["passed"] for r in results), case_count=len(results),
        subprocess_isolation=True, native_cases=results)
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

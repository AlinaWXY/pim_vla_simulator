"""Independent reconstruction of the striped input store and both CA buses.

The fixed geometry below is intentionally independent of the trace generator.
Only issue/completion evidence from the compiled controller is accepted.
"""
from collections import Counter, defaultdict

BASE = 524288
INPUTS = {
    "chain": [(16384, 8)], "v": [(32768, 16)], "pv": [(31744, 16)],
    "gate_up_gpu": [(32768, 16)], "down_gpu_input": [(131072, 16)],
    "combined_timing_chain": [(32768, 16), (131072, 16)],
    "serial_layer_control": [(16384, 8), (32768, 16), (31744, 16), (32768, 16), (131072, 16)],
}
WEIGHT_INPUTS = {
    "chain": [0], "v": [83968], "pv": [100352], "gate_up_gpu": [262144],
    "down_gpu_input": [327680], "combined_timing_chain": [262144, 327680],
    "serial_layer_control": [0, 83968, 100352, 262144, 327680],
}
WEIGHTS = ((0, 65536, 32), (83968, 100352, 8), (100352, 116736, 8),
           (131072, 163840, 16),
           (262144, 327680, 32), (327680, 360448, 4))


def input_geometry(case, v_token_sharding=False):
    """Independent per-operator geometry; never change legacy global tables."""
    inputs, weights = list(INPUTS[case]), list(WEIGHT_INPUTS[case])
    if v_token_sharding and case in ("v", "serial_layer_control"):
        index = 0 if case == "v" else 1
        inputs[index], weights[index] = (16384, 8), 131072
    return inputs, weights


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def counter_difference(expected, actual):
    missing, extra = expected-actual, actual-expected
    return (f"missing_count={sum(missing.values())}, extra_count={sum(extra.values())}; "
            f"first_missing={list(missing.items())[:5]}, first_extra={list(extra.items())[:5]}")


def offset(event):
    return event["Row"]*2048+event["Column"]*32


def source_bank(event):
    return event.get("BankGroup", 0)*4+event.get("Bank", 0)


def source_location(fragment, token, rows, staging_banks=1):
    """Fixed independent layout; do not import the producer's address helper."""
    require(staging_banks in (1, 2), "Only one or two staging banks are supported")
    return 4*(fragment%staging_banks), BASE+((fragment//staging_banks)*rows+token)*32


def source_packet(event, rows, staging_banks=1):
    require(staging_banks in (1, 2), "Only one or two staging banks are supported")
    bank = source_bank(event)
    require(event.get("Bank", 0) == 0 and bank in ((0,) if staging_banks == 1 else (0, 4)),
            "Invalid staging source bank")
    require(offset(event) >= BASE, "Staging source below reserved storage")
    fragment, token = divmod((offset(event)-BASE)//32, rows)
    return (fragment*staging_banks+bank//4)*rows+token


def audit_write_timing(writes, timings, staging_banks=1, require_short_gap=False):
    """Measure actual issued WR cadence, independently of trace declarations."""
    groups, gaps = defaultdict(list), Counter()
    same_bg_checks = 0
    short, long = timings["nCCDS_WR"], timings["nCCDL_WR"]
    for e in writes:
        groups[e["Channel"], e["stage_generation"]].append(e)
    for entries in groups.values():
        ordered = sorted(entries, key=lambda e: e["clock"])
        previous_bg = {}
        group_short_pairs = 0
        for e in ordered:
            bg = e["BankGroup"]
            if bg in previous_bg:
                require(e["clock"]-previous_bg[bg] >= long, "Staging WR violates same-BG write spacing")
                same_bg_checks += 1
            previous_bg[bg] = e["clock"]
        for a, b in zip(ordered, ordered[1:]):
            gap = b["clock"]-a["clock"]
            cross = a["BankGroup"] != b["BankGroup"]
            require(gap >= (short if cross else long), "Staging WR violates column issue spacing")
            gaps[("cross_BG" if cross else "same_BG", gap)] += 1
            group_short_pairs += cross and gap == short
        if require_short_gap:
            require(group_short_pairs > 0, "Two-bank writer never issues cross-BG WR at nCCDS_WR in an input stage")
    short_count = gaps["cross_BG", short]
    if require_short_gap:
        require(staging_banks == 2 and short_count > 0, "Two-bank writer never issues cross-BG WR at nCCDS_WR")
    return dict(same_BG_spacing_checks=same_bg_checks, cross_BG_short_WR_pairs=short_count,
        minimum_cross_BG_WR_gap_ck=min((gap for (kind, gap) in gaps if kind == "cross_BG"), default=None),
        issue_gap_histogram=[dict(kind=kind, gap_ck=gap, count=count) for (kind, gap), count in sorted(gaps.items())])


def audit_source_identity(staging, native, gemv, case, staging_banks=1, v_token_sharding=False):
    """A correct address multiset alone cannot detect permuted input fragments."""
    inputs, weights = input_geometry(case, v_token_sharding)
    source_at_fill = {}
    for e in staging:
        if e["operation"] != "BROADCAST":
            continue
        generation = e["stage_generation"]
        rows = inputs[generation-1][1]
        fragment, token = divmod(source_packet(e, rows, staging_banks), rows)
        key = e["Channel"], e["ready_at"], e["buffer"], 1 << token
        require(key not in source_at_fill, "Ambiguous simultaneous staging source fills")
        source_at_fill[key] = generation, fragment
    owners = {}
    for e in gemv:
        if e["operation"] != "GEMV_MULTICAST":
            continue
        key = e["Channel"], e["clock"], e["buffer"], e["token_mask"]
        require(key in source_at_fill, "SRF fill lacks staged source identity")
        source = source_at_fill[key]
        slot = e["Channel"], e["buffer"], e["generation"]
        require(owners.get(slot, source) == source, "Shared SRF generation mixes different source fragments")
        owners[slot] = source
    mac_addresses = {(e["Channel"], e["clock"]): offset(e)
                     for e in native if e["command"] == "MACAB"}
    checks = 0
    for e in gemv:
        if e["operation"] != "GEMV_MAC":
            continue
        slot = e["Channel"], e["buffer"], e["generation"]
        require(slot in owners, "MAC has no staged source fragment")
        generation, source_fragment = owners[slot]
        address = mac_addresses[e["Channel"], e["clock"]]
        layout = next(((base, columns) for base, end, columns in WEIGHTS if base <= address < end), None)
        require(layout is not None, "MAC weight has no independent operator layout")
        base, columns = layout
        require(weights[generation-1] == base,
                "MAC consumes input from the wrong staged operator")
        require(source_fragment == (address-base)//(columns*32),
                "Staged source fragment differs from MAC weight reduction fragment")
        checks += 1
    return checks


def audit_staging(events, native, gemv, case, timings, staging_banks=1, v_token_sharding=False):
    """Prove full array residency before the first bank->SRF transfer."""
    require(case in INPUTS, "Unknown staging geometry")
    inputs, _ = input_geometry(case, v_token_sharding)
    groups = defaultdict(list)
    allowed = {"BEGIN", "WR", "SEAL", "LOCALRD", "BROADCAST"}
    for event in events:
        require(event["operation"] in allowed, "Unknown staging operation")
        allowed_banks = (0,) if staging_banks == 1 or event["operation"] in ("BEGIN", "SEAL") else (0, 4)
        require(event["Rank"] == 1 and event["Bank"] == 0 and source_bank(event) in allowed_banks,
                "Input staging must use exactly physical rank1/BG0/bank0" if staging_banks == 1 else
                "Input staging must use rank1/BG0/bank0 and BG1/bank0, with canonical BEGIN/SEAL")
        require(0 <= event["Channel"] < 16 and event["ready_at"] >= event["clock"],
                "Invalid staging event channel/duration")
        groups[event["Channel"], event["stage_generation"]].append(event)
    expected_generations = set(range(1, len(inputs)+1))
    require(set(groups) == {(ch, generation) for ch in range(16) for generation in expected_generations},
            "Missing/extra staging generation or channel")
    writes, reads, broadcasts, input_bytes = [], [], [], 0
    last_complete = defaultdict(int)
    write_latency = timings["nCWL"]+timings["nBL_max"]+timings["nWR"]
    read_latency = timings["nCL"]+timings["nBL_min"]
    for (ch, generation), entries in sorted(groups.items()):
        payload, rows = inputs[generation-1]
        by_op = {op: [e for e in entries if e["operation"] == op] for op in allowed}
        require(len(by_op["BEGIN"]) == len(by_op["SEAL"]) == 1,
                "Every stage requires exactly one BEGIN and one SEAL")
        begin, seal = by_op["BEGIN"][0], by_op["SEAL"][0]
        require(offset(begin) == offset(seal) == BASE, "Staging base differs from reserved row256")
        require(begin["bytes"] == payload, "Staging declaration differs from independent tensor size")
        require(begin["clock"] >= last_complete[ch], "Staging storage overwritten before earlier reads complete")
        expected_offsets = Counter(source_location(f, t, rows, staging_banks)
                                   for f in range(payload//(rows*32)) for t in range(rows))
        for op in ("WR", "LOCALRD", "BROADCAST"):
            require(Counter((source_bank(e), offset(e)) for e in by_op[op]) == expected_offsets,
                    f"Staging {op} lacks exact full-input source-address coverage")
            require(all(e["bytes"] == 32 for e in by_op[op]), "Staging packet is not32 bytes")
        require(all(e["clock"] >= begin["clock"] for e in by_op["WR"]),
                "Staging write occurs before BEGIN")
        require(all(e["ready_at"] == e["clock"]+write_latency for e in by_op["WR"]),
                "Staging write completion omits CWL/burst/write recovery")
        require(seal["clock"] >= max(e["ready_at"] for e in by_op["WR"]),
                "Input sealed before every DRAM write completes")
        read_map = {(source_bank(e), offset(e)): e for e in by_op["LOCALRD"]}
        for e in by_op["LOCALRD"]:
            require(e["clock"] >= seal["ready_at"], "Local input read precedes full-input seal")
            require(e["ready_at"] == e["clock"]+read_latency,
                    "Internal input read must charge DRAM column read latency")
            packet = source_packet(e, rows, staging_banks)
            require(e["buffer"] == (packet//rows)%2, "Staged read targets the wrong shared slot")
        for e in by_op["BROADCAST"]:
            source = read_map[source_bank(e), offset(e)]
            require(e["clock"] >= source["clock"], "Broadcast dispatch precedes its source read")
            require(e["ready_at"] == max(source["ready_at"], e["clock"]),
                    "Broadcast must finish within the read latency unless CA dispatch is delayed")
            require(e["buffer"] == source["buffer"], "Broadcast slot differs from source read")
        last_complete[ch] = max(e["ready_at"] for e in by_op["BROADCAST"])
        input_bytes += payload
        writes.extend(by_op["WR"])
        reads.extend(by_op["LOCALRD"])
        broadcasts.extend(by_op["BROADCAST"])
    def physical_key(e):
        return e["Channel"], e["clock"], e["Rank"], e["BankGroup"], e["Bank"], e["Row"], e["Column"]
    native_writes = [e for e in native if e["command"] in ("WR", "WRA") and offset(e) >= BASE]
    native_reads = [e for e in native if e["command"] == "LOCALRD" and offset(e) >= BASE]
    require(Counter(map(physical_key, writes)) == Counter(map(physical_key, native_writes)),
            "Staging writes must correspond to actual external native array WR commands")
    require(Counter(map(physical_key, reads)) == Counter(map(physical_key, native_reads)),
            "Staging reads must correspond to actual native LOCALRD commands")
    require(not any(e["command"] in ("RD", "RDA", "WRGB") and offset(e) >= BASE for e in native),
            "Internal staging read/broadcast incorrectly uses external DQ")
    # A bank row may switch while the column's already-latched result is still
    # in flight. It must nevertheless retain the physical read-to-precharge
    # interval; waiting for nCL here would incorrectly serialize the pipeline.
    last_source_read, precharge_checks = {}, 0
    for e in sorted(native, key=lambda item: (item["Channel"], item["clock"])):
        ch = e["Channel"]
        if e["command"] == "LOCALRD" and offset(e) >= BASE:
            last_source_read[ch, source_bank(e)] = e["clock"]
        affects_rank = e["Rank"] in (-1, 1)
        for bank in ((0,) if staging_banks == 1 else (0, 4)):
            affects_source = (e["command"] == "PREab" or
                e["command"] == "PREpb" and e["BankGroup"] in (-1, bank//4) and e["Bank"] in (-1, 0) or
                e["command"] == "PREHB" and e["BankGroup"] in (-1, bank//4))
            if affects_rank and affects_source and (ch, bank) in last_source_read:
                require(e["clock"]-last_source_read[ch, bank] >= timings["nRTP"],
                        "Source-bank precharge violates internal read-to-precharge timing")
                precharge_checks += 1
    expected_fills = Counter()
    for e in broadcasts:
        rows = inputs[e["stage_generation"]-1][1]
        token = ((offset(e)-BASE)//32)%rows
        expected_fills[e["Channel"], e["ready_at"], e["buffer"], 1 << token, 65535, 65535] += 1
    actual_fills = Counter((e["Channel"], e["clock"], e["buffer"], e["token_mask"], e["lane_mask"], e["bank_mask"])
                          for e in gemv if e["operation"] == "GEMV_MULTICAST")
    require(actual_fills == expected_fills, "Staged source reads do not exactly cover physical SRF multicast completions")
    identity_checks = audit_source_identity(events, native, gemv, case, staging_banks, v_token_sharding)
    result = dict(stages_checked=len(groups), source_bank=0, staging_base_row=256,
        external_input_bytes=input_bytes, internal_read_bytes=len(reads)*32,
        multicast_destination_bytes=len(broadcasts)*32*16,
        write_completion_checks=len(writes), full_input_seal_checks=len(groups),
        source_read_to_precharge_checks=precharge_checks,
        read_to_SRF_multicast_checks=len(broadcasts),
        source_fragment_to_MAC_weight_checks=identity_checks,
        external_input_bytes_unchanged=True, internal_read_and_broadcast_use_DQ=False,
        SRAM_bytes_per_bank=1024,
        maximum_DRAM_staging_bytes_per_rank=max(size for size, _ in inputs),
        reserved_DRAM_staging_bytes_per_rank=131072)
    if v_token_sharding and case in ("v", "serial_layer_control"):
        result.update(external_input_bytes_unchanged=False,
                      V_input_bytes_saved_vs_feature_sharding=16*16384,
                      external_input_matches_independent_geometry=True)
    if staging_banks == 2:
        result.update(source_banks=[0, 4], source_bank_groups=[0, 1], staging_banks=2,
            reserved_DRAM_staging_bytes_per_source_bank=65536,
            maximum_DRAM_staging_bytes_per_source_bank=max(size for size, _ in inputs)//2,
            source_last_reserved_row=287,
            writer=audit_write_timing(writes, timings, staging_banks, require_short_gap=True))
        result.pop("source_bank")
    return result


def audit_rank_ca(events, native, staging, controller_events=(), records=None, gemv_events=()):
    """Check command intervals and coverage independently on each physical bus."""
    resources = defaultdict(list)
    fine = Counter()
    for event in events:
        require(event["scope"] in ("rank", "global"), "Unknown CA scope")
        require(event["ready_at"] > event["clock"], "CA command must occupy a positive interval")
        expected_cycles = 2 if event["command"] in ("ACTAB", "ACTHB") else 1
        require(event["ready_at"]-event["clock"] == expected_cycles,
                "CA duration differs from LPDDR command-cycle contract")
        require(0 <= event["Channel"] < 16, "Invalid CA channel")
        ranks = (0, 1) if event["Rank"] == -1 else (event["Rank"],)
        require(all(rank in (0, 1) for rank in ranks), "Invalid CA rank")
        if event["scope"] == "global":
            resources[event["Channel"], "global", -1].append(event)
        else:
            for rank in ranks:
                resources[event["Channel"], "rank", rank].append(event)
                fine[event["Channel"], event["clock"], rank, event["command"]] += 1
    for resource, commands in resources.items():
        ordered = sorted(commands, key=lambda e: (e["clock"], e["ready_at"]))
        for first, second in zip(ordered, ordered[1:]):
            require(first["ready_at"] <= second["clock"],
                    f"Overlapping command bus reservations on {resource}: {first} / {second}")
    expected = Counter()
    for e in native:
        ranks = (0, 1) if e["Rank"] == -1 else (e["Rank"],)
        for rank in ranks:
            expected[e["Channel"], e["clock"], rank, e["command"]] += 1
    for e in staging:
        if e["operation"] == "BROADCAST":
            expected[e["Channel"], e["clock"], e["Rank"], "BCAST"] += 1
    for e in controller_events:
        if e["operation"] == "PIM_ROPE_EXEC":
            expected[e["Channel"], e["clock"], e["Rank"], "PIM_ROPE_EXEC"] += 1
    for e in gemv_events:
        if e["operation"] == "GEMV_Q_LOCAL":
            expected[e["Channel"], e["clock"], e["Rank"], "Q_LOCAL_FILL"] += 1
    require(fine == expected,
            "Rank CA log must account for every native command and broadcast: "+counter_difference(expected, fine))
    coarse_count = None
    if records is not None:
        expected_coarse = Counter((e["address"] >> 5 & 15, e["operation"])
                                  for e in records if e["operation"].startswith("PIM_STAGE_"))
        actual_coarse = Counter((e["Channel"], e["command"]) for e in events
                               if e["scope"] == "global" and e["command"].startswith("PIM_STAGE_"))
        require(expected_coarse == actual_coarse,
                "Global CA must carry exactly the coarse staging descriptors from the trace")
        coarse_count = sum(count for (_, command), count in actual_coarse.items()
                           if command == "PIM_STAGE_FRAGMENT")
    return dict(command_intervals_checked=len(events), rank_command_coverage=sum(fine.values()),
        global_CA_resources=sum(key[1] == "global" for key in resources),
        rank_CA_resources=sum(key[1] == "rank" for key in resources),
        transfer_broadcast_compute_share_rank_CA=True, no_CA_interval_overlap=True,
        coarse_fragment_descriptors_checked=coarse_count)

"""Independent fused QKV, local V RMW and token-parity PV evidence checks."""
from collections import Counter, defaultdict

from audit_native import decode, require

CACHE = 163840


def cache_coordinates(head, token, feature):
    bank, pair = divmod(feature % 128, 8)
    lane = 2*pair+feature//128
    local = token//2
    return (2*head+token%2, 1, bank//4, bank%4,
            CACHE+(local//16*16+lane)*32+local%16*2)


def audit_cache_contract(contract, columns):
    require((contract["base_bytes"],contract["end_bytes_exclusive"],contract["used_bytes_per_bank"],
             contract["reserved_bytes_per_bank"]) == (CACHE,180224,15872,16384), "Token V cache geometry changed")
    require(contract["rank_tokens"] == [dict(channel=c,head=c//2,physical_rank=1,
             logical_tokens=list(range(c%2,992,2))) for c in range(16)], "V cache KV parity coverage changed")
    require(contract["feature_lanes_by_bank"] == [[8*b+l//2+128*(l%2) for l in range(16)] for b in range(16)],
            "Token V cache feature permutation changed")
    for key, expected in (("local_prefix",[0,484]),("local_suffix",[484,492]),("local_padding",[492,496]),
                           ("preserve_local_prefix",[480,484]),("global_suffix",[968,984]),("global_padding",[984,992])):
        require(contract[key] == expected, "Local V RMW prefix/suffix/padding ownership changed")
    spill = contract["spill"]
    require(spill["column_offsets"] == [CACHE+(480+c)*32 for c in range(16)]
            and spill["accumulator_columns"] == columns
            and spill["source_feature_column_first"] == (32 if columns == 48 else 0),
            "V spill selects incorrect accumulator columns or cache packets")
    require((spill["per_packet_preserve_bytes"],spill["per_packet_suffix_bytes"],spill["per_packet_zero_bytes"],
             spill["external_DQ_bytes"]) == ([0,8],[8,24],[24,32],0), "V spill RMW byte masks changed")


def audit_qkv(manifest, native, gemv):
    mapping = manifest["qkv_mapping"]
    w, accum = mapping["weights"],mapping["accumulator"]
    require(mapping["operators"] == ["q","k","v"] and
            (w["base_bytes"],w["end_bytes_exclusive"],w["columns_per_fragment"],w["fragment_count"],w["copies_per_head"])
            == (393216,491520,48,64,2), "Fused QKV weight geometry/order changed")
    require(w["column_operators"] == [dict(operator=p,column_first=i*16,column_end=(i+1)*16)
                for i,p in enumerate(("q","k","v"))], "Fused QKV operator columns overlap or omit a projection")
    require((accum["rows"],accum["columns"],accum["bytes_per_bank"]) == (8,48,1536), "QKV output footprint changed")
    require(mapping["input"]["rank_tokens"] == [dict(channel=c,physical_rank=1,head=c//2,
            logical_tokens=list(range(c%2,16,2))) for c in range(16)], "Fused QKV X token parity changed")
    selected = {(e["Channel"],e["clock"]) for e in native if e["command"] == "MACAB" and 192 <= e["Row"] < 240}
    consumes = [e for e in gemv if e["operation"] == "GEMV_MAC" and (e["Channel"],e["clock"]) in selected]
    require(len(selected) == len(consumes) == 49152, "Fused QKV does not consume all 64x48 weight columns")
    scalar = sum(e["token_mask"].bit_count()*e["bank_mask"].bit_count()*e["lane_mask"].bit_count() for e in consumes)
    require(scalar == 100663296 and all(e["service_cycles"] == 8 for e in consumes), "Fused QKV arithmetic or row service changed")
    return dict(logical_input_elements=131072, projection_MAC_commands=len(selected), scalar_MACs=scalar,
        weight_bytes_per_bank=98304, output_bytes_per_bank=1536, physical_output_capacity_bytes=2048,
        complete_Q_K_V_feature_coverage=True, numerical_compute=False)


def audit_local_cache(events, native, gemv, result, fused):
    columns, source_base, source_end = (48,393216,491520) if fused else (16,131072,163840)
    source = [e for e in native if e["command"] == "MACAB" and source_base <= e["Row"]*2048+e["Column"]*32 < source_end]
    require(Counter(e["Channel"] for e in source) == Counter({c:64*columns for c in range(16)}),
            "Local V spill lacks complete source projection")
    completion = {(e["Channel"],e["clock"]):e["ready_at"] for e in gemv if e["operation"] == "GEMV_MAC"}
    source_ready = {c:max(completion[e["Channel"],e["clock"]] for e in source if e["Channel"] == c) for c in range(16)}
    declared = [e for e in events if e["operation"] == "SOURCE_READY"]
    require(len(declared) == 16 and {e["Channel"] for e in declared} == set(range(16)), "V source-ready evidence incomplete")
    for e in declared:
        require(e["ready_at"] == source_ready[e["Channel"]] and e["source_columns"] == columns,
                "V source-ready event precedes full MAC completion or uses wrong accumulator")
    operations = ("LOCALRD","GATHER","CAST","LOCALWR")
    grouped = defaultdict(dict)
    for e in events:
        if e["operation"] in ("SOURCE_READY","RESULT_EXPORT"): continue
        require(e["operation"] in operations, "Unknown V cache operation")
        key = tuple(e[k] for k in ("Channel","Rank","BankGroup","Bank","Row","Column"))
        require(e["operation"] not in grouped[key], "Duplicate V cache RMW packet operation")
        grouped[key][e["operation"]] = e
    expected = {(c,1,b//4,b%4,87,32+l) for c in range(16) for b in range(16) for l in range(16)}
    require(set(grouped) == expected, "Local V RMW does not cover all 4096 final-block packets")
    t, config = result["resolved_timings_ck"],result["config"]
    expected_native = Counter()
    port_busy, last_store = defaultdict(int), defaultdict(int)
    for key, group in sorted(grouped.items(),key=lambda item: item[1].get("LOCALRD",{}).get("clock",-1)):
        require(set(group) == set(operations), "V cache RMW packet omits read, gather, cast or write")
        rd,gather,cast,wr = (group[op] for op in operations)
        require(len({e["request_id"] for e in group.values()}) == 1, "V cache packet changes request identity")
        lane = key[-1]-32
        for e in group.values():
            require(e["source_columns"] == columns and e["source_feature_column"] == lane+(32 if fused else 0)
                    and e["gather_elements"] == 8, "V cache gathers wrong FP32 accumulator cells")
            require((e["preserved_mask"],e["updated_mask"],e["zero_mask"]) == (15,4080,61440),
                    "V RMW corrupts prefix, suffix, or padding lanes")
        require(rd["clock"] >= source_ready[key[0]], "V local read begins before projection completes")
        require(rd["ready_at"] == rd["clock"]+t["nCL"]+t["nBL_min"], "V local read latency is wrong")
        require(gather["clock"] >= rd["ready_at"] and gather["ready_at"] == gather["clock"]+config["gemv_port_cycles"],
                "V gather precedes read completion or uses wrong local-port time")
        require(cast["clock"] >= gather["ready_at"] and cast["ready_at"] == cast["clock"]+config["pim_cast_cycles"],
                "V cast precedes gather completion or omits conversion latency")
        require(wr["clock"] >= cast["ready_at"] and wr["ready_at"] == wr["clock"]+t["nLOCALWR"]+t["nWR"],
                "V local write omits cast or write recovery")
        require((rd["bytes"],gather["bytes"],cast["bytes"],wr["bytes"]) == (32,32,16,32), "V local path byte counts changed")
        for command,e in (("LOCALRD",rd),("LOCALWR",wr)):
            expected_native[command,e["clock"],*key] += 1
        last_store[key[0]] = max(last_store[key[0]],wr["ready_at"])
    # Check the local gather port independently in issue order, not packet order.
    for e in sorted((e for e in events if e["operation"] == "GATHER"), key=lambda e:(e["Channel"],e["clock"])):
        require(e["clock"] >= port_busy[e["Channel"]], "Overlapping V gathers on the shared local port")
        port_busy[e["Channel"]] = e["ready_at"]
    actual = Counter((e["command"],e["clock"],*(e[k] for k in ("Channel","Rank","BankGroup","Bank","Row","Column")))
        for e in native if e["command"] in ("LOCALRD","LOCALWR") and 163840 <= e["Row"]*2048+e["Column"]*32 < 180224)
    require(actual == expected_native, "V cache event logs do not match actual native read/write issues")
    require(not any(e["command"] in ("RD","RDA","WR","WRA") and 163840 <= e["Row"]*2048+e["Column"]*32 < 180224
                    for e in native), "V cache local RMW unexpectedly uses external DQ")
    return dict(packets=4096, local_read_bytes=131072, local_write_bytes=131072, gathered_FP32_bytes=131072,
        cast_BF16_bytes=65536, external_DQ_bytes=0, preserved_prefix_lanes=4, suffix_lanes=8, zero_padding_lanes=4,
        source_columns=columns, store_ready_by_channel=dict(last_store), numerical_compute=False)


def audit_token_pv(manifest, records, native, gemv, cache_events, result):
    mapping = manifest["pv_token_mapping"]
    fused = bool(manifest.get("qkv_fusion"))
    columns = 48 if fused else 16
    audit_cache_contract(mapping["local_cache"], columns)
    contracts = mapping.get("stages", {manifest["case"]:mapping})
    report = dict(cache_base_bytes=163840, cache_reserved_bytes_per_bank=16384,
                  physical_output_capacity_bytes=2048, numerical_compute=False)
    exports = [e for e in cache_events if e["operation"] == "RESULT_EXPORT"]
    requests = {e["line"]-1:e for e in records if e["operation"] == "PIM_RD_RES"}
    require(Counter(e["request_id"] for e in exports) == Counter(requests.keys()), "Shared-output export completion log coverage differs")
    actual_exports = Counter()
    for e in exports:
        require(e["bytes"] == 32 and e["ready_at"] >= e["clock"], "Invalid output-pool export completion")
        geometry = tuple(e[k] for k in ("Channel","Rank","BankGroup","Bank","Row","Column"))
        require(geometry == decode(requests[e["request_id"]]["address"]), "Output-pool export request/address mismatch")
        actual_exports[e["Channel"],e["clock"],e["Row"],e["Column"]] += 1
    native_exports = Counter((e["Channel"],e["clock"],e["Row"],e["Column"]) for e in native if e["command"] == "RDRES")
    require(not (actual_exports-native_exports), "Output-pool export has no real native RDRES")
    report["output_exports_checked"] = len(exports)
    if any(stage in contracts for stage in ("chain","v")):
        report["local_cache"] = audit_local_cache(cache_events,native,gemv,result,fused)
    else:
        require(not any(e["operation"] != "RESULT_EXPORT" for e in cache_events), "Standalone PV unexpectedly produced V cache writes")
    if "pv" in contracts:
        contract = contracts["pv"]
        inputs = contract["input"]
        require((inputs["local_rows"],inputs["fragment_count"],inputs["bytes_per_rank"],inputs["query_batches"])
                == (16,31,15872,1), "PV must consume one complete sixteen-query batch")
        require(inputs["rank_kv_tokens"] == [dict(channel=c,head=c//2,logical_tokens=list(range(c%2,992,2))) for c in range(16)],
                "PV P input does not partition all KV tokens exactly by parity")
        require(inputs["queries"] == list(range(16)) and inputs["masked_global_padding"] == [984,992], "PV query/padding coverage changed")
        require(contract["gpu_reduction"]["operation"] == "sum" and contract["gpu_reduction"]["operands_per_element"] == 2,
                "PV rank partials require corresponding peer-parity sums")
        macs = [e for e in native if e["command"] == "MACAB" and 80 <= e["Row"] < 88]
        keys = {(e["Channel"],e["clock"]) for e in macs}
        consumes = [e for e in gemv if e["operation"] == "GEMV_MAC" and (e["Channel"],e["clock"]) in keys]
        require(len(macs) == len(consumes) == 7936 and all(e["service_cycles"] == 16 and e["token_mask"] == 65535 for e in consumes),
                "PV must consume 31 fragments x16columns x16queries on every rank")
        scalar = sum(e["token_mask"].bit_count()*e["bank_mask"].bit_count()*e["lane_mask"].bit_count() for e in consumes)
        require(scalar == 32505856, "PV padded scalar MAC count changed")
        export_range = contract["trace_ranges"]["exports"][0]
        exports = [e for e in records if export_range["line_start"] <= e["line"] <= export_range["line_end"]]
        require(len(exports) == 8192 and all(e["operation"] == "PIM_RD_RES" for e in exports), "PV partial output export incomplete")
        for index,e in enumerate(exports):
            ch,rank,bg,bank,row,col = decode(e["address"])
            require((ch,rank,bg,bank,row*2048+col*32) == (index%16,1,0,0,index//16*32), "PV partial-result packet ordering changed")
        if "local_cache" in report:
            require(not result["config"].get("v_cache_preinstalled",False), "Serial PV bypasses its actual local V producer")
            for e in macs:
                require(e["clock"] >= report["local_cache"]["store_ready_by_channel"][e["Channel"]], "PV MAC precedes full local V write recovery")
        else:
            require(result["config"].get("v_cache_preinstalled") is True, "Standalone PV lacks explicit full-cache precondition")
        report["PV"] = dict(projection_MAC_commands=7936, scalar_MACs=scalar, output_bytes_per_bank=1024,
            input_bytes_per_rank=15872, rank_partial_export_bytes=262144, GPU_sum_elements=32768,
            query_batches=1, query_rows=16, cache_preinstalled=result["config"].get("v_cache_preinstalled",False))
    return report

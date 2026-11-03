"""Independent token-sharded V tensor/packet oracle; no generator imports.

The simulator carries timing and addresses, not numerical tensor values. This
checks the declared packing contract against independent tensor geometry and
the actual trace/native addresses; it does not claim numerical GEMV accuracy.
"""
from collections import Counter

from audit_native import decode, require


def physical_address(channel, bg, bank, offset):
    return (offset//2048)<<20 | bank<<18 | bg<<16 | 1<<15 | (offset%2048//32)<<9 | channel<<5 | offset%32


def output_coordinates(head, token, feature):
    """Independent logical result -> FP32 stream and old BF16 cache address."""
    bank, pair = divmod(feature % 128, 8)
    lane = 2*pair+feature//128
    channel = 2*head+token%2
    result_offset = ((token//2*16+bank)*16+lane)*4
    cache_channel = 2*head+(feature%32)//16
    cache_bank = feature%16
    cached_token = 968+token
    cache_offset = 100352+(cached_token//16*8+feature//32)*32+cached_token%16*2
    return dict(channel=channel, bank=bank, output_lane=lane, result_offset_bytes=result_offset,
                trace_address=physical_address(channel, 0, 0, result_offset//32*32),
                cache_address=physical_address(cache_channel, cache_bank//4, cache_bank%4, cache_offset))


def audit_v_token_mapping(manifest, records):
    contract = manifest["v_token_mapping"]
    require(contract["enabled"] and contract["logical_shape"] == [8, 16, 256], "V logical shape changed")
    inputs = contract["input"]
    require(inputs["rank_tokens"] == [dict(channel=ch, physical_rank=1, head=ch//2,
                logical_tokens=list(range(ch%2, 16, 2))) for ch in range(16)], "V token parity ownership changed")
    require((inputs["local_rows"], inputs["fragment_count"], inputs["lanes_per_fragment"],
             inputs["bytes_per_rank"]) == (8, 64, 16, 16384), "V input geometry changed")
    require(contract["feature_lanes_by_bank"] == [[8*b+l//2+128*(l%2) for l in range(16)]
                for b in range(16)], "V must cover every pair-major projection feature")
    weights, export, cache = (contract[key] for key in ("weights", "export", "cache_scatter"))
    require((weights["base_bytes"], weights["end_bytes_exclusive"], weights["bytes_per_bank"],
             weights["copies_per_head"]) == (131072, 163840, 32768, 2), "V full-weight replica geometry changed")
    require((export["bytes_per_rank"], export["packets_per_rank"], export["elements_per_packet"],
             export["canonical_trace_bank"]) == (8192, 256, 8, 0), "V export packet geometry changed")
    require(cache["cache_base_bytes"] == 100352 and cache["valid_suffix"] == [968,984]
            and cache["preserve_prefix"] == [960,968] and cache["masked_padding"] == [984,992]
            and cache["tail_read_block_starts"] == [960] and cache["tail_write_block_starts"] == [960,976],
            "V cache boundary changes prefix, suffix or padding ownership")
    ranges = contract["trace_ranges"]
    def select(region):
        return [e for e in records if region["line_start"] <= e["line"] <= region["line_end"]
                and e["operation"] != "BARRIER"]
    fragments = ranges["input_fragments"]
    require([f["fragment"] for f in fragments] == list(range(64)), "V input fragment mapping incomplete")
    logical_inputs = Counter()
    staged = manifest.get("rank_staging", False)
    banks = manifest.get("staging_banks", 1)
    if staged:
        for e in select(ranges["v_gpu_input_stage"]):
            if e["operation"] != "ST":
                continue
            ch, rank, bg, bank, row, col = decode(e["address"])
            require(rank == 1 and bank == 0 and bg in range(banks), "V input uses an unreserved source bank")
            physical_fragment, local = divmod((row*2048+col*32-524288)//32, 8)
            fragment = physical_fragment*banks+bg
            require(0 <= fragment < 64, "V input store outside full X payload")
            for lane in range(16):
                logical_inputs[ch//2, 2*local+ch%2, 16*fragment+lane] += 1
    for f in fragments:
        fragment = f["fragment"]
        if not staged:
            for e in select(f):
                require(e["operation"] == "PIM_WR_GB", "V direct input range contains another producer")
                ch, rank, bg, bank, row, col = decode(e["address"])
                require((rank,bg,bank,row) == (1,0,0,0) and col < 8, "V direct row exceeds local token coverage")
                for lane in range(16):
                    logical_inputs[ch//2, 2*col+ch%2, 16*fragment+lane] += 1
        compute = select(dict(line_start=f["mac_line_first"], line_end=f["mac_line_last"]))
        expected = [("PIM_MAC_AB", physical_address(ch, 0, 0, 131072+(fragment*16+c)*32), fragment%2)
                    for c in range(16) for ch in range(16)]
        require([(e["operation"],e["address"],e["buffer"]) for e in compute] == expected,
                "V fragment lacks complete, ordered replicated Wv columns")
    require(logical_inputs == Counter((h,t,k) for h in range(8) for t in range(16) for k in range(1024)),
            "V X input coverage duplicates or drops logical token/features")
    packets = select(ranges["export_packets"])
    require([(e["operation"],e["address"]) for e in packets] ==
            [("PIM_RD_RES",physical_address(ch,0,0,offset)) for offset in range(0,8192,32) for ch in range(16)],
            "V export is not the full ordered rank result stream")
    destinations = set()
    for head in range(8):
        for token in range(16):
            for feature in range(256):
                item = output_coordinates(head, token, feature)
                index = item["result_offset_bytes"]//32*16+item["channel"]
                require(packets[index]["address"] == item["trace_address"], "V result-to-packet permutation mismatch")
                destinations.add(item["cache_address"])
    require(len(destinations) == 32768, "V output scatter aliases cache elements")
    for phase, operation, blocks in (("v_prefix_tail_read", "LD", (60,)), ("v_suffix_write", "ST", (60,61))):
        expected = Counter()
        for block in blocks:
            for feature in range(256):
                bank = feature%16
                for head in range(8):
                    expected[operation,physical_address(2*head+(feature%32)//16,bank//4,bank%4,
                        100352+(block*8+feature//32)*32)] += 1
        require(Counter((e["operation"],e["address"]) for e in select(ranges[phase])) == expected,
                "V scatter changed the original feature cache RMW packets")
    return dict(logical_input_elements_checked=sum(logical_inputs.values()), logical_output_elements_checked=32768,
        weight_copies_per_head=2, weight_bytes_per_bank=32768, export_bytes=len(packets)*32,
        cache_suffix_elements=len(destinations), preserved_prefix_tokens=[960,968], masked_padding=[984,992],
        legacy_cache_layout_preserved=True, numerical_compute=False)


def audit_v_token_native(manifest, records, native, gemv, barriers, result):
    report = audit_v_token_mapping(manifest, records)
    macs = [e for e in native if e["command"] == "MACAB" and 64 <= e["Row"] < 80]
    require(len(macs) == 16*64*16, "V must issue sixteen weight columns per fragment per rank")
    keys = {(e["Channel"],e["clock"]) for e in macs}
    consumes = [e for e in gemv if e["operation"] == "GEMV_MAC" and (e["Channel"],e["clock"]) in keys]
    require(len(consumes) == len(macs) and all(e["token_mask"] == 255 and e["service_cycles"] == 8 for e in consumes),
            "Token V native MAC must use exactly eight rows")
    arithmetic = sum(e["token_mask"].bit_count()*e["bank_mask"].bit_count()*e["lane_mask"].bit_count() for e in consumes)
    require(arithmetic == 16*1024*2048, "V scalar arithmetic count changed with token sharding")
    transition = None
    if manifest["case"] == "serial_layer_control":
        pv_keys = {(e["Channel"],e["clock"]) for e in native if e["command"] == "MACAB" and 49 <= e["Row"] < 57}
        pv = [e for e in gemv if e["operation"] == "GEMV_MAC" and (e["Channel"],e["clock"]) in pv_keys]
        for ch in range(16):
            for slot in (0,1):
                last = max(e["generation"] for e in consumes if e["Channel"] == ch and e["buffer"] == slot)
                first = min(e["generation"] for e in pv if e["Channel"] == ch and e["buffer"] == slot)
                require(first == last+1, "V to PV resets or skips the shared physical SRF generation")
        markers = {e["marker"]:e["frontend_tick"] for e in barriers}
        vpart = next(e for e in manifest["parts"] if e["stage"] == "v")
        pvpart = next(e for e in manifest["parts"] if e["stage"] == "pv")
        require(vpart["end_marker"] == pvpart["start_marker"], "V/PV lack one completion boundary")
        ratio = result["config"]["frontend_ratio"]
        timings = result["resolved_timings_ck"]
        writes = [e for e in native if e["command"] in ("WR","WRA") and 49 <= e["Row"] < 57]
        ready = max(e["clock"]+timings["nCWL"]+timings["nBL_max"]+timings["nWR"] for e in writes)
        require(markers[pvpart["start_marker"]] >= (ready-1)*ratio,
                "PV begins before the converted V cache suffix finishes DRAM write recovery")
        transition = True
    report.update(projection_MAC_commands=len(macs), scalar_MACs=arithmetic,
                  V_to_PV_same_slot_generation_transition=transition)
    return report

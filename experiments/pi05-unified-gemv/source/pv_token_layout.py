"""Enumerable token-parity V cache and rank-partial PV data contracts.

Addresses describe modeled BF16 storage and FP32 result packets.  No numerical
cast, GEMV, or GPU reduction is executed by these helpers.
"""
from copy import deepcopy
import horizon_attention_layout as L

HEADS, QUERIES, FEATURES, BANKS, LANES = 8, 16, 256, 16, 16
PREFIX, VALID, PADDED = 968, 984, 992
LOCAL_PREFIX, LOCAL_VALID, LOCAL_KV = 484, 492, 496
CACHE_BASE, CACHE_BYTES = 163840, 16384
KV_FRAGMENTS = 31


def cache_location(head, token, feature):
    """[local_KV_fragment31][feature_lane16][local_KV_lane16] BF16."""
    L.check(token, PADDED, "V cache token")
    owner, lane = L.feature_owner(head, token % 2, feature)
    local = token // 2
    return (*owner, CACHE_BASE + (local // 16 * 16 + lane) * 32 + local % 16 * 2)


def spill_element(head, token, feature, qkv_fusion=False):
    L.check(token, QUERIES, "V suffix token")
    owner, lane = L.feature_owner(head, token % 2, feature)
    destination = cache_location(head, PREFIX + token, feature)
    columns = 48 if qkv_fusion else 16
    column = 32 + lane if qkv_fusion else lane
    return dict(head=head, token=token, feature=feature,
        source=dict(channel=owner[0], physical_rank=1, bank_group=owner[2], bank=owner[3],
            local_token=token//2, output_lane=lane, accumulator_column=column,
            accumulator_index=token//2*columns+column, dtype="FP32"),
        destination=dict(channel=destination[0], physical_rank=1, bank_group=destination[2],
            bank=destination[3], offset_bytes=destination[4],
            physical_address=L.physical_address(*destination), token=PREFIX+token, dtype="BF16"))


def spill_packet(channel, flat_bank, feature_lane, qkv_fusion=False):
    L.check(channel, 16, "V spill channel")
    feature = L.feature_from_lane(flat_bank, feature_lane)
    offset = CACHE_BASE + (30*16+feature_lane)*32
    elements = []
    for lane in range(16):
        local = 480+lane
        global_token = 2*local+channel%2
        item = dict(byte_in_packet=2*lane, local_token=local, token=global_token,
            feature=feature, action="preserve" if local < LOCAL_PREFIX else
            "cast_accumulator" if local < LOCAL_VALID else "zero_padding")
        if item["action"] == "cast_accumulator":
            item["producer"] = spill_element(channel//2, global_token-PREFIX, feature, qkv_fusion)["source"]
        elements.append(item)
    return dict(channel=channel, physical_rank=1, flat_bank=flat_bank,
        feature_lane=feature_lane, offset_bytes=offset,
        physical_address=L.physical_address(channel, 1, flat_bank//4, flat_bank%4, offset),
        elements=elements)


def input_coordinates(channel, fragment, local_query, input_lane, query_rows):
    L.check(channel, 16, "PV channel")
    if query_rows not in (8, 16):
        raise ValueError("PV query rows must be 8 or 16")
    L.check(fragment, (QUERIES//query_rows)*KV_FRAGMENTS, "PV input fragment")
    L.check(local_query, query_rows, "PV query row")
    L.check(input_lane, 16, "PV input lane")
    batch, kv_fragment = divmod(fragment, KV_FRAGMENTS)
    local_kv = kv_fragment*16+input_lane
    return dict(tensor="P", head=channel//2, query=batch*query_rows+local_query,
        token=2*local_kv+channel%2, local_kv=local_kv,
        padding=local_kv >= LOCAL_VALID)


def result_element(head, query, feature, parity, query_rows, dtype="FP32"):
    """A rank partial O packet, requiring the peer parity's GPU sum."""
    L.check(query, QUERIES, "PV query")
    if query_rows not in (8, 16):
        raise ValueError("PV query rows must be 8 or 16")
    if dtype not in ("FP32","BF16"):
        raise ValueError("PV result dtype must be FP32 or BF16")
    owner, lane = L.feature_owner(head, parity, feature)
    local_query = query % query_rows
    flat_bank = 4*owner[2]+owner[3]
    element_bytes=4 if dtype=="FP32" else 2
    offset = ((local_query*16+flat_bank)*16+lane)*element_bytes
    return dict(head=head, query=query, feature=feature, parity=parity,
        channel=owner[0], physical_rank=1, bank_group=owner[2], bank=owner[3],
        batch=query//query_rows, local_query=local_query, output_lane=lane,
        accumulator_index=local_query*16+lane, result_offset_bytes=offset,
        packet_offset_bytes=offset//32*32, byte_in_packet=offset%32,
        trace_address=L.physical_address(owner[0],1,0,0,offset//32*32), dtype=dtype)


def result_packet(channel, batch, offset, query_rows, dtype="FP32"):
    L.check(channel, 16, "PV output channel")
    L.check(batch, QUERIES//query_rows, "PV output batch")
    if dtype not in ("FP32","BF16"):
        raise ValueError("PV result dtype must be FP32 or BF16")
    element_bytes=4 if dtype=="FP32" else 2
    if offset%32 or not 0 <= offset < query_rows*256*element_bytes:
        raise ValueError("PV result packet offset is outside the batch")
    local_query, inside = divmod(offset, 16*16*element_bytes)
    bank, byte = divmod(inside, 16*element_bytes)
    return [result_element(channel//2, batch*query_rows+local_query,
        L.feature_from_lane(bank,lane),channel%2,query_rows,dtype)
        for lane in range(byte//element_bytes,byte//element_bytes+32//element_bytes)]


def reduction_element(head, query, feature, query_rows, dtype="FP32"):
    return dict(head=head, query=query, feature=feature,
        operands=[result_element(head,query,feature,p,query_rows,dtype) for p in (0,1)],
        operation="sum", input_dtype=dtype, output_dtype="FP32", latency_us=None)


def cache_contract(qkv_fusion=False):
    return dict(base_bytes=CACHE_BASE, end_bytes_exclusive=CACHE_BASE+CACHE_BYTES,
        used_bytes_per_bank=31*16*32, reserved_bytes_per_bank=CACHE_BYTES,
        logical_shape=[HEADS,PADDED,FEATURES], rank_shape=[LOCAL_KV,FEATURES],
        rank_owner_formula="channel=2*head+token%2; local_token=token//2; physical_rank=1",
        feature_lanes_by_bank=[[L.feature_from_lane(b,l) for l in range(16)] for b in range(16)],
        element_offset_formula="163840 + ((token//2//16)*16+feature_lane)*32 + (token//2%16)*2",
        rank_tokens=[dict(channel=c,head=c//2,physical_rank=1,logical_tokens=list(range(c%2,PADDED,2))) for c in range(16)],
        local_prefix=[0,484], local_suffix=[484,492], local_padding=[492,496],
        global_prefix=[0,968], global_suffix=[968,984], global_padding=[984,992],
        local_tail_block=[480,496], preserve_local_prefix=[480,484],
        spill=dict(operation="PIM_SPILL", rank_descriptor="PIM_RANK_SPILL",
            column_offsets=[CACHE_BASE+(30*16+l)*32 for l in range(16)],
            packets_per_bank=16, bytes_per_packet=32, read_modify_write=True,
            per_packet_preserve_bytes=[0,8], per_packet_suffix_bytes=[8,24],
            per_packet_zero_bytes=[24,32], external_DQ_bytes=0,
            accumulator_columns=48 if qkv_fusion else 16,
            source_feature_column_first=32 if qkv_fusion else 0,
            conversion="FP32 accumulator -> BF16; gather eight suffix values into KV lanes4..11",
            timing="native LOCALRD32 then LOCALWR32; cast/gather included in modeled local path"),
        numerical_compute=False)


def trace_ranges(manifest):
    ranges = {s["name"]:dict(line_start=s["line_start"],line_end=s["line_end"])
              for s in manifest["segments"]}
    ranges["input_fragments"] = [dict(e) for e in manifest.get("input_fragments",[])]
    ranges["exports"] = [dict(e) for e in manifest.get("exports",[])]
    ranges["local_writes"] = [dict(e) for e in manifest.get("local_writes",[])]
    return ranges


def shift_trace_lines(contract, shift):
    result = deepcopy(contract)
    line_keys = {"line_start","line_end","mac_line_first","mac_line_last"}
    def visit(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key in line_keys: value[key]=child+shift
                else: visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    visit(result["trace_ranges"])
    return result

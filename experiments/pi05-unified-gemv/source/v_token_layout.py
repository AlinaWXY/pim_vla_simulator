"""Token-sharded V projection and its unchanged feature-sharded cache boundary.

These helpers describe logical tensor ownership and packing. They do not run
GPU arithmetic or assign a latency to FP32-to-BF16 conversion/scatter.
"""
from copy import deepcopy
import horizon_attention_layout as attention

HEADS, TOKENS, HIDDEN, FEATURES = 8, 16, 1024, 256
CHANNELS, BANKS, LANES, LOCAL_TOKENS = 16, 16, 16, 8
PREFIX, WEIGHT_BASE, WEIGHT_BYTES = 968, 131072, 32768
CACHE_BASE, RESULT_BYTES_PER_RANK = 100352, 8192


def token_owner(head, token):
    return attention.query_owner(head, token)


def feature_owner(head, token, feature):
    attention.check(token, TOKENS, "V token")
    return attention.feature_owner(head, token % 2, feature)


def weight_location(head, channel_in_head, reduction, feature):
    """[input_fragment64][output_feature_lane16][input_lane16] BF16."""
    attention.check(reduction, HIDDEN, "V reduction")
    owner, lane = attention.feature_owner(head, channel_in_head, feature)
    fragment, input_lane = divmod(reduction, LANES)
    return (*owner, WEIGHT_BASE + (fragment * LANES + lane) * 32 + input_lane * 2)


def input_coordinates(channel, local_token, fragment, input_lane):
    attention.check(channel, CHANNELS, "V input channel")
    attention.check(local_token, LOCAL_TOKENS, "V local token")
    attention.check(fragment, HIDDEN // LANES, "V input fragment")
    attention.check(input_lane, LANES, "V input lane")
    return dict(tensor="X", token=2 * local_token + channel % 2,
                feature=fragment * LANES + input_lane)


def export_element(head, token, feature):
    """One FP32 result in the declared rank result stream, not a DRAM array row."""
    owner, lane = feature_owner(head, token, feature)
    channel, rank, bg, physical_bank = owner
    local_token, flat_bank = token // 2, 4 * bg + physical_bank
    offset = ((local_token * BANKS + flat_bank) * LANES + lane) * 4
    packet = offset // 32 * 32
    return dict(head=head, token=token, feature=feature, channel=channel, physical_rank=rank,
                bank_group=bg, bank=physical_bank, output_lane=lane, local_token=local_token,
                accumulator_index=local_token * LANES + lane, result_offset_bytes=offset,
                packet_offset_bytes=packet, byte_in_packet=offset % 32,
                trace_address=attention.physical_address(channel, rank, 0, 0, packet), dtype="FP32")


def cache_scatter_element(horizon, head, token, feature):
    if horizon != TOKENS:
        raise ValueError("Token-sharded V supports H=16 only")
    # Lazy import avoids a module cycle and deliberately uses the old cache
    # mapper rather than deriving a new cache from the V projection ownership.
    from horizon_value import value_location, physical_address
    destination = value_location(horizon, head, PREFIX + token, feature)
    return dict(source=export_element(head, token, feature),
                destination=dict(head=head, token=PREFIX + token, feature=feature,
                    channel=destination[0], physical_rank=destination[1], bank_group=destination[2],
                    bank=destination[3], offset_bytes=destination[4],
                    physical_address=physical_address(*destination), dtype="BF16"))


def iter_cache_scatter(horizon=16):
    for head in range(HEADS):
        for token in range(TOKENS):
            for feature in range(FEATURES):
                yield cache_scatter_element(horizon, head, token, feature)


def export_packet(channel, offset):
    """Enumerate the eight FP32 elements carried by one rank RDRES packet."""
    attention.check(channel, CHANNELS, "V output channel")
    if type(offset) is not int or offset % 32 or not 0 <= offset < RESULT_BYTES_PER_RANK:
        raise ValueError("V result packet offset must be aligned within the rank result stream")
    local_token, inside = divmod(offset, BANKS * LANES * 4)
    flat_bank, byte_in_bank = divmod(inside, LANES * 4)
    first_lane = byte_in_bank // 4
    token, head = 2 * local_token + channel % 2, channel // 2
    return [cache_scatter_element(TOKENS, head, token, attention.feature_from_lane(flat_bank, lane))
            for lane in range(first_lane, first_lane + 8)]


def input_contract():
    return dict(tensor="X", shape=[TOKENS, HIDDEN], local_rows=LOCAL_TOKENS,
                fragment_count=64, lanes_per_fragment=16, bytes_per_fragment_per_rank=256,
                bytes_per_rank=16384, source_token_formula="2*local_token + channel%2",
                source_feature_formula="16*fragment + input_lane",
                source_element_formula="X[2*local_token+channel%2, 16*fragment+input_lane]",
                rank_tokens=[dict(channel=channel, physical_rank=1, head=channel // 2,
                    logical_tokens=[2 * local + channel % 2 for local in range(LOCAL_TOKENS)])
                    for channel in range(CHANNELS)])


def mapping_contract(manifest):
    """Compact, fully enumerable contract for all 32768 V output elements."""
    export = manifest["exports"][0]
    ranges = {segment["name"]: dict(line_start=segment["line_start"], line_end=segment["line_end"])
              for segment in manifest["segments"]}
    ranges["export_packets"] = dict(line_start=export["line_start"], line_end=export["line_end"])
    ranges["input_fragments"] = [{key: item[key] for key in
        ("fragment", "line_start", "line_end", "mac_line_first", "mac_line_last")}
        for item in manifest["input_fragments"]]
    return dict(schema_version=1, enabled=True, logical_shape=[HEADS, TOKENS, FEATURES],
        rank_owner_formula="channel=2*head+token%2; physical_rank=1; local_token=token//2",
        rank_tokens=input_contract()["rank_tokens"],
        feature_owner_formula="pair=feature%128; bank=pair//8; lane=2*(pair%8)+feature//128",
        feature_lanes_by_bank=[[attention.feature_from_lane(bank, lane) for lane in range(LANES)]
                              for bank in range(BANKS)],
        input=input_contract(),
        weights=dict(base_bytes=WEIGHT_BASE, end_bytes_exclusive=WEIGHT_BASE+WEIGHT_BYTES,
            bytes_per_bank=WEIGHT_BYTES, copies_per_head=2,
            order="[input_fragment64][output_feature_lane16][input_lane16] BF16",
            element_offset_formula="131072 + (fragment*16+output_lane)*32 + input_lane*2"),
        export=dict(dtype="FP32", order="[local_token8][bank16][feature_lane16]",
            bytes_per_rank=RESULT_BYTES_PER_RANK, packets_per_rank=RESULT_BYTES_PER_RANK // 32,
            packet_bytes=32, elements_per_packet=8, canonical_trace_bank=0,
            element_offset_formula="((local_token*16+bank)*16+output_lane)*4",
            trace_line_formula="export_packets.line_start + (packet_offset_bytes//32)*16 + channel",
            producer="bank FP32 accumulators", trace_address_is_result_tag=True),
        cache_scatter=dict(dtype="BF16", cache_base_bytes=CACHE_BASE,
            cache_order="[KV_fragment][feature_slot8][token_lane16]",
            destination_token_formula="968+logical_token",
            destination_channel_formula="2*head + (feature%32)//16",
            destination_bank_formula="feature%16",
            destination_offset_formula="100352 + (((968+logical_token)//16)*8 + feature//32)*32 + ((968+logical_token)%16)*2",
            elements=HEADS*TOKENS*FEATURES, valid_suffix=[968,984],
            preserve_prefix=[960,968], masked_padding=[984,992],
            tail_read_block_starts=[960], tail_write_block_starts=[960,976],
            boundary="GPU export -> FP32-to-BF16 cast/packing -> existing feature-cache scatter with prefix merge",
            latency_us=None, numerical_compute=False),
        trace_ranges=ranges)


def shift_trace_lines(contract, shift):
    result = deepcopy(contract)
    line_keys = {"line_start", "line_end", "mac_line_first", "mac_line_last"}
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in line_keys:
                    value[key] = child + shift
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result["trace_ranges"])
    return result

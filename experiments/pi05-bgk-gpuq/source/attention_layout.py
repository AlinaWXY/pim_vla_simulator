"""Fused feature-sharded QKV; GPU RoPE roundtrip; DRAM Q relay for QK.

Locations are (channel, rank, BG, bank-in-BG, bank-local byte offset).
The same output-feature owner produces Q, K and V. GPU repacks Q/K into
attention layouts, preserving the preceding BF16 projection rounding.
"""

import json
from pathlib import Path

HEADS, HEAD_DIM, HIDDEN, QUERIES, PREFIX = 8, 256, 1024, 20, 968
SEQUENCE = PREFIX + QUERIES
CHANNELS, RANKS, BGS, BANKS_PER_BG = 16, 2, 4, 4
TOTAL_BANKS = CHANNELS * RANKS * BGS * BANKS_PER_BG
ROW_BYTES, BURST_BYTES, ELEMENT_BYTES, LANES = 2048, 32, 2, 16
TOKEN_SLOTS = (SEQUENCE + 15) // 16
PADDED_TOKENS = TOKEN_SLOTS * 16
FEATURES_PER_K_BANK = HEAD_DIM // BANKS_PER_BG
MODES = ("rope_pair", "contiguous")


def check(value, limit, name):
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"{name} must be in [0, {limit})")


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def regions():
    result, base = {}, 0
    for name, used in (
        ("w_qkv", HIDDEN * 3 * 4 * ELEMENT_BYTES),
        ("q_cache", QUERIES * FEATURES_PER_K_BANK * ELEMENT_BYTES),
        ("k_cache", TOKEN_SLOTS * FEATURES_PER_K_BANK * ELEMENT_BYTES),
        ("v_cache", TOKEN_SLOTS * 4 * LANES * ELEMENT_BYTES),
    ):
        reserved = round_up(used, ROW_BYTES)
        result[name] = dict(base=base, used_bytes_per_bank=used,
                            reserved_bytes_per_bank=reserved,
                            row_first=base // ROW_BYTES,
                            row_last=(base + reserved - 1) // ROW_BYTES)
        base += reserved
    return result


REGIONS = regions()


def physical_address(channel, rank, bg, bank, offset):
    """Same RoBaRaCoCh encoding as the prior frozen layout."""
    for value, limit, name in ((channel, CHANNELS, "channel"), (rank, RANKS, "rank"),
                               (bg, BGS, "BG"), (bank, BANKS_PER_BG, "bank")):
        check(value, limit, name)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be nonnegative")
    row, within_row = divmod(offset, ROW_BYTES)
    column, byte = divmod(within_row, BURST_BYTES)
    return ((row << 20) | (bank << 18) | (bg << 16) | (rank << 15)
            | (column << 9) | (channel << 5) | byte)


def decode_address(address):
    if not isinstance(address, int) or address < 0:
        raise ValueError("address must be nonnegative")
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, (address >> 20) * ROW_BYTES
            + (address >> 9 & 63) * BURST_BYTES + (address & 31))


def k_feature(feature, mode="rope_pair"):
    """Return bank-in-BG and local feature; pair mode co-locates d and d+128.

    The pair convention matches split-half RoPE. Adjacent-pair RoPE must use a
    different feature permutation; it is not silently assumed equivalent.
    """
    check(feature, HEAD_DIM, "feature")
    if mode == "rope_pair":
        return (feature % 128) // 32, (feature // 128) * 32 + feature % 32
    if mode == "contiguous":
        return divmod(feature, FEATURES_PER_K_BANK)
    raise ValueError(f"mode must be one of {MODES}")


def k_feature_inverse(bank, local_feature, mode="rope_pair"):
    check(bank, BANKS_PER_BG, "bank")
    check(local_feature, FEATURES_PER_K_BANK, "local feature")
    if mode == "rope_pair":
        return 128 * (local_feature // 32) + 32 * bank + local_feature % 32
    if mode == "contiguous":
        return bank * FEATURES_PER_K_BANK + local_feature
    raise ValueError(f"mode must be one of {MODES}")


def token_group(head, token):
    check(head, HEADS, "head")
    check(token, PADDED_TOKENS, "token")
    return 2 * head + token % 2, (token // 2) % 2, (token // 4) % 4


def key_location(head, token, feature, mode="rope_pair"):
    """K is [token slot, local feature fragment, 16 BF16 lanes] per bank."""
    bank, local_feature = k_feature(feature, mode)
    offset = REGIONS["k_cache"]["base"] + ELEMENT_BYTES * (
        (token // 16) * FEATURES_PER_K_BANK + local_feature)
    return (*token_group(head, token), bank, offset)


def projection_weight_location(projection, head, reduction, feature):
    """All three matrices: [input fragment, Q/K/V, feature slot, 16 lanes]."""
    if projection not in ('q', 'k', 'v'):
        raise ValueError(projection)
    check(reduction, HIDDEN, "reduction")
    owner, slot = v_feature_owner(head, feature)
    fragment, lane = divmod(reduction, LANES)
    offset = REGIONS["w_qkv"]["base"] + BURST_BYTES * (
        fragment * 12 + ('q', 'k', 'v').index(projection) * 4 + slot) + ELEMENT_BYTES * lane
    return (*owner, offset)


def projection_owner(head, query, feature):
    """Each producer bank computes all 20 tokens for its four output features."""
    check(query, QUERIES, "query")
    owner, slot = v_feature_owner(head, feature)
    return owner, query * 4 + slot


def query_location(head, channel_in_head, query, feature, mode="rope_pair"):
    """A complete Q copy/channel, source rank0/BG0, feature shard per bank.

    Bank order [local feature fragment, query, lane] makes each input fragment
    a contiguous sequence of twenty 32B DRAM bursts. Other BG/rank source
    locations are reserved but unused; a native channel relay broadcasts to
    the eight consumers of the same bank index.
    """
    check(head, HEADS, "head")
    check(channel_in_head, 2, "channel_in_head")
    check(query, QUERIES, "query")
    bank, local_feature = k_feature(feature, mode)
    fragment, lane = divmod(local_feature, LANES)
    offset = REGIONS['q_cache']['base'] + BURST_BYTES * (fragment * QUERIES + query) + ELEMENT_BYTES * lane
    return 2 * head + channel_in_head, 0, 0, bank, offset


def k_queries_for_group(channel_in_head, rank, bg):
    for value, limit, name in ((channel_in_head, 2, "channel"), (rank, RANKS, "rank"),
                               (bg, BGS, "BG")):
        check(value, limit, name)
    return [q for q in range(QUERIES)
            if token_group(0, PREFIX + q) == (channel_in_head, rank, bg)]


def v_feature_owner(head, feature):
    check(head, HEADS, "head")
    check(feature, HEAD_DIM, "feature")
    local_slot, device_bank = divmod(feature, 64)
    channel_in_head, channel_bank = divmod(device_bank, 32)
    rank, rank_bank = divmod(channel_bank, 16)
    bg, bank = divmod(rank_bank, 4)
    return (2 * head + channel_in_head, rank, bg, bank), local_slot


def value_location(head, token, feature):
    """V preserves prior ownership and [token fragment, feature slot, lane]."""
    check(token, PADDED_TOKENS, "token")
    owner, slot = v_feature_owner(head, feature)
    fragment, lane = divmod(token, LANES)
    offset = REGIONS["v_cache"]["base"] + BURST_BYTES * (fragment * 4 + slot) + ELEMENT_BYTES * lane
    return (*owner, offset)


def v_weight_location(head, reduction, feature):
    return projection_weight_location('v', head, reduction, feature)


def q_input_features(bank, fragment, mode="rope_pair"):
    """Feature identities for one DRAM Q fragment and all 20 queries.

    The source DRAM read and shared relay are explicitly timed. The multicast
    network itself is assumed to sustain the relay's command throughput.
    """
    check(fragment, FEATURES_PER_K_BANK // LANES, "fragment")
    return [k_feature_inverse(bank, fragment * LANES + lane, mode)
            for lane in range(LANES)]


def manifest():
    return dict(
        status="structural layout for native xinyao_1025 workload",
        shape=dict(heads=HEADS, head_dim=HEAD_DIM, queries=QUERIES, prefix=PREFIX,
                   sequence=SEQUENCE, hidden=HIDDEN),
        topology=dict(channels=CHANNELS, ranks_per_channel=RANKS,
                      BGs_per_rank=BGS, banks_per_BG=BANKS_PER_BG, total_banks=TOTAL_BANKS),
        regions=REGIONS,
        attention_reserved_bytes_per_bank=sum(r['reserved_bytes_per_bank'] for r in REGIONS.values()),
        layer_reserved_bytes=TOTAL_BANKS * sum(r['reserved_bytes_per_bank'] for r in REGIONS.values()),
        projection_feature_formula="d=64*slot+32*channel_in_head+16*rank+4*BG+bank",
        projection_weight_order="[input fragment64][Q/K/V3][feature slot4][lane16]",
        projection_weight_copies=1,
        projection_weight_bytes_each=HEADS * HIDDEN * HEAD_DIM * ELEMENT_BYTES,
        qkv_weight_system_bytes=TOTAL_BANKS * REGIONS['w_qkv']['used_bytes_per_bank'],
        k_token_group_formula="channel=2h+t%2; rank=(t//2)%2; BG=(t//4)%4; slot=t//16",
        k_feature_formula="bank=(d%128)//32; local=(d//128)*32+d%32",
        k_feature_mode="rope_pair", supported_feature_modes=list(MODES),
        q_source_formula="channel=2h+c; rank=0; BG=0; bank=K-feature-bank; [local fragment4][query20][lane16]",
        q_logical_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
        q_dram_bytes=2 * HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
        q_dram_copies_per_head=2,
        q_residency="Post-RoPE BF16 Q stored once per head channel; native DRAM read then channel WRGB relay; no repeated GPU input during QK",
        q_projection="PIM, same feature owners as K and V",
        qkv_projection_input="One GPU X fragment per channel; scan Q then K then V before loading the next X fragment",
        effective_GPU_input_bytes=dict(qkv_projection=CHANNELS * QUERIES * HIDDEN * ELEMENT_BYTES,
            q_dram_write=2 * HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
            k_suffix_write=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES),
        projection_export_bytes_each=HEADS * QUERIES * HEAD_DIM * 4,
        cache_logical_bytes_each=HEADS * SEQUENCE * HEAD_DIM * ELEMENT_BYTES,
        cache_padded_bytes_each=HEADS * PADDED_TOKENS * HEAD_DIM * ELEMENT_BYTES,
        cache_reserved_bytes_each=TOTAL_BANKS * REGIONS['k_cache']['reserved_bytes_per_bank'],
        buffers=dict(projection_input_double_per_rank=1280,
            q_double_buffer_per_physical_bank=1280,
            qkv_projection_accumulator_per_bank=3 * QUERIES * 4 * 4,
            qk_partial_per_bank=QUERIES * TOKEN_SLOTS * 4,
            qk_partial_reserved_per_bank=5120,
            qk_export_double_per_BG=2560,
            q_dram_controller_relay_per_channel=32),
        qk_schedule=dict(loop_order="four local feature fragments, then 62 K token slots; 20 query batch",
            q_dram_read_passes=1, physical_token_slots_per_BG=TOKEN_SLOTS,
            physical_score_export_bytes=HEADS * QUERIES * PADDED_TOKENS * 4),
        score_ownership="Four bank partials reduce inside each BG; no cross-rank/channel numerical reduction",
        rope_pair_convention="split-half pairs (d,d+128); feature permutation retained for K and Q",
        numeric_contract="Projection output BF16 rounding precedes GPU FP32 RoPE, then BF16 Q/K storage; Q scales by1/sqrt(256). Exact floating-point parity not measured.",
        positions_contract="GPU receives model RoPE positions; physical cache token index is not assumed equal to model position",
        k_suffix_write_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
        k_suffix_partial_bursts=False,
        k_writeback="GPU-origin ordinary ST to BG-token K cache, no local producer-bank write assumption",
        v_writeback="Retain FP32 GPU-boundary export, prefix-tail read and BF16 RMW/store baseline",
        v_suffix=dict(rmw_read_bytes=HEADS * HEAD_DIM * BURST_BYTES,
            write_bytes=2 * HEADS * HEAD_DIM * BURST_BYTES,
            new_value_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
            preserved_prefix_bytes=HEADS * 8 * HEAD_DIM * ELEMENT_BYTES,
            zero_padding_bytes=HEADS * 4 * HEAD_DIM * ELEMENT_BYTES),
        unmeasured=["GPU Q/K RoPE/cast/pack arithmetic", "GPU softmax/O projection", "GPU V cast/repack", "GPU DMA/arrival overhead beyond modeled controller transfers"],
    )


if __name__ == "__main__":
    path = Path(__file__).with_name("layout.json")
    path.write_text(json.dumps(manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)

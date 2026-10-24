"""Candidate GPU-Q / BG-sharded-K / feature-sharded-V layout.

This is a structural specification, not a native simulator implementation or
latency result. Existing experiments and their frozen layouts are unchanged.
Locations are (global channel, rank, BG, bank-in-BG, bank-local byte offset).
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
        ("w_k", HIDDEN * FEATURES_PER_K_BANK * ELEMENT_BYTES),
        ("w_v", HIDDEN * 4 * ELEMENT_BYTES),
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


def k_weight_location(head, channel_in_head, rank, bg, reduction, feature,
                      mode="rope_pair"):
    """One full W_K head per BG, split over its four banks.

    Each bank orders weights as [input fragment, local output feature, lane].
    An X fragment stays buffered while all 64 local output features are scanned.
    """
    for value, limit, name in ((head, HEADS, "head"), (channel_in_head, 2, "channel"),
                               (rank, RANKS, "rank"), (bg, BGS, "BG"),
                               (reduction, HIDDEN, "reduction")):
        check(value, limit, name)
    bank, local_feature = k_feature(feature, mode)
    fragment, lane = divmod(reduction, LANES)
    offset = REGIONS["w_k"]["base"] + BURST_BYTES * (
        fragment * FEATURES_PER_K_BANK + local_feature) + ELEMENT_BYTES * lane
    return 2 * head + channel_in_head, rank, bg, bank, offset


def k_projection_location(head, query, feature, mode="rope_pair"):
    """The projection accumulator's owner is already the destination K bank."""
    check(query, QUERIES, "query")
    return key_location(head, PREFIX + query, feature, mode)


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
    check(reduction, HIDDEN, "reduction")
    owner, slot = v_feature_owner(head, feature)
    fragment, lane = divmod(reduction, LANES)
    offset = REGIONS["w_v"]["base"] + BURST_BYTES * (fragment * 4 + slot) + ELEMENT_BYTES * lane
    return (*owner, offset)


def q_input_features(bank, fragment, mode="rope_pair"):
    """GPU supplies these 16 post-RoPE Q features for all 20 queries.

    Same-bank-index consumers across BGs may share the input stream only if an
    actual within-rank broadcast path exists; no such latency is assumed zero.
    """
    check(fragment, FEATURES_PER_K_BANK // LANES, "fragment")
    return [k_feature_inverse(bank, fragment * LANES + lane, mode)
            for lane in range(LANES)]


def manifest():
    k_x_channel = QUERIES // 2 * HIDDEN * ELEMENT_BYTES
    v_x_channel = QUERIES * HIDDEN * ELEMENT_BYTES
    return dict(
        status="structural/numerical design audit only; no new native latency",
        shape=dict(heads=HEADS, head_dim=HEAD_DIM, queries=QUERIES, prefix=PREFIX,
                   sequence=SEQUENCE, hidden=HIDDEN),
        topology=dict(channels=CHANNELS, ranks_per_channel=RANKS,
                      BGs_per_rank=BGS, banks_per_BG=BANKS_PER_BG, total_banks=TOTAL_BANKS),
        k_feature_mode="rope_pair", supported_feature_modes=list(MODES),
        rope_pair_convention="split-half pairs (d,d+128), checked against Physical-Intelligence/openpi gemma._apply_rope",
        model_reference="https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py",
        numeric_contract="Preserve projection BF16 output rounding before FP32 RoPE and return to BF16; GPU Q also applies 1/sqrt(256). Exact floating-point kernel parity not tested.",
        positions_contract="GPU supplies model RoPE positions; physical cache token index is not assumed equal to RoPE position when prefix padding is present.",
        regions=REGIONS,
        layer_reserved_bytes=TOTAL_BANKS * sum(r["reserved_bytes_per_bank"] for r in REGIONS.values()),
        k_weight_copies_per_head=16,
        k_weight_system_bytes=TOTAL_BANKS * REGIONS["w_k"]["used_bytes_per_bank"],
        v_weight_system_bytes=TOTAL_BANKS * REGIONS["w_v"]["used_bytes_per_bank"],
        cache_logical_bytes_each=HEADS * SEQUENCE * HEAD_DIM * ELEMENT_BYTES,
        cache_padded_bytes_each=HEADS * PADDED_TOKENS * HEAD_DIM * ELEMENT_BYTES,
        cache_reserved_bytes_each=TOTAL_BANKS * REGIONS["k_cache"]["reserved_bytes_per_bank"],
        k_token_group_formula="channel=2h+t%2; rank=(t//2)%2; BG=(t//4)%4; slot=t//16",
        k_feature_formula="bank=(d%128)//32; local=(d//128)*32+d%32",
        v_feature_formula="d=64*slot+32*channel_in_head+16*rank+4*BG+bank",
        q_residency="GPU Q projection, RoPE and 1/16 scaling; stream to selected bank input buffers, no Q DRAM region",
        k_projection_input="X from GPU; each BG selects only its 1 or 2 assigned suffix tokens",
        v_projection_input="X from GPU in separate pass; all banks process all 20 tokens",
        effective_GPU_input_bytes=dict(
            k_projection=CHANNELS * k_x_channel,
            v_projection=CHANNELS * v_x_channel,
            q_attention=2 * HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES),
        input_bytes_assumption="pack only selected K queries/channel; Q once/channel then internal fanout; exclude headers and padding",
        buffers=dict(
            baseline="independent double buffer in each physical bank; rank-shared bank-index streams are an optional separate implementation",
            q_double_buffer_per_rank_with_shared_bank_index_streams=2 * 4 * QUERIES * LANES * ELEMENT_BYTES,
            q_double_buffer_per_physical_bank=2 * QUERIES * LANES * ELEMENT_BYTES,
            q_double_buffer_per_rank_if_duplicated_at_all_banks=16 * 2 * QUERIES * LANES * ELEMENT_BYTES,
            k_projection_double_buffer_per_BG_max=2 * 2 * LANES * ELEMENT_BYTES,
            k_projection_double_buffer_per_channel_if_all_10_queries_staged=2 * 10 * LANES * ELEMENT_BYTES,
            v_projection_double_buffer_per_rank=2 * QUERIES * LANES * ELEMENT_BYTES,
            BG_score_accumulator_bytes_per_token_tile=QUERIES * 4,
            BG_score_partial_bytes_four_banks_per_token_tile=4 * QUERIES * 4),
        qk_schedule=dict(
            loop_order="local feature fragment (4), then all 62 K token slots; all 20 queries per MAC batch",
            q_transfer_passes=1,
            physical_token_slots_per_BG=TOKEN_SLOTS,
            bank_partial_score_used_bytes=QUERIES * TOKEN_SLOTS * 4,
            bank_partial_score_reserved_bytes=QUERIES * round_up(TOKEN_SLOTS, 16) * 4,
            BG_reduction_vector_bytes=QUERIES * 4,
            BG_export_tile_tokens=16,
            BG_export_double_buffer_bytes=2 * QUERIES * 16 * 4,
            physical_score_export_bytes=HEADS * QUERIES * PADDED_TOKENS * 4,
            caveat="Bank partial state persists across all four Q fragments. Reduction/output can drain token tiles only after their final feature fragment. Overlap, ports, FIFO backpressure and local timings remain unmodeled."),
        score_ownership="complete score belongs to one BG after four-bank reduction; no cross-rank/channel numerical reduction",
        PV_ownership="V unchanged; each output feature completes token reduction in its bank",
        k_suffix_write_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
        k_suffix_partial_bursts=False,
        k_writeback="Local path required; payload timing uses independent-bank bandwidth estimate in timing-policy.json; row timing and conversion remain separate, no new native run",
        command_timing_policy="timing-policy.json: AttAcc-level multicast/reduction assumptions plus analytical K local writeback",
        v_writeback="Keep prior GPU-boundary export and RMW/store path for baseline; unchanged ownership does not imply local packing is implemented",
        v_suffix=dict(rmw_read_bytes=HEADS * HEAD_DIM * BURST_BYTES,
                      write_bytes=2 * HEADS * HEAD_DIM * BURST_BYTES,
                      new_value_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
                      preserved_prefix_bytes=HEADS * 8 * HEAD_DIM * ELEMENT_BYTES,
                      zero_padding_bytes=HEADS * 4 * HEAD_DIM * ELEMENT_BYTES),
        native_missing=["bank-specific Q input streams", "BG-specific K projection X selection",
                        "four-bank score reduction network and backpressure",
                        "bank-local RoPE/cast/packing/writeback",
                        "GPU Q projection and real GPU/PIM input arrival timeline"],
        caveat="Weights are statically duplicated; BGs process different tokens and cannot use the prior broadcast-only MACAB semantics unchanged.",
    )


if __name__ == "__main__":
    path = Path(__file__).with_name("layout.json")
    path.write_text(json.dumps(manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)

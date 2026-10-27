"""Whole-batch horizon-scaled, token-sharded PIM ranks with bank-local RoPE and rank-local QK.

Physical rank 0 is reserved for GPU traffic.  The two PIM ranks of a head
are physical rank 1 in two different channels, not the two ranks of one
channel.  Locations are (channel, physical_rank, BG, bank, byte_offset).
"""

import json
from pathlib import Path

HEADS, HEAD_DIM, HIDDEN, QUERIES, PREFIX = 8, 256, 1024, 16, 968
CHANNELS, RANKS, BGS, BANKS_PER_BG = 16, 2, 4, 4
GPU_RANK, PIM_RANK = 0, 1
PIM_RANKS_PER_HEAD = 2
BANKS_PER_RANK = BGS * BANKS_PER_BG
ACTIVE_RANKS = CHANNELS
ACTIVE_BANKS = CHANNELS * BANKS_PER_RANK
TOTAL_BANKS = CHANNELS * RANKS * BANKS_PER_RANK
ROW_BYTES, BURST_BYTES, ELEMENT_BYTES, LANES = 2048, 32, 2, 16
PAIRS_PER_BANK = HEAD_DIM // 2 // BANKS_PER_RANK
FEATURES_PER_BANK = 2 * PAIRS_PER_BANK
LOCAL_QUERIES = QUERIES // PIM_RANKS_PER_HEAD
LOCAL_PREFIX = PREFIX // PIM_RANKS_PER_HEAD
SEQUENCE = PREFIX + QUERIES
PADDED_TOKENS = ((PREFIX + QUERIES + 31) // 32) * 32
LOCAL_VALID_TOKENS = SEQUENCE // PIM_RANKS_PER_HEAD
LOCAL_TOKENS = PADDED_TOKENS // PIM_RANKS_PER_HEAD
LOCAL_PADDING = LOCAL_TOKENS - LOCAL_VALID_TOKENS
PROJECTIONS = ("q", "k")
WORK_COLUMNS = {"coef": (0, 8), "q_local": (8, 16),
                "k_suffix": (16, 24), "k_padding": (24, 28)}


def check(value, limit, name):
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"{name} must be an integer in [0,{limit})")


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def regions():
    result, base = {}, 0
    sizes = (("w_qk", HIDDEN * len(PROJECTIONS) * FEATURES_PER_BANK * ELEMENT_BYTES),
             ("k_prefix", LOCAL_PREFIX * BURST_BYTES),
             ("work", WORK_COLUMNS["k_padding"][1] * BURST_BYTES))
    for name, used in sizes:
        reserved = round_up(used, ROW_BYTES)
        result[name] = dict(base=base, used_bytes_per_bank=used,
                            reserved_bytes_per_bank=reserved,
                            row_first=base // ROW_BYTES,
                            row_last=(base + reserved - 1) // ROW_BYTES)
        base += reserved
    return result


REGIONS = regions()


def configure(horizon):
    """Resize capacity and token ownership without changing the 8 attention heads."""
    if type(horizon) is not int or horizon < 2 or horizon > 64 or horizon % 2:
        raise ValueError("horizon must be an even integer from2 through64")
    global QUERIES, LOCAL_QUERIES, SEQUENCE, PADDED_TOKENS
    global LOCAL_VALID_TOKENS, LOCAL_TOKENS, LOCAL_PADDING, WORK_COLUMNS, REGIONS
    QUERIES = horizon
    LOCAL_QUERIES = horizon // 2
    SEQUENCE = PREFIX + horizon
    PADDED_TOKENS = round_up(SEQUENCE, 32)
    LOCAL_VALID_TOKENS, LOCAL_TOKENS = SEQUENCE // 2, PADDED_TOKENS // 2
    LOCAL_PADDING = LOCAL_TOKENS - LOCAL_VALID_TOKENS
    n = LOCAL_QUERIES
    WORK_COLUMNS = {"coef": (0,n), "q_local": (n,2*n),
                    "k_suffix": (2*n,3*n), "k_padding": (3*n,3*n+LOCAL_PADDING)}
    REGIONS = regions()


def physical_address(channel, rank, bg, bank, offset):
    """RoBaRaCoCh encoding used by the LPDDR5X native backend."""
    for value, limit, name in ((channel, CHANNELS, "channel"), (rank, RANKS, "rank"),
                               (bg, BGS, "BG"), (bank, BANKS_PER_BG, "bank")):
        check(value, limit, name)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    row, within_row = divmod(offset, ROW_BYTES)
    column, byte = divmod(within_row, BURST_BYTES)
    return ((row << 20) | (bank << 18) | (bg << 16) | (rank << 15)
            | (column << 9) | (channel << 5) | byte)


def decode_address(address):
    if not isinstance(address, int) or address < 0:
        raise ValueError("address must be a nonnegative integer")
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, (address >> 20) * ROW_BYTES
            + (address >> 9 & 63) * BURST_BYTES + (address & 31))


def rank_owner(head, channel_in_head):
    check(head, HEADS, "head")
    check(channel_in_head, PIM_RANKS_PER_HEAD, "channel_in_head")
    return 2 * head + channel_in_head, PIM_RANK


def bank_coordinates(bank_in_rank):
    check(bank_in_rank, BANKS_PER_RANK, "bank_in_rank")
    return divmod(bank_in_rank, BANKS_PER_BG)


def feature_owner(head, channel_in_head, feature):
    """Return the physical owner and pair-major output-feature lane."""
    check(feature, HEAD_DIM, "feature")
    component, pair = divmod(feature, HEAD_DIM // 2)
    bank_in_rank, pair_in_bank = divmod(pair, PAIRS_PER_BANK)
    bg, bank = bank_coordinates(bank_in_rank)
    return (*rank_owner(head, channel_in_head), bg, bank), 2 * pair_in_bank + component


def feature_from_lane(bank_in_rank, lane):
    check(bank_in_rank, BANKS_PER_RANK, "bank_in_rank")
    check(lane, FEATURES_PER_BANK, "lane")
    pair, component = divmod(lane, 2)
    return PAIRS_PER_BANK * bank_in_rank + pair + (HEAD_DIM // 2) * component


def token_owner(head, token):
    check(token, PADDED_TOKENS, "token")
    return rank_owner(head, token % PIM_RANKS_PER_HEAD), token // PIM_RANKS_PER_HEAD


def query_owner(head, query):
    check(query, QUERIES, "query")
    return rank_owner(head, query % PIM_RANKS_PER_HEAD), query // PIM_RANKS_PER_HEAD


def projection_owner(head, query, feature):
    check(query, QUERIES, "query")
    owner, lane = feature_owner(head, query % 2, feature)
    return owner, (query // 2) * FEATURES_PER_BANK + lane


def projection_weight_location(projection, head, channel_in_head, reduction, feature):
    """[input_fragment64][Q/K2][pair8][component2][input_lane16]."""
    if projection not in PROJECTIONS:
        raise ValueError(projection)
    check(reduction, HIDDEN, "reduction")
    owner, output_lane = feature_owner(head, channel_in_head, feature)
    fragment, input_lane = divmod(reduction, LANES)
    column = ((fragment * len(PROJECTIONS) + PROJECTIONS.index(projection))
              * FEATURES_PER_BANK + output_lane)
    return (*owner, REGIONS["w_qk"]["base"] + BURST_BYTES * column
            + ELEMENT_BYTES * input_lane)


def work_offset(kind, local_token):
    if kind not in WORK_COLUMNS:
        raise ValueError(kind)
    first, stop = WORK_COLUMNS[kind]
    check(local_token, stop - first, "local_token")
    return REGIONS["work"]["base"] + (first + local_token) * BURST_BYTES


def query_location(head, query, feature):
    """Only the producing rank stores Q in DRAM; remote Q goes to SRAM."""
    owner, lane = feature_owner(head, query % 2, feature)
    check(query, QUERIES, "query")
    return (*owner, work_offset("q_local", query // 2) + lane * ELEMENT_BYTES)


def key_location(head, token, feature):
    check(token, PADDED_TOKENS, "token")
    owner, lane = feature_owner(head, token % 2, feature)
    local_token = token // 2
    if local_token < LOCAL_PREFIX:
        offset = REGIONS["k_prefix"]["base"] + local_token * BURST_BYTES
    elif local_token < LOCAL_VALID_TOKENS:
        offset = work_offset("k_suffix", local_token - LOCAL_PREFIX)
    else:
        offset = work_offset("k_padding", local_token - LOCAL_VALID_TOKENS)
    return (*owner, offset + lane * ELEMENT_BYTES)


def coefficient_location(head, query, pair, kind):
    """BF16 sin/cos are precomputed from actual model position IDs."""
    check(query, QUERIES, "query")
    check(pair, HEAD_DIM // 2, "pair")
    if kind not in ("cos", "sin"):
        raise ValueError(kind)
    owner, lane = feature_owner(head, query % 2, pair)
    coefficient_lane = lane + (kind == "sin")
    return (*owner, work_offset("coef", query // 2) + coefficient_lane * ELEMENT_BYTES)


def projection_spill_location(projection, head, query, feature):
    if projection == "q":
        return query_location(head, query, feature)
    if projection == "k":
        return key_location(head, PREFIX + query, feature)
    raise ValueError(projection)


def query_buffer_slot(query, feature):
    """Byte index inside the bank's horizon*32 B SRAM, not a DRAM allocation."""
    check(query, QUERIES, "query")
    check(feature, HEAD_DIM, "feature")
    lane = 2 * (feature % (HEAD_DIM // 2) % PAIRS_PER_BANK) + feature // (HEAD_DIM // 2)
    return query * BURST_BYTES + lane * ELEMENT_BYTES


def q_forward_destination(head, query, feature):
    owner, _ = feature_owner(head, 1 - query % 2, feature)
    return owner, query_buffer_slot(query, feature)


def qk_column_offset(local_token):
    check(local_token, LOCAL_TOKENS, "local_token")
    if local_token < LOCAL_PREFIX:
        return REGIONS["k_prefix"]["base"] + local_token * BURST_BYTES
    if local_token < LOCAL_VALID_TOKENS:
        return work_offset("k_suffix", local_token - LOCAL_PREFIX)
    return work_offset("k_padding", local_token - LOCAL_VALID_TOKENS)


def active_bank_locations(offset=0):
    for channel in range(CHANNELS):
        for bg in range(BGS):
            for bank in range(BANKS_PER_BG):
                yield channel, PIM_RANK, bg, bank, offset


def traffic_contract():
    unique_q = HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES
    x_input = ACTIVE_RANKS * LOCAL_QUERIES * HIDDEN * ELEMENT_BYTES
    scores = HEADS * QUERIES * PADDED_TOKENS * 4
    return dict(x_gpu_to_pim=x_input, q_pim_to_gpu=unique_q,
                q_gpu_to_remote_buffer=unique_q, k_rank_exchange=0,
                rope_rank_exchange=0, score_pim_to_gpu=scores,
                steady_external_total=x_input + 2 * unique_q + scores,
                optional_unique_x_source_read=QUERIES * HIDDEN * ELEMENT_BYTES,
                score_logical_bytes=HEADS * QUERIES * SEQUENCE * 4,
                q_logical_interrank_payload=unique_q,
                qk_bank_internal_k_read=HEADS * PADDED_TOKENS * HEAD_DIM * ELEMENT_BYTES)


def manifest():
    return dict(
        variant=f"H={QUERIES} bank-local RoPE", action_horizon=QUERIES,
        shape=dict(heads=HEADS, head_dim=HEAD_DIM, queries=QUERIES,
                   hidden=HIDDEN, prefix=PREFIX, sequence=SEQUENCE,
                   padded_sequence=PADDED_TOKENS),
        topology=dict(channels=CHANNELS, physical_ranks_per_channel=RANKS,
                      gpu_rank=GPU_RANK, pim_rank=PIM_RANK,
                      active_pim_ranks=ACTIVE_RANKS, active_pim_banks=ACTIVE_BANKS,
                      BGs_per_rank=BGS, banks_per_BG=BANKS_PER_BG),
        rank_token_counts=dict(prefix=LOCAL_PREFIX, suffix=LOCAL_QUERIES,
                               valid=LOCAL_VALID_TOKENS, padded=LOCAL_TOKENS),
        feature_formula="pair=8*bank_in_rank+pair_in_bank; feature=pair+128*component",
        token_owner_formula="channel=2*head+token%2; physical_rank=1; local_token=token//2",
        projection_weight_order="[input_fragment64][Q/K2][pair8][component2][input_lane16]",
        projection_weight_copies=2,
        projection_weight_bytes_each=2 * HEADS * HIDDEN * HEAD_DIM * ELEMENT_BYTES,
        qk_weight_system_bytes=ACTIVE_BANKS * REGIONS["w_qk"]["used_bytes_per_bank"],
        regions=REGIONS, work_columns=WORK_COLUMNS,
        end=max(v["base"]+v["reserved_bytes_per_bank"] for v in REGIONS.values()),
        reserved_bytes_per_bank=sum(v["reserved_bytes_per_bank"] for v in REGIONS.values()),
        initialization_bytes=dict(weights_qk=ACTIVE_BANKS * REGIONS["w_qk"]["used_bytes_per_bank"],
                                  coefficients=ACTIVE_BANKS * LOCAL_QUERIES * BURST_BYTES,
                                  prefix_k=ACTIVE_BANKS * LOCAL_PREFIX * BURST_BYTES,
                                  padding_k=ACTIVE_BANKS * LOCAL_PADDING * BURST_BYTES),
        buffers=dict(projection_input_double_per_rank=2 * LOCAL_QUERIES * BURST_BYTES,
                     projection_accumulator_per_bank=2 * LOCAL_QUERIES * FEATURES_PER_BANK * 4,
                     rope_input_double_per_bank=2 * BURST_BYTES,
                     rope_coefficient_per_bank=BURST_BYTES,
                     rope_output_per_bank=2 * BURST_BYTES,
                     rope_output_entries_per_bank=2,
                     qk_query_per_bank=QUERIES * BURST_BYTES,
                     score_tile_per_rank=16 * QUERIES * 4),
        coefficient_dtype="BF16 (generated offline from FP32 sin/cos)",
        coefficient_positions="Actual model position IDs; cache token indices are not assumed equal to positions",
        coefficient_lifetime="Reusable work region per active bank; table reused across layers/steps while positions agree",
        projection_numeric_contract="FP32 accumulation followed by BF16 rounding before local spill and RoPE",
        numeric_claim="Timing/traffic model; no claim of bitwise PyTorch or JAX equivalence",
        q_residency=f"Local{LOCAL_QUERIES} Q columns in DRAM; all{QUERIES} queries in{QUERIES*32}B bank SRAM after missing-half forward",
        k_residency=f"{LOCAL_PREFIX} prefix columns then{LOCAL_QUERIES} suffix and{LOCAL_PADDING} zero-padding columns in work region",
        qk_loop_order=f"Fill all{QUERIES} Q columns once, then scan{LOCAL_TOKENS} local K columns once",
        score_ownership="Bank->BG->rank reduction completes256feature dot; ranks produce different K tokens, not partial scores",
        traffic_bytes=traffic_contract(),
        exclusions=["GPU softmax and downstream PV", "GPU arrival/dispatch overhead", "weight/prefix/coefficient initialization"],
    )


if __name__ == "__main__":
    output = Path(__file__).with_name("layout.json")
    output.write_text(json.dumps(manifest(), indent=2) + "\n", encoding="utf-8")
    print(output)

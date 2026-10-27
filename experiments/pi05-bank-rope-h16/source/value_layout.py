"""V/PV feature ownership on the256 active PIM banks of the H=16 design.

Unlike token-sharded Q/K, V is split by output feature across both channels
of a head. Each bank owns eight features and every token of those features.
The GPU-owned physical rank0 is never part of this mapping.
"""
import attention_layout as A


HEADS, HEAD_DIM, HIDDEN, QUERIES, PREFIX = 8, 256, 1024, 16, 968
CHANNELS, RANKS, BGS, BANKS_PER_BG, PIM_RANK = 16, 2, 4, 4, 1
ROW_BYTES, BURST_BYTES, ELEMENT_BYTES, LANES = 2048, 32, 2, 16
SEQUENCE, PADDED_TOKENS, TOKEN_FRAGMENTS = 984, 992, 62
FEATURES_PER_BANK, BANKS_PER_HEAD, ACTIVE_BANKS = 8, 32, 256
FFN_BASE = 256 * 1024
physical_address, decode_address = A.physical_address, A.decode_address


def check(value, limit, name):
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f'{name} must be in [0,{limit})')


def align(value, size=ROW_BYTES):
    return (value + size - 1) // size * size


ATTENTION_END = max(r['base'] + r['reserved_bytes_per_bank'] for r in A.REGIONS.values())


def regions():
    base, result = align(ATTENTION_END), {}
    for name, size in (('w_v', HIDDEN * FEATURES_PER_BANK * ELEMENT_BYTES),
                       ('v_cache', TOKEN_FRAGMENTS * FEATURES_PER_BANK * BURST_BYTES)):
        reserved = align(size)
        result[name] = dict(base=base, used_bytes_per_bank=size,
                            reserved_bytes_per_bank=reserved,
                            row_first=base // ROW_BYTES,
                            row_last=(base + reserved - 1) // ROW_BYTES)
        base += reserved
    if base > FFN_BASE:
        raise ValueError('V weights/cache overlap the reserved FFN base')
    return result


REGIONS = regions()


def feature_owner(head, feature):
    """d=32*slot+16*channel_in_head+4*BG+bank, slot0..7."""
    check(head, HEADS, 'head')
    check(feature, HEAD_DIM, 'feature')
    slot, bank_in_head = divmod(feature, BANKS_PER_HEAD)
    channel_in_head, bank_in_rank = divmod(bank_in_head, BGS * BANKS_PER_BG)
    bg, bank = divmod(bank_in_rank, BANKS_PER_BG)
    return (2 * head + channel_in_head, PIM_RANK, bg, bank), slot


def feature_from_slot(channel_in_head, bg, bank, slot):
    for value, limit, name in ((channel_in_head, 2, 'channel_in_head'),
                               (bg, BGS, 'BG'), (bank, BANKS_PER_BG, 'bank'),
                               (slot, FEATURES_PER_BANK, 'slot')):
        check(value, limit, name)
    return 32 * slot + 16 * channel_in_head + 4 * bg + bank


def projection_owner(head, query, feature):
    check(query, QUERIES, 'query')
    owner, slot = feature_owner(head, feature)
    return owner, query * FEATURES_PER_BANK + slot


def weight_location(head, reduction, feature):
    """BF16 V weights: [64 input fragments][8 output slots][16 input lanes]."""
    check(reduction, HIDDEN, 'reduction')
    owner, slot = feature_owner(head, feature)
    fragment, lane = divmod(reduction, LANES)
    return (*owner, REGIONS['w_v']['base'] +
            BURST_BYTES * (fragment * FEATURES_PER_BANK + slot) + lane * ELEMENT_BYTES)


def value_location(head, token, feature):
    """BF16 V cache: [62 token fragments][8 output slots][16 token lanes]."""
    check(token, PADDED_TOKENS, 'token')
    owner, slot = feature_owner(head, feature)
    fragment, lane = divmod(token, LANES)
    return (*owner, REGIONS['v_cache']['base'] +
            BURST_BYTES * (fragment * FEATURES_PER_BANK + slot) + lane * ELEMENT_BYTES)


def manifest():
    return dict(action_horizon=QUERIES, attention_heads=HEADS,
                shape=dict(hidden=HIDDEN, head_dim=HEAD_DIM, queries=QUERIES,
                           prefix=PREFIX, sequence=SEQUENCE, padded_sequence=PADDED_TOKENS),
                topology=dict(channels=CHANNELS, physical_ranks_per_channel=RANKS,
                              active_rank=PIM_RANK, active_banks=ACTIVE_BANKS,
                              banks_per_head=BANKS_PER_HEAD, features_per_bank=FEATURES_PER_BANK),
                feature_formula='d=32*slot+16*channel_in_head+4*BG+bank',
                regions=REGIONS, preceding_attention_end=ATTENTION_END,
                end_before_ffn=max(r['base'] + r['reserved_bytes_per_bank'] for r in REGIONS.values()),
                ffn_reserved_base=FFN_BASE,
                weight_copies=1, weight_bytes=HEADS * HIDDEN * HEAD_DIM * ELEMENT_BYTES,
                cache_payload_bytes=HEADS * PADDED_TOKENS * HEAD_DIM * ELEMENT_BYTES,
                new_v_payload_bytes=HEADS * QUERIES * HEAD_DIM * ELEMENT_BYTES,
                tail_blocks=[960, 976], preserved_prefix_tokens=8, padding_tokens=8,
                cache_order='[token_fragment62][feature_slot8][token_lane16]',
                pv_input='Each head sends the full16x992 P tensor to both channels; no cross-rank broadcast',
                output_ownership='Each bank produces complete values for its8 output features; no reduction across banks/ranks')

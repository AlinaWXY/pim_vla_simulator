"""Rank-local FFN placement after the frozen attention allocation.

Down full and tiled are alternative layouts of the same weight region.
Mappings were exhaustively validated in pi05-rank-local-ffn-design; these
integrated addresses add the existing attention stride to the scratch bases.
"""

DEVICES, CHANNELS, RANKS, BANKS_PER_RANK = 8, 16, 2, 16
TOTAL_RANKS, TOTAL_BANKS = 32, 512
HORIZON, HIDDEN, INTERMEDIATE, LOCAL_INTERMEDIATE = 16, 1024, 4096, 128
BURST_BYTES, ROW_BYTES, BF16_BYTES, LANES = 32, 2048, 2, 16
ATTENTION_BYTES_PER_BANK = 43008
GU_BASE, GU_BYTES_PER_BANK = ATTENTION_BYTES_PER_BANK, 32768
DOWN_BASE, DOWN_BYTES_PER_BANK = GU_BASE + GU_BYTES_PER_BANK, 16384
LAYER_STRIDE_BYTES_PER_BANK = DOWN_BASE + DOWN_BYTES_PER_BANK


def _bounds(value, limit, name):
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f'{name} must be an integer in [0,{limit})')


def physical_address(channel, rank, bg, bank, offset):
    for value, limit, name in ((channel, CHANNELS, 'channel'), (rank, RANKS, 'rank'),
                               (bg, 4, 'BG'), (bank, 4, 'bank')):
        _bounds(value, limit, name)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('offset must be nonnegative integer')
    row, within = divmod(offset, ROW_BYTES)
    burst, byte = divmod(within, BURST_BYTES)
    return ((row << 20) | (bank << 18) | (bg << 16) | (rank << 15)
            | (burst << 9) | (channel << 5) | byte)


def decode_address(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3,
            (address >> 20) * ROW_BYTES + (address >> 9 & 63) * BURST_BYTES + (address & 31))


def rank_bank_coords(global_rank, bank):
    _bounds(global_rank, TOTAL_RANKS, 'global rank')
    _bounds(bank, BANKS_PER_RANK, 'rank bank')
    channel, rank = divmod(global_rank, RANKS)
    bg, local_bank = divmod(bank, 4)
    return channel, rank, bg, local_bank


def gate_up_location(projection, reduction, output):
    if projection in ('gate', 'up'):
        projection = ('gate', 'up').index(projection)
    _bounds(projection, 2, 'projection')
    _bounds(reduction, HIDDEN, 'reduction')
    _bounds(output, INTERMEDIATE, 'output')
    global_rank, local = divmod(output, LOCAL_INTERMEDIATE)
    slot, bank = divmod(local, BANKS_PER_RANK)
    fragment, lane = divmod(reduction, LANES)
    return (*rank_bank_coords(global_rank, bank),
            GU_BASE + BURST_BYTES * (16 * fragment + 2 * slot + projection) + 2 * lane)


def down_location(reduction, output, tiled=False):
    _bounds(reduction, INTERMEDIATE, 'reduction')
    _bounds(output, HIDDEN, 'output')
    global_rank, local = divmod(reduction, LOCAL_INTERMEDIATE)
    fragment, lane = divmod(local, LANES)
    if tiled:
        tile, local_output = divmod(output, 256)
        slot, bank = divmod(local_output, BANKS_PER_RANK)
        burst = (tile * 8 + fragment) * 16 + slot
    else:
        slot, bank = divmod(output, BANKS_PER_RANK)
        burst = fragment * 64 + slot
    return (*rank_bank_coords(global_rank, bank), DOWN_BASE + BURST_BYTES * burst + 2 * lane)


def weight_location(operator, reduction, output):
    if operator == 'gate_up':
        _bounds(output, 2 * INTERMEDIATE, 'fused output')
        projection, feature = divmod(output, INTERMEDIATE)
        return gate_up_location(projection, reduction, feature)
    if operator in ('gate', 'up'):
        return gate_up_location(operator, reduction, output)
    if operator in ('down_full', 'down_tiled'):
        return down_location(reduction, output, operator == 'down_tiled')
    raise ValueError(operator)


def build_layout():
    common = dict(kind='weight', M=HORIZON, active_banks=TOTAL_BANKS,
                  physical_weight_copies=1, input_fragment_bytes_per_rank=HORIZON*BURST_BYTES,
                  input_double_buffer_bytes_per_rank=2*HORIZON*BURST_BYTES)
    gu = dict(**common, K=HIDDEN, N=2*INTERMEDIATE, padded_K=HIDDEN,
              padded_N=2*INTERMEDIATE, slots_per_bank=16, base=GU_BASE,
              bytes_per_bank=GU_BYTES_PER_BANK, weight_bytes=16*1024**2,
              accumulator_bytes_per_bank=HORIZON*16*4, full_reduction_per_bank=True,
              mapping='rank-contiguous 128 features; paired Gate/Up slots',
              offset_formula='GU_BASE+32*(16*(k//16)+2*((j%128)//16)+p)+2*(k%16)')
    regions = {'gate_up': gu}
    for tiled in (False, True):
        name = 'down_tiled' if tiled else 'down_full'
        regions[name] = dict(**common, K=LOCAL_INTERMEDIATE, N=HIDDEN,
            global_K=INTERMEDIATE, reduction_groups=TOTAL_RANKS,
            padded_K=LOCAL_INTERMEDIATE, padded_N=HIDDEN, base=DOWN_BASE,
            bytes_per_bank=DOWN_BYTES_PER_BANK, weight_bytes=8*1024**2,
            stored_output_slots_per_bank=64, slots_per_bank=16 if tiled else 64,
            output_tiles=4 if tiled else 1,
            outputs_per_tile_per_rank=256 if tiled else HIDDEN,
            accumulator_bytes_per_bank=HORIZON*(16 if tiled else 64)*4,
            rank_local_z_sram_bytes=HORIZON*LOCAL_INTERMEDIATE*BF16_BYTES,
            partial_result_fp32_bytes=TOTAL_RANKS*HORIZON*HIDDEN*4,
            mapping='rank-local 128 reduction terms; each rank produces all 1024 outputs',
            offset_formula=('DOWN_BASE+32*((tile*8+f)*16+slot)+2*lane' if tiled else
                            'DOWN_BASE+32*(f*64+slot)+2*lane'))
    for region in regions.values():
        region['row_first'] = region['base']//ROW_BYTES
        region['row_last'] = (region['base']+region['bytes_per_bank']-1)//ROW_BYTES
    return dict(devices=DEVICES, channels=CHANNELS, ranks=RANKS, total_ranks=TOTAL_RANKS,
                banks_per_rank=BANKS_PER_RANK, total_banks=TOTAL_BANKS,
                horizon=HORIZON, hidden=HIDDEN, intermediate=INTERMEDIATE,
                burst_bytes=BURST_BYTES, row_bytes=ROW_BYTES, regions=regions,
                attention_reserved_bytes_per_bank=ATTENTION_BYTES_PER_BANK,
                layer_stride_bytes_per_bank=LAYER_STRIDE_BYTES_PER_BANK,
                ffn_weight_bytes=24*1024**2,
                alternatives='down_full and down_tiled share a reserved region but are separate controls',
                gpu_partial_count=TOTAL_RANKS,
                unified_memory='GPU global memory uses the same LPDDR; materialization is additional traffic')


LAYOUT = build_layout()
REGIONS = LAYOUT['regions']

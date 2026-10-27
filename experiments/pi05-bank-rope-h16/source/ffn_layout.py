"""Single-PIM-rank FFN mapping; physical rank0 remains reserved for GPU.

Gate/Up output features and Down reduction features are contiguous per
active rank.  Full and tiled Down are alternative views of the same region.
"""

CHANNELS, RANKS, PIM_RANK, BANKS_PER_RANK = 16, 2, 1, 16
TOTAL_RANKS, TOTAL_BANKS = CHANNELS, CHANNELS * BANKS_PER_RANK
HORIZON, HIDDEN, INTERMEDIATE = 16, 1024, 4096
LOCAL_INTERMEDIATE = INTERMEDIATE // TOTAL_RANKS
BURST_BYTES, ROW_BYTES, BF16_BYTES, LANES = 32, 2048, 2, 16
ATTENTION_BYTES_PER_BANK = 256 * 1024
GU_BYTES_PER_BANK = 2 * HIDDEN * (INTERMEDIATE // TOTAL_BANKS) * BF16_BYTES
DOWN_BYTES_PER_BANK = LOCAL_INTERMEDIATE * (HIDDEN // BANKS_PER_RANK) * BF16_BYTES
GU_BASE = ATTENTION_BYTES_PER_BANK
DOWN_BASE = GU_BASE + GU_BYTES_PER_BANK
LAYER_STRIDE_BYTES_PER_BANK = DOWN_BASE + DOWN_BYTES_PER_BANK


def _bounds(value, limit, name):
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"{name} must be an integer in [0,{limit})")


def _bases(attention_bytes_per_bank):
    if (not isinstance(attention_bytes_per_bank, int)
            or attention_bytes_per_bank < 0 or attention_bytes_per_bank % ROW_BYTES):
        raise ValueError("attention reservation must be a nonnegative whole number of DRAM rows")
    return attention_bytes_per_bank, attention_bytes_per_bank + GU_BYTES_PER_BANK


def physical_address(channel, rank, bg, bank, offset):
    for value, limit, name in ((channel, CHANNELS, "channel"), (rank, RANKS, "rank"),
                               (bg, 4, "BG"), (bank, 4, "bank")):
        _bounds(value, limit, name)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    row, within = divmod(offset, ROW_BYTES)
    column, byte = divmod(within, BURST_BYTES)
    return ((row << 20) | (bank << 18) | (bg << 16) | (rank << 15)
            | (column << 9) | (channel << 5) | byte)


def decode_address(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3,
            (address >> 20) * ROW_BYTES + (address >> 9 & 63) * BURST_BYTES + (address & 31))


def rank_bank_coords(global_rank, bank):
    _bounds(global_rank, TOTAL_RANKS, "active rank")
    _bounds(bank, BANKS_PER_RANK, "rank bank")
    bg, local_bank = divmod(bank, 4)
    return global_rank, PIM_RANK, bg, local_bank


def gate_up_location(projection, reduction, output,
                     attention_bytes_per_bank=ATTENTION_BYTES_PER_BANK):
    if projection in ("gate", "up"):
        projection = ("gate", "up").index(projection)
    _bounds(projection, 2, "projection")
    _bounds(reduction, HIDDEN, "reduction")
    _bounds(output, INTERMEDIATE, "output")
    gu_base, _ = _bases(attention_bytes_per_bank)
    rank, local_output = divmod(output, LOCAL_INTERMEDIATE)
    slot, bank = divmod(local_output, BANKS_PER_RANK)
    fragment, lane = divmod(reduction, LANES)
    return (*rank_bank_coords(rank, bank),
            gu_base + BURST_BYTES * (32*fragment + 2*slot + projection) + BF16_BYTES*lane)


def down_location(reduction, output, tiled=False,
                  attention_bytes_per_bank=ATTENTION_BYTES_PER_BANK):
    _bounds(reduction, INTERMEDIATE, "reduction")
    _bounds(output, HIDDEN, "output")
    _, down_base = _bases(attention_bytes_per_bank)
    rank, local_reduction = divmod(reduction, LOCAL_INTERMEDIATE)
    fragment, lane = divmod(local_reduction, LANES)
    if tiled:
        tile, local_output = divmod(output, 256)
        slot, bank = divmod(local_output, BANKS_PER_RANK)
        burst = (tile*16 + fragment)*16 + slot
    else:
        slot, bank = divmod(output, BANKS_PER_RANK)
        burst = fragment*64 + slot
    return (*rank_bank_coords(rank, bank), down_base + BURST_BYTES*burst + BF16_BYTES*lane)


def weight_location(operator, reduction, output,
                    attention_bytes_per_bank=ATTENTION_BYTES_PER_BANK):
    if operator == "gate_up":
        _bounds(output, 2*INTERMEDIATE, "fused output")
        projection, feature = divmod(output, INTERMEDIATE)
        return gate_up_location(projection, reduction, feature, attention_bytes_per_bank)
    if operator in ("gate", "up"):
        return gate_up_location(operator, reduction, output, attention_bytes_per_bank)
    if operator in ("down_full", "down_tiled"):
        return down_location(reduction, output, operator == "down_tiled", attention_bytes_per_bank)
    raise ValueError(operator)


def build_layout(attention_bytes_per_bank=ATTENTION_BYTES_PER_BANK):
    gu_base, down_base = _bases(attention_bytes_per_bank)
    common = dict(kind="weight", M=HORIZON, active_banks=TOTAL_BANKS,
                  physical_weight_copies=1,
                  input_fragment_bytes_per_rank=HORIZON*BURST_BYTES,
                  input_double_buffer_bytes_per_rank=2*HORIZON*BURST_BYTES)
    regions = {"gate_up": dict(**common, K=HIDDEN, N=2*INTERMEDIATE,
        padded_K=HIDDEN, padded_N=2*INTERMEDIATE, slots_per_bank=32,
        base=gu_base, bytes_per_bank=GU_BYTES_PER_BANK, weight_bytes=16*1024**2,
        accumulator_bytes_per_bank=HORIZON*32*4, full_reduction_per_bank=True,
        mapping="rank-contiguous256 intermediate features; paired Gate/Up output slots",
        offset_formula="GU_BASE+32*(32*(k//16)+2*((j%256)//16)+p)+2*(k%16)")}
    for tiled in (False, True):
        name = "down_tiled" if tiled else "down_full"
        slots = 16 if tiled else 64
        regions[name] = dict(**common, K=LOCAL_INTERMEDIATE, N=HIDDEN,
            global_K=INTERMEDIATE, reduction_groups=TOTAL_RANKS,
            padded_K=LOCAL_INTERMEDIATE, padded_N=HIDDEN,
            base=down_base, bytes_per_bank=DOWN_BYTES_PER_BANK, weight_bytes=8*1024**2,
            stored_output_slots_per_bank=64, slots_per_bank=slots,
            output_tiles=4 if tiled else 1,
            outputs_per_tile_per_rank=slots*BANKS_PER_RANK,
            accumulator_bytes_per_bank=HORIZON*slots*4,
            rank_local_z_sram_bytes=HORIZON*LOCAL_INTERMEDIATE*BF16_BYTES,
            partial_result_fp32_bytes=TOTAL_RANKS*HORIZON*HIDDEN*4,
            mapping="256 reduction terms/rank;16 rank partials for each of1024 output features",
            offset_formula=("DOWN_BASE+32*((tile*16+f)*16+slot)+2*lane" if tiled
                            else "DOWN_BASE+32*(f*64+slot)+2*lane"))
    for region in regions.values():
        region["row_first"] = region["base"] // ROW_BYTES
        region["row_last"] = (region["base"]+region["bytes_per_bank"]-1) // ROW_BYTES
    return dict(channels=CHANNELS, physical_ranks_per_channel=RANKS,
                active_physical_rank=PIM_RANK, total_active_ranks=TOTAL_RANKS,
                banks_per_rank=BANKS_PER_RANK, total_active_banks=TOTAL_BANKS,
                horizon=HORIZON, hidden=HIDDEN, intermediate=INTERMEDIATE,
                local_intermediate=LOCAL_INTERMEDIATE,
                burst_bytes=BURST_BYTES, row_bytes=ROW_BYTES, regions=regions,
                attention_reserved_bytes_per_bank=attention_bytes_per_bank,
                layer_stride_bytes_per_bank=down_base+DOWN_BYTES_PER_BANK,
                ffn_weight_bytes=24*1024**2, gpu_partial_count=TOTAL_RANKS,
                gate_up_accumulator_note="2KiB/bank, explicitly increased from old32-rank1KiB arrangement",
                alternatives="down_full and down_tiled share a region; do not sum both controls",
                unified_memory="Additional GPU partial materialization/readback would consume shared LPDDR")


LAYOUT = build_layout()
REGIONS = LAYOUT["regions"]

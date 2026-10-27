"""Existing FFN weight ownership with bank-resident DRAM Z columns.

Each column contains one intermediate feature across all16 tokens. Sixteen
bank columns are transposed into the bounded rank input fragment for Down.
"""

import ffn_layout as F

CHANNELS, RANKS, PIM_RANK = F.CHANNELS, F.RANKS, F.PIM_RANK
BANKS_PER_RANK, TOTAL_RANKS, TOTAL_BANKS = F.BANKS_PER_RANK, F.TOTAL_RANKS, F.TOTAL_BANKS
HORIZON, HIDDEN, INTERMEDIATE = F.HORIZON, F.HIDDEN, F.INTERMEDIATE
LOCAL_INTERMEDIATE = F.LOCAL_INTERMEDIATE
BURST_BYTES, ROW_BYTES, BF16_BYTES, LANES = F.BURST_BYTES, F.ROW_BYTES, F.BF16_BYTES, F.LANES
ATTENTION_BYTES_PER_BANK = F.ATTENTION_BYTES_PER_BANK
GU_BASE, DOWN_BASE = F.GU_BASE, F.DOWN_BASE
GU_BYTES_PER_BANK, DOWN_BYTES_PER_BANK = F.GU_BYTES_PER_BANK, F.DOWN_BYTES_PER_BANK
Z_BASE = F.LAYER_STRIDE_BYTES_PER_BANK
Z_COLUMNS_PER_BANK = LOCAL_INTERMEDIATE // BANKS_PER_RANK
Z_BYTES_PER_BANK = Z_COLUMNS_PER_BANK * HORIZON * BF16_BYTES
Z_RESERVED_BYTES_PER_BANK = ROW_BYTES
LAYER_STRIDE_BYTES_PER_BANK = Z_BASE + Z_RESERVED_BYTES_PER_BANK
physical_address, decode_address = F.physical_address, F.decode_address
rank_bank_coords = F.rank_bank_coords
gate_up_location, down_location, weight_location = F.gate_up_location, F.down_location, F.weight_location


def z_location(query, feature):
    """[feature_slot16][token16] BF16 in the producing bank, no GPU spill."""
    F._bounds(query, HORIZON, "query")
    F._bounds(feature, INTERMEDIATE, "intermediate feature")
    rank, local_feature = divmod(feature, LOCAL_INTERMEDIATE)
    slot, bank = divmod(local_feature, BANKS_PER_RANK)
    return (*rank_bank_coords(rank, bank), Z_BASE + slot*BURST_BYTES + query*BF16_BYTES)


def z_column_location(rank, fragment, bank):
    F._bounds(fragment, Z_COLUMNS_PER_BANK, "Z fragment")
    return (*rank_bank_coords(rank, bank), Z_BASE + fragment*BURST_BYTES)


def z_fragment_feature(rank, fragment, lane):
    F._bounds(rank, TOTAL_RANKS, "active rank")
    F._bounds(fragment, Z_COLUMNS_PER_BANK, "Z fragment")
    F._bounds(lane, LANES, "input lane")
    return rank*LOCAL_INTERMEDIATE + fragment*LANES + lane


def z_fragment_buffer_offset(query, source_bank):
    F._bounds(query, HORIZON, "query")
    F._bounds(source_bank, BANKS_PER_RANK, "source bank")
    return query*BURST_BYTES + source_bank*BF16_BYTES


def build_layout():
    layout = F.build_layout()
    for name in ("down_tiled", "down_full"):
        layout["regions"][name]["rank_local_z_sram_bytes"] = 0
        layout["regions"][name]["z_source"] = "bank DRAM,16-column gather/transpose per input fragment"
    layout["regions"]["z_dram"] = dict(kind="activation", base=Z_BASE,
        bytes_per_bank=Z_BYTES_PER_BANK, reserved_bytes_per_bank=Z_RESERVED_BYTES_PER_BANK,
        row_first=Z_BASE//ROW_BYTES, row_last=Z_BASE//ROW_BYTES,
        layout="[local_feature_slot16][token16] BF16",
        bank_feature_formula="feature=channel*256+slot*16+bank_in_rank",
        system_payload_bytes=TOTAL_BANKS*Z_BYTES_PER_BANK)
    layout["layer_stride_bytes_per_bank"] = LAYER_STRIDE_BYTES_PER_BANK
    layout["rank_local_z_sram_bytes"] = 0
    layout["rank_bulk_GU_staging_bytes"] = 0
    layout["buffers"] = dict(
        GU_accumulator_per_bank=HORIZON*32*4,
        Down_tiled_accumulator_per_bank=HORIZON*16*4,
        rank_input_double=2*HORIZON*BURST_BYTES,
        bank_column_staging=BURST_BYTES,
        rank_nonlinear_input_staging=2*HORIZON*4,
        rank_nonlinear_output_staging=HORIZON*BF16_BYTES,
        rank_full_Z=0, rank_full_GU=0)
    layout["z_store_semantics"] = (
        "One feature's16-token Gate/Up128B streamed from bank accumulators to rank staging; "
        "unmeasured nonlinear arithmetic; returned BF16 Z32B is locally written to its producer bank")
    layout["z_load_semantics"] = (
        "Sixteen source banks each read one32B feature column; rank local path transposes "
        "columns into[token16][feature_lane16]512B buffer before Down MAC")
    return layout


LAYOUT = build_layout()
REGIONS = LAYOUT["regions"]

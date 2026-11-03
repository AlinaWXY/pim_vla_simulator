"""Full per-rank GPU input staging in one bank before GEMV consumption.

The DRAM reservation is reused only after the previous stage releases both
SRF slots. Fragment-major storage preserves the GPU packet order; source
packing is outside the measured memory schedule.
"""
BASE = 256*2048
CAPACITY = 128*1024


def input_offset(fragment, token, rows):
    return BASE + (fragment*rows+token)*32


def fragment_batches(fragments, rank_staging):
    """Use both physical SRF slots before returning bank0 to weight rows."""
    width = 2 if rank_staging else 1
    for first in range(0, fragments, width):
        yield tuple(range(first, min(first+width, fragments)))


def prefetch_metadata(batch, rows):
    return dict(prefetch_batch=batch[0]//2, prefetch_fragments=list(batch),
                prefetched_bytes_per_bank=len(batch)*rows*32)


def stage_input(writer, name, rows, fragments):
    payload = rows*fragments*32
    if rows not in (8, 16) or not 0 < payload <= CAPACITY:
        raise ValueError("Invalid staged input geometry")
    if not hasattr(writer, "staging_inputs"):
        writer.staging_inputs = []
    def body():
        first = writer.line_no+1
        writer.ranks("PIM_STAGE_BEGIN", BASE, f"{payload} {rows}")
        store_first = writer.line_no+1
        for offset in range(BASE, BASE+payload, 32):
            writer.ranks("ST", offset)
        store_last = writer.line_no
        writer.ranks("PIM_STAGE_SEAL", BASE)
        writer.staging_inputs.append(dict(name=name, base=BASE, bank=0,
            bytes_per_rank=payload, rows_per_fragment=rows, fragments=fragments,
            first_line=first, store_first_line=store_first, store_last_line=store_last,
            last_line=writer.line_no, copies=16, source_layout="[fragment][token][lane16] BF16",
            srf_fragments_per_batch=2, srf_batch_bytes_per_bank=rows*64,
            srf_slots_per_bank=2, srf_slot_bytes=512,
            consumption_order="load both slots, then compute both fragments in order"))
    writer.phase(name+"_gpu_input_stage", body, kind="input_staging",
        bytes_per_rank=payload, source_bank=0, source_base=BASE,
        rows_per_fragment=rows, fragments=fragments,
        completion="All ordinary WR data and write recovery complete before SEAL")


def reservation():
    return dict(bank=0, bank_group=0, physical_rank=1, base_bytes=BASE,
        capacity_bytes=CAPACITY, row_first=256, row_last=319,
        storage="DRAM array; excluded from GEMV SRAM capacity",
        layout="[fragment][token][lane16] BF16; reused between operators")

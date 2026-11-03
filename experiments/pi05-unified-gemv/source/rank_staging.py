"""Full per-rank GPU input staging before GEMV consumption.

The DRAM reservation is reused only after the previous stage releases both
SRF slots. Two-bank mode stripes fragments across different bank groups;
source packing is outside the measured memory schedule.
"""
BASE = 256*2048
CAPACITY = 128*1024


def check_staging_banks(staging_banks):
    if type(staging_banks) is not int or staging_banks not in (1, 2):
        raise ValueError("staging_banks must be 1 or 2")
    return staging_banks


def source_bank(fragment, staging_banks=1):
    return 4*(fragment % check_staging_banks(staging_banks))


def input_offset(fragment, token, rows, staging_banks=1):
    return BASE + (fragment//check_staging_banks(staging_banks)*rows+token)*32


def input_location(fragment, token, rows, staging_banks=1):
    return source_bank(fragment, staging_banks), input_offset(fragment, token, rows, staging_banks)


def fragment_metadata(fragment, rows, staging_banks=1):
    bank, offset = input_location(fragment, 0, rows, staging_banks)
    return dict(staging_source=dict(logical_fragment=fragment,
        physical_fragment=fragment//staging_banks, physical_rank=1,
        flat_bank=bank, bank_group=bank//4, bank=bank%4,
        offset_bytes=offset, bytes=rows*32))


def fragment_batches(fragments, rank_staging):
    """Use both physical SRF slots before returning source banks to weights."""
    width = 2 if rank_staging else 1
    for first in range(0, fragments, width):
        yield tuple(range(first, min(first+width, fragments)))


def prefetch_metadata(batch, rows):
    return dict(prefetch_batch=batch[0]//2, prefetch_fragments=list(batch),
                prefetched_bytes_per_bank=len(batch)*rows*32)


def stage_input(writer, name, rows, fragments, staging_banks=1):
    check_staging_banks(staging_banks)
    payload = rows*fragments*32
    if rows not in (8, 16) or not 0 < payload <= CAPACITY:
        raise ValueError("Invalid staged input geometry")
    if not hasattr(writer, "staging_inputs"):
        writer.staging_inputs = []
    def body():
        first = writer.line_no+1
        writer.ranks("PIM_STAGE_BEGIN", BASE, f"{payload} {rows}")
        store_first = writer.line_no+1
        stores = []
        for first_fragment in range(0, fragments, staging_banks):
            for token in range(rows):
                for fragment in range(first_fragment, min(first_fragment+staging_banks, fragments)):
                    bank, offset = input_location(fragment, token, rows, staging_banks)
                    line_first = writer.line_no+1
                    writer.ranks("ST", offset, bank=bank)
                    stores.append(dict(fragment=fragment, token=token, flat_bank=bank,
                        offset_bytes=offset, first_line=line_first, last_line=writer.line_no))
        store_last = writer.line_no
        writer.ranks("PIM_STAGE_SEAL", BASE)
        writer.staging_inputs.append(dict(name=name, base=BASE, bank=0 if staging_banks == 1 else None,
            bytes_per_rank=payload, rows_per_fragment=rows, fragments=fragments,
            first_line=first, store_first_line=store_first, store_last_line=store_last,
            last_line=writer.line_no, copies=16, source_layout="[fragment][token][lane16] BF16",
            staging_banks=staging_banks, source_banks=[4*i for i in range(staging_banks)],
            reservation=reservation(staging_banks),
            fragment_mapping=[fragment_metadata(f, rows, staging_banks)["staging_source"] for f in range(fragments)],
            store_order="fragment group, token, source bank, channel",
            store_sequence=stores,
            store_ranges=[dict(flat_bank=4*s, offset_first=BASE,
                offset_end=BASE+len(range(s, fragments, staging_banks))*rows*32,
                bytes_per_rank=len(range(s, fragments, staging_banks))*rows*32)
                for s in range(staging_banks)],
            srf_fragments_per_batch=2, srf_batch_bytes_per_bank=rows*64,
            srf_slots_per_bank=2, srf_slot_bytes=512,
            consumption_order="load both slots, then compute both fragments in order"))
    writer.phase(name+"_gpu_input_stage", body, kind="input_staging",
        bytes_per_rank=payload, source_bank=0 if staging_banks == 1 else None,
        source_banks=[4*i for i in range(staging_banks)], staging_banks=staging_banks, source_base=BASE,
        rows_per_fragment=rows, fragments=fragments,
        completion="All ordinary WR data and write recovery complete before SEAL")


def reservation(staging_banks=1):
    check_staging_banks(staging_banks)
    capacity_per_bank = CAPACITY//staging_banks
    return dict(bank=0 if staging_banks == 1 else None,
        bank_group=0 if staging_banks == 1 else None,
        physical_rank=1, base_bytes=BASE, bank_count=staging_banks,
        capacity_bytes=CAPACITY, capacity_bytes_per_bank=capacity_per_bank,
        row_first=256, row_last=(BASE+capacity_per_bank-1)//2048,
        banks=[dict(physical_rank=1, flat_bank=4*s, bank_group=s, bank=0,
            base_bytes=BASE, capacity_bytes=capacity_per_bank,
            row_first=256, row_last=(BASE+capacity_per_bank-1)//2048)
            for s in range(staging_banks)],
        storage="DRAM array; excluded from GEMV SRAM capacity",
        layout="[fragment//bank_count][token][lane16] BF16 per bank; reused between operators",
        logical_to_physical="flat_bank=4*(fragment%bank_count); offset=BASE+(fragment//bank_count*rows+token)*32")

"""Append a local two-bank PV-result staging and two-tile Wo GEMV sequence."""
import o_projection_layout as O
from rank_staging import input_offset,source_bank,fragment_batches,fragment_metadata,prefetch_metadata


def append(writer):
    def spill():
        first=writer.line_no+1
        writer.ranks("PIM_STAGE_BEGIN",O.INPUT_BASE,"8192 16")
        for fragment_pair in range(0,16,2):
            for query in range(16):
                for fragment in (fragment_pair,fragment_pair+1):
                    writer.ranks("PIM_PV_SPILL",input_offset(fragment,query,16,2),bank=source_bank(fragment,2))
        writer.ranks("PIM_STAGE_SEAL",O.INPUT_BASE)
        writer.local_writes.append(dict(phase="pv_result_dram_spill",line_start=first,line_end=writer.line_no,
            bytes=16*8192,bytes_per_rank=8192,external_bytes=0,
            source="PV FP32 accumulator; gather 16 features from source bank, cast to BF16",
            destination="bank0/4 staged PV partial; fragment identifies source feature bank"))
    writer.phase("pv_result_dram_spill",spill,kind="local_write",operator="pv",
        input_dtype="FP32",output_dtype="BF16",source_banks=list(range(16)),destination_banks=[0,4],
        bytes_per_rank=8192,external_bytes=0)
    for tile in range(2):
        if tile:
            writer.phase("o_projection_input_rewind",lambda:writer.ranks("PIM_O_REWIND",O.INPUT_BASE),
                kind="gemv_release",operator="o_projection",retains_dram_input=True)
        def compute(tile=tile):
            for batch in fragment_batches(16,True,views=2):
                inputs={}
                for fragment in batch:
                    first=writer.line_no+1
                    writer.ranks("PIM_STAGE_FRAGMENT",input_offset(fragment,0,16,2),fragment%2,
                        bank=source_bank(fragment,2))
                    inputs[fragment]=(first,writer.line_no)
                for fragment in batch:
                    first,last=inputs[fragment]
                    mac_first=writer.line_no+1
                    for column in range(32):
                        writer.ranks("PIM_MAC_AB",O.WEIGHT_BASE+tile*O.WEIGHT_TILE_BYTES+(fragment*32+column)*32,fragment%2)
                    writer.input_fragments.append(dict(operator="o_projection",output_tile=tile,
                        fragment=fragment,buffer=fragment%2,line_start=first,line_end=last,
                        mac_line_first=mac_first,mac_line_last=writer.line_no,command_count=16,
                        external_bytes=0,internal_load_bursts=16*16,query_count=16,
                        source="BF16 PV partial in bank0/4 DRAM",whole_srf_batch=True,
                        output_feature_slots=32,**prefetch_metadata(batch,16,views=2),
                        **fragment_metadata(fragment,16,2)))
        writer.phase(f"o_projection_tile_{tile}_compute",compute,kind="local_compute",operator="o_projection",
            output_tile=tile,mac_service_ck=16,query_count=16,whole_srf_batch=True,
            accumulator_bytes_per_bank=2048)
        def release():
            for slot in range(2):
                writer.ranks("PIM_GEMV_RELEASE",0,slot)
        writer.phase(f"o_projection_tile_{tile}_release",release,kind="gemv_release",operator="o_projection")
        def export(tile=tile):
            first=writer.line_no+1
            for offset in range(0,16384,32):
                writer.ranks("PIM_RD_RES",offset)
            writer.exports.append(dict(phase=f"o_projection_tile_{tile}_export",line_start=first,
                line_end=writer.line_no,bytes=262144,bytes_per_rank=16384,output_tile=tile,
                dtype="BF16",query_count=16,elements_per_packet=16,rank_partial=True,
                output_order="[query16][bank16][output_column32]",payload_contract="o_projection_mapping.output"))
        writer.phase(f"o_projection_tile_{tile}_export",export,kind="external_read",operator="o_projection",
            output_dtype="BF16",rank_partial=True,output_tile=tile)

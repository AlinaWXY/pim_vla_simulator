"""FFN data/compute-bank separation with two 512B input ping-pong slots."""
from horizon_ffn import Writer, address
import ffn_split_layout as L


def begin_pass(w,operator):
    def body():
        first=w.line_no+1
        w.ranks("PIM_FFN_PASS",L.WEIGHT_BASES[operator])
        w.ffn_passes.append(dict(operator=operator,pass_id=L.PASS_IDS[operator],
            line_start=first,line_end=w.line_no,weight_base=L.WEIGHT_BASES[operator],
            replay_sealed_input=operator=="up",input_usage=1 if operator=="up" else 0))
    w.phase(operator+"_pass_begin",body,kind="control",operator=operator,
        replay_sealed_input=operator=="up")


def stage_input(w,operator,count):
    fragments=L.input_width(operator)//16
    payload=fragments*512
    def body():
        first=w.line_no+1
        w.ranks("PIM_STAGE_BEGIN",L.INPUT_BASE,f"{payload} 16")
        store_first=w.line_no+1
        stores=[]
        logical_rows=fragments//4
        for physical_row in range((logical_rows+count-1)//count):
            for column in range(64):
                for source_index,bank in enumerate(L.data_banks(count)):
                    logical_row=physical_row*count+source_index
                    if logical_row>=logical_rows:continue
                    fragment=logical_row*4+column//16;token=column%16
                    offset=L.INPUT_BASE+physical_row*2048+column*32
                    start=w.line_no+1
                    w.ranks("ST",offset,bank=bank)
                    stores.append(dict(fragment=fragment,token=token,flat_bank=bank,
                        offset_bytes=offset,first_line=start,last_line=w.line_no))
        store_last=w.line_no
        w.ranks("PIM_STAGE_SEAL",L.INPUT_BASE)
        w.staging_inputs.append(dict(name=operator,base=L.INPUT_BASE,bank=0 if count==1 else None,
            bytes_per_rank=payload,rows_per_fragment=16,fragments=fragments,
            first_line=first,store_first_line=store_first,store_last_line=store_last,last_line=w.line_no,
            copies=16,staging_banks=count,source_banks=L.data_banks(count),ffn_row_striped=True,
            source_layout="[row][fragment4][token16][lane16] BF16 per data bank",
            store_order="physical row, column, data bank, channel",store_sequence=stores,
            fragment_mapping=[dict(logical_fragment=f,flat_bank=L.input_packet(operator,0,f,0,count)["source_bank"],
                offset_bytes=L.input_packet(operator,0,f,0,count)["offset_bytes"],bytes=512)
                for f in range(fragments)],
            store_ranges=[dict(flat_bank=bank,offset_first=L.INPUT_BASE,
                offset_end=L.INPUT_BASE+payload//count,bytes_per_rank=payload//count)
                for bank in L.data_banks(count)],
            rows_fully_written_per_source_bank=logical_rows//count,
            complete_row_bytes=2048,srf_slots_per_bank=2,srf_slot_bytes=512,
            consumption_order="ping-pong: next fragment reads/broadcasts while current fragment computes"))
    w.phase(("gate_up" if operator=="gate" else "down")+"_gpu_input_stage",body,
        kind="input_staging",bytes_per_rank=payload,fragments=fragments,rows_per_fragment=16,
        source_banks=L.data_banks(count),ffn_data_banks=count,source_base=L.INPUT_BASE,
        completion="Entire payload WR and recovery complete before SEAL")


def compute(w,operator,count):
    fragments=L.input_width(operator)//16;cols=L.columns(operator,count)
    name=operator+"_local_input_and_compute"
    def body():
        inputs={}
        def fill(fragment):
            first=w.line_no+1;packet=L.input_packet(operator,0,fragment,0,count)
            w.ranks("PIM_STAGE_FRAGMENT",packet["offset_bytes"],fragment%2,bank=packet["source_bank"])
            inputs[fragment]=(first,w.line_no,packet)
        fill(0)
        for fragment in range(fragments):
            if fragment+1<fragments:fill(fragment+1)
            first,last,packet=inputs[fragment];mac_first=w.line_no+1
            for column in range(cols):
                w.ranks("PIM_MAC_AB",L.WEIGHT_BASES[operator]+(fragment*cols+column)*32,fragment%2)
            w.input_fragments.append(dict(operator=operator,pass_id=L.PASS_IDS[operator],
                input_usage=1 if operator=="up" else 0,phase=name,fragment=fragment,buffer=fragment%2,
                line_start=first,line_end=last,input_line_first=first,input_line_last=last,
                mac_line_first=mac_first,mac_line_last=w.line_no,
                source_tensor=packet["source_tensor"],source_feature_first=fragment*16,
                source_tensor_width=L.input_width(operator),same_source_fragment_all_ranks=True,
                command_count=16,internal_load_bursts=16*16,external_bytes=0,
                bursts_per_rank=16,bytes_per_rank=512,useful_input_bytes_per_rank=512,
                slots_per_fragment=cols,consumer_mac_count=cols*16,
                valid_bank_masks=[L.valid_bank_mask(operator,c,count) for c in range(cols)],
                compute_bank_mask=sum(1<<b for b in L.compute_banks(count)),
                staging_source=dict(logical_fragment=fragment,physical_rank=1,flat_bank=packet["source_bank"],
                    bank_group=packet["source_bank"]//4,bank=packet["source_bank"]%4,
                    offset_bytes=packet["offset_bytes"],bytes=512,logical_row=fragment//4,
                    fragment_in_row=fragment%4),
                input_broadcast_extra_cycles=0))
        w.fence()
    w.phase(name,body,kind="local_compute",operator=operator,M=16,fragment_count=fragments,
        slots_per_fragment=cols,weight_base=L.WEIGHT_BASES[operator],
        source_tensor="Z_gpu" if operator=="down" else "X",double_buffer_slots=2,
        scheduling="prefetch next slot before current MAC requests; dependency-aware rank scheduler overlaps independent work")
    def release():
        for slot in (0,1):w.ranks("PIM_GEMV_RELEASE",0,slot)
    w.phase(name+"_gemv_release",release,kind="gemv_release")


def export(w,operator):
    features=L.rank_features(operator);name=operator+"_result_export"
    def body():
        first=w.line_no+1
        for offset in range(0,features*64,32):w.ranks("PIM_FFN_RDRES",offset)
        w.exports.append(dict(phase=name,operator=operator,pass_id=L.PASS_IDS[operator],
            line_start=first,line_end=w.line_no,bytes_fp32=features*64*16,
            bytes_per_rank=features*64,output_count_per_rank=features,projection_count=1,
            output_order="[valid local feature][token16] FP32",rank_partials=0,destination="GPU",
            elements_per_packet=8,padding_exported=False))
        w.fence()
    w.phase(name,body,kind="export",operator=operator,output_count_per_rank=features,
        rank_partials=0,destination="GPU",padding_exported=False)


def generate(path,horizon,case,ffn_data_banks,ffn_overlap=True):
    if horizon!=16 or case not in ("gate_up_gpu","down_gpu_input","combined_timing_chain"):
        raise ValueError((horizon,case))
    count=L.check_data_banks(ffn_data_banks)
    gu,down=case!="down_gpu_input",case!="gate_up_gpu"
    w=Writer(path,horizon);w.staging_inputs=[];w.ffn_passes=[]
    try:
        if gu:
            begin_pass(w,"gate");stage_input(w,"gate",count);compute(w,"gate",count);export(w,"gate")
            begin_pass(w,"up");compute(w,"up",count);export(w,"up")
        if gu and down:
            w.phase("gpu_gelu_multiply_boundary",lambda:None,kind="excluded_boundary",measured_latency_us=None,
                operation="Z=BF16(GeLU(Gate)*Up) on GPU",policy="GPU arithmetic remains unmeasured")
        if down:
            begin_pass(w,"down");stage_input(w,"down",count);compute(w,"down",count);export(w,"down")
    finally:w.stream.close()
    dq=dict(GU_input_gpu_to_pim=int(gu)*524288,GU_gate_up_pim_to_gpu=int(gu)*524288,
        Down_Z_gpu_to_pim=int(down)*2097152,Down_output_pim_to_gpu=int(down)*65536)
    dq["total"]=sum(dq.values())
    written=int(gu)*524288+int(down)*2097152;read=int(gu)*1048576+int(down)*2097152
    info=L.layout(count)
    result=dict(case=case,horizon=16,action_horizon=16,ffn_data_banks=count,ffn_overlap=bool(ffn_overlap),
        whole_batch=False,token_microbatches=1,padded_compute_tokens=16,layout=info,
        buffers=dict(input_srf_bytes_per_bank=1024,bank_input_double=1024,input_slots_per_bank=2,
            input_slot_bytes=512,output_pool_bytes_per_bank=2048,
            GU_accumulator_per_bank=16*L.columns("gate",count)*4,Down_accumulator_per_bank=320,
            separate_rank_input_bytes=0,rank_full_Z=0,rank_full_GU=0),
        counts=dict(w.counts),segments=w.segments,input_fragments=w.input_fragments,exports=w.exports,
        ffn_passes=w.ffn_passes,staging_inputs=w.staging_inputs,rank_staging=True,
        staging_banks=count,ffn_staging_banks=count,expected_dq_bytes=dq,
        expected_internal_bytes=dict(input_staging_write=written,input_staging_read=read,
            input_multicast_source_bytes=read,input_multicast_destination_compute_bank_bytes=read*(16-count),
            input_multicast_additional_service_cycles=0),
        useful_scalar_MACs=int(gu)*16*1024*4096*2+int(down)*16*4096*1024,padding_MACs=0,
        mac_service_ck=16,gpu_partial_reduction_required=False,numeric_compute=False,
        required_backend=dict(pim_horizon=16,ffn_data_banks=count,ffn_overlap=bool(ffn_overlap),
            pim_mac_scope="rank",pim_input_broadcast=False,nCCDAB=16,ffn_z_preinstalled=False),
        timing_chain=gu and down,integrated_chain=False,
        padding_policy="Uniform column address reservation; invalid compute banks masked per MAC; export valid features only",
        exclusions=["GPU GeLU/multiply/BF16 cast and producer availability","Weight and initial data placement"],
        input_resource="same two 512B SRF slots; opposite-slot refill may overlap current-slot MAC")
    result["ffn_split_mapping"]=L.mapping_contract(result)
    return result

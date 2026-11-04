"""Local V cache writeback and token-parity PV on one whole 1KiB SRF."""
from horizon_attention_workload import Writer
from rank_staging import (stage_input,input_offset,source_bank,fragment_metadata,
                          fragment_batches,prefetch_metadata,reservation)
import pv_token_layout as T
import v_token_layout as V


def local_cache_spill(writer, qkv_fusion=False):
    def spill():
        first=writer.line_no+1
        for lane in range(16):
            writer.banks("PIM_SPILL",T.CACHE_BASE+(30*16+lane)*32)
        writer.local_writes.append(dict(phase="v_local_cache_spill",line_start=first,line_end=writer.line_no,
            bytes=16*16*16*32,internal_read_bytes=16*16*16*32,external_bytes=0,
            source="fused QKV V accumulator columns32..47" if qkv_fusion else "V FP32 accumulator columns0..15",
            destination="token-parity V cache tail block480..495; preserve0..3, cast4..11, zero12..15"))
    writer.phase("v_local_cache_spill",spill,kind="local_write",
        cache_base=T.CACHE_BASE,local_prefix_preserved=[480,484],local_suffix=[484,492],
        local_padding=[492,496],read_modify_write=True,external_bytes=0)


def p_input_contract():
    return dict(tensor="P",logical_shape=[8,16,992],rank_shape=[16,496],
        local_rows=16,fragment_count=31,lanes_per_fragment=16,bytes_per_rank=15872,
        source_element_formula="P[channel//2, query, 2*(16*fragment+input_lane)+channel%2]",
        rank_kv_tokens=[dict(channel=c,head=c//2,logical_tokens=list(range(c%2,992,2))) for c in range(16)],
        queries=list(range(16)),masked_global_padding=[984,992],
        bytes_per_fragment_per_rank=512,query_batches=1)


def mapping_contract(manifest):
    is_v=manifest["case"]=="v"
    result=dict(schema_version=1,enabled=True,operator=manifest["case"],
        local_cache=T.cache_contract(),input=V.input_contract() if is_v else p_input_contract(),
        trace_ranges=T.trace_ranges(manifest),numerical_compute=False)
    if is_v:
        result["projection_weights"]=dict(base_bytes=131072,end_bytes_exclusive=163840,
            bytes_per_bank=32768,copies_per_head=2,columns_per_fragment=16,fragment_count=64)
    else:
        bf16=manifest.get("pv_bf16_output",False)
        element_bytes=2 if bf16 else 4
        dtype="BF16" if bf16 else "FP32"
        result.update(query_batches=1,query_rows=16,kv_fragments=31,columns_per_fragment=16,
            weight_offset_formula="163840+(fragment*16+feature_lane)*32",
            export=dict(dtype=dtype,accumulator_dtype="FP32",order="[query16][bank16][feature_lane16]",bytes_per_rank=4096*element_bytes,
                packets_per_rank=128*element_bytes,elements_per_packet=32//element_bytes,packet_bytes=32,
                element_offset_formula=f"((query*16+bank)*16+feature_lane)*{element_bytes}",
                trace_line_formula="exports[0].line_start+(packet_offset_bytes//32)*16+channel",
                trace_address_is_result_tag=True),
            gpu_reduction=dict(operation="sum",operands_per_element=2,logical_shape=[8,16,256],
                operand_channels_formula="[2*head,2*head+1]",elements=32768,
                operand_bytes=65536*element_bytes,output_bytes=131072,dtype=dtype,
                input_dtype=dtype,output_dtype="FP32",latency_us=None,
                meaning="Each rank sums its 496 KV-token parity shard; GPU adds corresponding rank partial O values"))
        if manifest.get("o_projection"):
            result["export"].update(enabled=False,bytes_per_rank=0,packets_per_rank=0,
                destination="local two-bank DRAM input of Wo",trace_line_formula=None)
            result["gpu_reduction"].update(enabled=False,
                meaning="PV rank partials remain local; reduction is deferred until after Wo")
    return result


def generate(path,horizon,case,rank_staging=False,staging_banks=1,rank_descriptors=False,
             o_projection=False,pv_bf16_output=False):
    if horizon != 16 or case not in ("v","pv"):
        raise ValueError((horizon,case))
    if o_projection and (not rank_staging or staging_banks!=2 or not pv_bf16_output):
        raise ValueError("O projection requires two-bank rank staging and BF16 PV output")
    o_projection=bool(o_projection and case=="pv")
    pv_bf16_output=bool(pv_bf16_output and case=="pv")
    from horizon_value import layout
    geometry=layout(horizon,v_token_sharding=True,pv_token_sharding=True)
    geometry["input_staging"]=reservation(staging_banks) if rank_staging else None
    rows,fragments,views=(8,64,4) if case=="v" else (16,31,2)
    base=131072 if case=="v" else T.CACHE_BASE
    w=Writer(path,rank_descriptors=rank_descriptors)
    try:
        if rank_staging:
            stage_input(w,case,rows,fragments,staging_banks,views=views)
            w.staging_inputs[-1]["logical_input"]=V.input_contract() if case=="v" else p_input_contract()
            w.staging_inputs[-1]["whole_srf_batch"]=True
        def compute():
            for batch in fragment_batches(fragments,True,views=views):
                inputs={}
                for fragment in batch:
                    buffer,first=fragment%views,w.line_no+1
                    if rank_staging:
                        w.ranks("PIM_STAGE_FRAGMENT",input_offset(fragment,0,rows,staging_banks),buffer,
                            bank=source_bank(fragment,staging_banks))
                    else:
                        for row in range(rows):
                            w.ranks("PIM_WR_GB",row*32,buffer)
                    inputs[fragment]=(buffer,first,w.line_no)
                for fragment in batch:
                    buffer,first,last=inputs[fragment]
                    mac_first=w.line_no+1
                    for lane in range(16):
                        w.ranks("PIM_MAC_AB",base+(fragment*16+lane)*32,buffer)
                    w.input_fragments.append(dict(operator=case,fragment=fragment,buffer=buffer,
                        line_start=first,line_end=last,mac_line_first=mac_first,mac_line_last=w.line_no,
                        command_count=16 if rank_staging else rows*16,
                        external_bytes=0 if rank_staging else rows*16*32,
                        internal_load_bursts=rows*16 if rank_staging else 0,
                        query_count=rows,source="X" if case=="v" else "post-softmax P token-parity shard",
                        whole_srf_batch=True,output_feature_slots=16,
                        **(prefetch_metadata(batch,rows,views=views) if rank_staging else {}),
                        **(fragment_metadata(fragment,rows,staging_banks) if rank_staging else {})))
            w.fence()
        w.phase(f"{case}_local_input_compute" if rank_staging else f"{case}_input_compute",compute,
            kind="local_compute" if rank_staging else "pipeline",operator=case,
            mac_service_ck=rows,native_mac_completion=True,query_count=rows,whole_srf_batch=True)
        if case=="v":
            local_cache_spill(w)
        def release():
            for view in range(views):
                w.ranks("PIM_GEMV_RELEASE",0,view)
        w.phase(f"{case}_gemv_release",release,kind="gemv_release")
        if o_projection:
            from o_projection_workload import append
            append(w)
        elif case=="pv":
            output_bytes=8192 if pv_bf16_output else 16384
            output_dtype="BF16" if pv_bf16_output else "FP32"
            def export():
                first=w.line_no+1
                for offset in range(0,output_bytes,32):
                    w.ranks("PIM_RD_RES",offset)
                w.exports.append(dict(phase="pv_result_export",line_start=first,line_end=w.line_no,
                    bytes=output_bytes*16,bytes_per_rank=output_bytes,query_count=16,dtype=output_dtype,
                    output_order="[query16][bank16][feature_lane16]",elements_per_packet=16 if pv_bf16_output else 8,
                    payload_contract="pv_token_mapping.export",rank_partial=True))
                w.fence()
            w.phase("pv_result_export",export,kind="external_read",output_dtype=output_dtype,rank_partial=True)
    finally:
        w.close()
    dq=dict(pim_input_write=rows*fragments*16*32,pim_result_read=w.counts["PIM_RD_RES"]*32,
        dram_read=0,dram_write=0)
    dq["total"]=sum(dq.values())
    matrix=dict(M=16,K=1024 if case=="v" else 984,N=2048 if case=="v" else 256,
        heads=1 if case=="v" else 8,padded_K=1024 if case=="v" else 992,
        padded_N=2048 if case=="v" else 256,slots_per_bank=16,base=base,mac_service_ck=rows,
        local_rows_per_rank=rows,output_features_per_bank=16,
        rank_partition="query token parity" if case=="v" else "KV token parity",
        query_batches=1,rank_partial_reduction=case=="pv")
    matrix["logical_macs"]=matrix["heads"]*matrix["M"]*matrix["K"]*matrix["N"]
    matrix["padded_macs"]=matrix["heads"]*matrix["M"]*matrix["padded_K"]*matrix["N"]
    result=dict(case=case,horizon=16,action_horizon=16,attention_heads=8,schedule="whole_srf_batch",
        matrix=matrix,layout=geometry,counts=dict(w.counts),expanded_counts=dict(w.expanded_counts),
        rank_batches=w.rank_batches,segments=w.segments,input_fragments=w.input_fragments,
        exports=w.exports,local_writes=w.local_writes,staging_inputs=getattr(w,"staging_inputs",[]),
        rank_staging=rank_staging,staging_banks=staging_banks if rank_staging else 0,
        rank_descriptors=rank_descriptors,v_token_sharding=True,pv_token_sharding=True,
        expected_dq_bytes=dq,expected_internal_array_read=w.counts["PIM_MAC_AB"]*16*32,
        expected_local_bytes=dict(v_cache_rmw_read=131072 if case=="v" else 0,
            v_cache_rmw_write=131072 if case=="v" else 0),
        buffers=dict(input_srf_bytes_per_bank=1024,output_pool_bytes_per_bank=2048,
            accumulator_per_bank=rows*16*4,input_rows=rows,gemv_views_per_bank=views,
            gemv_view_bytes=rows*32,separate_rank_input_bytes=0),
        required_backend=dict(pim_horizon=16,pim_rank_rope=False,pim_ffn_dram=False,
            pim_input_broadcast=False,pim_mac_scope="rank",pim_data_path="attacc",nCCDAB=rows),
        physical_active_rank=1,execution="Whole 1KiB SRF batch; one complete 16-query PV scan" if case=="pv" else
            "Token-parity V projection followed by bank-local BF16 cache tail RMW",
        query_padding=0,kv_padding=8,gpu_boundary="Post-softmax P availability and rank-partial O sum remain unmeasured",
        cache_precondition="Token-parity prefix resident; full cache preinstalled for standalone PV, local V spill required in serial",
        unknown_latency_us=dict(GPU_softmax=None,GPU_PV_partial_sum=None,GPU_source_DMA_arrival=None),
        numeric_compute=False,complete_ae_latency=False)
    if pv_bf16_output:
        result["pv_bf16_output"]=True
        result["matrix"].update(input_dtype="BF16",weight_dtype="BF16",accumulator_dtype="FP32",output_dtype="BF16")
        result["required_backend"]["pim_pv_bf16_output"]=True
    if o_projection:
        from o_projection_layout import contract
        result.update(o_projection=True,o_projection_mapping=contract())
        result["o_projection_mapping"]["trace_ranges"]=T.trace_ranges(result)
        result["required_backend"]["pim_o_projection"]=True
        result["buffers"]["o_projection_accumulator_per_bank"]=2048
        result["expected_local_bytes"].update(pv_result_dram_write=131072,
            o_projection_input_read=262144,o_projection_broadcast_destination=4194304)
        result["unknown_latency_us"].pop("GPU_PV_partial_sum")
        result["unknown_latency_us"]["GPU_O_projection_partial_sum"]=None
        result["gpu_boundary"]="Post-softmax P availability and 16-way Wo-partial GPU sum remain unmeasured"
    result["pv_token_mapping"]=mapping_contract(result)
    if case=="v":
        result["v_token_mapping"]=dict(schema_version=2,enabled=True,logical_shape=[8,16,256],
            input=V.input_contract(),weights=result["pv_token_mapping"]["projection_weights"],
            feature_lanes_by_bank=result["pv_token_mapping"]["local_cache"]["feature_lanes_by_bank"],
            cache_local=T.cache_contract(),trace_ranges=T.trace_ranges(result),external_export=False,
            numerical_compute=False)
    return result

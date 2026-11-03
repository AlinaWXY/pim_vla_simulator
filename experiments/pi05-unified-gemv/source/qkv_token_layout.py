"""Fused Q/K/V weights and outputs in unchanged token/feature ownership."""
from copy import deepcopy
import horizon_attention_layout as L
from v_token_layout import input_contract
from pv_token_layout import cache_contract, trace_ranges

WEIGHT_BASE, WEIGHT_BYTES = 393216, 98304
PROJECTIONS = ("q","k","v")


def weight_location(projection, head, parity, reduction, feature):
    if projection not in PROJECTIONS:
        raise ValueError(projection)
    L.check(reduction,1024,"QKV reduction")
    owner,lane=L.feature_owner(head,parity,feature)
    fragment,input_lane=divmod(reduction,16)
    return (*owner,WEIGHT_BASE+(fragment*48+PROJECTIONS.index(projection)*16+lane)*32+input_lane*2)


def output_element(projection, head, query, feature):
    if projection not in PROJECTIONS:
        raise ValueError(projection)
    L.check(query,16,"QKV query")
    owner,lane=L.feature_owner(head,query%2,feature)
    column=PROJECTIONS.index(projection)*16+lane
    return dict(projection=projection,head=head,query=query,feature=feature,
        channel=owner[0],physical_rank=1,bank_group=owner[2],bank=owner[3],
        local_token=query//2,accumulator_column=column,
        accumulator_index=query//2*48+column,dtype="FP32")


def layout(original):
    result=deepcopy(original)
    result["regions"].pop("w_qk")
    result["regions"]["w_qkv"]=dict(base=WEIGHT_BASE,used_bytes_per_bank=WEIGHT_BYTES,
        reserved_bytes_per_bank=WEIGHT_BYTES,row_first=192,row_last=239)
    result.update(qkv_fusion=True,projection_weight_order="[fragment64][Q/K/V3][feature_lane16][input_lane16] BF16",
        qkv_weight_system_bytes=256*WEIGHT_BYTES,end=WEIGHT_BASE+WEIGHT_BYTES,
        reserved_bytes_per_bank=sum(r["reserved_bytes_per_bank"] for r in result["regions"].values()))
    result.pop("qk_weight_system_bytes",None)
    result["initialization_bytes"].pop("weights_qk")
    result["initialization_bytes"]["weights_qkv"]=256*WEIGHT_BYTES
    result["buffers"]["projection_accumulator_per_bank"]=8*48*4
    return result


def mapping_contract(manifest):
    return dict(schema_version=1,enabled=True,operators=list(PROJECTIONS),
        logical_shapes=dict(X=[16,1024],W_per_operator_per_head=[1024,256],QKV_each=[8,16,256]),
        input=input_contract(),weights=dict(base_bytes=WEIGHT_BASE,end_bytes_exclusive=WEIGHT_BASE+WEIGHT_BYTES,
            bytes_per_bank=WEIGHT_BYTES,copies_per_head=2,fragment_count=64,columns_per_fragment=48,
            column_operators=[dict(operator=p,column_first=i*16,column_end=(i+1)*16) for i,p in enumerate(PROJECTIONS)],
            element_offset_formula="393216+(fragment*48+operator_index*16+feature_lane)*32+input_lane*2"),
        feature_lanes_by_bank=[[L.feature_from_lane(b,l) for l in range(16)] for b in range(16)],
        accumulator=dict(rows=8,columns=48,bytes_per_bank=1536,
            index_formula="local_token*48+operator_index*16+feature_lane"),
        local_cache=cache_contract(qkv_fusion=True),trace_ranges=trace_ranges(manifest),
        numerical_compute=False)

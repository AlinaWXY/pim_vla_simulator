"""Dedicated FFN data banks, dense logical outputs and row-striped inputs.

Uniform weight columns reserve padding cells, but invalid banks are masked
from each MAC and padded features never enter the GPU result stream.
"""
from copy import deepcopy
from horizon_ffn import address

CHANNELS, TOKENS, ROW, BURST = 16, 16, 2048, 32
INPUT_BASE = 524288
WEIGHT_BASES = dict(gate=1048576, up=1087488, down=2097152)
PASS_IDS = dict(gate=0, up=1, down=2)


def check_data_banks(count):
    if type(count) is not int or count not in (1,2):
        raise ValueError("FFN dedicated data-bank count must be 1 or 2")
    return count


def data_banks(count):
    return [0,4][:check_data_banks(count)]


def compute_banks(count):
    source=set(data_banks(count))
    return [bank for bank in range(16) if bank not in source]


def rank_features(operator):
    if operator not in PASS_IDS:
        raise ValueError(operator)
    return 64 if operator=="down" else 256


def input_width(operator):
    rank_features(operator)
    return 4096 if operator=="down" else 1024


def columns(operator,count):
    n=len(compute_banks(count))
    return (rank_features(operator)+n-1)//n


def valid_bank_mask(operator,column,count):
    banks=compute_banks(count)
    if not 0<=column<columns(operator,count):
        raise ValueError("FFN output column outside pass")
    return sum(1<<bank for ordinal,bank in enumerate(banks)
        if column*len(banks)+ordinal<rank_features(operator))


def feature_location(operator,feature,count):
    width=rank_features(operator)
    if type(feature) is not int or not 0<=feature<16*width:
        raise ValueError("FFN feature outside logical output")
    channel,local=divmod(feature,width)
    column,ordinal=divmod(local,16-check_data_banks(count))
    bank=compute_banks(count)[ordinal]
    return dict(channel=channel,physical_rank=1,bank=bank,bank_group=bank//4,
        bank_in_group=bank%4,compute_bank_ordinal=ordinal,column=column,
        local_feature=local,feature=feature)


def weight_location(operator,reduction,feature,count):
    if type(reduction) is not int or not 0<=reduction<input_width(operator):
        raise ValueError("FFN reduction outside input")
    owner=feature_location(operator,feature,count)
    fragment,lane=divmod(reduction,16)
    offset=WEIGHT_BASES[operator]+(fragment*columns(operator,count)+owner["column"])*32+lane*2
    return address(owner["channel"],owner["bank"],offset)


def input_packet(operator,channel,fragment,token,count):
    width=input_width(operator)
    if not 0<=channel<16 or not 0<=fragment<width//16 or not 0<=token<16:
        raise ValueError("FFN input coordinate outside payload")
    logical_row,in_row=divmod(fragment,4)
    source=data_banks(count)[logical_row%count]
    offset=INPUT_BASE+logical_row//count*2048+in_row*512+token*32
    return dict(source_tensor="Z_gpu" if operator=="down" else "X",token=token,
        feature_first=fragment*16,source_byte_offset=(token*width+fragment*16)*2,
        channel=channel,physical_rank=1,source_bank=source,offset_bytes=offset,
        destination=address(channel,source,offset),logical_fragment=fragment,
        logical_row=logical_row,physical_row=offset//2048,fragment_in_row=in_row,
        slot=fragment%2,srf_row=token,bytes=32,
        target_bank_mask=sum(1<<b for b in compute_banks(count)))


def result_element(operator,channel,token,local_feature,count):
    if not 0<=channel<16 or not 0<=token<16 or not 0<=local_feature<rank_features(operator):
        raise ValueError("FFN result coordinate outside logical output")
    owner=feature_location(operator,channel*rank_features(operator)+local_feature,count)
    offset=local_feature*16*4+token*4
    return dict(operator=operator,channel=channel,physical_rank=1,token=token,
        feature=owner["feature"],local_feature=local_feature,source_bank=owner["bank"],
        source_column=owner["column"],accumulator_index=token*columns(operator,count)+owner["column"],
        result_offset_bytes=offset,packet_offset_bytes=offset//32*32,byte_in_packet=offset%32,
        trace_address=address(channel,0,offset//32*32),dtype="FP32")


def result_packet(operator,channel,offset,count):
    if offset%32 or not 0<=offset<rank_features(operator)*64:
        raise ValueError("FFN export packet outside valid result stream")
    feature,inside=divmod(offset,64)
    return [result_element(operator,channel,t,feature,count) for t in range(inside//4,inside//4+8)]


def layout(count):
    banks=compute_banks(count)
    passes={}
    for operator in PASS_IDS:
        n=columns(operator,count);fragments=input_width(operator)//16
        used=fragments*n*32;reserved=20*2048 if operator=="down" else 19*2048
        passes[operator]=dict(pass_id=PASS_IDS[operator],weight_base=WEIGHT_BASES[operator],
            weight_used_bytes_per_compute_bank=used,weight_reserved_bytes_per_compute_bank=reserved,
            weight_row_first=WEIGHT_BASES[operator]//2048,
            weight_row_last=(WEIGHT_BASES[operator]+used-1)//2048,
            input_fragments=fragments,input_width=input_width(operator),columns_per_fragment=n,
            logical_features_per_rank=rank_features(operator),reserved_features_per_rank=n*len(banks),
            padding_features_per_rank=n*len(banks)-rank_features(operator),
            valid_bank_masks=[valid_bank_mask(operator,c,count) for c in range(n)],
            valid_features_per_bank=[dict(bank=b,features=[c*len(banks)+j for c in range(n)
                if c*len(banks)+j<rank_features(operator)]) for j,b in enumerate(banks)],
            output_bytes_per_compute_bank=16*n*4,valid_export_bytes_per_rank=rank_features(operator)*64,
            weight_order="[input_fragment][output_column][input_lane16] BF16",
            zero_weight_padding_allocated=True,padded_bank_MACs_issued=False,
            source_tensor="Z_gpu" if operator=="down" else "X")
    return dict(ffn_data_banks=count,data_banks=data_banks(count),compute_banks=banks,
        data_bank_mask=sum(1<<b for b in data_banks(count)),
        compute_bank_mask=sum(1<<b for b in banks),physical_rank=1,channels=16,horizon=16,
        rank_partition_preserved=True,gu_features_per_rank=256,down_features_per_rank=64,
        input_srf_bytes_per_bank=1024,input_slots_per_bank=2,input_slot_bytes=512,
        output_capacity_bytes_per_bank=2048,passes=passes,
        input_staging=dict(base_bytes=INPUT_BASE,capacity_bytes_per_rank=131072,
            capacity_bytes_per_source_bank=131072//count,fragments_per_row=4,
            layout="[logical_row//data_banks][fragment_in_row4][token16][lane16] BF16",
            mapping="logical_row=fragment//4; source=data_banks[logical_row%D]; offset=524288+(logical_row//D)*2048+(fragment%4)*512+token*32",
            store_order="physical row, column0..63, source bank, channel",
            read_order="logical fragment; four consecutive 512B fragments per open source row"),
        output_layout="[valid local feature][token16] FP32; padding omitted",
        gu_input_stores=1,gu_input_reads=2,gpu_partial_reduction_required=False,
        numerical_compute=False)


def mapping_contract(manifest):
    result=deepcopy(manifest["layout"])
    result.update(schema_version=1,enabled=True,ffn_overlap=manifest["ffn_overlap"],
        trace_ranges=dict(phases=[dict(name=s["name"],line_start=s["line_start"],line_end=s["line_end"])
            for s in manifest["segments"]],input_fragments=deepcopy(manifest["input_fragments"]),
            exports=deepcopy(manifest["exports"]),passes=deepcopy(manifest["ffn_passes"])))
    return result


def shift_trace_lines(contract,shift):
    result=deepcopy(contract)
    keys={"line_start","line_end","mac_line_first","mac_line_last","input_line_first","input_line_last"}
    def visit(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key in keys:value[key]=child+shift
                else:visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(result["trace_ranges"])
    return result

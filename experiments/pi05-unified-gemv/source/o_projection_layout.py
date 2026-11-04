"""Token-PV partials to two-bank DRAM, then head-local output projection.

These are enumerable tensor/address contracts. The simulator models FP32
accumulation and BF16 boundary storage, rather than executing tensor arithmetic.
"""
import horizon_attention_layout as L
from rank_staging import input_offset, source_bank, BASE as INPUT_BASE

HEADS, RANKS, QUERIES, INPUT_FEATURES, OUTPUT_FEATURES = 8, 16, 16, 256, 1024
WEIGHT_BASE = 786432
TILES, COLUMNS, FRAGMENTS = 2, 32, 16
WEIGHT_TILE_BYTES = FRAGMENTS*COLUMNS*32
WEIGHT_HEAD_BYTES_PER_BANK = TILES*WEIGHT_TILE_BYTES
RESIDENT_WEIGHT_BYTES_PER_BANK = HEADS*WEIGHT_HEAD_BYTES_PER_BANK
RESIDENT_WEIGHT_END = WEIGHT_BASE+RESIDENT_WEIGHT_BYTES_PER_BANK


def input_feature(fragment, lane):
    return L.feature_from_lane(fragment, lane)


def input_location(channel, fragment, query):
    L.check(channel, RANKS, "O input channel")
    L.check(fragment, FRAGMENTS, "O input fragment")
    L.check(query, QUERIES, "O input query")
    bank=source_bank(fragment,2)
    return channel,1,bank//4,bank%4,input_offset(fragment,query,16,2)


def input_packet(channel, fragment, query):
    location=input_location(channel,fragment,query)
    return dict(channel=channel,head=channel//2,parity=channel%2,query=query,
        source_flat_bank=fragment,source_accumulator_first=query*16,
        source_dtype="FP32",output_dtype="BF16",bytes=32,
        destination=location,destination_address=L.physical_address(*location),
        elements=[dict(feature=input_feature(fragment,lane),source_accumulator_index=query*16+lane,
                       byte_in_packet=lane*2) for lane in range(16)])


def weight_location(head, parity, reduction, output):
    L.check(head, HEADS, "O head")
    L.check(parity, 2, "O rank parity")
    L.check(reduction, INPUT_FEATURES, "O reduction feature")
    L.check(output, OUTPUT_FEATURES, "O output feature")
    tile,inside=divmod(output,512)
    bank,column=divmod(inside,COLUMNS)
    fragment=reduction%128//8
    lane=2*(reduction%8)+reduction//128
    offset=WEIGHT_BASE+tile*WEIGHT_TILE_BYTES+(fragment*COLUMNS+column)*32+lane*2
    return 2*head+parity,1,bank//4,bank%4,offset


def decode_weight(location):
    channel,rank,bg,ba,offset=location
    if rank!=1 or not 0<=channel<16 or not 0<=bg<4 or not 0<=ba<4:
        raise ValueError("Invalid O weight owner")
    relative=offset-WEIGHT_BASE
    if relative<0 or relative>=2*WEIGHT_TILE_BYTES or relative%2:
        raise ValueError("Invalid O weight address")
    tile,inside=divmod(relative,WEIGHT_TILE_BYTES)
    column,lane_byte=divmod(inside,32)
    fragment,slot=divmod(column,COLUMNS)
    return channel//2,channel%2,input_feature(fragment,lane_byte//2),tile*512+(bg*4+ba)*32+slot


def resident_weight_location(channel, global_reduction, output):
    """Locate one cell of the complete [2048,1024] Wo copy in a PIM rank.

    Each rank rotates the eight head blocks so its active head occupies slot0.
    Thus existing MAC addresses remain unchanged; slots1..7 are resident but
    are not read by this rank's head-partial projection.
    """
    L.check(channel, RANKS, "resident O channel")
    L.check(global_reduction, HEADS*INPUT_FEATURES, "global O reduction feature")
    L.check(output, OUTPUT_FEATURES, "O output feature")
    head,feature=divmod(global_reduction,INPUT_FEATURES)
    owner,parity=divmod(channel,2)
    slot=(head-owner)%HEADS
    location=weight_location(owner,parity,feature,output)
    return (*location[:-1],location[-1]+slot*WEIGHT_HEAD_BYTES_PER_BANK)


def decode_resident_weight(location):
    """Invert the resident mapping to (channel, global Wo row, output)."""
    channel,rank,bg,bank,offset=location
    if rank!=1 or not 0<=channel<RANKS or not 0<=bg<4 or not 0<=bank<4:
        raise ValueError("Invalid resident O weight owner")
    relative=offset-WEIGHT_BASE
    if relative<0 or relative>=RESIDENT_WEIGHT_BYTES_PER_BANK or relative%2:
        raise ValueError("Invalid resident O weight address")
    slot,within=divmod(relative,WEIGHT_HEAD_BYTES_PER_BANK)
    owner,_,feature,output=decode_weight((channel,rank,bg,bank,WEIGHT_BASE+within))
    head=(owner+slot)%HEADS
    return channel,head*INPUT_FEATURES+feature,output


def result_element(channel, tile, query, output):
    L.check(channel, RANKS, "O result channel")
    L.check(tile, TILES, "O output tile")
    L.check(query, QUERIES, "O result query")
    L.check(output, OUTPUT_FEATURES, "O output feature")
    if output//512!=tile:
        raise ValueError("Output feature does not belong to requested tile")
    bank,column=divmod(output%512,COLUMNS)
    offset=((query*16+bank)*COLUMNS+column)*2
    return dict(head=channel//2,parity=channel%2,channel=channel,physical_rank=1,
        query=query,output=output,tile=tile,flat_bank=bank,output_column=column,
        accumulator_index=query*COLUMNS+column,accumulator_dtype="FP32",dtype="BF16",
        result_offset_bytes=offset,packet_offset_bytes=offset//32*32,byte_in_packet=offset%32,
        trace_address=L.physical_address(channel,1,0,0,offset//32*32))


def contract():
    return dict(schema_version=2,enabled=True,logical_operation="Y=sum(head,parity, PV_partial[head,parity] @ Wo[head])",
        input_shape_per_rank=[16,256],weight_shape_per_rank=[2048,1024],
        active_weight_shape_per_rank=[256,1024],output_shape_per_rank=[16,1024],
        input_dtype="BF16",weight_dtype="BF16",accumulator_dtype="FP32",output_dtype="BF16",
        input=dict(base_bytes=INPUT_BASE,bytes_per_rank=8192,source_banks=[0,4],
            source="PIM PV FP32 accumulators cast to BF16 and gathered into two DRAM banks",
            layout="[fragment//2][query16][lane16] BF16 per source bank",
            fragment_feature_order=[[input_feature(f,l) for l in range(16)] for f in range(16)],
            bank_formula="4*(fragment%2)",offset_formula="524288+(fragment//2*16+query)*32",
            fills=1,reads=2,external_input_bytes=0),
        weights=dict(base_bytes=WEIGHT_BASE,end_bytes_exclusive=RESIDENT_WEIGHT_END,
            copies_per_head=16,bytes_per_bank=RESIDENT_WEIGHT_BYTES_PER_BANK,
            bytes_per_rank=4194304,total_bytes=67108864,
            active_copies_per_head=2,active_head_formula="channel//2",active_head_slot=0,
            active_end_bytes_exclusive=WEIGHT_BASE+WEIGHT_HEAD_BYTES_PER_BANK,
            active_bytes_per_bank=WEIGHT_HEAD_BYTES_PER_BANK,
            active_bytes_per_rank=524288,active_total_bytes=8388608,
            inactive_bytes_per_rank=3670016,inactive_total_bytes=58720256,
            resident_head_slot_formula="(weight_head-channel//2)%8",
            head_from_slot_formula="(channel//2+resident_head_slot)%8",
            order="[resident_head_slot8][output_tile2][input_fragment16][output_column32][input_lane16] BF16 per bank",
            active_order="[output_tile2][input_fragment16][output_column32][input_lane16] BF16 per bank",
            output_feature_formula="tile*512+bank*32+column",
            input_feature_formula="8*fragment+(lane//2)+128*(lane%2)",
            global_input_feature_formula="((channel//2+resident_head_slot)%8)*256+8*fragment+(lane//2)+128*(lane%2)",
            initialization_excluded=True,initialization_bytes_per_rank=4194304,
            initialization_total_bytes=67108864,initialization_latency_us=None,
            inactive_weight_reads_per_step=0,
            residency_scope="Post-run capacity/layout extension: full copies are assumed preinstalled. Native timing and address traces validate only the active head block; inactive initialization is not simulated."),
        output=dict(tiles=2,columns_per_bank_per_tile=32,accumulator_bytes_per_bank=2048,
            bytes_per_rank_per_tile=16384,bytes_per_rank=32768,total_bytes=524288,
            order="[tile2][query16][bank16][output_column32] BF16",
            gpu_reduction_operands=16,gpu_reduction_latency_us=None,
            bias="Apply once after reduction, if present"),
        scalar_macs=16*16*256*1024, numerical_compute=False,
        precision_note="All four variants round PV rank partials to BF16. O-enabled variants additionally round each Wo rank partial to BF16 before the final GPU sum; this is not bitwise equivalent to one FP32 projection after the GPU PV sum. The historical FP32 PV export remains a separate legacy control.")

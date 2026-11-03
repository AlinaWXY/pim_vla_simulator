"""Whole-horizon V/PV: buffers scale with H; weights and feature owners do not.

All valid V outputs cross the existing GPU conversion/packing boundary before
one cache tail read-modify-write. Physical KV padding is shared with Q/K.
"""
from collections import Counter
from pathlib import Path
from rank_staging import stage_input, input_offset, source_bank, fragment_metadata, reservation, fragment_batches, prefetch_metadata


HEADS, CHANNELS, BANKS_PER_RANK, HEAD_DIM, HIDDEN = 8, 16, 16, 256, 1024
PREFIX, RANK, BURST, ROW, FEATURES = 968, 1, 32, 2048, 8
SUPPORTED_HORIZONS = (16,)


def align(value, multiple):
    return (value+multiple-1)//multiple*multiple


def physical_address(channel, rank, bg, bank, offset):
    if not (0 <= channel < 16 and rank in (0, 1) and 0 <= bg < 4 and 0 <= bank < 4
            and isinstance(offset, int) and offset >= 0):
        raise ValueError((channel, rank, bg, bank, offset))
    row, inside = divmod(offset, ROW)
    column, byte = divmod(inside, BURST)
    return row << 20 | bank << 18 | bg << 16 | rank << 15 | column << 9 | channel << 5 | byte


def layout(horizon, v_token_sharding=False, pv_token_sharding=False):
    if pv_token_sharding and not v_token_sharding:
        raise ValueError('PV token sharding requires token-sharded V')
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f'Unsupported horizon: {horizon}')
    padded = align(PREFIX+horizon, 32)
    # Work holds H/2 coefficients, H/2 local Q, H/2 suffix K, then local K padding.
    work_columns = horizon+padded//2-PREFIX//2
    from horizon_attention import layout_for
    attention = layout_for(horizon)
    attention_end = max(r['base']+r['reserved_bytes_per_bank'] for r in attention['regions'].values())
    if attention_end != 81920+align(work_columns*BURST, ROW):
        raise ValueError('Shared attention work layout disagrees with V region placement')
    regions = {}
    base = attention_end
    for name, used in (('w_v', HIDDEN*FEATURES*2), ('v_cache', padded*FEATURES*2)):
        reserved = align(used, ROW)
        regions[name] = dict(base=base, used_bytes_per_bank=used,
            reserved_bytes_per_bank=reserved, row_first=base//ROW, row_last=(base+reserved-1)//ROW)
        base += reserved
    if base > 256*1024:
        raise ValueError('V regions overlap the FFN weight reservation')
    result = dict(horizon=horizon, sequence=PREFIX+horizon, padded_sequence=padded,
        prefix=PREFIX, kv_fragments=padded//16, preceding_attention_end=attention_end,
        attention_work_columns=work_columns, regions=regions, end_before_ffn=base,
        ffn_reserved_base=256*1024, channels=16, physical_ranks_per_channel=2,
        active_rank=1, active_banks=256, features_per_bank=8, banks_per_head=32,
        feature_formula='d=32*slot+16*channel_in_head+4*BG+bank',
        cache_order='[KV_fragment][feature_slot8][token_lane16]',
        tail_block_starts=list(range(PREFIX//16*16, padded, 16)),
        preserved_prefix_tokens=list(range(960, 968)),
        suffix_tokens=[PREFIX, PREFIX+horizon], padding_tokens=[PREFIX+horizon, padded],
        weight_bytes=8*1024*256*2, cache_payload_bytes=8*padded*256*2,
        input_double_buffer_bytes_per_bank=2*horizon*32, separate_rank_input_bytes=0,
        accumulator_bytes_per_bank=horizon*8*4,
        execution='one whole-H batch; no query microbatch or query padding')
    if v_token_sharding:
        import v_token_layout as token
        result['regions']['w_v'] = dict(base=token.WEIGHT_BASE,
            used_bytes_per_bank=token.WEIGHT_BYTES, reserved_bytes_per_bank=token.WEIGHT_BYTES,
            row_first=token.WEIGHT_BASE//ROW, row_last=(token.WEIGHT_BASE+token.WEIGHT_BYTES-1)//ROW)
        result.update(v_token_sharding=True, features_per_bank=16, input_rows_per_rank=8,
            projection_weight_copies=2, weight_bytes=2*HEADS*HIDDEN*HEAD_DIM*2,
            cache_features_per_bank=8, cache_feature_formula=result['feature_formula'],
            feature_formula='pair=feature%128; bank=pair//8; lane=2*(pair%8)+feature//128',
            token_owner_formula='channel=2*head+token%2; physical_rank=1; local_token=token//2',
            end_before_ffn=max(base, token.WEIGHT_BASE+token.WEIGHT_BYTES),
            used_input_bytes_per_slot=256, used_input_bytes_per_bank=512,
            execution='One token-parity shard per rank; all H tokens covered once per head')
        if result['regions']['v_cache']['base'] != token.CACHE_BASE:
            raise ValueError('Token V must preserve the existing PV cache base')
    if pv_token_sharding:
        from pv_token_layout import CACHE_BASE, CACHE_BYTES, cache_contract
        result['regions']['v_cache']=dict(base=CACHE_BASE,used_bytes_per_bank=31*16*32,
            reserved_bytes_per_bank=CACHE_BYTES,row_first=80,row_last=87)
        result.update(pv_token_sharding=True,cache_features_per_bank=16,banks_per_head=32,
            cache_feature_formula=result['feature_formula'],
            cache_order='[local_KV_fragment31][feature_lane16][local_KV_lane16]',
            cache_token_owner_formula='channel=2*head+token%2; local_token=token//2',
            kv_fragments=31,local_cache_tokens=496,end_before_ffn=CACHE_BASE+CACHE_BYTES,
            cache_mapping=cache_contract(),input_srf_bytes_per_bank=1024,
            output_pool_bytes_per_bank=2048,cache_writeback='bank-local RMW; no GPU V export/scatter',
            tail_block_starts=[480],tail_block_coordinates='rank-local token indices',
            preserved_prefix_tokens=list(range(480,484)),
            pv_accumulator_bytes_per_bank=1024,pv_query_batches=1,
            execution='Whole 1KiB SRF batches; V token parity and PV full-query rank-partial outputs')
        result.pop('input_double_buffer_bytes_per_bank',None)
    return result


def feature_owner(head, feature):
    if not (0 <= head < HEADS and 0 <= feature < HEAD_DIM):
        raise ValueError((head, feature))
    slot, physical_bank = divmod(feature, 32)
    channel_in_head, bank = divmod(physical_bank, 16)
    return (2*head+channel_in_head, RANK, bank//4, bank % 4), slot


def value_location(horizon, head, token, feature):
    geometry = layout(horizon)
    if not 0 <= token < geometry['padded_sequence']:
        raise ValueError(token)
    owner, slot = feature_owner(head, feature)
    fragment, lane = divmod(token, 16)
    return (*owner, geometry['regions']['v_cache']['base']+(fragment*8+slot)*32+lane*2)


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('w', encoding='utf-8', newline='\n')
        self.counts = Counter()
        self.segments, self.input_fragments, self.exports = [], [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, operation, address, buffer=None):
        self.stream.write(f'{operation} {address}'+('' if buffer is None else f' {buffer}')+'\n')
        self.line_no += 1
        self.counts[operation] += 1

    def fence(self):
        marker = self.marker
        self.marker += 1
        self.emit('BARRIER', marker)
        return marker

    def phase(self, name, callback, **metadata):
        first, before = self.line_no+1, self.counts.copy()
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
            line_start=first, line_end=self.line_no, counts=dict(self.counts-before), **metadata))
        self.boundary = end

    def ranks(self, operation, offset, buffer=None, bank=0):
        for channel in range(16):
            self.emit(operation, physical_address(channel, 1, bank//4, bank%4, offset), buffer)


def generate(path, horizon, case, rank_staging=False, staging_banks=1, v_token_sharding=False,
             pv_token_sharding=False, rank_descriptors=False):
    if pv_token_sharding:
        if not v_token_sharding:
            raise ValueError('PV token sharding requires token-sharded V')
        from pv_token_workload import generate as token_generate
        return token_generate(path,horizon,case,rank_staging,staging_banks,rank_descriptors)
    if case not in ('v', 'pv'):
        raise ValueError(case)
    token_sharded = bool(v_token_sharding and case == 'v')
    geometry = layout(horizon, v_token_sharding=token_sharded)
    geometry['input_staging'] = reservation(staging_banks) if rank_staging else None
    input_rows, output_slots = (horizon//2, 16) if token_sharded else (horizon, 8)
    fragments = 64 if case == 'v' else geometry['kv_fragments']
    base = geometry['regions']['w_v' if case == 'v' else 'v_cache']['base']
    matrix = dict(M=horizon, K=1024 if case == 'v' else PREFIX+horizon,
        N=2048 if case == 'v' else 256, heads=1 if case == 'v' else 8,
        padded_K=1024 if case == 'v' else geometry['padded_sequence'],
        padded_N=2048 if case == 'v' else 256, slots_per_bank=output_slots,
        base=base, mac_service_ck=input_rows)
    if token_sharded:
        matrix.update(local_rows_per_rank=input_rows, output_features_per_bank=output_slots,
                      rank_partition='token parity', weight_copies_per_head=2)
    matrix['logical_macs'] = matrix['heads']*matrix['M']*matrix['K']*matrix['N']
    matrix['padded_macs'] = matrix['heads']*matrix['M']*matrix['padded_K']*matrix['N']
    w = Writer(path)
    try:
        if rank_staging:
            stage_input(w, case, input_rows, fragments, staging_banks)
            if token_sharded:
                from v_token_layout import input_contract
                w.staging_inputs[-1]['logical_input'] = input_contract()
        def compute():
            for batch in fragment_batches(fragments, rank_staging):
                inputs = {}
                for fragment in batch:
                    buffer, first = fragment%2, w.line_no+1
                    if rank_staging:
                        w.ranks('PIM_STAGE_FRAGMENT', input_offset(fragment, 0, input_rows, staging_banks), buffer,
                            bank=source_bank(fragment, staging_banks))
                    else:
                        for query in range(input_rows):
                            w.ranks('PIM_WR_GB', query*32, buffer)
                    inputs[fragment] = (buffer, first, w.line_no)
                for fragment in batch:
                    buffer, first, last = inputs[fragment]
                    mac_first = w.line_no+1
                    for slot in range(output_slots):
                        w.ranks('PIM_MAC_AB', base+(fragment*output_slots+slot)*32, buffer)
                    w.input_fragments.append(dict(operator=case, fragment=fragment, buffer=buffer,
                        line_start=first, line_end=last, mac_line_first=mac_first, mac_line_last=w.line_no,
                        command_count=16 if rank_staging else input_rows*16, external_bytes=0 if rank_staging else input_rows*16*32,
                        internal_load_bursts=input_rows*16 if rank_staging else 0,
                        query_count=input_rows, source='X' if case == 'v' else 'post-softmax P',
                        **(dict(logical_token_formula='2*local_token+channel%2',
                                input_feature_first=fragment*16, input_feature_count=16,
                                output_feature_slots=output_slots) if token_sharded else {}),
                        **(prefetch_metadata(batch, input_rows) if rank_staging else {}),
                        **(fragment_metadata(fragment, input_rows, staging_banks) if rank_staging else {})))
            w.fence()
        w.phase(f'{case}_local_input_compute' if rank_staging else f'{case}_input_compute', compute, kind='local_compute' if rank_staging else 'pipeline', operator=case,
                mac_service_ck=input_rows, native_mac_completion=True, query_count=input_rows)
        def release():
            for slot in (0, 1):
                w.ranks('PIM_GEMV_RELEASE', 0, slot)
        w.phase(f'{case}_gemv_release', release, kind='gemv_release')

        def export():
            first = w.line_no+1
            for offset in range(0, input_rows*16*output_slots*4, 32):
                w.ranks('PIM_RD_RES', offset)
            w.exports.append(dict(phase=f'{case}_result_export', line_start=first, line_end=w.line_no,
                bytes=(w.line_no-first+1)*32, query_count=input_rows, dtype='FP32',
                **(dict(bytes_per_rank=8192, output_order='[local_token8][bank16][feature_lane16]',
                        payload_contract='v_token_mapping.export', elements_per_packet=8) if token_sharded else {})))
            w.fence()
        w.phase(f'{case}_result_export', export, kind='external_read', output_dtype='FP32')
        if case == 'v':
            def read_tail():
                for feature in range(256):
                    for head in range(8):
                        w.emit('LD', physical_address(*value_location(horizon, head, 960, feature)))
            def write_tail():
                for token in geometry['tail_block_starts']:
                    for feature in range(256):
                        for head in range(8):
                            w.emit('ST', physical_address(*value_location(horizon, head, token, feature)))
            w.phase('v_prefix_tail_read', read_tail, kind='dram_read', preserved_prefix_tokens='960..967')
            w.phase('v_suffix_write', write_tail, kind='dram_write',
                valid_suffix=[968, 968+horizon], masked_padding=[968+horizon, geometry['padded_sequence']],
                required_unmeasured_stage='GPU BF16 conversion/packing and prefix-tail merge once for all H tokens')
    finally:
        w.stream.close()
    dq = dict(pim_input_write=(fragments*input_rows*16 if rank_staging else w.counts['PIM_WR_GB'])*32,
              pim_result_read=w.counts['PIM_RD_RES']*32,
              dram_read=w.counts['LD']*32, dram_write=(w.counts['ST']-(fragments*input_rows*16 if rank_staging else 0))*32)
    dq['total'] = sum(dq.values())
    result = dict(case=case, horizon=horizon, action_horizon=horizon, attention_heads=8,
        schedule='double', matrix=matrix, layout=geometry, counts=dict(w.counts),
        segments=w.segments, input_fragments=w.input_fragments, exports=w.exports,
        staging_inputs=getattr(w, 'staging_inputs', []), rank_staging=rank_staging,
        staging_banks=staging_banks if rank_staging else 0,
        v_token_sharding=token_sharded,
        expected_dq_bytes=dq, expected_internal_array_read=w.counts['PIM_MAC_AB']*16*32,
        buffers=dict(input_double_per_bank=64*horizon, separate_rank_input_bytes=0,
            accumulator_per_bank=input_rows*output_slots*4,
            **(dict(input_rows=input_rows, used_input_bytes_per_slot=input_rows*32) if token_sharded else {})),
        required_backend=dict(pim_horizon=horizon, pim_rank_rope=False, pim_ffn_dram=False,
            pim_input_broadcast=False, pim_mac_scope='rank', pim_data_path='attacc', nCCDAB=input_rows),
        physical_active_rank=1, execution=geometry['execution'] if token_sharded else 'one whole-H batch with buffers expanded to H',
        query_padding=0, kv_padding=geometry['padded_sequence']-968-horizon,
        gpu_boundary='V cast/packing, prefix RMW and post-softmax P availability remain unmeasured',
        cache_precondition='Full prefix resident; PV sees all H suffix tokens and masked physical KV padding',
        unknown_latency_us=dict(GPU_V_cast_pack=None, GPU_softmax=None, GPU_source_DMA_arrival=None),
        numeric_compute=False, complete_ae_latency=False)
    if token_sharded:
        from v_token_layout import mapping_contract
        result['v_token_mapping'] = mapping_contract(result)
    return result

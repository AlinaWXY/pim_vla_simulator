"""Whole-horizon V/PV: buffers scale with H; weights and feature owners do not.

All valid V outputs cross the existing GPU conversion/packing boundary before
one cache tail read-modify-write. Physical KV padding is shared with Q/K.
"""
from collections import Counter
from pathlib import Path
from rank_staging import stage_input, input_offset, reservation, fragment_batches, prefetch_metadata


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


def layout(horizon):
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
    return dict(horizon=horizon, sequence=PREFIX+horizon, padded_sequence=padded,
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

    def ranks(self, operation, offset, buffer=None):
        for channel in range(16):
            self.emit(operation, physical_address(channel, 1, 0, 0, offset), buffer)


def generate(path, horizon, case, rank_staging=False):
    if case not in ('v', 'pv'):
        raise ValueError(case)
    geometry = layout(horizon)
    geometry['input_staging'] = reservation() if rank_staging else None
    fragments = 64 if case == 'v' else geometry['kv_fragments']
    base = geometry['regions']['w_v' if case == 'v' else 'v_cache']['base']
    matrix = dict(M=horizon, K=1024 if case == 'v' else PREFIX+horizon,
        N=2048 if case == 'v' else 256, heads=1 if case == 'v' else 8,
        padded_K=1024 if case == 'v' else geometry['padded_sequence'],
        padded_N=2048 if case == 'v' else 256, slots_per_bank=8,
        base=base, mac_service_ck=horizon)
    matrix['logical_macs'] = matrix['heads']*matrix['M']*matrix['K']*matrix['N']
    matrix['padded_macs'] = matrix['heads']*matrix['M']*matrix['padded_K']*matrix['N']
    w = Writer(path)
    try:
        if rank_staging:
            stage_input(w, case, horizon, fragments)
        def compute():
            for batch in fragment_batches(fragments, rank_staging):
                inputs = {}
                for fragment in batch:
                    buffer, first = fragment%2, w.line_no+1
                    if rank_staging:
                        w.ranks('PIM_STAGE_FRAGMENT', input_offset(fragment, 0, horizon), buffer)
                    else:
                        for query in range(horizon):
                            w.ranks('PIM_WR_GB', query*32, buffer)
                    inputs[fragment] = (buffer, first, w.line_no)
                for fragment in batch:
                    buffer, first, last = inputs[fragment]
                    mac_first = w.line_no+1
                    for slot in range(8):
                        w.ranks('PIM_MAC_AB', base+(fragment*8+slot)*32, buffer)
                    w.input_fragments.append(dict(operator=case, fragment=fragment, buffer=buffer,
                        line_start=first, line_end=last, mac_line_first=mac_first, mac_line_last=w.line_no,
                        command_count=16 if rank_staging else horizon*16, external_bytes=0 if rank_staging else horizon*16*32,
                        internal_load_bursts=horizon*16 if rank_staging else 0,
                        query_count=horizon, source='X' if case == 'v' else 'post-softmax P',
                        **(prefetch_metadata(batch, horizon) if rank_staging else {})))
            w.fence()
        w.phase(f'{case}_local_input_compute' if rank_staging else f'{case}_input_compute', compute, kind='local_compute' if rank_staging else 'pipeline', operator=case,
                mac_service_ck=horizon, native_mac_completion=True, query_count=horizon)
        def release():
            for slot in (0, 1):
                w.ranks('PIM_GEMV_RELEASE', 0, slot)
        w.phase(f'{case}_gemv_release', release, kind='gemv_release')

        def export():
            first = w.line_no+1
            for offset in range(0, horizon*16*8*4, 32):
                w.ranks('PIM_RD_RES', offset)
            w.exports.append(dict(phase=f'{case}_result_export', line_start=first, line_end=w.line_no,
                bytes=(w.line_no-first+1)*32, query_count=horizon, dtype='FP32'))
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
    dq = dict(pim_input_write=(fragments*horizon*16 if rank_staging else w.counts['PIM_WR_GB'])*32,
              pim_result_read=w.counts['PIM_RD_RES']*32,
              dram_read=w.counts['LD']*32, dram_write=(w.counts['ST']-(fragments*horizon*16 if rank_staging else 0))*32)
    dq['total'] = sum(dq.values())
    return dict(case=case, horizon=horizon, action_horizon=horizon, attention_heads=8,
        schedule='double', matrix=matrix, layout=geometry, counts=dict(w.counts),
        segments=w.segments, input_fragments=w.input_fragments, exports=w.exports,
        staging_inputs=getattr(w, 'staging_inputs', []), rank_staging=rank_staging,
        expected_dq_bytes=dq, expected_internal_array_read=w.counts['PIM_MAC_AB']*16*32,
        buffers=dict(input_double_per_bank=64*horizon, separate_rank_input_bytes=0,
            accumulator_per_bank=32*horizon),
        required_backend=dict(pim_horizon=horizon, pim_rank_rope=False, pim_ffn_dram=False,
            pim_input_broadcast=False, pim_mac_scope='rank', pim_data_path='attacc', nCCDAB=horizon),
        physical_active_rank=1, execution='one whole-H batch with buffers expanded to H',
        query_padding=0, kv_padding=geometry['padded_sequence']-968-horizon,
        gpu_boundary='V cast/packing, prefix RMW and post-softmax P availability remain unmeasured',
        cache_precondition='Full prefix resident; PV sees all H suffix tokens and masked physical KV padding',
        unknown_latency_us=dict(GPU_V_cast_pack=None, GPU_softmax=None, GPU_source_DMA_arrival=None),
        numeric_compute=False, complete_ae_latency=False)

"""H=16 action horizon: fused QKV projection, explicit GPU RoPE transfers, DRAM-resident Q.

Native trace timing covers controller/DRAM/PIM service. GPU RoPE, conversion,
packing and arrival arithmetic are explicit unmeasured dependency boundaries.
No latency in this workload substitutes for those required stages.
"""
from collections import Counter, defaultdict
from pathlib import Path

import attention_layout as L

OPERATORS = ('qkv', 'qk', 'pv')
CONTROL_OPERATORS = ('q', 'k', 'v')
NATIVE_REQUEST_MAP = {
    'MACK': 'PIM_MAC_K', 'MACQK': 'PIM_MAC_QK',
    'WRGB': 'PIM_WR_GB', 'MACAB': 'PIM_MAC_AB', 'RDRES': 'PIM_RD_RES',
    'RD': 'LD', 'WR': 'ST',
}
NATIVE_REQUEST_GROUPS = {
    'WRGB': ('PIM_WR_BG', 'PIM_WR_Q', 'PIM_WR_GB', 'PIM_RD_Q'),
    'RDRES': ('PIM_RD_BG', 'PIM_RD_RES'),
    'MACK': ('PIM_MAC_K',), 'MACQK': ('PIM_MAC_QK',),
    'MACAB': ('PIM_MAC_AB',), 'RD': ('LD', 'PIM_RD_Q'), 'WR': ('ST',),
}
MOVEMENT_TAIL_CK = 4


class Writer:
    def __init__(self, path):
        self.stream = Path(path).open('w', newline='\n')
        self.counts = Counter()
        self.segments, self.input_fragments, self.exports = [], [], []
        self.local_writes, self.dependencies = [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, op, address, buffer=None):
        self.stream.write(f'{op} {address}' + ('' if buffer is None else f' {buffer}') + '\n')
        self.counts[op] += 1
        self.line_no += 1

    def fence(self):
        marker = self.marker
        self.marker += 1
        self.emit('BARRIER', marker)
        return marker

    def wait(self, cycles):
        if cycles:
            self.emit('WAIT', cycles)

    def phase(self, name, callback, **metadata):
        before, first = self.counts.copy(), self.line_no + 1
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
            line_start=first, line_end=self.line_no, counts=dict(self.counts-before), **metadata))
        self.boundary = end

    def all_channels(self, command, offset, buffer=None):
        for channel in range(L.CHANNELS):
            self.emit(command, L.physical_address(channel, 0, 0, 0, offset), buffer)

    def drain_mac(self, native_completion, service):
        self.fence()
        # Legacy MACAB retires at issue, while new MACK/MACQK retire at completion.
        if not native_completion:
            self.wait(service)

    def close(self):
        self.stream.close()


def matrix_for(op):
    if op in ('qkv', 'q', 'k', 'v'):
        multiplicity = 3 if op == 'qkv' else 1
        k, n, heads = L.HIDDEN, multiplicity * L.HEADS * L.HEAD_DIM, 1
        padded_k, padded_n = k, n
        region, slots = L.REGIONS['w_qkv'], multiplicity * 4
    elif op == 'qk':
        k, n, padded_k, padded_n, heads = L.HEAD_DIM, L.SEQUENCE, L.HEAD_DIM, L.PADDED_TOKENS, L.HEADS
        region, slots = L.REGIONS['k_cache'], L.TOKEN_SLOTS
    elif op == 'pv':
        k, n, padded_k, padded_n, heads = L.SEQUENCE, L.HEAD_DIM, L.PADDED_TOKENS, L.HEAD_DIM, L.HEADS
        region, slots = L.REGIONS['v_cache'], 4
    else:
        raise ValueError(op)
    return dict(M=L.QUERIES, K=k, N=n, padded_K=padded_k, padded_N=padded_n,
        heads=heads, slots=slots, slots_per_bank=slots, base=region['base'],
        logical_macs=L.QUERIES*k*n*heads,
        padded_macs=L.QUERIES*padded_k*padded_n*heads,
        bytes_per_bank=(L.HIDDEN*slots*2 if op in ('qkv','q','k','v') else region['used_bytes_per_bank']),
        mac_service_ck=L.QUERIES, weight_copies_per_head=1 if op in ('qkv','q','k','v') else None)


def _fill(w, op, fragment, schedule, phase):
    buffer = fragment % 2 if schedule == 'double' else 0
    first = w.line_no+1
    if op == 'qk':
        for bank in range(L.BANKS_PER_BG):
            for query in range(L.QUERIES):
                feature = L.k_feature_inverse(bank, fragment*L.LANES)
                for channel in range(L.CHANNELS):
                    location = L.query_location(channel//2, channel%2, query, feature)
                    w.emit('PIM_RD_Q', L.physical_address(*location), buffer)
        command = 'PIM_RD_Q'
    else:
        for query in range(L.QUERIES):
            w.all_channels('PIM_WR_GB', query*L.BURST_BYTES, buffer)
        command = 'PIM_WR_GB'
    count = w.line_no-first+1
    record = dict(phase=phase, operator=op, fragment=fragment, buffer=buffer,
        input_command=command, line_start=first, line_end=w.line_no,
        command_count=count, external_bytes=0 if op=='qk' else count*L.BURST_BYTES,
        dram_source_bytes=count*L.BURST_BYTES if op=='qk' else 0,
        relay_bytes=count*L.BURST_BYTES if op=='qk' else 0)
    w.input_fragments.append(record)
    return buffer, record


def _macs(w, op, fragment, buffer, start_slot, stop_slot):
    base = matrix_for(op)['base']
    command = 'PIM_MAC_QK' if op == 'qk' else 'PIM_MAC_AB'
    for slot in range(start_slot, stop_slot):
        if op == 'qk':
            offset = base + L.BURST_BYTES*(slot*4+fragment)
        elif op in ('qkv','q','k','v'):
            projection_slot = slot if op == 'qkv' else ('q','k','v').index(op)*4+slot
            offset = base + L.BURST_BYTES*(fragment*12+projection_slot)
        else:
            offset = base + L.BURST_BYTES*(fragment*matrix_for(op)['slots']+slot)
        w.all_channels(command, offset, buffer)


def projection(w, op, schedule, prefix=''):
    matrix = matrix_for(op)
    phase = prefix+'input_and_compute'
    def body():
        for fragment in range(L.HIDDEN//L.LANES):
            buffer, record = _fill(w, op, fragment, schedule, phase)
            record['mac_line_first'] = w.line_no+1
            _macs(w, op, fragment, buffer, 0, matrix['slots'])
            record['mac_line_last'] = w.line_no
            if schedule == 'serial':
                w.drain_mac(False, matrix['mac_service_ck'])
        w.drain_mac(False, matrix['mac_service_ck'])
    w.phase(phase, body, kind='pipeline', operator=op,
        mac_service_ck=L.QUERIES, native_mac_completion=False,
        shared_input_matrices=['q','k','v'] if op=='qkv' else [op])


def k_suffix_bursts():
    # Interleave physical destinations for the ordinary GPU-origin ST stream.
    # H=16 places one suffix token in every BG: four aligned 32B bursts/bank.
    by_bank = defaultdict(list)
    for head in range(L.HEADS):
        for query in range(L.QUERIES):
            for bank in range(L.BANKS_PER_BG):
                for fragment in range(4):
                    feature = L.k_feature_inverse(bank, fragment*L.LANES)
                    location = L.key_location(head, L.PREFIX+query, feature)
                    by_bank[location[:4]].append(location)
    banks = sorted(by_bank, key=lambda p: (p[1], p[2], p[3], p[0]))
    for burst in range(max(map(len, by_bank.values()))):
        for bank in banks:
            if burst < len(by_bank[bank]):
                yield by_bank[bank][burst]


def q_writeback(w, prefix=''):
    def body():
        for fragment in range(4):
            for bank in range(4):
                feature = L.k_feature_inverse(bank, fragment*16)
                for query in range(L.QUERIES):
                    for channel in range(L.CHANNELS):
                        w.emit('ST', L.physical_address(*L.query_location(
                            channel//2,channel%2,query,feature)))
    w.phase(prefix+'q_rope_dram_write', body, kind='dram_write',
        source='GPU post-RoPE BF16 Q, two complete channel copies per head',
        nonlinear_boundary='GPU projection-output rounding, Q RoPE, scaling and packing unmeasured')


def k_writeback(w, prefix=''):
    def body():
        for location in k_suffix_bursts():
            w.emit('ST', L.physical_address(*location))
    w.phase(prefix+'k_rope_dram_write', body, kind='dram_write',
        source='GPU post-RoPE BF16 K in token/BG layout',
        nonlinear_boundary='GPU projection-output rounding, K RoPE and packing unmeasured')


def _rank_export(w, *, name, slots=4, projection=None):
    def body():
        first = w.line_no+1
        for offset in range(slots*L.QUERIES*16*4//32):
            for rank in range(L.RANKS):
                for channel in range(L.CHANNELS):
                    w.emit('PIM_RD_RES', L.physical_address(channel, rank, 0, 0, offset*32))
        w.exports.append(dict(phase=name, line_start=first, line_end=w.line_no,
                             bytes=(w.line_no-first+1)*32, scope='rank bank-complete results', projection=projection))
        w.fence()
        w.wait(MOVEMENT_TAIL_CK)
    w.phase(name, body, kind='export')


def v_writeback(w, prefix='', include_export=True):
    if include_export:
        _rank_export(w, name=prefix+'v_result_export_to_gpu_boundary', projection='v')
    def read():
        for feature in range(L.HEAD_DIM):
            for head in range(L.HEADS):
                location = L.value_location(head, L.PREFIX//16*16, feature)
                w.emit('LD', L.physical_address(*location))
    def write():
        for token in range(L.PREFIX//16*16, L.PADDED_TOKENS, 16):
            for feature in range(L.HEAD_DIM):
                for head in range(L.HEADS):
                    w.emit('ST', L.physical_address(*L.value_location(head, token, feature)))
    w.phase(prefix+'v_prefix_tail_read', read, kind='dram_read')
    w.phase(prefix+'v_suffix_write', write, kind='dram_write',
            nonlinear_boundary='GPU BF16 conversion/repacking required and unmeasured')


def qk(w, schedule, prefix=''):
    # Fragment-outer keeps one Q DRAM read pass with native channel relay. 3968B/bank partial state
    # persists across four fragments. Final-fragment output drains at most
    # sixteen tokens (1024B) per BG before a later tile can reuse output SRAM.
    phase = prefix+'q_dram_input_and_compute'
    def first_three():
        for fragment in range(3):
            buffer, record = _fill(w, 'qk', fragment, schedule, phase)
            record['mac_line_first'] = w.line_no+1
            _macs(w, 'qk', fragment, buffer, 0, L.TOKEN_SLOTS)
            record['mac_line_last'] = w.line_no
            if schedule == 'serial':
                w.drain_mac(True, L.QUERIES)
        # Final Q fragment is loaded once and retained through all four tiles.
        _, record = _fill(w, 'qk', 3, schedule, phase)
        record['mac_line_first'] = record['mac_line_last'] = None
        w.drain_mac(True, L.QUERIES)
    w.phase(phase, first_three, kind='pipeline', operator='qk', mac_service_ck=L.QUERIES,
            native_mac_completion=True)
    buffer = 1 if schedule == 'double' else 0
    for tile, start in enumerate(range(0, L.TOKEN_SLOTS, 16)):
        stop = min(start+16, L.TOKEN_SLOTS)
        def compute(start=start, stop=stop):
            _macs(w, 'qk', 3, buffer, start, stop)
            w.drain_mac(True, L.QUERIES)
        w.phase(prefix+f'final_tile_{tile}_compute', compute, kind='pipeline',
                operator='qk', mac_service_ck=L.QUERIES, native_mac_completion=True,
                token_slot_start=start, token_slot_stop=stop, buffer=buffer)
        def export(start=start, stop=stop):
            first = w.line_no+1
            for offset in range(start*L.QUERIES*4, stop*L.QUERIES*4, 32):
                for rank in range(L.RANKS):
                    for bg in range(L.BGS):
                        for channel in range(L.CHANNELS):
                            w.emit('PIM_RD_BG', L.physical_address(channel, rank, bg, 0, offset))
            w.exports.append(dict(phase=prefix+f'score_tile_{tile}_export',
                line_start=first, line_end=w.line_no, bytes=(w.line_no-first+1)*32,
                scope='complete four-bank-reduced BG scores', token_slot_start=start,
                token_slot_stop=stop, maximum_staged_bytes_per_BG=(stop-start)*L.QUERIES*4))
        w.phase(prefix+f'score_tile_{tile}_export', export, kind='export',
                BG_reduction='implicit AttAcc throughput assumption; no extra network/adder penalty')


def pv(w, schedule, prefix=''):
    phase = prefix+'input_and_compute'
    def body():
        for fragment in range(L.PADDED_TOKENS//16):
            buffer, record = _fill(w, 'pv', fragment, schedule, phase)
            record['mac_line_first'] = w.line_no+1
            _macs(w, 'pv', fragment, buffer, 0, 4)
            record['mac_line_last'] = w.line_no
            if schedule == 'serial':
                w.drain_mac(False, L.QUERIES)
        w.drain_mac(False, L.QUERIES)
    w.phase(phase, body, kind='pipeline', operator='pv', mac_service_ck=L.QUERIES,
            native_mac_completion=False)
    _rank_export(w, name=prefix+'result_export_to_gpu_boundary')


def generate(path, op, mode='complete', schedule='double'):
    if op not in (*OPERATORS, *CONTROL_OPERATORS, 'chain_qkv_qk'):
        raise ValueError(op)
    if mode not in ('complete','kernel') or schedule not in ('double','serial'):
        raise ValueError((mode,schedule))
    w = Writer(path)
    try:
        if op in ('qkv','q','k','v','chain_qkv_qk'):
            projection_op = 'qkv' if op=='chain_qkv_qk' else op
            prefix = 'qkv_' if op=='chain_qkv_qk' else ''
            projection(w, projection_op, schedule, prefix)
            projected = ('q','k','v') if projection_op=='qkv' else (projection_op,)
            if mode=='complete':
                for item in projected:
                    _rank_export(w, name=prefix+item+'_result_export_to_gpu_boundary', projection=item)
                gpu_boundary = w.boundary
                if 'q' in projected:
                    q_writeback(w,prefix)
                if 'k' in projected:
                    k_writeback(w,prefix)
                if 'v' in projected:
                    v_writeback(w,prefix,include_export=False)
                w.dependencies.append(dict(producer=prefix+'projection_results_exported',
                    consumer=prefix+'GPU_postprocess_and_DRAM_stores', fence=gpu_boundary,
                    scope='native transfers only; required GPU transform not numerically executed',
                    unknown_required_stage='GPU Q/K RoPE, projection BF16 rounding, Q scaling and repacking; V cast/repacking'))
            if op=='chain_qkv_qk':
                w.dependencies.append(dict(producer='qkv_q_and_k_dram_stores',
                    consumer='qk_q_dram_input_and_compute', fence=w.boundary,
                    scope='native memory visibility and relay dependency only',
                    unknown_required_stage='GPU RoPE/cast/repacking before preceding stores'))
                qk(w,schedule,'qk_')
        elif op=='qk':
            qk(w,schedule)
        else:
            pv(w,schedule)
    finally:
        w.close()
    q_read = w.counts['PIM_RD_Q'] * 32
    return dict(case=op,mode=mode,schedule=schedule,variant='H=16',
        action_horizon=L.QUERIES,attention_heads=L.HEADS,counts=dict(w.counts),
        segments=w.segments,input_fragments=w.input_fragments,exports=w.exports,
        local_writes=w.local_writes,dependencies=w.dependencies,
        matrix=None if op=='chain_qkv_qk' else matrix_for(op),
        native_request_map=NATIVE_REQUEST_MAP,native_request_groups=NATIVE_REQUEST_GROUPS,
        native_expansion={'PIM_RD_Q':['RD','WRGB']},
        attention_reserved_bytes_per_bank=sum(r['reserved_bytes_per_bank'] for r in L.REGIONS.values()),
        new_native_mac_commands_retire_at_completion=True,
        external_input_bytes=w.counts['PIM_WR_GB']*32,
        external_output_bytes=(w.counts['PIM_RD_BG']+w.counts['PIM_RD_RES'])*32,
        q_dram_read_bytes=q_read,q_internal_broadcast_bytes=q_read,
        q_internal_delivered_bytes=q_read*L.RANKS*L.BGS,
        local_k_write_bytes=0,
        explicit_dram_read_bytes=w.counts['LD']*32,
        dram_read_bytes=(w.counts['LD']+w.counts['PIM_RD_Q'])*32,
        dram_write_bytes=w.counts['ST']*32,
        buffers=dict(q_per_bank=(2 if schedule=='double' else 1)*L.QUERIES*L.LANES*L.ELEMENT_BYTES,
            qk_partial_per_bank=L.QUERIES*L.TOKEN_SLOTS*4,
            qk_partial_reservation_per_bank=L.round_up(L.QUERIES*L.TOKEN_SLOTS*4,1024),
            qk_export_double_per_BG=2*L.LANES*L.QUERIES*4,
            qk_export_tile_per_BG=L.LANES*L.QUERIES*4,
            qkv_input_double_per_rank=(2 if schedule=='double' else 1)*L.QUERIES*L.LANES*L.ELEMENT_BYTES,
            qkv_projection_accumulator_per_bank=3*L.QUERIES*4*4,
            single_projection_accumulator_per_bank=L.QUERIES*4*4,
            v_input_double_per_rank=2*L.QUERIES*L.LANES*L.ELEMENT_BYTES,
            v_or_pv_accumulator_per_bank=L.QUERIES*4*4,
            q_dram_controller_relay_per_channel=32),
        buffer_allocation_note='H=16 action tokens; Q/K/V share each X fragment; three16x4 FP32 bank accumulators total768B. QK uses per-bank double512B input and a32B serialized controller relay.',
        qk_output_schedule='four Q fragments read from DRAM; final fragment tiles16tokens then BG export',
        multicast_model='ordinary Q DRAM RD, CL+burst completion, then internal WRGB relay; shared channel movement timing; internal fanout sustains command stream',
        BG_reduction_model='four-bank partials reduce locally; throughput implicit, independent adder/network timing unmeasured',
        unknown_latency_us=dict(GPU_QK_RoPE_cast_pack=None,GPU_softmax=None,
            GPU_V_cast_pack=None,GPU_O_projection=None,GPU_arrival_processing=None),
        input_precondition='GPU X/P available at controller; weights and prefix resident; standalone QK starts with post-RoPE Q/K already in DRAM',
        output_boundary='controller/LPDDR transfer completion; GPU arithmetic/arrival beyond boundary unmeasured',
        exclusions=['GPU Q/K RoPE, scaling, conversion and packing arithmetic',
            'GPU source DMA and arrival overhead beyond modeled transfers',
            'GPU V conversion/repacking arithmetic','prefix installation',
            'FFN, norms, residuals and action I/O'],
        integrated_numeric_chain=False,
        chain_semantics='native projection exports, GPU-origin Q/K stores and Q DRAM relay; GPU transforms explicit unmeasured boundaries')

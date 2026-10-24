"""GPU-Q / BG-sharded-K traces for the extended LPDDR Ramulator backend.

These traces time native memory/PIM service, not GPU arithmetic. Unknown K
RoPE/cast and GPU transformations are explicit dependency boundaries. They are
not assigned zero measured latency. The numerical ownership proof lives in
the companion layout tests; Ramulator itself does not evaluate tensor values.
"""
from collections import Counter, defaultdict
from pathlib import Path

import attention_layout as L

OPERATORS = ('k', 'v', 'qk', 'pv')
NATIVE_REQUEST_MAP = {
    'MACK': 'PIM_MAC_K', 'MACQK': 'PIM_MAC_QK',
    'WRGB': 'PIM_WR_GB', 'MACAB': 'PIM_MAC_AB', 'RDRES': 'PIM_RD_RES',
    'RD': 'LD', 'WR': 'ST',
}
NATIVE_REQUEST_GROUPS = {
    'WRGB': ('PIM_WR_BG', 'PIM_WR_Q', 'PIM_WR_GB'),
    'RDRES': ('PIM_RD_BG', 'PIM_RD_RES'),
    'MACK': ('PIM_MAC_K',), 'MACQK': ('PIM_MAC_QK',),
    'MACAB': ('PIM_MAC_AB',), 'RD': ('LD',), 'WR': ('ST',),
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
    if op in ('k', 'v'):
        k, n, padded_k, padded_n, heads = L.HIDDEN, L.HEADS*L.HEAD_DIM, L.HIDDEN, L.HEADS*L.HEAD_DIM, 1
        region = L.REGIONS['w_'+op]
        slots = L.FEATURES_PER_K_BANK if op == 'k' else 4
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
                bytes_per_bank=region['used_bytes_per_bank'],
                mac_service_ck=2 if op == 'k' else 20,
                weight_copies_per_head=16 if op == 'k' else 1 if op == 'v' else None)


def _fill(w, op, fragment, schedule, phase):
    buffer = fragment % 2 if schedule == 'double' else 0
    first = w.line_no+1
    if op == 'k':
        # Each channel receives its ten selected X tokens once; the target BG
        # receives one or two vectors and broadcasts each to its four banks.
        for rank in range(L.RANKS):
            for bg in range(L.BGS):
                for local_query in range(2):
                    for channel in range(L.CHANNELS):
                        queries = L.k_queries_for_group(channel % 2, rank, bg)
                        if local_query < len(queries):
                            w.emit('PIM_WR_BG', L.physical_address(channel, rank, bg, 0,
                                   local_query*L.BURST_BYTES), buffer)
        command = 'PIM_WR_BG'
    elif op == 'qk':
        for bank in range(L.BANKS_PER_BG):
            for query in range(L.QUERIES):
                for channel in range(L.CHANNELS):
                    w.emit('PIM_WR_Q', L.physical_address(channel, 0, 0, bank,
                           query*L.BURST_BYTES), buffer)
        command = 'PIM_WR_Q'
    else:
        for query in range(L.QUERIES):
            w.all_channels('PIM_WR_GB', query*L.BURST_BYTES, buffer)
        command = 'PIM_WR_GB'
    record = dict(phase=phase, fragment=fragment, buffer=buffer,
        input_command=command, line_start=first, line_end=w.line_no,
        command_count=w.line_no-first+1, external_bytes=(w.line_no-first+1)*L.BURST_BYTES)
    w.input_fragments.append(record)
    return buffer, record


def _macs(w, op, fragment, buffer, start_slot, stop_slot):
    base = matrix_for(op)['base']
    command = {'k': 'PIM_MAC_K', 'qk': 'PIM_MAC_QK'}.get(op, 'PIM_MAC_AB')
    for slot in range(start_slot, stop_slot):
        if op == 'qk':
            # K is token-major, then the four 16-feature local fragments.
            offset = base + L.BURST_BYTES*(slot*4+fragment)
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
                w.drain_mac(op == 'k', matrix['mac_service_ck'])
        w.drain_mac(op == 'k', matrix['mac_service_ck'])
    w.phase(phase, body, kind='pipeline', operator=op,
            mac_service_ck=matrix['mac_service_ck'], native_mac_completion=op == 'k')


def k_suffix_bursts():
    # Round-robin physical banks so a local independent bank-port model can
    # overlap writes to different banks. Each bank receives four/eight bursts.
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


def k_writeback(w, prefix=''):
    def body():
        first = w.line_no+1
        histogram = Counter()
        for location in k_suffix_bursts():
            w.emit('PIM_LOCAL_WR', L.physical_address(*location))
            histogram[location[:4]] += 32
        w.local_writes.append(dict(phase=prefix+'k_local_writeback', line_start=first,
            line_end=w.line_no, bytes=sum(histogram.values()), active_banks=len(histogram),
            per_bank_byte_histogram=dict(Counter(histogram.values())),
            source='producer-bank K projection accumulator after local RoPE/cast boundary'))
    w.phase(prefix+'k_local_writeback', body, kind='local_write',
            local_port_bytes=32, local_port_cycles=4,
            nonlinear_boundary='K RoPE, intermediate BF16 rounding and packing are required but unmeasured')


def _rank_export(w, *, name, slots=4):
    def body():
        first = w.line_no+1
        for offset in range(slots*L.QUERIES*16*4//32):
            for rank in range(L.RANKS):
                for channel in range(L.CHANNELS):
                    w.emit('PIM_RD_RES', L.physical_address(channel, rank, 0, 0, offset*32))
        w.exports.append(dict(phase=name, line_start=first, line_end=w.line_no,
                             bytes=(w.line_no-first+1)*32, scope='rank bank-complete results'))
        w.fence()
        w.wait(MOVEMENT_TAIL_CK)
    w.phase(name, body, kind='export')


def v_writeback(w, prefix=''):
    _rank_export(w, name=prefix+'result_export_to_gpu_boundary')
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
    # Fragment-outer keeps one Q transmission pass. 4960B/bank partial state
    # persists across four fragments. Final-fragment output drains at most
    # sixteen tokens (1280B) per BG before a later tile can reuse output SRAM.
    phase = prefix+'q_input_and_compute'
    def first_three():
        for fragment in range(3):
            buffer, record = _fill(w, 'qk', fragment, schedule, phase)
            record['mac_line_first'] = w.line_no+1
            _macs(w, 'qk', fragment, buffer, 0, L.TOKEN_SLOTS)
            record['mac_line_last'] = w.line_no
            if schedule == 'serial':
                w.drain_mac(True, 20)
        # Final Q fragment is loaded once and retained through all four tiles.
        _, record = _fill(w, 'qk', 3, schedule, phase)
        record['mac_line_first'] = record['mac_line_last'] = None
        w.drain_mac(True, 20)
    w.phase(phase, first_three, kind='pipeline', operator='qk', mac_service_ck=20,
            native_mac_completion=True)
    buffer = 1 if schedule == 'double' else 0
    for tile, start in enumerate(range(0, L.TOKEN_SLOTS, 16)):
        stop = min(start+16, L.TOKEN_SLOTS)
        def compute(start=start, stop=stop):
            _macs(w, 'qk', 3, buffer, start, stop)
            w.drain_mac(True, 20)
        w.phase(prefix+f'final_tile_{tile}_compute', compute, kind='pipeline',
                operator='qk', mac_service_ck=20, native_mac_completion=True,
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
                token_slot_stop=stop, maximum_staged_bytes_per_BG=(stop-start)*80))
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
                w.drain_mac(False, 20)
        w.drain_mac(False, 20)
    w.phase(phase, body, kind='pipeline', operator='pv', mac_service_ck=20,
            native_mac_completion=False)
    _rank_export(w, name=prefix+'result_export_to_gpu_boundary')


def generate(path, op, mode='complete', schedule='double'):
    if op not in (*OPERATORS, 'chain_k_qk'):
        raise ValueError(op)
    if mode not in ('complete', 'kernel') or schedule not in ('double', 'serial'):
        raise ValueError((mode, schedule))
    w = Writer(path)
    try:
        if op in ('k', 'chain_k_qk'):
            prefix = 'k_' if op == 'chain_k_qk' else ''
            projection(w, 'k', schedule, prefix)
            if mode == 'complete':
                k_writeback(w, prefix)
            if op == 'chain_k_qk':
                w.dependencies.append(dict(producer='k_local_writeback', consumer='qk_q_input_and_compute',
                    fence=w.boundary, scope='native memory dependency only',
                    unknown_required_stage='local K RoPE and conversion before local writes'))
                qk(w, schedule, 'qk_')
        elif op == 'v':
            projection(w, 'v', schedule)
            if mode == 'complete':
                v_writeback(w)
        elif op == 'qk':
            qk(w, schedule)
        else:
            pv(w, schedule)
    finally:
        w.close()
    return dict(case=op, mode=mode, schedule=schedule, counts=dict(w.counts),
        segments=w.segments, input_fragments=w.input_fragments, exports=w.exports,
        local_writes=w.local_writes, dependencies=w.dependencies,
        matrix=None if op == 'chain_k_qk' else matrix_for(op),
        native_request_map=NATIVE_REQUEST_MAP,
        native_request_groups=NATIVE_REQUEST_GROUPS,
        local_write_observation='PIM_LOCAL_WR bgk_events; no external CA/DQ command',
        new_native_mac_commands_retire_at_completion=True,
        external_input_bytes=(w.counts['PIM_WR_BG']+w.counts['PIM_WR_Q']+w.counts['PIM_WR_GB'])*32,
        external_output_bytes=(w.counts['PIM_RD_BG']+w.counts['PIM_RD_RES'])*32,
        local_k_write_bytes=w.counts['PIM_LOCAL_WR']*32,
        dram_read_bytes=w.counts['LD']*32, dram_write_bytes=w.counts['ST']*32,
        k_BG_query_counts={f'{c}:{r}:{bg}': len(L.k_queries_for_group(c,r,bg))
                           for c in range(2) for r in range(2) for bg in range(4)},
        buffers=dict(q_per_bank=1280 if schedule == 'double' else 640,
                     qk_partial_per_bank=4960, qk_partial_reservation_per_bank=5120,
                     qk_export_double_per_BG=2560, qk_export_tile_per_BG=1280,
                     k_input_double_per_BG_staging_max=128,
                     k_input_double_per_bank_if_fanout_materialized_max=128,
                     k_input_double_per_BG_physical_if_bank_materialized_max=512,
                     k_projection_accumulator_per_bank_max=512,
                     v_input_double_per_rank=1280, v_or_pv_accumulator_per_bank=320),
        buffer_allocation_note='K logical BG stream is at most128B double-buffered; four physical bank copies use512B/BG. Banks may reuse their1280B Q double-buffer allocation.',
        qk_output_schedule='final-feature-fragment tiles16tokens; MAC completion then BG export; conservative no MAC/export overlap',
        multicast_model='external WRBG/WRQ transactions plus shared movement timing; internal fanout provisioned to keep pace',
        BG_reduction_model='four-bank partials reduce locally; throughput implicit in MAC/result schedule, no independent adder model',
        K_local_write_model='native bank-local32B/4CK ports; array row-state and recovery timed separately',
        unknown_latency_us=dict(GPU_Q_projection=None, GPU_Q_RoPE_scale=None,
            K_local_RoPE_cast_pack=None, GPU_softmax=None, GPU_V_cast_pack=None,
            GPU_O_projection=None, GPU_arrival_processing=None),
        input_precondition='GPU input operands available at LPDDR controller; weights/prefix already resident',
        output_boundary='native controller/LPDDR transfer completion; GPU processing/arrival beyond this boundary unmeasured',
        exclusions=['GPU arithmetic and additional DMA/controller arrival overhead',
                    'local K RoPE/cast/packing arithmetic and position metadata delivery',
                    'V GPU conversion/repacking arithmetic', 'prefix installation',
                    'FFN, norms, residuals and action I/O'],
        integrated_numeric_chain=False,
        chain_semantics='native producer completion and local DRAM visibility, not completed nonlinear/GPU numerical execution')

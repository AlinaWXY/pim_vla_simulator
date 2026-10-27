"""Legacy V/PV timing with only physical rank1 participating in PIM.

V projection is measured separately from the fixed Q/K/RoPE chain. The GPU
conversion/packing boundary and its required tail read-modify-write remain
explicit and unmeasured as arithmetic. No numerical values are simulated.
"""
from collections import Counter
from pathlib import Path

import value_layout as L


OPERATORS = ('v', 'pv')
MOVEMENT_TAIL_CK = 4


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('w', encoding='utf-8', newline='\n')
        self.counts = Counter()
        self.segments, self.input_fragments, self.exports, self.dependencies = [], [], [], []
        self.marker = self.line_no = 0
        self.boundary = self.fence()

    def emit(self, operation, address, buffer=None):
        self.stream.write(f'{operation} {address}' + ('' if buffer is None else f' {buffer}') + '\n')
        self.line_no += 1
        self.counts[operation] += 1

    def fence(self):
        marker = self.marker
        self.marker += 1
        self.emit('BARRIER', marker)
        return marker

    def phase(self, name, callback, **metadata):
        first, before = self.line_no + 1, self.counts.copy()
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
                                  line_start=first, line_end=self.line_no,
                                  counts=dict(self.counts - before), **metadata))
        self.boundary = end

    def ranks(self, operation, offset, buffer=None):
        for channel in range(L.CHANNELS):
            self.emit(operation, L.physical_address(channel, L.PIM_RANK, 0, 0, offset), buffer)

    def drain_mac(self):
        # Legacy MACAB retires at issue. Keep the old explicit16CK tail;
        # runner must use pim_rank_rope=False so it is not double-counted.
        self.fence()
        self.emit('WAIT', L.QUERIES)


def matrix_for(case):
    if case == 'v':
        return dict(M=16, K=1024, N=2048, heads=1, padded_K=1024, padded_N=2048,
                    logical_macs=16 * 1024 * 2048, padded_macs=16 * 1024 * 2048,
                    slots_per_bank=8, base=L.REGIONS['w_v']['base'], mac_service_ck=16)
    if case == 'pv':
        return dict(M=16, K=984, N=256, heads=8, padded_K=992, padded_N=256,
                    logical_macs=8 * 16 * 984 * 256, padded_macs=8 * 16 * 992 * 256,
                    slots_per_bank=8, base=L.REGIONS['v_cache']['base'], mac_service_ck=16)
    raise ValueError(case)


def compute(w, case, schedule):
    fragments = L.HIDDEN // L.LANES if case == 'v' else L.TOKEN_FRAGMENTS
    base = matrix_for(case)['base']

    def body():
        for fragment in range(fragments):
            buffer = fragment % 2 if schedule == 'double' else 0
            first = w.line_no + 1
            for query in range(L.QUERIES):
                w.ranks('PIM_WR_GB', query * L.BURST_BYTES, buffer)
            last = w.line_no
            mac_first = w.line_no + 1
            for slot in range(L.FEATURES_PER_BANK):
                offset = base + (fragment * L.FEATURES_PER_BANK + slot) * L.BURST_BYTES
                w.ranks('PIM_MAC_AB', offset, buffer)
            w.input_fragments.append(dict(operator=case, fragment=fragment, buffer=buffer,
                line_start=first, line_end=last, mac_line_first=mac_first, mac_line_last=w.line_no,
                command_count=last-first+1, external_bytes=(last-first+1)*L.BURST_BYTES,
                input_tensor='X' if case == 'v' else 'P',
                scope='rank1 shared input, all16 queries, eight complete output features per bank'))
            if schedule == 'serial':
                w.drain_mac()
        w.drain_mac()
    w.phase(f'{case}_input_compute', body, kind='pipeline', operator=case,
            mac_service_ck=16, native_mac_completion=False,
            data_read='Each weight/cache burst is reused across all16 input query vectors')


def export_results(w, case):
    def body():
        first = w.line_no + 1
        bytes_per_rank = L.QUERIES * L.BGS * L.BANKS_PER_BG * L.FEATURES_PER_BANK * 4
        for offset in range(0, bytes_per_rank, L.BURST_BYTES):
            w.ranks('PIM_RD_RES', offset)
        w.exports.append(dict(phase=f'{case}_result_export', line_start=first, line_end=w.line_no,
                              bytes=(w.line_no-first+1)*L.BURST_BYTES,
                              scope='FP32 complete bank outputs; each physical rank1 exports once'))
        w.fence()
        w.emit('WAIT', MOVEMENT_TAIL_CK)
    w.phase(f'{case}_result_export', body, kind='external_read', operator=case,
            output_dtype='FP32', legacy_result_tail_ck=MOVEMENT_TAIL_CK)


def v_writeback(w):
    producer_fence = w.boundary

    def read_tail():
        for feature in range(L.HEAD_DIM):
            for head in range(L.HEADS):
                w.emit('LD', L.physical_address(*L.value_location(head, 960, feature)))

    def write_suffix():
        for token in (960, 976):
            for feature in range(L.HEAD_DIM):
                for head in range(L.HEADS):
                    w.emit('ST', L.physical_address(*L.value_location(head, token, feature)))

    w.phase('v_prefix_tail_read', read_tail, kind='dram_read',
            preserved_prefix_tokens='960..967', burst_tokens='960..975')
    w.phase('v_suffix_write', write_suffix, kind='dram_write',
            new_suffix_tokens='968..983', zero_padding_tokens='984..991',
            required_unmeasured_stage='GPU BF16 conversion/packing and prefix-tail merge')
    w.dependencies.append(dict(producer='v_result_export', consumer='v_prefix_tail_read/v_suffix_write',
        producer_fence=producer_fence, scope='Native transfers and cache visibility only',
        unknown_required_stage='GPU V BF16 conversion, packing, and read-modify-write merge'))


def generate(path, case='v', schedule='double'):
    if case not in OPERATORS or schedule not in ('double', 'serial'):
        raise ValueError((case, schedule))
    w = Writer(path)
    try:
        compute(w, case, schedule)
        export_results(w, case)
        if case == 'v':
            v_writeback(w)
    finally:
        w.stream.close()
    dq = dict(pim_input_write=w.counts['PIM_WR_GB'] * L.BURST_BYTES,
              pim_result_read=w.counts['PIM_RD_RES'] * L.BURST_BYTES,
              dram_read=w.counts['LD'] * L.BURST_BYTES,
              dram_write=w.counts['ST'] * L.BURST_BYTES)
    dq['total'] = sum(dq.values())
    return dict(case=case, variant='H16 V/PV single PIM rank per channel', schedule=schedule,
                action_horizon=16, attention_heads=8, physical_active_rank=1,
                matrix=matrix_for(case), layout=L.manifest(),
                counts=dict(w.counts), segments=w.segments, input_fragments=w.input_fragments,
                exports=w.exports, dependencies=w.dependencies, expected_dq_bytes=dq,
                traffic_labels=dict(pim_input_write='GPU X input' if case == 'v' else 'GPU softmax P input, two copies/head',
                                    pim_result_read='FP32 V projection' if case == 'v' else 'FP32 attention output',
                                    dram_read='Existing V prefix-tail RMW read',
                                    dram_write='BF16 V suffix/tail/padding writes'),
                expected_internal_array_read=w.counts['PIM_MAC_AB'] * 16 * L.BURST_BYTES,
                buffers=dict(input_double_per_rank=2 * 16 * 16 * 2,
                             accumulator_per_bank=16 * 8 * 4),
                required_backend=dict(pim_rank_rope=False, pim_input_broadcast=False,
                                      pim_mac_scope='rank', pim_data_path='attacc', nCCDAB=16),
                projection_fusion='V measured separately; no claim of sharing X with locked Q/K chain',
                cache_precondition='Weights and V prefix already resident in rank1; initialization excluded',
                input_precondition='X available at memory controller' if case == 'v' else
                                   'Post-softmax BF16 P available at controller; updated full V cache resident',
                unknown_latency_us=dict(GPU_V_cast_pack=None, GPU_softmax=None,
                                        GPU_source_DMA_arrival=None),
                numeric_compute=False,
                exclusions=['GPU arithmetic, including V conversion/packing or softmax',
                            'V weight/prefix installation', 'O projection, norm, residual and action I/O'])

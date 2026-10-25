"""Native-service FFN traces with an explicitly assumed local gather port.

GELU/multiply/cast remains pending. Down starts with rank-local BF16 Z ready;
there is no executable end-to-end GU-to-Down chain in this generator.
"""
from collections import Counter
from pathlib import Path

from ffn_layout import (REGIONS, HORIZON, CHANNELS, RANKS, TOTAL_RANKS, LOCAL_INTERMEDIATE,
                        BURST_BYTES, physical_address)

OPERATORS = ('gate_up', 'down_tiled', 'down_full')
MAC_SERVICE_CK = HORIZON
MOVEMENT_TAIL_CK = 4
GATHER_BYTES_PER_RANK = 2*HORIZON*LOCAL_INTERMEDIATE*4
GATHER_BURSTS_PER_RANK = GATHER_BYTES_PER_RANK//BURST_BYTES
Z_BYTES_PER_RANK = HORIZON*LOCAL_INTERMEDIATE*2


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('w', newline='\n')
        self.counts, self.segments = Counter(), []
        self.input_fragments, self.exports = [], []
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
        before, line_start = self.counts.copy(), self.line_no+1
        callback()
        end = self.fence()
        self.segments.append(dict(name=name, start=self.boundary, end=end,
                                  line_start=line_start, line_end=self.line_no,
                                  counts=dict(self.counts-before), **metadata))
        self.boundary = end

    def all_channels(self, op, offset, buffer=None):
        for channel in range(CHANNELS):
            self.emit(op, physical_address(channel, 0, 0, 0, offset), buffer)

    def drain_mac(self):
        self.fence()
        self.wait(MAC_SERVICE_CK)

    def drain_movement(self):
        self.fence()
        self.wait(MOVEMENT_TAIL_CK)

    def close(self):
        self.stream.close()


def matrix_for(op):
    region = REGIONS[op]
    logical_macs = HORIZON*region['K']*region['N']*region.get('reduction_groups', 1)
    return {**region, 'slots': region['slots_per_bank'],
            'logical_macs': logical_macs, 'padded_macs': logical_macs}


def emit_fragment(w, *, phase, fragment, slots, base, tile, source):
    buffer, line_start = fragment % 2, w.line_no+1
    op = 'PIM_WR_GB' if source == 'gpu_wrgb' else 'PIM_LOCAL_GB'
    for query in range(HORIZON):
        if source == 'gpu_wrgb':
            w.all_channels(op, query*BURST_BYTES, buffer)
        else:
            for rank in range(RANKS):
                for channel in range(CHANNELS):
                    w.emit(op, physical_address(channel, rank, 0, 0, query*BURST_BYTES), buffer)
    input_line_end = w.line_no
    mac_line_start = w.line_no+1
    for slot in range(slots):
        w.all_channels('PIM_MAC_AB', base+(fragment*slots+slot)*BURST_BYTES, buffer)
    w.input_fragments.append(dict(phase=phase, tile=tile, fragment=fragment, buffer=buffer,
        input_source=source, input_command=op, line_start=line_start, line_end=input_line_end,
        input_line_first=line_start, input_line_last=input_line_end,
        mac_line_first=mac_line_start, mac_line_last=w.line_no,
        command_count=HORIZON*CHANNELS*(1 if source == 'gpu_wrgb' else RANKS),
        bursts_per_rank=HORIZON, bytes_per_rank=HORIZON*BURST_BYTES,
        consumer_mac_count=slots*CHANNELS, slots_per_fragment=slots,
        local_source_fragment=fragment if source == 'rank_local_sram' else None))


def pipeline(w, *, name, base, fragments, slots, tile, source):
    def body():
        for fragment in range(fragments):
            emit_fragment(w, phase=name, fragment=fragment, slots=slots,
                          base=base, tile=tile, source=source)
        w.drain_mac()
    w.phase(name, body, kind='pipeline', tile=tile, fragment_count=fragments,
            slots_per_fragment=slots, input_source=source, M=HORIZON,
            weight_base=base, weight_bursts_per_channel=fragments*slots)


def export_results(w, *, name, slots, tile, output_start):
    def body():
        line_start = w.line_no+1
        slot_bytes = HORIZON*16*4
        for slot in range(slots):
            for offset in range(slot*slot_bytes, (slot+1)*slot_bytes, BURST_BYTES):
                for rank in range(RANKS):
                    for channel in range(CHANNELS):
                        w.emit('PIM_RD_RES', physical_address(channel, rank, 0, 0, offset))
        w.exports.append(dict(phase=name, tile=tile, output_start=output_start,
            output_count_per_rank=slots*16, line_start=line_start, line_end=w.line_no,
            bytes_fp32=slots*HORIZON*16*4*TOTAL_RANKS,
            rank_partials=TOTAL_RANKS, accumulator_slots_reused_from_zero=True))
        w.drain_movement()
    w.phase(name, body, kind='export', tile=tile, slots=slots,
            output_start=output_start, output_count_per_rank=slots*16,
            output_source='rank_partial_accumulators', rank_partials=TOTAL_RANKS)


def generate(path, op, local_ck=2):
    if op not in OPERATORS:
        raise ValueError(op)
    if not isinstance(local_ck, int) or local_ck <= 0:
        raise ValueError('local_ck must be a positive integer memory-clock count')
    matrix = matrix_for(op)
    w = Writer(path)
    try:
        if op == 'gate_up':
            pipeline(w, name='input_and_compute', base=matrix['base'], fragments=64,
                     slots=16, tile=None, source='gpu_wrgb')
            w.phase('local_gather', lambda: w.wait(GATHER_BURSTS_PER_RANK*local_ck), kind='local_gather',
                    analytical_wait=True, assumed_service_ck=GATHER_BURSTS_PER_RANK*local_ck,
                    wait_cycles=GATHER_BURSTS_PER_RANK*local_ck, budget_ck=GATHER_BURSTS_PER_RANK*local_ck,
                    bytes_per_rank=GATHER_BYTES_PER_RANK, bytes_system=GATHER_BYTES_PER_RANK*TOTAL_RANKS, parallel_ranks=TOTAL_RANKS,
                    port_bytes=32, port_cycles=local_ck,
                    excluded_operations=['GELU', 'multiply', 'cast'])
        else:
            tiled = op == 'down_tiled'
            tiles, slots = (4, 16) if tiled else (1, 64)
            for tile in range(tiles):
                prefix = f'tile_{tile}_' if tiled else ''
                base = matrix['base']+tile*8*slots*BURST_BYTES
                pipeline(w, name=prefix+'input_and_compute', base=base, fragments=8,
                         slots=slots, tile=tile, source='rank_local_sram')
                export_results(w, name=prefix+'result_export', slots=slots,
                               tile=tile, output_start=tile*slots*16)
    finally:
        w.close()
    return dict(case=op, horizon=HORIZON, mac_service_ck=MAC_SERVICE_CK, mode='native_service', counts=dict(w.counts), segments=w.segments,
        matrix=matrix, input_fragments=w.input_fragments, exports=w.exports,
        local_ck=local_ck, pim_local_transfer_cycles=local_ck,
        local_input_command='PIM_LOCAL_GB', local_input_external_dq_bytes=0,
        local_input_rank_selected=True, global_wrgb_rank_broadcast=True,
        input_double_buffer_bytes_per_rank=2*HORIZON*BURST_BYTES,
        local_input_bytes=w.counts['PIM_LOCAL_GB']*BURST_BYTES,
        wrgb_input_dq_bytes=w.counts['PIM_WR_GB']*BURST_BYTES,
        result_dq_bytes=w.counts['PIM_RD_RES']*BURST_BYTES,
        assumed_local_gather_ck=GATHER_BURSTS_PER_RANK*local_ck if op == 'gate_up' else None,
        local_gather_included=op == 'gate_up',
        local_gather_model='declared analytical WAIT for dedicated 32B/local_ck port per rank; not independently validated hardware',
        nonlinear_latency_ck=None, nonlinear_status='pending; GELU/multiply/cast excluded, not zero-cost',
        rank_local_z_sram_bytes=Z_BYTES_PER_RANK, gpu_partial_count=TOTAL_RANKS if op != 'gate_up' else 0,
        output_policy=('Gate/Up results remain local; assumed gather to rank nonlinear input boundary'
                       if op == 'gate_up' else 'MAC drained before FP32 RDRES; native export, not GPU arrival'),
        preconditions=('resident weights; GPU X available at WRGB input; local nonlinear consumer can accept gathered operands'
                       if op == 'gate_up' else 'resident weights; all 4 KiB BF16 Z per rank ready in rank-local SRAM; GELU/multiply/cast already completed'),
        exclusions=['GPU input/link/DMA latency', 'GELU, multiply and cast',
                    'GPU partial reduction and residual/norm',
                    'GPU global-memory partial materialization and shared LPDDR traffic'],
        integrated_chain=False, gpu_memory_policy='Unified LPDDR; GPU global writes/reads are not free and not included here')

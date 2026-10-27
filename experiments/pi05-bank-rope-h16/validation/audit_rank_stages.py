"""Independent native audit of the legacy-scheduled rank1 V/PV/FFN stages.

No workload/layout module is imported. Geometry, work counts and addresses are
derived here from the declared H16 matrices. Native logs, not manifest totals,
are the source of measured traffic. This validates timing, not tensor values.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


CHANNELS, BANKS, HORIZON, BURST, MAC_CK = 16, 16, 16, 32, 16
MOVEMENT_TAIL_CK = 4  # Legacy explicit WAIT; distinct from native nBL_min.
W_V, V_CACHE, GATE_UP, DOWN = 83968, 100352, 262144, 327680
NATIVE_TO_TRACE = dict(WRGB='PIM_WR_GB', MACAB='PIM_MAC_AB', RDRES='PIM_RD_RES',
                       RD='LD', RDA='LD', WR='ST', WRA='ST')
WORK_OPS = set(NATIVE_TO_TRACE.values()) | {'PIM_LOCAL_GB'}
EXPECTED_COUNTS = {
    'v': dict(PIM_WR_GB=16384, PIM_MAC_AB=8192, PIM_RD_RES=4096, LD=2048, ST=4096),
    'pv': dict(PIM_WR_GB=15872, PIM_MAC_AB=7936, PIM_RD_RES=4096),
    'gate_up': dict(PIM_WR_GB=16384, PIM_MAC_AB=32768),
    'down_tiled': dict(PIM_LOCAL_GB=16384, PIM_MAC_AB=16384, PIM_RD_RES=32768),
}
EXPECTED_DQ = dict(v=832*1024, pv=624*1024, gate_up=512*1024, down_tiled=1024*1024)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return [{key: value if key == 'command' else int(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def tick(clock, ratio):
    return (clock-1)*ratio+1


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, address >> 20, address >> 9 & 63)


def address_key(op, channel, rank, bg, bank, row, column, buffer=-1):
    # Native all-bank commands replace canonical BG0/bank0 by -1/-1.
    if op in ('PIM_WR_GB', 'PIM_MAC_AB', 'PIM_RD_RES', 'PIM_LOCAL_GB'):
        bg = bank = -1
    return channel, rank, bg, bank, row, column, buffer


def pipeline_shapes(case):
    if case == 'v':
        return [(W_V, 64, 8, 'PIM_WR_GB')]
    if case == 'pv':
        return [(V_CACHE, 62, 8, 'PIM_WR_GB')]
    if case == 'gate_up':
        return [(GATE_UP, 64, 32, 'PIM_WR_GB')]
    require(case == 'down_tiled', f'Unsupported stage: {case}')
    return [(DOWN+tile*16*16*BURST, 16, 16, 'PIM_LOCAL_GB') for tile in range(4)]


def expected_addresses(case):
    """Independent matrix coverage, including all V tail RMW locations."""
    result = defaultdict(Counter)

    def add(op, channel, offset, buffer=-1, bg=0, bank=0):
        row, within = divmod(offset, 2048)
        result[op][address_key(op, channel, 1, bg, bank, row, within//BURST, buffer)] += 1

    for base, fragments, slots, input_op in pipeline_shapes(case):
        for fragment in range(fragments):
            for channel in range(CHANNELS):
                for query in range(HORIZON):
                    add(input_op, channel, query*BURST, fragment % 2)
                for slot in range(slots):
                    add('PIM_MAC_AB', channel, base+(fragment*slots+slot)*BURST, fragment % 2)
        if case != 'gate_up':
            for channel in range(CHANNELS):
                for offset in range(0, slots*HORIZON*BANKS*4, BURST):
                    add('PIM_RD_RES', channel, offset)
    if case == 'v':
        for channel in range(CHANNELS):
            for bg in range(4):
                for bank in range(4):
                    for slot in range(8):
                        add('LD', channel, V_CACHE+(60*8+slot)*BURST, bg=bg, bank=bank)
                        for fragment in (60, 61):
                            add('ST', channel, V_CACHE+(fragment*8+slot)*BURST, bg=bg, bank=bank)
    return result


def audit_external_payload(native, local_count, case):
    """Calculate traffic from actual issued native commands."""
    counts = Counter(e['command'] for e in native)
    require(not any(counts[op] for op in ('MACK', 'MACQK', 'MACHB', 'LOCALRD', 'LOCALWR')),
            'Unexpected nonlegacy PIM operation in rank-stage trace')
    actual = Counter()
    for command, op in NATIVE_TO_TRACE.items():
        actual[op] += counts[command]
    actual['PIM_LOCAL_GB'] = local_count
    require(+actual == Counter(EXPECTED_COUNTS[case]), 'Native work counts differ from independent matrix dimensions')
    dq = dict(pim_input_write=counts['WRGB']*BURST,
              pim_result_read=counts['RDRES']*BURST,
              dram_read=(counts['RD']+counts['RDA'])*BURST,
              dram_write=(counts['WR']+counts['WRA'])*BURST)
    dq['total'] = sum(dq.values())
    require(dq['total'] == EXPECTED_DQ[case], 'Unexpected native external DQ payload')
    return dq


def audit_rank_scope(native, local):
    work_count = 0
    for event in native:
        command = event['command']
        if command in NATIVE_TO_TRACE:
            require(event['Rank'] == 1, 'Workload accesses reserved physical rank0')
            work_count += 1
        if command in ('WRGB', 'MACAB', 'RDRES', 'ACTAB'):
            require(event['Rank'] == 1 and event['pim_rank_mask'] == 2,
                    'All-bank command does not exclusively target rank1')
            require(event['BankGroup'] == event['Bank'] == -1,
                    'Rank command does not address all sixteen banks')
    require(all(e['Rank'] == 1 for e in local), 'LOCAL_GB accesses reserved physical rank0')
    return work_count


def audit_rows(native):
    """Replay actual ACT/PRE/refresh and expand MACAB into sixteen banks."""
    opened, checked = {}, 0
    for event in native:
        command = event['command']
        ranks = range(2) if event['Rank'] < 0 else [event['Rank']]
        groups = range(4) if event['BankGroup'] < 0 else [event['BankGroup']]
        banks = range(4) if event['Bank'] < 0 else [event['Bank']]
        if command in ('ACTAB', 'PREab', 'REFab'):
            groups, banks = range(4), range(4)
        targets = [(event['Channel'], r, g, b) for r in ranks for g in groups for b in banks]
        if command in ('ACT2', 'ACTAB', 'ACTHB'):
            for target in targets:
                opened[target] = event['Row']
        elif command in ('PREpb', 'PREab', 'PREHB', 'REFab', 'REFpb'):
            for target in targets:
                opened.pop(target, None)
        elif command in ('MACAB', 'RD', 'RDA', 'WR', 'WRA'):
            for target in targets:
                require(opened.get(target) == event['Row'], 'Native array command accesses closed/wrong row')
                checked += 1
            if command in ('RDA', 'WRA'):
                for target in targets:
                    opened.pop(target, None)
    require(checked > 0, 'No native array accesses audited')
    return checked


def audit_local_events(local, native, local_ck):
    """Check recorded port state against both previous fills and actual MACs."""
    previous_ready = defaultdict(int)
    mac_clocks = defaultdict(list)
    for event in native:
        if event['command'] == 'MACAB':
            mac_clocks[event['Channel'], event['pim_buffer']].append(event['clock'])
    for clocks in mac_clocks.values():
        clocks.sort()
    for event in local:
        channel, buffer = event['Channel'], event['pim_buffer']
        require(event['Rank'] == 1 and buffer in (0, 1), 'Invalid local rank/buffer')
        require(event['bytes'] == BURST and event['port_cycles'] == local_ck,
                'Local port service differs from configured 32B transfer')
        require(event['ready_at'] == event['clock']+local_ck, 'Incorrect local completion latency')
        require(event['port_ready_before'] == previous_ready[channel], 'Incorrect local port predecessor')
        require(event['clock'] >= previous_ready[channel], 'Overlapping transfers on one local port')
        clocks = mac_clocks[channel, buffer]
        position = bisect_right(clocks, event['clock'])
        busy = clocks[position-1]+MAC_CK if position else 0
        require(event['buffer_in_use_until'] == busy, 'Local buffer ownership disagrees with actual native MAC')
        require(event['clock'] >= busy, 'Local input overwrites a buffer before MAC completion')
        previous_ready[channel] = event['ready_at']


def audit_fragment_dependencies(writes, macs, *, base, fragments, slots, local, local_ck, burst_ck):
    """One channel/phase: all16 inputs ready, correct slots, no early reuse."""
    require(len(writes) == fragments*HORIZON and len(macs) == fragments*slots,
            'Incomplete fragment/query/slot coverage')
    require(all(b['clock']-a['clock'] >= MAC_CK for a, b in zip(macs, macs[1:])),
            'Overlapping MAC service windows')
    for fragment in range(fragments):
        group = writes[fragment*HORIZON:(fragment+1)*HORIZON]
        consumers = macs[fragment*slots:(fragment+1)*slots]
        buffer = fragment % 2
        require(all(e['pim_buffer'] == buffer for e in group+consumers), 'Incorrect ping-pong buffer selection')
        offsets = [e['offset_bytes'] if local else e['Row']*2048+e['Column']*BURST for e in group]
        require(offsets == list(range(0, HORIZON*BURST, BURST)), 'Incomplete sixteen-query input slice')
        require([e['Row']*2048+e['Column']*BURST for e in consumers] ==
                [base+(fragment*slots+slot)*BURST for slot in range(slots)],
                'MAC fragment/output-column order mismatch')
        ready = max(e['ready_at'] if local else e['clock']+burst_ck for e in group)
        require(consumers[0]['clock'] >= ready, 'MAC begins before complete input fragment is ready')
        if fragment >= 2:
            previous = macs[(fragment-1)*slots-1]
            require(group[0]['clock'] >= previous['clock']+MAC_CK,
                    'Ping-pong buffer reused before previous consumer completed')
    return fragments


def audit_export_dependency(macs, exports, start, stop, ratio, burst_ck):
    require(macs and exports, 'Missing export producer or output commands')
    ready = max(tick(e['clock'], ratio)+MAC_CK*ratio for e in macs)
    require(start >= ready and min(tick(e['clock'], ratio) for e in exports) >= ready,
            'Export precedes complete producer MAC+16CK')
    require(stop >= max(tick(e['clock'], ratio) for e in exports)+max(burst_ck, MOVEMENT_TAIL_CK)*ratio,
            'Missing final RDRES movement tail')


def audit_run(directory, manifest=None, result=None):
    directory = Path(directory)
    simulation = directory/'simulation'
    manifest = manifest if manifest is not None else read_json(directory/'manifest.json')
    result = result if result is not None else read_json(simulation/'result.json')
    case = manifest['case']
    require(case in EXPECTED_COUNTS, 'Unsupported rank stage case')
    cfg, timings = result['config'], result['resolved_timings_ck']
    require((cfg['channels'], cfg['ranks'], cfg['memory_ratio']) == (16, 2, 1), 'Incorrect physical topology')
    require(cfg['pim'] and not cfg.get('pim_rank_rope', False) and not cfg['pim_input_broadcast']
            and cfg['pim_mac_scope'] == 'rank' and cfg['pim_data_path'] == 'attacc',
            'Stage must retain legacy rank-scoped AttAcc scheduling')
    ratio, local_ck, burst_ck = cfg['frontend_ratio'], cfg['pim_local_transfer_cycles'], timings['nBL_min']
    require(ratio == 32 and timings['nCCDAB'] == 16 and 0 < burst_ck <= MOVEMENT_TAIL_CK and local_ck == 2,
            'Unexpected legacy MAC/movement/local-port timing')
    require(manifest.get('schedule', 'double') == 'double', 'This audit covers the double-buffer primary schedule')
    trace_path = directory/'workload.trace'
    require(sha(trace_path) == sha(simulation/'input.trace') == result['requests']['sha256'], 'Trace hash mismatch')
    trace, counts, expected = [], Counter(), defaultdict(Counter)
    for line in trace_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        op, values = parts[0], [int(v, 0) for v in parts[1:]]
        require(op in WORK_OPS | {'BARRIER', 'WAIT'}, f'Unexpected request {op}')
        trace.append((op, *values)); counts[op] += 1
        if op in WORK_OPS:
            channel, rank, bg, bank, row, column = decode(values[0])
            require(values[0] % BURST == 0 and rank == 1, 'Unaligned request or source trace accesses rank0')
            if op.startswith('PIM_'):
                require(bg == bank == 0, 'Noncanonical rank input/MAC/output source address')
            buffer = values[1] if len(values) > 1 else -1
            expected[op][address_key(op, channel, rank, bg, bank, row, column, buffer)] += 1
    require(counts == Counter(manifest['counts']), 'Manifest request counts differ from actual trace')
    require(Counter({op: counts[op] for op in WORK_OPS if counts[op]}) == Counter(EXPECTED_COUNTS[case]),
            'Trace work count differs from independent matrix dimensions')
    oracle = expected_addresses(case)
    for op in WORK_OPS:
        require(expected[op] == oracle[op], f'Trace address coverage differs from independent layout: {op}')

    native, local, native_hashes, local_hashes = [], [], {}, {}
    for channel in range(CHANNELS):
        np, lp = simulation/f'timeline.csv.ch{channel}', simulation/f'local_input.csv.ch{channel}'
        nr, lr = read_csv(np), read_csv(lp)
        require(all(e['Channel'] == channel for e in nr+lr), 'Native log channel mismatch')
        require(all(a['clock'] <= b['clock'] for a, b in zip(nr, nr[1:])), 'Nonmonotonic native clock')
        require(all(a['clock'] <= b['clock'] for a, b in zip(lr, lr[1:])), 'Nonmonotonic local clock')
        native.extend(nr); local.extend(lr)
        native_hashes[str(channel)], local_hashes[str(channel)] = sha(np), sha(lp)
    audit_rank_scope(native, local)
    row_checks = audit_rows(native)
    audit_local_events(local, native, local_ck)
    native_counts = Counter(e['command'] for e in native)
    require(native_counts == Counter({k: v for k, v in result['issued_commands'].items() if v}),
            'Native command counters differ from result')
    actual = defaultdict(Counter)
    for event in native:
        op = NATIVE_TO_TRACE.get(event['command'])
        if op:
            key = address_key(op, *(event[k] for k in ('Channel', 'Rank', 'BankGroup', 'Bank', 'Row', 'Column')),
                              event['pim_buffer'])
            actual[op][key] += 1
    for event in local:
        row, within = divmod(event['offset_bytes'], 2048)
        actual['PIM_LOCAL_GB'][address_key('PIM_LOCAL_GB', event['Channel'], event['Rank'], 0, 0,
                                         row, within//BURST, event['pim_buffer'])] += 1
    for op in WORK_OPS:
        require(actual[op] == expected[op], f'Native address multiset differs from source trace: {op}')
    require(len(local) == result['requests'].get('pim_local_gb', 0), 'Native local-event count differs from result')
    dq = audit_external_payload(native, len(local), case)
    require(manifest['expected_dq_bytes'] == dq, 'Manifest DQ differs from native-derived traffic')
    for key, value in dq.items():
        if key != 'total':
            require(result['traffic_bytes'][key] == value, f'Result traffic mismatch: {key}')
    internal = native_counts['MACAB']*BANKS*BURST
    require(result['traffic_bytes']['pim_internal'] == internal, 'Internal array traffic not sixteen banks/MAC')
    require(result['traffic_bytes']['pim_input_delivered'] == dq['pim_input_write'], 'Unexpected cross-rank input broadcast')
    require(result['traffic_bytes']['pim_local_input_write'] == len(local)*BURST, 'Incorrect local input bytes')

    barriers = {e['marker']: e['frontend_tick'] for e in read_csv(simulation/'barriers.csv')}
    trace_barriers = {e[1]: i for i, e in enumerate(trace) if e[0] == 'BARRIER'}
    require(list(barriers) == list(trace_barriers) == list(range(len(barriers)))
            and len(trace_barriers) == counts['BARRIER'], 'Barrier coverage mismatch')
    shapes, pipeline_index, export_count, gather_count = pipeline_shapes(case), 0, 0, 0
    fragments_checked, phases, previous_stop, flow = 0, [], None, []
    last_producers = {}
    for segment in manifest['segments']:
        start, stop = barriers[segment['start']], barriers[segment['end']]
        require(start < stop and (previous_stop is None or start == previous_stop), 'Noncontiguous phase bounds')
        previous_stop = stop
        body = trace[trace_barriers[segment['start']]+1:trace_barriers[segment['end']]+1]
        body_counts = Counter(e[0] for e in body)
        require(body_counts == Counter(segment['counts']), 'Segment counts differ from source trace')
        selected = [e for e in native if start <= tick(e['clock'], ratio) < stop]
        selected_local = [e for e in local if start <= tick(e['clock'], ratio) < stop]
        issued = Counter(NATIVE_TO_TRACE[e['command']] for e in selected if e['command'] in NATIVE_TO_TRACE)
        issued['PIM_LOCAL_GB'] = len(selected_local)
        require(+issued == Counter({op: body_counts[op] for op in WORK_OPS if body_counts[op]}),
                'Native work crosses its declared phase boundary')
        waits = [e[1] for e in body if e[0] == 'WAIT']
        if body_counts['PIM_MAC_AB']:
            require(pipeline_index < len(shapes), 'Too many compute phases')
            base, fragments, slots, input_op = shapes[pipeline_index]
            require(+issued == Counter({input_op: fragments*HORIZON*CHANNELS,
                                        'PIM_MAC_AB': fragments*slots*CHANNELS}),
                    'Compute phase contains unrelated/missing work')
            require(waits == [MAC_CK], 'Missing legacy explicit WAIT16 compute tail')
            for channel in range(CHANNELS):
                macs = [e for e in selected if e['Channel'] == channel and e['command'] == 'MACAB']
                writes = ([e for e in selected_local if e['Channel'] == channel] if input_op == 'PIM_LOCAL_GB'
                          else [e for e in selected if e['Channel'] == channel and e['command'] == 'WRGB'])
                fragments_checked += audit_fragment_dependencies(writes, macs, base=base,
                    fragments=fragments, slots=slots, local=input_op == 'PIM_LOCAL_GB',
                    local_ck=local_ck, burst_ck=burst_ck)
                require(stop >= tick(macs[-1]['clock'], ratio)+MAC_CK*ratio, 'Phase closes before MAC completion')
                last_producers[channel] = macs
            pipeline_index += 1
            flow.append('compute')
        elif body_counts['PIM_RD_RES']:
            expected_results = (8 if case in ('v', 'pv') else 16)*HORIZON*BANKS*4//BURST*CHANNELS
            require(+issued == Counter(PIM_RD_RES=expected_results), 'Export phase contains unrelated/missing work')
            require(waits == [MOVEMENT_TAIL_CK], 'Missing legacy explicit WAIT4 result tail')
            for channel in range(CHANNELS):
                exports = [e for e in selected if e['Channel'] == channel and e['command'] == 'RDRES']
                audit_export_dependency(last_producers.get(channel, []), exports, start, stop, ratio, burst_ck)
            export_count += 1
            flow.append('export')
        elif waits:
            require(case == 'gate_up' and waits == [2*16*256*4//32*local_ck] and not +issued,
                    'Incorrect assumed Gate/Up local gather budget')
            require(stop-start >= waits[0]*ratio, 'Gather shorter than its explicit port budget')
            require(all(start >= tick(e[-1]['clock'], ratio)+MAC_CK*ratio for e in last_producers.values()),
                    'Gather starts before complete Gate/Up production')
            gather_count += 1
            flow.append('gather')
        elif body_counts['LD']:
            require(case == 'v' and +issued == Counter(LD=2048), 'Incorrect V prefix-tail read phase')
            flow.append('read')
        elif body_counts['ST']:
            require(case == 'v' and +issued == Counter(ST=4096), 'Incorrect V suffix write phase')
            flow.append('write')
        else:
            raise AssertionError('Unrecognized or empty workload phase')
        phases.append(dict(name=segment['name'], start_tick=start, stop_tick=stop,
                           latency_us=(stop-start)*result['tCK_ns']/(ratio*1000)))
    require(pipeline_index == len(shapes), 'Missing compute phases')
    require(export_count == (0 if case == 'gate_up' else len(shapes)) and gather_count == (case == 'gate_up'),
            'Missing output/gather phase')
    expected_flow = dict(v=['compute', 'export', 'read', 'write'], pv=['compute', 'export'],
                         gate_up=['compute', 'gather'], down_tiled=['compute', 'export']*4)
    require(flow == expected_flow[case], 'Stage producer/consumer phase order differs from contract')
    require(phases and phases[0]['start_tick'] == barriers[0] and phases[-1]['stop_tick'] == barriers[max(barriers)],
            'Unaudited leading/trailing trace phases')
    boundary = result['memory_cycles']*ratio-sum(p['stop_tick']-p['start_tick'] for p in phases)
    require(0 <= boundary <= 2*ratio, 'Unexpected unaccounted outer boundary time')
    return dict(status='passed', case=case, active_rank=1, active_banks=256,
        native_timelines_checked=CHANNELS, local_event_files_checked=CHANNELS,
        independent_matrix_address_coverage=True, native_address_multisets_match_trace=True,
        rank0_workload_accesses=0, bank_array_accesses_checked=row_checks,
        channel_fragment_dependencies_checked=fragments_checked, local_events_checked=len(local),
        native_counts=dict(native_counts), native_external_dq_bytes=dq,
        native_internal_array_bytes=internal, local_input_bytes=len(local)*BURST,
        legacy_mac_tail_ck=MAC_CK, legacy_result_tail_ck=MOVEMENT_TAIL_CK,
        native_input_burst_ck=burst_ck,
        assumed_gather_ck=2048 if case == 'gate_up' else None,
        trace_sha256=sha(trace_path), native_timeline_sha256=native_hashes,
        local_event_sha256=local_hashes, phases=phases, boundary_ticks=boundary,
        numerical_compute=False, complete_ae_latency=False,
        unmeasured='GPU conversion/packing, softmax, O, GELU/multiply/cast, final partial merge, norm/residual/action I/O')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run)
    (args.run/'native-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

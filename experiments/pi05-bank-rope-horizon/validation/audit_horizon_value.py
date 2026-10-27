"""Independent whole-H V/PV audit; no generator/layout imports."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

_legacy = Path(__file__).resolve().parents[2]/'pi05-bank-rope-h16/validation/audit_rank_stages.py'
_spec = importlib.util.spec_from_file_location('_frozen_rank_audit', _legacy)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
require, read_json, read_csv, sha, tick, decode = R.require, R.read_json, R.read_csv, R.sha, R.tick, R.decode
NATIVE = dict(WRGB='PIM_WR_GB', MACAB='PIM_MAC_AB', RDRES='PIM_RD_RES',
              RD='LD', RDA='LD', WR='ST', WRA='ST')
WORK = set(NATIVE.values())


def shape(horizon, case):
    require(horizon in (16, 20, 30, 40) and case in ('v', 'pv'), 'Unsupported whole-H value case')
    padded = (968+horizon+31)//32*32
    work_columns = 3*(horizon//2)+(padded-968-horizon)//2
    vbase = 81920+(work_columns*32+2047)//2048*2048
    cache = vbase+16384
    fragments = 64 if case == 'v' else padded//16
    counts = Counter(PIM_WR_GB=fragments*horizon*16, PIM_MAC_AB=fragments*8*16,
                     PIM_RD_RES=horizon*256)
    if case == 'v':
        counts.update(LD=2048, ST=(padded-960)//16*2048)
    return dict(horizon=horizon, padded=padded, vbase=vbase, cache=cache,
                base=vbase if case == 'v' else cache, fragments=fragments, counts=counts)


def key(op, geometry, buffer=-1):
    return R.address_key(op, *geometry, buffer)


def expected_addresses(horizon, case):
    s, addresses = shape(horizon, case), defaultdict(Counter)
    def add(op, channel, byte, buffer=-1, bg=0, bank=0):
        addresses[op][key(op, (channel, 1, bg, bank, byte//2048, byte % 2048//32), buffer)] += 1
    for channel in range(16):
        for fragment in range(s['fragments']):
            for query in range(horizon):
                add('PIM_WR_GB', channel, query*32, fragment % 2)
            for slot in range(8):
                add('PIM_MAC_AB', channel, s['base']+(fragment*8+slot)*32, fragment % 2)
        for byte in range(0, horizon*16*8*4, 32):
            add('PIM_RD_RES', channel, byte)
        if case == 'v':
            for bg in range(4):
                for bank in range(4):
                    for slot in range(8):
                        add('LD', channel, s['cache']+(60*8+slot)*32, bg=bg, bank=bank)
                        for fragment in range(60, s['padded']//16):
                            add('ST', channel, s['cache']+(fragment*8+slot)*32, bg=bg, bank=bank)
    return addresses


def audit_fragments(writes, macs, *, horizon, base, fragments, burst_ck):
    require(len(writes) == fragments*horizon and len(macs) == fragments*8,
            'Input or MAC count differs from the complete whole-H matrix')
    require(all(b['clock']-a['clock'] >= horizon for a, b in zip(macs, macs[1:])),
            'MAC service overlaps before all H query vectors finish')
    for fragment in range(fragments):
        inputs = writes[fragment*horizon:(fragment+1)*horizon]
        outputs = macs[fragment*8:(fragment+1)*8]
        require(all(e['pim_buffer'] == fragment % 2 for e in inputs+outputs), 'Wrong ping-pong buffer')
        require([e['Row']*2048+e['Column']*32 for e in inputs] == list(range(0, horizon*32, 32)),
                'Input fragment omits/duplicates an H query row')
        require([e['Row']*2048+e['Column']*32 for e in outputs] ==
                [base+(fragment*8+slot)*32 for slot in range(8)], 'Weight/cache columns changed or repeated')
        require(outputs[0]['clock'] >= max(e['clock']+burst_ck for e in inputs),
                'MAC begins before all H input rows complete')
        if fragment >= 2:
            last_previous = macs[(fragment-1)*8-1]
            require(inputs[0]['clock'] >= last_previous['clock']+horizon,
                    'Whole-H input buffer overwritten before prior MAC completion')


def audit_payload(native, horizon, case):
    counts = Counter(e['command'] for e in native)
    require(not any(counts[op] for op in ('MACK', 'MACQK', 'MACHB', 'LOCALRD', 'LOCALWR')),
            'Unexpected non-value PIM command')
    work = Counter()
    for cmd, op in NATIVE.items():
        work[op] += counts[cmd]
    require(+work == shape(horizon, case)['counts'], 'Native work count differs from whole-H oracle')
    dq = dict(pim_input_write=counts['WRGB']*32, pim_result_read=counts['RDRES']*32,
              dram_read=(counts['RD']+counts['RDA'])*32, dram_write=(counts['WR']+counts['WRA'])*32)
    dq['total'] = sum(dq.values())
    return dq


def audit_run(directory, manifest=None, result=None):
    directory = Path(directory)
    simulation = directory/'simulation'
    manifest = manifest if manifest is not None else read_json(directory/'manifest.json')
    result = result if result is not None else read_json(simulation/'result.json')
    horizon, case = manifest.get('horizon', manifest.get('action_horizon')), manifest['case']
    s = shape(horizon, case)
    config, timings = result['config'], result['resolved_timings_ck']
    require((config['channels'], config['ranks'], config['memory_ratio'], config['frontend_ratio']) == (16, 2, 1, 32),
            'Wrong memory geometry/clock ratio')
    require(config.get('pim_horizon', 16) == horizon and config['pim'] and
            not config.get('pim_rank_rope', False) and not config.get('pim_ffn_dram', False) and
            not config['pim_input_broadcast'] and config['pim_mac_scope'] == 'rank' and
            config['pim_data_path'] == 'attacc', 'Incorrect whole-H native value mode')
    require(timings['nCCDAB'] == horizon and 0 < timings['nBL_min'] <= 4, 'Wrong whole-H MAC/burst timing')
    ratio, burst_ck = config['frontend_ratio'], timings['nBL_min']
    trace_path = directory/'workload.trace'
    require(sha(trace_path) == sha(simulation/'input.trace') == result['requests']['sha256'], 'Trace hash mismatch')
    trace, counts, source = [], Counter(), defaultdict(Counter)
    for line in trace_path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        op, values = fields[0], [int(v, 0) for v in fields[1:]]
        require(op in WORK | {'BARRIER', 'WAIT'}, 'Unexpected trace operation')
        trace.append((op, *values)); counts[op] += 1
        if op in WORK:
            geometry = decode(values[0])
            require(values[0] % 32 == 0 and geometry[1] == 1, 'Unaligned or rank0 source work')
            if op.startswith('PIM_'):
                require(geometry[2] == geometry[3] == 0, 'Noncanonical rank operation')
            source[op][key(op, geometry, values[1] if len(values) > 1 else -1)] += 1
    require(counts == Counter(manifest['counts']), 'Manifest counts differ from source trace')
    oracle = expected_addresses(horizon, case)
    for op in WORK:
        require(source[op] == oracle[op], f'Source work differs from independent address oracle: {op}')
    native, hashes = [], {}
    for channel in range(16):
        path = simulation/f'timeline.csv.ch{channel}'
        events = read_csv(path)
        require(all(e['Channel'] == channel for e in events), 'Timeline channel mismatch')
        require(all(a['clock'] <= b['clock'] for a, b in zip(events, events[1:])), 'Nonmonotonic native timeline')
        native.extend(events); hashes[str(channel)] = sha(path)
    R.audit_rank_scope(native, [])
    array_checks = R.audit_rows(native)
    counts_native = Counter(e['command'] for e in native)
    require(counts_native == Counter({k: v for k, v in result['issued_commands'].items() if v}),
            'Result issued counters disagree with actual native commands')
    actual = defaultdict(Counter)
    for event in native:
        op = NATIVE.get(event['command'])
        if op:
            geometry = tuple(event[k] for k in ('Channel', 'Rank', 'BankGroup', 'Bank', 'Row', 'Column'))
            actual[op][key(op, geometry, event['pim_buffer'])] += 1
    for op in WORK:
        require(actual[op] == source[op], f'Native address/buffer multiset differs from source: {op}')
    dq = audit_payload(native, horizon, case)
    require(dq == manifest['expected_dq_bytes'], 'Manifest DQ disagrees with native work')
    for name, amount in dq.items():
        if name != 'total':
            require(result['traffic_bytes'][name] == amount, f'Result traffic mismatch: {name}')
    internal = counts_native['MACAB']*16*32
    require(result['traffic_bytes']['pim_internal'] == internal, 'Weight/cache array read count mismatch')
    require(result['traffic_bytes']['pim_input_delivered'] == dq['pim_input_write'] and
            result['traffic_bytes']['pim_local_input_write'] == 0, 'Unexpected implicit input broadcast/local fill')
    barriers = {e['marker']: e['frontend_tick'] for e in read_csv(simulation/'barriers.csv')}
    source_barriers = {e[1]: i for i, e in enumerate(trace) if e[0] == 'BARRIER'}
    require(list(barriers) == list(source_barriers) == list(range(len(barriers))) and
            len(source_barriers) == counts['BARRIER'], 'Barrier coverage mismatch')
    phases, flow, previous_stop, producers = [], [], None, {}
    for segment in manifest['segments']:
        start, stop = barriers[segment['start']], barriers[segment['end']]
        require(start < stop and (previous_stop is None or start == previous_stop), 'Noncontiguous phase bounds')
        previous_stop = stop
        body = trace[source_barriers[segment['start']]+1:source_barriers[segment['end']]+1]
        body_counts = Counter(e[0] for e in body)
        require(body_counts == Counter(segment['counts']), 'Phase counts disagree with trace')
        selected = [e for e in native if start <= tick(e['clock'], ratio) < stop]
        issued = Counter(NATIVE[e['command']] for e in selected if e['command'] in NATIVE)
        require(issued == Counter({op: body_counts[op] for op in WORK if body_counts[op]}),
                'Native work crosses phase boundary')
        waits = [e[1] for e in body if e[0] == 'WAIT']
        if body_counts['PIM_MAC_AB']:
            require(issued == Counter(PIM_WR_GB=s['fragments']*horizon*16,
                                      PIM_MAC_AB=s['fragments']*8*16) and waits == [horizon],
                    'Compute phase does not contain exactly one whole-H pass and its completion tail')
            for channel in range(16):
                macs = [e for e in selected if e['command'] == 'MACAB' and e['Channel'] == channel]
                writes = [e for e in selected if e['command'] == 'WRGB' and e['Channel'] == channel]
                audit_fragments(writes, macs, horizon=horizon, base=s['base'],
                                fragments=s['fragments'], burst_ck=burst_ck)
                require(stop >= tick(macs[-1]['clock']+horizon, ratio), 'Compute phase ends before whole-H MAC completion')
                producers[channel] = macs
            flow.append('compute')
        elif body_counts['PIM_RD_RES']:
            require(issued == Counter(PIM_RD_RES=horizon*256) and waits == [4],
                    'Export omits H query outputs or result movement tail')
            for channel in range(16):
                exported = [e for e in selected if e['command'] == 'RDRES' and e['Channel'] == channel]
                ready = max(tick(e['clock']+horizon, ratio) for e in producers[channel])
                require(start >= ready and tick(exported[0]['clock'], ratio) >= ready, 'Export before producer completion')
                require(stop >= tick(exported[-1]['clock']+4, ratio), 'Missing result movement tail')
            flow.append('export')
        elif body_counts['LD']:
            require(case == 'v' and issued == Counter(LD=2048) and not waits, 'Incorrect single prefix tail RMW read')
            flow.append('read')
        elif body_counts['ST']:
            require(case == 'v' and issued == Counter(ST=s['counts']['ST']) and not waits,
                    'Suffix/padding cache writes missing or duplicated')
            flow.append('write')
        else:
            raise AssertionError('Unrecognized or empty phase')
        phases.append(dict(name=segment['name'], start_tick=start, stop_tick=stop,
                           latency_us=(stop-start)*result['tCK_ns']/(ratio*1000)))
    require(flow == (['compute', 'export', 'read', 'write'] if case == 'v' else ['compute', 'export']),
            'Whole-H producer/cache phase order differs from contract')
    require(phases[0]['start_tick'] == barriers[0] and phases[-1]['stop_tick'] == barriers[max(barriers)],
            'Unaudited leading/trailing work')
    boundary = result['memory_cycles']*ratio-sum(p['stop_tick']-p['start_tick'] for p in phases)
    require(0 <= boundary <= 2*ratio, 'Unexpected unaccounted outer boundary')
    return dict(status='passed', case=case, horizon=horizon, physical_active_rank=1,
        whole_horizon_batch=True, query_padding=0, kv_padded_tokens=s['padded'],
        input_double_buffer_bytes_per_rank=64*horizon, accumulator_bytes_per_bank=32*horizon,
        native_external_dq_bytes=dq, native_internal_array_bytes=internal,
        bank_array_accesses_checked=array_checks, channel_fragment_dependencies_checked=16*s['fragments'],
        native_counts=dict(counts_native), trace_sha256=sha(trace_path), native_timeline_sha256=hashes,
        phases=phases, boundary_ticks=boundary, numerical_compute=False,
        GPU_conversion_and_softmax_latency_us=None, complete_ae_latency=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run)
    (args.run/'native-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

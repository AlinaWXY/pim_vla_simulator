"""Independent audit of fused QKV and DRAM-read/relay Q attention traces.

Replays request identities, complete vector fills, buffer lifetimes, DRAM row
state and transfer dependencies. This is a timing/data-placement audit, not a
numerical implementation of GPU RoPE or the PIM multiply/accumulate datapath.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path

RICH = ('PIM_RD_Q', 'PIM_RELAY_Q', 'PIM_MAC_QK', 'PIM_RD_BG')
TARGETS = dict(PIM_RD_Q=1, PIM_RELAY_Q=8, PIM_MAC_QK=32, PIM_RD_BG=4)
RICH_NATIVE = dict(PIM_RD_Q='RD', PIM_RELAY_Q='WRGB', PIM_MAC_QK='MACQK', PIM_RD_BG='RDRES')
LEGACY_NATIVE = dict(WRGB='PIM_WR_GB', MACAB='PIM_MAC_AB', RDRES='PIM_RD_RES', RD='LD', WR='ST')
Q_BASE, K_BASE, V_BASE = 24576, 28672, 36864


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, address >> 20, address >> 9 & 63)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return [{key: value if key in ('operation', 'command') else int(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def trace_records(path):
    records = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        fields = text.split()
        if not fields or fields[0].startswith('#'):
            continue
        records.append(dict(line=line, operation=fields[0], address=int(fields[1], 0),
                            buffer=int(fields[2]) if len(fields) > 2 else -1))
    return records


def key(record, operation=None):
    op = operation or record['operation']
    if 'address' in record:
        c, r, g, b, row, col = decode(record['address'])
    else:
        c, r, g, b, row, col = (record[k] for k in ('Channel', 'Rank', 'BankGroup', 'Bank', 'Row', 'Column'))
    if op in ('PIM_MAC_AB', 'PIM_MAC_QK'):
        r = g = b = -1
    elif op in ('PIM_WR_GB', 'PIM_RD_RES'):
        g = b = -1
    elif op == 'PIM_RD_BG':
        b = -1
    return op, c, r, g, b, row, col, record.get('buffer', record.get('pim_buffer', -1))


def audit_run(directory, manifest=None, result=None, write=True):
    directory = Path(directory).resolve()
    simulation = directory / 'simulation'
    manifest = manifest or load_json(directory / 'manifest.json')
    result = result or load_json(simulation / 'result.json')
    records = trace_records(directory / 'workload.trace')
    require(digest(directory / 'workload.trace') == digest(simulation / 'input.trace') == result['requests']['sha256'],
            'Native input and source trace hash mismatch')
    counts = Counter(r['operation'] for r in records)
    require(counts == Counter(manifest['counts']), 'Manifest/trace request counts mismatch')
    cfg, timing = result['config'], result['resolved_timings_ck']
    require((cfg['channels'], cfg['ranks'], cfg['frontend_ratio'], cfg['memory_ratio']) == (16, 2, 32, 1),
            'Unsupported audit topology/clocks')
    ratio = 32
    tick = lambda c: (c - 1) * ratio + 1
    barriers = {r['marker']: r['frontend_tick'] for r in read_csv(simulation / 'barriers.csv')}
    require(set(barriers) == {r['address'] for r in records if r['operation'] == 'BARRIER'}, 'Barrier coverage mismatch')
    phase_by_line = {}
    for phase in manifest['segments']:
        for line in range(phase['line_start'], phase['line_end'] + 1):
            phase_by_line[line] = phase

    queues = defaultdict(deque)
    for record in records:
        if record['operation'] in ('PIM_RD_Q', 'PIM_MAC_QK', 'PIM_RD_BG', *LEGACY_NATIVE.values()):
            queues[key(record)].append(record)
        if record['operation'] == 'PIM_RD_Q':
            queues[key(record, 'PIM_RELAY_Q')].append(record)
    rich, legacy, native, hashes = [], [], [], {}
    for channel in range(16):
        ep, np = simulation / f'bgk_events.csv.ch{channel}', simulation / f'timeline.csv.ch{channel}'
        require(ep.exists() and np.exists(), f'Missing channel {channel} traces')
        hashes[ep.name], hashes[np.name] = digest(ep), digest(np)
        erows, nrows = read_csv(ep), read_csv(np)
        require(all(e['Channel'] == channel for e in erows+nrows), f'Wrong channel trace {channel}')
        # RD_Q is logged when data returns but records the original RD clock;
        # callbacks may therefore appear after a later-issued MAC in the file.
        used_native = Counter()
        native_index = defaultdict(list)
        for e in nrows:
            native_index[e['clock'], e['command']].append(e)
        for event in erows:
            op = event['operation']
            require(op in RICH, f'Unexpected rich operation {op}')
            k = key(event)
            require(bool(queues[k]), f'Unexpected event identity {k}')
            source = queues[k].popleft()
            event.update(source_line=source['line'], source_address=source['address'])
            require(event['target_banks'] == TARGETS[op], f'Wrong multicast/compute fanout {event}')
            require(event['bytes'] == 32, f'Wrong transaction size {event}')
            service = (timing['nCL'] + timing['nBL_min'] if op == 'PIM_RD_Q' else
                       timing['nCCDQK'] if op == 'PIM_MAC_QK' else timing['nBL_min'])
            require(event['service_cycles'] == service, f'Wrong service cycles {event}')
            require(event['ready_at'] == event['clock'] + service, f'Missing arrival/completion service {event}')
            nk = event['clock'], RICH_NATIVE[op]
            matched = native_index[nk]
            require(len(matched) == 1, f'Rich event missing unique native command {event}')
            n = matched[0]
            # The source transaction is an ordinary Read (no PIM buffer
            # selector). Its owning composite/relay retains the destination.
            expected_buffer = -1 if op == 'PIM_RD_Q' else event['buffer']
            require((n['Row'], n['Column'], n['pim_buffer']) == (event['Row'], event['Column'], expected_buffer),
                    f'Rich/native address or buffer mismatch {event}')
            # MAC scopes have wildcard bank selectors. RD source geometry is exact.
            if op == 'PIM_RD_Q':
                require((n['Rank'], n['BankGroup'], n['Bank']) == (event['Rank'], event['BankGroup'], event['Bank']),
                        'Q read native source-bank mismatch')
            used_native[nk] += 1
            rich.append(event)
        for event in nrows:
            nk = event['clock'], event['command']
            if used_native[nk]:
                used_native[nk] -= 1
                continue
            op = LEGACY_NATIVE.get(event['command'])
            if op is None:
                continue
            k = key(event, op)
            require(bool(queues[k]), f'Unexpected legacy native identity {k}')
            source = queues[k].popleft()
            item = dict(event, operation=op, source_line=source['line'], source_address=source['address'],
                        buffer=event['pim_buffer'])
            service = (timing['nCCDAB'] if op == 'PIM_MAC_AB' else
                       timing['nCL'] + timing['nBL_min'] if op == 'LD' else
                       timing['nCWL'] + timing['nBL_max'] + timing['nWR'] if op == 'ST' else timing['nBL_min'])
            item.update(service_cycles=service, ready_at=item['clock']+service)
            legacy.append(item)
        require(not any(used_native.values()), 'Rich/native command multiplicity mismatch')
        native.extend(nrows)
    require(not any(queues.values()), 'Native dropped requested operation/address')
    events = rich + legacy
    native_counts = Counter(e['command'] for e in native)
    require(native_counts == Counter({k: v for k, v in result['issued_commands'].items() if v}),
            'Native/result issued-command counters mismatch')
    groups = dict(WRGB=('PIM_WR_GB', 'PIM_RD_Q'), MACAB=('PIM_MAC_AB',), MACQK=('PIM_MAC_QK',),
                  RDRES=('PIM_RD_RES', 'PIM_RD_BG'), RD=('LD', 'PIM_RD_Q'), WR=('ST',))
    for command, ops in groups.items():
        require(native_counts[command] == sum(counts[o] for o in ops), f'Wrong native {command} work count')
    require(counts['PIM_WR_Q'] == counts['PIM_WR_BG'] == counts['PIM_LOCAL_WR'] == 0,
            'Obsolete GPU-Q/BG-X/local-K path present')
    for event in events:
        phase = phase_by_line[event['source_line']]
        require(barriers[phase['start']] <= tick(event['clock']), f'Event precedes phase {event}')
        require(tick(event['ready_at']) <= barriers[phase['end']], f'Phase ends before operation completion {event}')

    # Each Q read requires a return before its relay. The explicit 32B staging
    # slot cannot accept another read until the previous relay has completed.
    reads = {e['source_line']: e for e in rich if e['operation'] == 'PIM_RD_Q'}
    relays = {e['source_line']: e for e in rich if e['operation'] == 'PIM_RELAY_Q'}
    require(reads.keys() == relays.keys(), 'Q read/relay identity coverage mismatch')
    for line, read in reads.items():
        relay = relays[line]
        require(key(read)[1:] == key(relay)[1:], 'Q read/relay source or destination mismatch')
        require(relay['clock'] >= read['ready_at'], 'Q relay starts before DRAM data return')
        require(read['Rank'] == read['BankGroup'] == 0, 'Q source must reside at rank0 BG0')
    for c in range(16):
        ordered = sorted((e for e in reads.values() if e['Channel'] == c), key=lambda e:e['clock'])
        for a, b in zip(ordered, ordered[1:]):
            require(b['clock'] >= relays[a['source_line']]['ready_at'], 'Q staging buffer overwritten before relay completes')

    fragment_by_line, declared = {}, {}
    for fragment in manifest['input_fragments']:
        require((fragment['operator'], fragment['fragment']) not in declared,
                'Repeated whole input fragment declaration')
        declared[fragment['operator'], fragment['fragment']] = fragment
        for line in range(fragment['line_start'], fragment['line_end']+1):
            fragment_by_line[line] = fragment
    operators = {operator for operator, _ in declared}
    for operator in operators:
        expected = set(range(4 if operator == 'qk' else 62 if operator == 'pv' else 64))
        require({fragment for op, fragment in declared if op == operator} == expected,
                f'Missing/extra whole input fragment for {operator}')
    fills, macs = defaultdict(list), defaultdict(list)
    for event in events:
        op = event['operation']
        if op in ('PIM_WR_GB', 'PIM_RELAY_Q'):
            require(event['source_line'] in fragment_by_line, 'Input outside declared fragment')
            fragment = fragment_by_line[event['source_line']]
            generation = fragment['operator'], fragment['fragment'], event['Channel']
            require(event['buffer'] == fragment['buffer'], 'Fragment/issued buffer mismatch')
            fills[generation].append(event)
        elif op in ('PIM_MAC_AB', 'PIM_MAC_QK'):
            phase = phase_by_line[event['source_line']]
            operator = phase['operator']
            offset = event['Row']*2048+event['Column']*32
            fragment = ((offset-K_BASE)//32)%4 if operator == 'qk' else (
                (offset-V_BASE)//128 if operator == 'pv' else offset//384)
            generation = operator, fragment, event['Channel']
            require((operator, fragment) in declared, 'MAC uses undeclared input fragment')
            macs[generation].append(event)
    for generation, consumers in macs.items():
        operator, fragment, channel = generation
        require(generation in fills, f'MAC without full input generation {generation}')
        inputs = fills[generation]
        require(min(e['clock'] for e in consumers) >= max(e['ready_at'] for e in inputs),
                f'MAC starts before whole input fragment ready {generation}')
        require({e['buffer'] for e in consumers} == {e['buffer'] for e in inputs}, f'Wrong MAC input slot {generation}')
        expected_count = 62 if operator == 'qk' else 12 if operator == 'qkv' else 4
        require(len(consumers) == expected_count, f'Missing/duplicate MAC outputs {generation}')
        input_offsets = [e['Row']*2048+e['Column']*32 for e in inputs]
        if operator == 'qk':
            expected = [Q_BASE+32*(fragment*20+q) for q in range(20)]
            for bank in range(4):
                delivered = [e for e in inputs if e['Bank'] == bank]
                require(sorted(e['Row']*2048+e['Column']*32 for e in delivered) == expected,
                        f'Missing/wrong Q vectors for bank {bank}, {generation}')
            expected_macs = [K_BASE+32*(slot*4+fragment) for slot in range(62)]
        else:
            require(sorted(input_offsets) == list(range(0, 640, 32)), f'Missing/wrong X/P vector fill {generation}')
            require(all(e['Rank'] == 0 and e.get('pim_rank_mask') == 3 for e in inputs),
                    'Legacy X/P multicast must target both rank input buffers')
            if operator == 'pv':
                expected_macs = [V_BASE+32*(fragment*4+slot) for slot in range(4)]
            else:
                slots = range(12) if operator == 'qkv' else range(('q','k','v').index(operator)*4, ('q','k','v').index(operator)*4+4)
                expected_macs = [32*(fragment*12+slot) for slot in slots]
        require(sorted(e['Row']*2048+e['Column']*32 for e in consumers) == expected_macs,
                f'Wrong weights/cache columns {generation}')
    require(set(fills) == set(macs), 'Filled inputs lack consumers or vice versa')
    require(set(fills) == {(op, fragment, c) for op, fragment in declared for c in range(16)},
            'A complete fragment is missing from one or more channels')
    for c in range(16):
        for buffer in range(2):
            generations = [g for g in fills if g[2] == c and fills[g][0]['buffer'] == buffer]
            generations.sort(key=lambda g:min(e['source_line'] for e in fills[g]))
            for earlier, later in zip(generations, generations[1:]):
                first_fill = min(e['clock'] for e in fills[later])
                require(first_fill >= max(e['ready_at'] for e in macs[earlier]),
                        f'Input slot overwritten while prior MAC consumes it {earlier}->{later}')

    # Four feature partials must exist before each complete BG score is sent.
    qk_ready = defaultdict(dict)
    for (operator, fragment, c), consumers in macs.items():
        if operator == 'qk':
            for e in consumers:
                slot = (e['Row']*2048+e['Column']*32-K_BASE)//128
                qk_ready[c, slot][fragment] = e['ready_at']
    for event in rich:
        if event['operation'] == 'PIM_RD_BG':
            offset = event['Row']*2048+event['Column']*32
            for slot in range(offset//80, (offset+31)//80+1):
                ready = qk_ready[event['Channel'], slot]
                require(set(ready) == {0,1,2,3}, 'BG score export lacks all feature fragments')
                require(event['clock'] >= max(ready.values()), 'BG score exported before final partial completes')

    # Ordinary Q/K/V reads and stores use genuine DRAM commands. Check open
    # row state independently, including the source bank of every Q read.
    row_checks = 0
    for c in range(16):
        opened = {}
        for e in sorted((e for e in native if e['Channel'] == c), key=lambda e:e['clock']):
            cmd = e['command']
            ranks = range(2) if e['Rank'] < 0 else [e['Rank']]
            groups_ = range(4) if e['BankGroup'] < 0 else [e['BankGroup']]
            banks_ = range(4) if e['Bank'] < 0 else [e['Bank']]
            targets = [(r,g,b) for r in ranks for g in groups_ for b in banks_]
            if cmd in ('PREab','REFab','ACTAB'):
                targets = [(r,g,b) for r in ranks for g in range(4) for b in range(4)]
            if cmd in ('ACT2','ACTAB'):
                for bank in targets:
                    opened[bank] = e['Row']
            elif cmd in ('PREpb','PREab','PREHB','REFab'):
                for bank in targets:
                    opened.pop(bank, None)
            elif cmd in ('RD','WR'):
                bank = e['Rank'], e['BankGroup'], e['Bank']
                require(opened.get(bank) == e['Row'], f'{cmd} wrong/closed row ch{c} bank{bank}')
                row_checks += 1
    require(row_checks == counts['LD']+counts['ST']+counts['PIM_RD_Q'], 'Not every ordinary data command row audited')
    traffic = result['traffic_bytes']
    expected_traffic = dict(pim_input_write=counts['PIM_WR_GB']*32,
                           q_dram_read=counts['PIM_RD_Q']*32,
                           q_internal_relay=counts['PIM_RD_Q']*32,
                           pim_result_read=(counts['PIM_RD_RES']+counts['PIM_RD_BG'])*32,
                           dram_read=(counts['LD']+counts['PIM_RD_Q'])*32,
                           dram_write=counts['ST']*32)
    for name, value in expected_traffic.items():
        require(traffic.get(name,0) == value, f'Traffic accounting mismatch {name}')

    # Verify GPU-origin cache stores against independent logical ownership,
    # not just the manifest or the workload's own address helper.
    expected_stores, expected_reads = Counter(), Counter()
    def location(c, r, g, b, offset):
        return c, r, g, b, offset//2048, offset%2048//32
    projection_ops = operators & {'qkv','q','k','v'}
    if manifest.get('mode') == 'complete' and projection_ops:
        projected = {'q','k','v'} if 'qkv' in projection_ops else projection_ops
        if 'q' in projected:
            expected_stores.update(location(c,0,0,b,Q_BASE+32*(fragment*20+q))
                for c in range(16) for b in range(4) for fragment in range(4) for q in range(20))
        if 'k' in projected:
            expected_stores.update(location(2*h+t%2,t//2%2,t//4%4,b,K_BASE+32*(t//16*4+fragment))
                for h in range(8) for t in range(968,988) for b in range(4) for fragment in range(4))
        if 'v' in projected:
            for token in (960,976):
                for h in range(8):
                    for d in range(256):
                        loc = location(2*h+d//32%2,d//16%2,d//4%4,d%4,V_BASE+32*(token//16*4+d//64))
                        expected_stores[loc] += 1
                        if token == 960:
                            expected_reads[loc] += 1
    actual_stores = Counter(decode(e['source_address']) for e in legacy if e['operation'] == 'ST')
    actual_reads = Counter(decode(e['source_address']) for e in legacy if e['operation'] == 'LD')
    require(actual_stores == expected_stores, 'Cache store layout does not match independent Q/K/V ownership')
    require(actual_reads == expected_reads, 'V tail preservation reads do not match independent ownership')
    for c in range(16):
        projection_macs = [e for e in legacy if e['Channel'] == c and e['operation'] == 'PIM_MAC_AB']
        exports = [e for e in legacy if e['Channel'] == c and e['operation'] == 'PIM_RD_RES']
        if projection_macs and exports:
            require(min(e['clock'] for e in exports) >= max(e['ready_at'] for e in projection_macs),
                    'Result export before full input reduction completes')
        if manifest['case'] == 'chain_qkv_qk':
            stores = [e for e in legacy if e['Channel'] == c and e['operation'] == 'ST']
            require(min(e['clock'] for e in reads.values() if e['Channel'] == c) >= max(e['ready_at'] for e in stores),
                    'Chained QK starts before preceding DRAM stores complete')
    report = dict(status='passed', case=manifest['case'], schedule=manifest.get('schedule'),
        native_event_count=len(rich), native_command_count=len(native), source_request_counts=dict(counts),
        new_event_counts=dict(Counter(e['operation'] for e in rich)), input_generations_checked=len(fills),
        q_read_relay_pairs_checked=len(reads), ordinary_data_row_checks=row_checks, local_store_row_checks=0,
        cache_store_layout_bursts_checked=sum(expected_stores.values()),
        channels_checked=16, legacy_wrgb_completion_checked=True,
        checks=['exact source/native address and multiplicity', 'native RD return before Q relay',
                'single-entry controller staging lifetime', 'bank-index Q multicast/full 20-vector fragment',
                'shared X/P whole-fragment readiness and double-buffer overwrite',
                'projection and QK full reduction before export', 'ordinary RD/WR row state',
                'phase completion fences and chain DRAM visibility', 'GPU input excludes internal Q relay'],
        limitations=['Timing-only backend does not compute data values',
                     'GPU RoPE/conversion/repacking arithmetic and host DMA unmeasured',
                     'Internal multicast and BG reduction sustain command stream; independent network delay not measured'],
        trace_sha256=digest(directory/'workload.trace'), native_trace_hashes=hashes)
    if write:
        (directory/'independent-native-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    cases = [args.path] if (args.path/'manifest.json').exists() else sorted(p for p in args.path.iterdir()
        if p.is_dir() and (p/'manifest.json').exists() and (p/'simulation/result.json').exists())
    require(bool(cases), 'No completed cases found')
    reports = []
    for directory in cases:
        manifest = load_json(directory/'manifest.json')
        if manifest.get('case') not in ('qkv','q','k','v','qk','pv','chain_qkv_qk'):
            continue
        report = audit_run(directory, manifest)
        reports.append(dict(directory=str(directory), case=report['case'], status=report['status'],
                            events=report['native_event_count'], q_pairs=report['q_read_relay_pairs_checked']))
    require(bool(reports), 'No attention cases audited')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()

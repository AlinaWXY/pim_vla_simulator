"""Independent native-event audit for whole-horizon bank-local RoPE.

This module deliberately does not import the trace generator or its address
helpers.  Expected geometry and payloads are derived here from the experiment's
architectural contract.  A successful audit establishes timing/data placement,
not numerical RoPE correctness or measured GPU execution.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path


CHANNELS, HEADS, QUERIES, HEAD_DIM = 16, 8, 16, 256
ACTIVE_RANK, BGS, BANKS = 1, 4, 4
BURST, ROW_BYTES = 32, 2048
WEIGHT_BASE, PREFIX_BASE, WORK_BASE = 0, 65536, 81920
WORK_ROW = WORK_BASE // ROW_BYTES
LOCAL_PREFIX, LOCAL_SUFFIX, LOCAL_PADDED = 484, 8, 496
GEOMETRY = ('Channel', 'Rank', 'BankGroup', 'Bank', 'Row', 'Column')
ROPE_OPS = ('PIM_CAST', 'PIM_SPILL', 'PIM_ROPE_COEF', 'PIM_ROPE_LOAD',
            'PIM_ROPE_EXEC', 'PIM_ROPE_STORE')
EXTERNAL_COMMANDS = {'RD', 'RDA', 'WR', 'WRA', 'WRGB', 'RDRES'}
NATIVE_FOR_EVENT = dict(PIM_WR_GB='WRGB', PIM_MAC_AB='MACAB',
                       PIM_SPILL='LOCALWR', PIM_ROPE_COEF='LOCALRD',
                       PIM_ROPE_LOAD='LOCALRD', PIM_ROPE_STORE='LOCALWR',
                       PIM_Q_EXPORT='RD', PIM_Q_REMOTE='WRGB',
                       PIM_MAC_QK='MACQK', PIM_RD_BG='RDRES')
INTERNAL_EVENTS = {'PIM_CAST', 'PIM_ROPE_EXEC', 'PIM_RANK_REDUCE'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return [{k: v if k in ('operation', 'command') else int(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, address >> 20, address >> 9 & 63)


def location(event):
    return tuple(event[k] for k in GEOMETRY)


def offset(event):
    return event['Row'] * ROW_BYTES + event['Column'] * BURST


def trace_records(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        record = dict(line=number, operation=fields[0], address=int(fields[1], 0),
                      buffer=int(fields[2]) if len(fields) > 2 else -1)
        record.update(zip(GEOMETRY, decode(record['address'])))
        result.append(record)
    return result


def bank_work_locations(columns):
    return Counter((c, ACTIVE_RANK, g, b, WORK_ROW, column)
                   for c in range(CHANNELS) for g in range(BGS)
                   for b in range(BANKS) for column in columns)


def audit_peer_dependencies(exports, remotes):
    """A remote fill must use the completed export of its exact peer bank."""
    source = {}
    for event in exports:
        key = location(event)
        require(key not in source, f'Duplicate Q export {key}')
        source[key] = event
    consumed = Counter()
    for event in remotes:
        peer = (event['Channel'] ^ 1, *location(event)[1:])
        require(peer in source, f'Remote Q fill has no peer export {peer}')
        require(event['clock'] >= source[peer]['ready_at'],
                f'Remote Q fill precedes peer export completion {peer}')
        require(event['ready_at'] > event['clock'], 'Remote fill has zero completion latency')
        consumed[peer] += 1
    require(consumed == Counter({key: 1 for key in source}),
            'Q forwarding is not one-to-one between peer-channel banks')


def audit_k_scan(macs):
    """Independently require one physical K burst per bank for all16 queries."""
    expected = Counter()
    for c in range(CHANNELS):
        for token in range(LOCAL_PADDED):
            byte = (PREFIX_BASE + token * BURST if token < LOCAL_PREFIX else
                    WORK_BASE + (2 * LOCAL_SUFFIX + token - LOCAL_PREFIX) * BURST)
            expected[c, byte // ROW_BYTES, byte % ROW_BYTES // BURST] += 1
    actual = Counter((e['Channel'], e['Row'], e['Column']) for e in macs)
    require(actual == expected, 'K scan missing/duplicates a token or scans K once per query')
    for e in macs:
        require(e['Rank'] == ACTIVE_RANK and e['target_banks'] == 16,
                'QK compute does not cover exactly one PIM rank')
        require(e['service_cycles'] == QUERIES and e['ready_at'] == e['clock'] + QUERIES,
                'QK must reuse each K burst for all16 query vectors with completion')
    for channel in range(CHANNELS):
        ordered = sorted((e for e in macs if e['Channel'] == channel), key=lambda e: e['clock'])
        require(all(b['clock'] >= a['ready_at'] for a, b in zip(ordered, ordered[1:])),
                'QK MAC service overlaps within one rank')
    return len(macs) * 16 * BURST


def audit_score_readiness(scores, reductions):
    """Each exported score depends on its own K token, including tiled output.

    Score storage is token-major: each token has16 FP32 query scores (64B),
    exported as two32B bursts. Later K tiles need not have executed when an
    earlier completed tile is exported.
    """
    ready = {}
    for reduction in reductions:
        byte = offset(reduction)
        if PREFIX_BASE <= byte < PREFIX_BASE + LOCAL_PREFIX * BURST:
            token = (byte - PREFIX_BASE) // BURST
        elif WORK_BASE + 2 * LOCAL_SUFFIX * BURST <= byte < WORK_BASE + (2 * LOCAL_SUFFIX + LOCAL_PADDED - LOCAL_PREFIX) * BURST:
            token = LOCAL_PREFIX + (byte - WORK_BASE) // BURST - 2 * LOCAL_SUFFIX
        else:
            raise AssertionError('Rank reduction refers to an unknown K storage column')
        key = reduction['Channel'], token
        require(key not in ready, 'Duplicate completion for one rank K token')
        ready[key] = reduction['ready_at']
    for score in scores:
        byte = offset(score)
        require(byte % BURST == 0 and 0 <= byte < LOCAL_PADDED * QUERIES * 4,
                'Score burst outside the token-major output range')
        first = byte // (QUERIES * 4)
        last = (byte + BURST - 1) // (QUERIES * 4)
        for token in range(first, last + 1):
            key = score['Channel'], token
            require(key in ready, 'Score export has no corresponding completed rank reduction')
            require(score['clock'] >= ready[key],
                    f'Score export precedes its K-token rank reduction: channel{key[0]} token{key[1]}')


def audit_external_payload(native, exports, remotes, score_exports, x_count):
    """Count issued native DQ commands; no manifest traffic totals are trusted."""
    native_dq = Counter(e['command'] for e in native if e['command'] in EXTERNAL_COMMANDS)
    expected_commands = Counter(RD=len(exports), WRGB=x_count + len(remotes),
                                RDRES=len(score_exports))
    require(native_dq == expected_commands,
            f'Unexpected/unaccounted external DQ command: {native_dq} != {expected_commands}')
    payloads = dict(x_input=x_count * BURST,
                    q_export=len(exports) * BURST,
                    q_remote_fill=len(remotes) * BURST,
                    score_export=len(score_exports) * BURST)
    expected_bytes = dict(x_input=16 * 64 * LOCAL_SUFFIX * BURST,
                          q_export=256 * LOCAL_SUFFIX * BURST,
                          q_remote_fill=256 * LOCAL_SUFFIX * BURST,
                          score_export=16 * LOCAL_PADDED * QUERIES * 4)
    require(payloads == expected_bytes, f'External payload conservation failed: {payloads}')
    payloads['total'] = sum(payloads.values())
    require(payloads['total'] == 24576 * QUERIES + 64 * LOCAL_PADDED * QUERIES, 'Steady external DQ payload mismatch')
    return payloads


def audit_rows(native):
    """Replay actual ACT/PRE/refresh commands rather than trusting addresses."""
    checks = 0
    for channel in range(CHANNELS):
        opened = {}
        events = sorted((e for e in native if e['Channel'] == channel), key=lambda e: e['clock'])
        for event in events:
            command = event['command']
            ranks = range(2) if event['Rank'] < 0 else [event['Rank']]
            groups = range(BGS) if event['BankGroup'] < 0 else [event['BankGroup']]
            banks = range(BANKS) if event['Bank'] < 0 else [event['Bank']]
            if command in ('ACTAB', 'PREab', 'REFab'):
                groups, banks = range(BGS), range(BANKS)
            targets = [(r, g, b) for r in ranks for g in groups for b in banks]
            if command in ('ACT2', 'ACTAB', 'ACTHB'):
                for bank in targets:
                    opened[bank] = event['Row']
            elif command in ('PREpb', 'PREab', 'PREHB', 'REFab', 'REFpb'):
                for bank in targets:
                    opened.pop(bank, None)
            elif command in ('RD', 'RDA', 'WR', 'WRA', 'LOCALRD', 'LOCALWR', 'MACAB', 'MACQK'):
                require(event['Rank'] == ACTIVE_RANK,
                        f'Workload array access touches reserved GPU rank: {event}')
                for bank in targets:
                    require(opened.get(bank) == event['Row'],
                            f'{command} accesses closed/wrong row ch{channel} bank{bank}')
                    checks += 1
                if command in ('RDA', 'WRA'):
                    for bank in targets:
                        opened.pop(bank, None)
    require(checks > 0, 'No native DRAM row accesses audited')
    return checks


def audit_rope_dependencies(events, pair_ii, pair_latency):
    require(pair_ii > 0 and pair_latency > 0, 'RoPE pair II/latency must both be positive')
    by_bank = defaultdict(list)
    for event in events:
        if event['operation'] in ROPE_OPS:
            require(event['Row'] == WORK_ROW, 'Projection spill/RoPE left the local work row')
            require(event['Rank'] == ACTIVE_RANK and event['target_banks'] == 1,
                    'RoPE local operation is not bank-local in PIM rank1')
            require(event['service_cycles'] > 0 and event['ready_at'] == event['clock'] + event['service_cycles'],
                    'Local operation lacks positive completion latency')
            by_bank[location(event)[:4]].append(event)
    require(len(by_bank) == CHANNELS * BGS * BANKS, 'Missing active RoPE bank')
    jobs = 0
    for bank, bank_events in by_bank.items():
        casts, spilled, executed = {}, {}, {}
        coefficient, loaded = None, {}
        input_busy = defaultdict(lambda: -1)
        last_exec = None
        for event in sorted(bank_events, key=lambda e: (e['clock'], e['operation'])):
            op, column, slot = event['operation'], event['Column'], event['buffer']
            if op == 'PIM_CAST':
                require(LOCAL_SUFFIX <= column < 3 * LOCAL_SUFFIX and column not in casts, 'Wrong/duplicate FP32-to-BF16 cast')
                casts[column] = event['ready_at']
            elif op == 'PIM_SPILL':
                require(LOCAL_SUFFIX <= column < 3 * LOCAL_SUFFIX and column not in spilled, 'Wrong/duplicate projection spill')
                require(column in casts and event['clock'] >= casts[column],
                        'Projection spill precedes required BF16 rounding')
                spilled[column] = event['ready_at']
            elif op == 'PIM_ROPE_COEF':
                require(0 <= column < LOCAL_SUFFIX, 'RoPE coefficient address outside coefficient columns')
                if last_exec is not None:
                    require(event['clock'] >= last_exec['clock'] + 8 * pair_ii,
                            'Coefficients overwritten before the final pair consumes them')
                coefficient = event
            elif op == 'PIM_ROPE_LOAD':
                require(LOCAL_SUFFIX <= column < 3 * LOCAL_SUFFIX and column in spilled, 'RoPE consumes an unproduced Q/K projection')
                require(event['clock'] >= spilled[column], 'RoPE load precedes projection spill completion')
                require(event['clock'] >= input_busy[slot], 'RoPE input buffer overwritten while active')
                require(slot not in loaded, 'RoPE input buffer overwrites an unconsumed vector')
                loaded[slot] = event
            elif op == 'PIM_ROPE_EXEC':
                require(coefficient is not None and slot in loaded, 'RoPE executes without input/coefficient')
                require(column == loaded[slot]['Column'], 'RoPE executes on the wrong Q/K input')
                require(coefficient['Column'] == (column - LOCAL_SUFFIX) % LOCAL_SUFFIX, 'Wrong token coefficient selected for RoPE')
                require(event['clock'] >= max(coefficient['ready_at'], loaded[slot]['ready_at'], input_busy[slot]),
                        'RoPE executes before local input/coefficient is ready')
                if last_exec is not None:
                    require(event['clock'] >= last_exec['clock'] + 8 * pair_ii,
                            'RoPE issues more than one pair per initiation interval')
                require(event['service_cycles'] == 7 * pair_ii + pair_latency,
                        'Eight-pair RoPE latency does not include seven initiation intervals and final latency')
                input_busy[slot] = event['clock'] + 8 * pair_ii
                require(column not in executed, 'Duplicate RoPE execution for one output')
                executed[column] = event
                del loaded[slot]
                last_exec = event
                jobs += 1
            elif op == 'PIM_ROPE_STORE':
                require(column in executed, 'RoPE stores wrong/missing result')
                require(event['clock'] >= executed[column]['ready_at'], 'RoPE output stored before final pair completes')
                del executed[column]
        require(not executed, 'Computed RoPE output was never stored')
    require(jobs == CHANNELS * BGS * BANKS * LOCAL_SUFFIX * 2, 'Missing/duplicate Q/K RoPE operations')
    return jobs


def audit_output_fifo(events, local_write_cycles):
    """A finished vector occupies its output FIFO until its write burst drains."""
    stores = {location(e): e for e in events if e['operation'] == 'PIM_ROPE_STORE'}
    intervals = defaultdict(list)
    for e in events:
        if e['operation'] != 'PIM_ROPE_EXEC':
            continue
        require(location(e) in stores, 'RoPE result has no store to release its output FIFO')
        release = stores[location(e)]['clock'] + local_write_cycles
        intervals[location(e)[:4]].extend(((e['clock'], 1), (release, -1)))
    for bank, changes in intervals.items():
        occupancy = 0
        for clock, delta in sorted(changes):
            occupancy += delta
            require(0 <= occupancy <= 2, f'RoPE output FIFO exceeds two entries in {bank} at {clock}')
        require(occupancy == 0, 'RoPE output FIFO never drains')


def expected_geometry(events, operation, columns):
    require(Counter(location(e) for e in events if e['operation'] == operation) == bank_work_locations(columns),
            f'{operation} does not cover each expected bank/token exactly once')


def request_key(event):
    return event['operation'], *location(event), event.get('buffer', -1)


def match_native(events, native):
    """Associate data events with issued commands, preserving multiplicity."""
    index = defaultdict(list)
    for number, event in enumerate(native):
        index[event['Channel'], event['clock'], event['command']].append((number, event))
    used = set()
    for event in events:
        operation = event['operation']
        if operation in INTERNAL_EVENTS:
            continue
        require(operation in NATIVE_FOR_EVENT, f'Unknown native event operation {operation}')
        candidates = []
        for number, command in index[event['Channel'], event['clock'], NATIVE_FOR_EVENT[operation]]:
            if number in used:
                continue
            if command['Rank'] != ACTIVE_RANK or command['Row'] != event['Row'] or command['Column'] != event['Column']:
                continue
            if operation in ('PIM_SPILL', 'PIM_ROPE_COEF', 'PIM_ROPE_LOAD', 'PIM_ROPE_STORE', 'PIM_Q_EXPORT', 'PIM_Q_REMOTE'):
                if any(command[k] not in (event[k], -1) for k in ('BankGroup', 'Bank')):
                    continue
            candidates.append(number)
        require(len(candidates) == 1, f'Event lacks unique matching issued command: {event}')
        used.add(candidates[0])
    relevant = EXTERNAL_COMMANDS | {'LOCALRD', 'LOCALWR', 'MACAB', 'MACQK'}
    require(used == {i for i, e in enumerate(native) if e['command'] in relevant},
            'Issued native data/compute commands lack matching completion events')


def audit_projection(events):
    grouped = defaultdict(list)
    for event in events:
        if event['operation'] in ('PIM_WR_GB', 'PIM_MAC_AB'):
            grouped[event['Channel'], event['operation']].append(event)
    projection_done = {}
    for c in range(CHANNELS):
        fills = sorted(grouped[c, 'PIM_WR_GB'], key=lambda e: e['source_line'])
        macs = grouped[c, 'PIM_MAC_AB']
        require(len(fills) == 64 * LOCAL_SUFFIX, 'Projection X input fragment count is not64x8')
        require(all(e['service_cycles'] == LOCAL_SUFFIX and e['target_banks'] == 16 for e in macs),
                'Projection must process8 local tokens across exactly16 banks')
        require(Counter(offset(e) for e in macs) == Counter(range(0, 65536, BURST)),
                'Q/K projection omitted/repeated weights or output features')
        ordered = sorted(macs, key=lambda e: e['clock'])
        require(all(b['clock'] >= a['ready_at'] for a, b in zip(ordered, ordered[1:])),
                'Projection MAC service overlaps within one rank')
        previous_consumers = {}
        for fragment in range(64):
            inputs = fills[fragment * LOCAL_SUFFIX: (fragment + 1) * LOCAL_SUFFIX]
            consumers = [e for e in macs if offset(e) // 1024 == fragment]
            require(len(consumers) == 32, 'Projection fragment does not compute both16-feature outputs')
            require(Counter(offset(e) for e in inputs) == Counter(range(0, LOCAL_SUFFIX * BURST, BURST)),
                    'Projection fragment lacks one or more of its8 distinct input vectors')
            require({e['buffer'] for e in inputs + consumers} == {fragment % 2},
                    'Projection uses wrong double-buffer generation')
            require(min(e['clock'] for e in consumers) >= max(e['ready_at'] for e in inputs),
                    'Projection MAC starts before all8 input vectors arrive')
            if fragment % 2 in previous_consumers:
                require(min(e['clock'] for e in inputs) >= previous_consumers[fragment % 2],
                        'Projection input buffer overwritten before prior MAC completes')
            previous_consumers[fragment % 2] = max(e['ready_at'] for e in consumers)
        projection_done[c] = max(e['ready_at'] for e in macs)
    for event in events:
        if event['operation'] == 'PIM_CAST':
            require(event['clock'] >= projection_done[event['Channel']],
                    'Projection cast begins before the full hidden-dimension reduction completes')
    return projection_done


def audit_fences(records, events, simulation, config):
    barriers = [r for r in records if r['operation'] == 'BARRIER']
    require(barriers, 'Full chain must expose completion fences')
    observed = {r['marker']: r['frontend_tick'] for r in read_csv(simulation / 'barriers.csv')}
    require(set(observed) == {r['address'] for r in barriers}, 'Native barrier coverage mismatch')
    ratio = config['frontend_ratio']
    require(config['memory_ratio'] == 1, 'Audit clock conversion assumes memory_ratio=1')
    tick = lambda clock: (clock - 1) * ratio + 1
    spill_line = {location(e): e['source_line'] for e in events if e['operation'] == 'PIM_SPILL'}
    mac_line = {(e['Channel'], e['Row'], e['Column']): e['source_line']
                for e in events if e['operation'] == 'PIM_MAC_QK'}
    for event in events:
        if event['operation'] == 'PIM_CAST':
            event['source_line'] = spill_line[location(event)]
        elif event['operation'] == 'PIM_RANK_REDUCE':
            event['source_line'] = mac_line[event['Channel'], event['Row'], event['Column']]
        line = event['source_line']
        before = [r for r in barriers if r['line'] < line]
        after = [r for r in barriers if r['line'] > line]
        if before:
            require(tick(event['clock']) >= observed[before[-1]['address']],
                    'Native operation begins before its preceding completion fence')
        require(after, 'Native operation has no final completion fence')
        require(tick(event['ready_at']) <= observed[after[0]['address']],
                'Frontend fence retires before native operation completes')
    return len(observed)


def audit_run(directory, write=True):
    global QUERIES, LOCAL_SUFFIX, LOCAL_PADDED
    directory = Path(directory).resolve()
    simulation = directory / 'simulation'
    result = load_json(simulation / 'result.json')
    require(digest(directory / 'workload.trace') == digest(simulation / 'input.trace') == result['requests']['sha256'],
            'Native input and workload trace hashes differ')
    config, timing = result['config'], result['resolved_timings_ck']
    QUERIES = config.get('pim_horizon', 16)
    require(QUERIES in (16, 20, 30, 40), 'Unsupported horizon')
    LOCAL_SUFFIX = QUERIES // 2
    LOCAL_PADDED = ((968 + QUERIES + 31) // 32) * 16
    require(config['channels'] == 16 and config['ranks'] == 2, 'Wrong physical topology')
    require(config['pim_rank_rope'] and config['pim_mac_scope'] == 'rank' and not config['pim_input_broadcast'],
            'Experiment enables PIM on the reserved GPU rank or lacks rank isolation')
    records = trace_records(directory / 'workload.trace')
    request_queues = defaultdict(deque)
    for record in records:
        if record['operation'] != 'BARRIER':
            require(record['Rank'] == ACTIVE_RANK, 'Workload targets reserved GPU rank')
            request_queues[request_key(record)].append(record)
    events, native, hashes = [], [], {}
    for channel in range(CHANNELS):
        for stem, target in (('bgk_events.csv', events), ('timeline.csv', native)):
            path = simulation / f'{stem}.ch{channel}'
            require(path.exists(), f'Missing channel log {path.name}')
            rows = read_csv(path)
            require(all(e['Channel'] == channel for e in rows), 'Per-channel file contains another channel')
            target.extend(rows)
            hashes[path.name] = digest(path)
    for event in events:
        require(event['Rank'] == ACTIVE_RANK, 'Native PIM completion event targets GPU rank')
        require(event['service_cycles'] > 0 and event['ready_at'] == event['clock'] + event['service_cycles'],
                'Native event has absent/invalid completion timing')
        require(event['bytes'] == BURST, 'Native event transaction width is not32B')
        if event['operation'] in ('PIM_CAST', 'PIM_RANK_REDUCE'):
            continue
        queue = request_queues[request_key(event)]
        require(queue, f'Native event has no matching source request {request_key(event)}')
        event['source_line'] = queue.popleft()['line']
    require(not any(request_queues.values()), 'Source requests missing from native completion logs')
    actual_commands = Counter(e['command'] for e in native)
    workload_commands = EXTERNAL_COMMANDS | {'LOCALRD', 'LOCALWR', 'MACAB', 'MACQK'}
    require(all(e['Rank'] == ACTIVE_RANK and e['pim_rank_mask'] == 2
                for e in native if e['command'] in workload_commands),
            'Issued workload command rank mask is not exactly physical rank1')
    require(actual_commands == Counter({k: v for k, v in result['issued_commands'].items() if v}),
            'Issued native counters disagree with timeline')
    match_native(events, native)
    fence_checks = audit_fences(records, events, simulation, config)
    audit_projection(events)
    for operation in ('PIM_CAST', 'PIM_SPILL', 'PIM_ROPE_LOAD', 'PIM_ROPE_EXEC', 'PIM_ROPE_STORE'):
        expected_geometry(events, operation, range(LOCAL_SUFFIX, 3 * LOCAL_SUFFIX))
    for operation in ('PIM_Q_EXPORT', 'PIM_Q_REMOTE'):
        expected_geometry(events, operation, range(LOCAL_SUFFIX, 2 * LOCAL_SUFFIX))
    coefficients = [e for e in events if e['operation'] == 'PIM_ROPE_COEF']
    coefficient_counts = Counter(location(e) for e in coefficients)
    coefficient_unique = bank_work_locations(range(LOCAL_SUFFIX))
    require(set(coefficient_counts) == set(coefficient_unique) and set(coefficient_counts.values()) <= {1, 2},
            'Coefficients missing or reread more than once per Q/K rotation')
    pair_ii = config['rope_ii_cycles']
    pair_latency = config['rope_latency_cycles']
    services = dict(PIM_CAST=config['pim_cast_cycles'],
                    PIM_SPILL=timing['nLOCALWR'] + timing['nWR'],
                    PIM_ROPE_STORE=timing['nLOCALWR'] + timing['nWR'],
                    PIM_ROPE_COEF=timing['nCL'] + timing['nBL_min'],
                    PIM_ROPE_LOAD=timing['nCL'] + timing['nBL_min'],
                    PIM_Q_EXPORT=timing['nCL'] + timing['nBL_min'],
                    PIM_Q_REMOTE=timing['nBL_min'], PIM_WR_GB=timing['nBL_min'],
                    PIM_RD_BG=timing['nBL_min'], PIM_RANK_REDUCE=config['rank_reduce_cycles'])
    for event in events:
        if event['operation'] in services:
            require(event['service_cycles'] == services[event['operation']],
                    f'Wrong native completion service for {event["operation"]}')
    rope_jobs = audit_rope_dependencies(events, pair_ii, pair_latency)
    audit_output_fifo(events, timing['nLOCALWR'])
    exports = [e for e in events if e['operation'] == 'PIM_Q_EXPORT']
    remotes = [e for e in events if e['operation'] == 'PIM_Q_REMOTE']
    stores = {location(e): e for e in events if e['operation'] == 'PIM_ROPE_STORE'}
    for event in exports:
        require(event['clock'] >= stores[location(event)]['ready_at'], 'Q export precedes rotated DRAM store')
    audit_peer_dependencies(exports, remotes)
    macs = [e for e in events if e['operation'] == 'PIM_MAC_QK']
    internal_k_bytes = audit_k_scan(macs)
    reductions = [e for e in events if e['operation'] == 'PIM_RANK_REDUCE']
    by_column = {(e['Channel'], e['Row'], e['Column']): e for e in macs}
    require(Counter((e['Channel'], e['Row'], e['Column']) for e in reductions) == Counter(by_column.keys()),
            'Each K column needs exactly one complete rank reduction')
    for event in reductions:
        mac = by_column[event['Channel'], event['Row'], event['Column']]
        require(event['clock'] >= mac['ready_at'] and event['target_banks'] == 16,
                'Rank reduction precedes bank MAC completion or omits bank partials')
    score_exports = [e for e in events if e['operation'] == 'PIM_RD_BG']
    audit_score_readiness(score_exports, reductions)
    for channel in range(CHANNELS):
        q_ready = max(e['ready_at'] for e in exports + remotes if e['Channel'] == channel)
        k_ready = max(e['ready_at'] for e in stores.values() if e['Channel'] == channel and e['Column'] >= 2 * LOCAL_SUFFIX)
        require(min(e['clock'] for e in macs if e['Channel'] == channel) >= max(q_ready, k_ready),
                'QK starts before all16 Q vectors and local rotated K are ready')
        scores = [e for e in score_exports if e['Channel'] == channel]
        require(len(scores) == LOCAL_PADDED * QUERIES * 4 // BURST, 'Score stream does not contain16x496 FP32 scores')
        require(Counter(offset(e) for e in scores) == Counter(range(0, LOCAL_PADDED * QUERIES * 4, BURST)),
                'Score export has duplicate/missing logical output offsets')
    payloads = audit_external_payload(native, exports, remotes, score_exports,
                                     sum(e['operation'] == 'PIM_WR_GB' for e in events))
    row_checks = audit_rows(native)
    report = dict(status='passed', variant=f'H{QUERIES} whole-horizon bank-local RoPE', horizon=QUERIES,
                  native_commands=len(native), native_events=len(events),
                  active_pim_rank=1, reserved_gpu_rank=0, channels_checked=16,
                  native_event_counts=dict(Counter(e['operation'] for e in events)),
                  external_payload_bytes=payloads, bank_internal_k_read_bytes=internal_k_bytes,
                  q_peer_transfers_checked=len(remotes), rope_jobs_checked=rope_jobs,
                  rope_pair_ii_cycles=pair_ii, rope_pair_latency_cycles=pair_latency,
                  array_bank_row_checks=row_checks, completion_fences_checked=fence_checks,
                  source_trace_sha256=digest(directory / 'workload.trace'), native_trace_sha256=hashes,
                  limitations=['Timing/data-placement audit; numerical values are not computed',
                               'GPU dispatch overhead and preinstalled weights/prefix/coefficients excluded'])
    if write:
        (directory / 'native-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.path), indent=2))


if __name__ == '__main__':
    main()

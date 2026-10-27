"""Independent native audit for streamed GU->Z DRAM and tiled Down.

The oracle deliberately imports no workload or layout generator. It verifies
issued array commands, rank-local port completion and buffer ownership, not
the numerical GELU/multiply/cast or its unmeasured latency.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path



CHANNELS, BANKS, QUERIES, BURST, MAC_CK = 16, 16, 16, 32, 16
GU_BASE, DOWN_BASE, Z_BASE = 262144, 327680, 360448
Z_ROW = 176
GEOMETRY = ('Channel', 'Rank', 'BankGroup', 'Bank', 'Row', 'Column')
CASE_ALIASES = {'gate_up': 'gate_up_z_store', 'down_tiled': 'down_tiled_dram'}
NATIVE_OPS = dict(WRGB='PIM_WR_GB', MACAB='PIM_MAC_AB', RDRES='PIM_RD_RES',
                 LOCALWR='PIM_Z_STORE', LOCALRD='PIM_Z_LOAD')
WORK_OPS = set(NATIVE_OPS.values())
REQUIRED_COUNTS = {
    'gate_up_z_store': Counter(PIM_WR_GB=16384, PIM_MAC_AB=32768, PIM_Z_STORE=4096),
    'down_tiled_dram': Counter(PIM_Z_LOAD=16384, PIM_MAC_AB=16384, PIM_RD_RES=32768),
}
REQUIRED_COUNTS['combined_timing_chain'] = sum(REQUIRED_COUNTS.values(), Counter())
def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tick(clock, ratio):
    return (clock-1)*ratio+1


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, address >> 20, address >> 9 & 63)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return [{k: v if k in ('operation', 'command') else int(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def location(event):
    return tuple(event[k] for k in GEOMETRY)


def offset(event):
    return event['Row'] * 2048 + event['Column'] * BURST


def address_key(op, geometry, buffer=-1):
    channel, rank, bg, bank, row, column = geometry
    if op in ('PIM_WR_GB', 'PIM_MAC_AB', 'PIM_RD_RES'):
        bg = bank = -1
    return channel, rank, bg, bank, row, column, buffer



def required_counts(case, horizon):
    blocks = (horizon+15)//16
    gu = Counter(PIM_WR_GB=1024*horizon, PIM_MAC_AB=32768, PIM_Z_STORE=4096*blocks)
    down = Counter(PIM_Z_LOAD=16384*blocks, PIM_MAC_AB=16384, PIM_RD_RES=2048*horizon)
    return gu if case == 'gate_up_z_store' else down if case == 'down_tiled_dram' else gu+down


def gather_bytes(column, horizon):
    block = column % ((horizon+15)//16)
    return ((min(16, horizon-16*block)*8+31)//32)*32


def audit_gu_dependencies(writes, macs, *, horizon, burst_ck):
    require(len(writes)==64*horizon and len(macs)==64*32, 'GU incomplete input/weight coverage')
    require(all(b['clock']-a['clock'] >= horizon for a,b in zip(macs,macs[1:])), 'Overlapping GU MAC service')
    for fragment in range(64):
        group, consumers = writes[fragment*horizon:(fragment+1)*horizon], macs[fragment*32:(fragment+1)*32]
        require(all(e['pim_buffer']==fragment%2 for e in group+consumers), 'GU wrong input buffer')
        require([offset(e) for e in group]==list(range(0,horizon*32,32)), 'GU token coverage incomplete')
        require([offset(e) for e in consumers]==[GU_BASE+(fragment*32+s)*32 for s in range(32)], 'GU weight order wrong')
        require(consumers[0]['clock']>=max(e['clock']+burst_ck for e in group), 'GU MAC before input ready')
        if fragment>=2:
            require(group[0]['clock']>=macs[(fragment-1)*32-1]['clock']+horizon, 'GU input buffer overwritten early')


def audit_export_dependency(macs, exports, start, stop, ratio, burst_ck, horizon):
    require(macs and exports, 'Missing Down export producers')
    ready=max(tick(e['clock'],ratio)+horizon*ratio for e in macs)
    require(start>=ready and min(tick(e['clock'],ratio) for e in exports)>=ready, 'Export before MAC completion')
    require(stop>=max(tick(e['clock'],ratio) for e in exports)+max(burst_ck,4)*ratio, 'Missing export tail')

def expected_addresses(case, horizon=16):
    blocks = (horizon+15)//16
    """Fixed original weight mapping, with Z in row176, feature-major columns."""
    case = CASE_ALIASES.get(case, case)
    require(case in REQUIRED_COUNTS, f'Unsupported FFN DRAM case: {case}')
    result = defaultdict(Counter)

    def add(op, ch, byte, buffer=-1, bank=0):
        row, within = divmod(byte, 2048)
        geometry = ch, 1, bank//4, bank % 4, row, within//BURST
        result[op][address_key(op, geometry, buffer)] += 1

    if case != 'down_tiled_dram':
        for fragment in range(64):
            for channel in range(CHANNELS):
                for query in range(horizon):
                    add('PIM_WR_GB', channel, query*BURST, fragment % 2)
                for slot in range(32):
                    add('PIM_MAC_AB', channel, GU_BASE+(fragment*32+slot)*BURST, fragment % 2)
        for channel in range(CHANNELS):
            for bank in range(BANKS):
                for column in range(16*blocks):
                    add('PIM_Z_STORE', channel, Z_BASE+column*BURST, bank=bank)
    if case != 'gate_up_z_store':
        for tile in range(4):
            for fragment in range(16):
                for channel in range(CHANNELS):
                    for bank in range(BANKS):
                        for block in range(blocks):
                            add('PIM_Z_LOAD', channel, Z_BASE+(fragment*blocks+block)*BURST, fragment % 2, bank)
                    for slot in range(16):
                        add('PIM_MAC_AB', channel, DOWN_BASE+((tile*16+fragment)*16+slot)*BURST,
                            fragment % 2)
            for channel in range(CHANNELS):
                for byte in range(0, horizon*16*16*4, BURST):
                    add('PIM_RD_RES', channel, byte)
    return result


def audit_native_scope_rows(native, horizon=16):
    """Replay actual ACT/PRE/refresh; all LOCAL accesses must target only Z."""
    opened, checks = {}, 0
    for event in native:
        command = event['command']
        if command in NATIVE_OPS:
            require(event['Rank'] == 1, 'Workload touches reserved physical rank0')
        if command in ('WRGB', 'MACAB', 'RDRES', 'ACTAB'):
            require(event['Rank'] == 1 and event['pim_rank_mask'] == 2,
                    'Rank command broadcasts beyond physical rank1')
            require(event['BankGroup'] == event['Bank'] == -1,
                    'Rank command does not cover all16 banks')
        if command in ('LOCALRD', 'LOCALWR'):
            require(event['Rank'] == 1 and 0 <= event['BankGroup'] < 4 and 0 <= event['Bank'] < 4,
                    'Z array command must target exactly one rank1 bank')
            require(event['Row'] == Z_ROW and 0 <= event['Column'] < 16*((horizon+15)//16),
                    'Z array access overwrites weights/attention or escapes its allocated row')
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
        elif command in ('MACAB', 'LOCALRD', 'LOCALWR', 'RD', 'RDA', 'WR', 'WRA'):
            for target in targets:
                require(opened.get(target) == event['Row'], 'Native array command accesses closed/wrong row')
                checks += 1
            if command in ('RDA', 'WRA'):
                for target in targets:
                    opened.pop(target, None)
    require(checks, 'No native array operations audited')
    return checks


def audit_native_payload(native, case, horizon=16):
    """Local Z traffic is excluded from DQ only for real LOCALRD/LOCALWR."""
    case = CASE_ALIASES.get(case, case)
    counts = Counter(e['command'] for e in native)
    require(not any(counts[c] for c in ('RD', 'RDA', 'WR', 'WRA', 'MACQK', 'MACK', 'MACHB')),
            'Unexpected external/other array command in FFN DRAM stage')
    work = Counter({op: counts[cmd] for cmd, op in NATIVE_OPS.items() if counts[cmd]})
    require(work == required_counts(case, horizon), 'Native work counts differ from independent FFN dimensions')
    dq = dict(pim_input_write=counts['WRGB']*BURST,
              pim_result_read=counts['RDRES']*BURST, dram_read=0, dram_write=0)
    dq['total'] = sum(dq.values())
    expected = {'gate_up_z_store': 512*1024, 'down_tiled_dram': 1024*1024,
                'combined_timing_chain': 1536*1024}[case]*horizon//16
    require(dq['total'] == expected, 'Incorrect independently derived external DQ')
    return dq, dict(z_array_write=counts['LOCALWR']*BURST,
                    z_array_read=counts['LOCALRD']*BURST,
                    gu_gather=sum(gather_bytes(e['Column'], horizon) for e in native if e['command']=='LOCALWR'),
                    z_return=counts['LOCALWR']*BURST,
                    z_input_gather=counts['LOCALRD']*BURST)


def audit_load_fragment_dependencies(loads, macs, *, tile=0, fragments=16, slots=16,
                                     previous_buffers=None, horizon=16):
    MAC_CK = horizon
    blocks = (horizon+15)//16
    """All16 feature-bank vectors must finish transposing before a MAC reads Z."""
    previous_buffers = {} if previous_buffers is None else previous_buffers
    require(len(loads) == fragments*16*blocks and len(macs) == fragments*slots,
            'Incomplete Z bank/fragment or Down output-slot coverage')
    require(all(b['clock']-a['clock'] >= MAC_CK for a, b in zip(macs, macs[1:])),
            'Overlapping native MAC service windows')
    for fragment in range(fragments):
        group = loads[fragment*16*blocks:(fragment+1)*16*blocks]
        consumers = macs[fragment*slots:(fragment+1)*slots]
        buffer = fragment % 2
        require(all(e['buffer'] == buffer for e in group) and
                all(e['pim_buffer'] == buffer for e in consumers), 'Incorrect Down ping-pong buffer')
        require(Counter((e['BankGroup'], e['Bank'], e['Row'], e['Column']) for e in group) ==
                Counter((bank//4, bank % 4, Z_ROW, fragment*blocks+block) for bank in range(16) for block in range(blocks)),
                'Missing/duplicated bank or wrong feature in complete sixteen-bank input')
        require([offset(e) for e in consumers] ==
                [DOWN_BASE+((tile*16+fragment)*16+slot)*BURST for slot in range(slots)],
                'Down MAC weights changed relative to original feature mapping')
        ready = max(e['ready_at'] for e in group)
        require(consumers[0]['clock'] >= ready, 'Down MAC starts before all16 Z-bank inputs complete')
        if buffer in previous_buffers:
            require(min(e['clock'] for e in group) >= previous_buffers[buffer],
                    'Z input gather overwrites a buffer before its previous MAC completes')
        previous_buffers[buffer] = consumers[-1]['clock']+MAC_CK
    return previous_buffers


def audit_stream_ports(gathers, returns, loads, stores, reads, *, local_ck, read_ck, write_ck,
                       write_consume_ck, producer_ready=None, horizon=16):
    """Validate array<->port causality and bounded staging independently of fields.

    Rich records use native memory clocks. Store gather/return and load gather
    records identify the exact bank/column whose 16-token vector they move.
    """
    producer_ready = producer_ready or {}
    require(local_ck == 2 and read_ck > 0 and write_ck > 0 and write_consume_ck > 0,
            'Invalid local/array service timing')
    for events, byte_count, duration in ((gathers, None, None),
                                         (returns, 32, local_ck), (loads, 32, local_ck)):
        for event in events:
            actual_bytes = gather_bytes(event['Column'], horizon) if byte_count is None else byte_count
            actual_duration = actual_bytes//32*local_ck if duration is None else duration
            require(event['Rank'] == 1 and 0 <= event['BankGroup'] < 4 and 0 <= event['Bank'] < 4,
                    'Rank-local port event has invalid bank/rank')
            require(event['Row'] == Z_ROW and 0 <= event['Column'] < 16*((horizon+15)//16),
                    'Rank-local port event escapes Z feature allocation')
            require(event['bytes'] == actual_bytes and event['ready_at'] == event['clock']+actual_duration,
                    'Incorrect/nonpositive rank-local port completion')
            require(event['bank_mask'] == 1 << (event['BankGroup']*4+event['Bank']) and
                    event['service_cycles'] == actual_duration,
                    'Rank-local transfer has incorrect service/target count')
    for events, duration in ((stores, write_ck), (reads, read_ck)):
        for event in events:
            require(event['bytes'] == 32 and
                    event['bank_mask'] == 1 << (event['BankGroup']*4+event['Bank']) and
                    event['ready_at'] == event['clock']+duration and event['service_cycles'] == duration,
                    'Incorrect/nonpositive native Z array completion')

    def groups(events):
        grouped = defaultdict(list)
        for event in events:
            grouped[location(event), event['buffer']].append(event)
        for records in grouped.values():
            records.sort(key=lambda e: e['clock'])
        return grouped

    ga, re, st = groups(gathers), groups(returns), groups(stores)
    require({k: len(v) for k, v in ga.items()} == {k: len(v) for k, v in re.items()} ==
            {k: len(v) for k, v in st.items()}, 'Missing/duplicated G/U gather, Z return or Z STORE event')
    staging, nonlinear = defaultdict(list), defaultdict(list)
    for key, gs in ga.items():
        for gather, back, store in zip(gs, re[key], st[key]):
            require(gather['clock'] >= producer_ready.get(gather['Channel'], 0),
                    'G/U gather starts before complete GU producer MAC')
            require(back['clock'] >= gather['ready_at'], 'Z returned before complete G/U gather')
            require(store['clock'] >= back['ready_at'], 'LOCALWR starts before Z return completes')
            # Bank32B staging holds returned Z until the local write consumes it.
            staging[location(store)[:4]].append((back['clock'], store['clock']+write_consume_ck))
            # One128B input + one32B output slice; no whole-rank GU/Z SRAM.
            nonlinear[gather['Channel']].append((gather['clock'], back['ready_at']))
    rd, ld = groups(reads), groups(loads)
    require({k: len(v) for k, v in rd.items()} == {k: len(v) for k, v in ld.items()},
            'Missing/duplicated Z array read or local input gather event')
    for key, rs in rd.items():
        for read, load in zip(rs, ld[key]):
            require(load['clock'] >= read['ready_at'], 'Z input gather starts before LOCALRD data completion')
            staging[location(read)[:4]].append((read['ready_at'], load['ready_at']))
    ports = defaultdict(list)
    for event in gathers+returns+loads:
        ports[event['Channel']].append((event['clock'], event['ready_at']))
    for intervals in ports.values():
        ordered = sorted(intervals)
        require(all(b[0] >= a[1] for a, b in zip(ordered, ordered[1:])),
                'Overlapping transfers on the same rank-local32B port')
    for intervals in nonlinear.values():
        ordered = sorted(intervals)
        require(all(b[0] >= a[1] for a, b in zip(ordered, ordered[1:])),
                'More than one G/U->Z slice resident in rank nonlinear staging')
    for intervals in staging.values():
        ordered = sorted(intervals)
        require(all(b[0] >= a[1] for a, b in zip(ordered, ordered[1:])),
                'More than one32B Z slice resident in a bank staging buffer')
    return len(gathers)+len(returns)+len(loads)


def audit_batches(events, native, horizon=16):
    MAC_CK = horizon
    blocks = (horizon+15)//16
    """Generation/mask metadata must agree with the independent bank/weight oracle."""
    selected = defaultdict(list)
    for event in events:
        selected[event['operation']].append(event)
    groups = defaultdict(lambda: defaultdict(list))
    for operation in ('PIM_Z_LOAD', 'PIM_Z_LOAD_PORT', 'PIM_Z_BATCH_READY'):
        for event in selected[operation]:
            require(event['buffer'] in (0, 1) and event['generation'] > 0,
                    'Z input has invalid buffer generation')
            groups[event['Channel'], event['buffer'], event['generation']][operation].append(event)
    for event in selected['PIM_MAC_AB']:
        require(event['ready_at'] == event['clock']+MAC_CK and event['service_cycles'] == MAC_CK,
                'MAC does not have a complete16CK service window')
        require(event['bank_mask'] == 65535, 'MAC does not cover all16 banks')
        if event['generation'] == 0:
            require(GU_BASE <= offset(event) < DOWN_BASE and event['mac_columns'] == 0,
                    'GU generation metadata or weight address is invalid')
        else:
            require(DOWN_BASE <= offset(event) < Z_BASE, 'Z input consumed by wrong weight region')
            groups[event['Channel'], event['buffer'], event['generation']]['PIM_MAC_AB'].append(event)
    generations = defaultdict(list)
    for key, records in groups.items():
        channel, buffer, generation = key
        generations[channel, buffer].append(generation)
        reads, ports, batches, macs = (records[op] for op in
            ('PIM_Z_LOAD', 'PIM_Z_LOAD_PORT', 'PIM_Z_BATCH_READY', 'PIM_MAC_AB'))
        require(len(reads) == len(ports) == 16*blocks and len(macs) == 16 and len(batches) == 1,
                'Z generation lacks16 reads/ports/MAC columns or one batch-ready event')
        column = 2*((generation-1) % 8)+buffer
        tile = (generation-1)//8
        require(tile < 4, 'Too many Down input generations')
        expected_banks = Counter((b//4, b % 4, Z_ROW, column*blocks+block) for b in range(16) for block in range(blocks))
        for rows in (reads, ports):
            require(Counter((e['BankGroup'], e['Bank'], e['Row'], e['Column']) for e in rows) == expected_banks,
                    'Z generation bank coverage is not a full matching feature column')
        batch = batches[0]
        last_port = max(ports, key=lambda e: e['ready_at'])
        require(batch['bank_mask'] == 65535 and batch['mac_columns'] == 0 and batch['bytes'] == horizon*32 and
                batch['clock'] == last_port['clock'] and batch['ready_at'] == last_port['ready_at'] and
                location(batch) == location(last_port), 'Batch readiness disagrees with actual final bank transfer')
        ordered_macs = sorted(macs, key=lambda e: e['clock'])
        require([e['mac_columns'] for e in ordered_macs] == list(range(1, 17)),
                'MAC count does not describe exactly sixteen consumers of this Z generation')
        require([offset(e) for e in ordered_macs] ==
                [DOWN_BASE+((tile*16+column)*16+s)*BURST for s in range(16)],
                'Z generation consumed by wrong Down tile/fragment/weight column')
        require(ordered_macs[0]['clock'] >= batch['ready_at'], 'MAC precedes recorded complete bank batch')
    for values in generations.values():
        require(sorted(values) == list(range(1, 33)), 'Missing/duplicate per-buffer Z generations')
    if groups:
        require(set(generations) == {(ch, b) for ch in range(16) for b in range(2)},
                'Missing channel or ping-pong buffer input generations')
    # Check lifetime against actual native MAC issue clocks, not only rich metadata.
    fragments = 0
    for channel in range(CHANNELS):
        previous = {}
        for tile in range(4):
            macs = sorted([e for e in native if e['Channel'] == channel and e['command'] == 'MACAB'
                           and DOWN_BASE+tile*8192 <= offset(e) < DOWN_BASE+(tile+1)*8192],
                          key=lambda e: e['clock'])
            if not macs:
                continue
            loads = []
            for fragment in range(16):
                key = channel, fragment % 2, tile*8+fragment//2+1
                loads.extend(sorted(groups[key]['PIM_Z_LOAD_PORT'], key=lambda e: e['clock']))
            previous = audit_load_fragment_dependencies(loads, macs, tile=tile,
                                                        previous_buffers=previous, horizon=horizon)
            fragments += 16
    return fragments


def audit_rich_identity(native, events):
    """Every local array/MAC event must have an actual native command at its clock."""
    expected, actual = Counter(), Counter()
    for event in native:
        op = NATIVE_OPS.get(event['command'])
        if op in ('PIM_MAC_AB', 'PIM_Z_STORE', 'PIM_Z_LOAD'):
            expected[(op, event['clock'], address_key(op, location(event), event['pim_buffer']))] += 1
    for event in events:
        op = event['operation']
        require(op in ('PIM_Z_GATHER', 'PIM_Z_RETURN', 'PIM_Z_STORE', 'PIM_Z_LOAD',
                       'PIM_Z_LOAD_PORT', 'PIM_Z_BATCH_READY', 'PIM_MAC_AB'),
                f'Unexpected FFN rich operation: {op}')
        require(event['Rank'] == 1, 'Rich event touches rank0')
        if op in ('PIM_MAC_AB', 'PIM_Z_STORE', 'PIM_Z_LOAD'):
            actual[(op, event['clock'], address_key(op, location(event), event['buffer']))] += 1
        if op in ('PIM_Z_GATHER', 'PIM_Z_RETURN', 'PIM_Z_STORE'):
            require(event['generation'] == 0 and event['mac_columns'] == 0,
                    'Z materialization carries spurious Down generation state')
    require(actual == expected, 'Rich completion event has no matching actual native issue')


def audit_run(directory, manifest=None, result=None):
    directory = Path(directory)
    simulation = directory/'simulation'
    manifest = manifest if manifest is not None else read_json(directory/'manifest.json')
    result = result if result is not None else read_json(simulation/'result.json')
    case = CASE_ALIASES.get(manifest['case'], manifest['case'])
    horizon = manifest['horizon']
    MAC_CK = horizon
    require(horizon in (16, 20, 30, 40), 'Unsupported horizon')
    blocks = (horizon+15)//16
    require(case in REQUIRED_COUNTS, 'Unsupported FFN DRAM stage')
    cfg, timings = result['config'], result['resolved_timings_ck']
    expected_buffers = dict(GU_accumulator_per_bank=128*horizon,
        Down_tiled_accumulator_per_bank=64*horizon, rank_input_double=64*horizon,
        bank_column_staging=32, rank_nonlinear_input_staging=128,
        rank_nonlinear_output_staging=32, rank_full_Z=0, rank_full_GU=0)
    require(manifest['buffers'] == expected_buffers, 'Manifest assumes wrong whole-horizon buffer capacities')
    model = result['ffn_dram_model']
    require(model['input_double_buffer_bytes_per_rank'] == 64*horizon and
            model['z_payload_bytes_per_bank'] == 32*horizon and
            model.get('z_physical_payload_bytes_per_bank', model['z_payload_bytes_per_bank']) == 512*blocks and
            model['rank_resident_z_sram_bytes'] == 0,
            'Native model does not implement declared whole-horizon Z/input capacity')
    require((cfg['channels'], cfg['ranks'], cfg['memory_ratio'], cfg['frontend_ratio']) == (16, 2, 1, 32),
            'Incorrect FFN DRAM physical topology/clock ratio')
    require(cfg['pim'] and cfg.get('pim_ffn_dram', False) and not cfg.get('pim_rank_rope', False)
            and not cfg['pim_input_broadcast'] and cfg['pim_mac_scope'] == 'rank'
            and cfg['pim_data_path'] == 'attacc', 'Incorrect opt-in FFN DRAM native mode')
    require(bool(cfg.get('ffn_z_preinstalled', False)) == (case == 'down_tiled_dram'),
            'Only standalone Down may assume Z is preinstalled')
    ratio, local_ck, burst_ck = cfg['frontend_ratio'], cfg['pim_local_transfer_cycles'], timings['nBL_min']
    require(timings['nCCDAB'] == horizon and cfg.get('pim_horizon',16) == horizon and local_ck == 2 and 0 < burst_ck <= 4,
            'Unexpected MAC/rank-port/DQ timing')
    read_ck, write_consume_ck = timings['nCL']+burst_ck, timings['nLOCALWR']
    write_ck = write_consume_ck+timings['nWR']
    trace_path = directory/'workload.trace'
    require(digest(trace_path) == digest(simulation/'input.trace') == result['requests']['sha256'],
            'Trace hash mismatch')
    trace, source, counts = [], defaultdict(Counter), Counter()
    for line in trace_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        op, values = parts[0], [int(v, 0) for v in parts[1:]]
        require(op in WORK_OPS | {'BARRIER', 'WAIT'}, f'Unexpected source operation: {op}')
        trace.append((op, *values)); counts[op] += 1
        if op in WORK_OPS:
            geometry = decode(values[0])
            require(values[0] % BURST == 0 and geometry[1] == 1, 'Unaligned/rank0 source work')
            if op in ('PIM_WR_GB', 'PIM_MAC_AB', 'PIM_RD_RES'):
                require(geometry[2] == geometry[3] == 0, 'Noncanonical rank source address')
            source[op][address_key(op, geometry, values[1] if len(values) > 1 else -1)] += 1
    require(counts == Counter(manifest['counts']), 'Manifest counts disagree with actual source trace')
    require(Counter({op: counts[op] for op in WORK_OPS if counts[op]}) == required_counts(case, horizon),
            'Source counts disagree with independent matrix dimensions')
    require(not any(e[0] == 'WAIT' and e[1] != MAC_CK and e[1] != 4 for e in trace),
            'Legacy whole-rank gather WAIT or unexpected artificial delay remains')
    oracle = expected_addresses(case, horizon)
    for op in WORK_OPS:
        require(source[op] == oracle[op], f'Source address coverage differs from independent oracle: {op}')

    native, events, hashes = [], [], {}
    for channel in range(CHANNELS):
        np, ep = simulation/f'timeline.csv.ch{channel}', simulation/f'ffn_dram_events.csv.ch{channel}'
        ns, es = read_csv(np), read_csv(ep)
        require(all(e['Channel'] == channel for e in ns+es), 'Native file channel mismatch')
        require(all(a['clock'] <= b['clock'] for a, b in zip(ns, ns[1:])), 'Nonmonotonic native issue clock')
        native.extend(ns); events.extend(es)
        hashes[str(channel)] = dict(timeline=digest(np), ffn_events=digest(ep))
    native_counts = Counter(e['command'] for e in native)
    require(native_counts == Counter({k: v for k, v in result['issued_commands'].items() if v}),
            'Native issued counters disagree with result')
    observed = defaultdict(Counter)
    for event in native:
        op = NATIVE_OPS.get(event['command'])
        if op:
            observed[op][address_key(op, location(event), event['pim_buffer'])] += 1
    for op in WORK_OPS:
        require(observed[op] == source[op], f'Native work differs from source address/buffer multiset: {op}')
    row_checks = audit_native_scope_rows(native, horizon)
    dq, internal = audit_native_payload(native, case, horizon)
    native_traffic = result['traffic_bytes']
    for native_key, oracle_key in (
            ('ffn_z_array_write','z_array_write'), ('ffn_z_array_read','z_array_read'),
            ('ffn_gu_local_gather','gu_gather'), ('ffn_z_local_return','z_return'),
            ('ffn_z_local_load','z_input_gather')):
        require(native_traffic[native_key] == internal[oracle_key],
                f'Result local traffic disagrees with actual native work: {native_key}')
    gu, down = case != 'down_tiled_dram', case != 'gate_up_z_store'
    expected_internal = dict(GU_gather_to_nonlinear=internal['gu_gather'],
        GU_gather_valid=gu*horizon*4096*8,
        GU_gather_padding=internal['gu_gather']-gu*horizon*4096*8,
        Z_return_to_producer_bank=internal['z_return'], Z_dram_write=internal['z_array_write'],
        Z_dram_read=internal['z_array_read'], Z_transpose_to_rank_input=internal['z_input_gather'],
        Z_write_valid=gu*horizon*4096*2,
        Z_write_padding=internal['z_array_write']-gu*horizon*4096*2,
        Z_read_valid=down*4*horizon*4096*2,
        Z_read_padding=internal['z_array_read']-down*4*horizon*4096*2)
    require(all(expected_internal.get(k) == v for k,v in manifest['expected_internal_bytes'].items()),
            'Manifest local valid/padding traffic disagrees with independent native oracle')
    require(manifest['expected_dq_bytes'] == dq, 'Manifest external DQ disagrees with actual native commands')
    for key, value in dq.items():
        if key != 'total':
            require(result['traffic_bytes'][key] == value, f'Result traffic mismatch: {key}')
    require(result['traffic_bytes']['pim_input_delivered'] == dq['pim_input_write'],
            'Unexpected cross-rank input broadcast')
    require(result['traffic_bytes']['pim_internal'] == native_counts['MACAB']*16*BURST,
            'Weight-array bytes differ from sixteen-bank MAC work')
    require(result['traffic_bytes'].get('pim_local_input_write', 0) == 0,
            'Old SRAM-only PIM_LOCAL_GB path remains in DRAM experiment')
    audit_rich_identity(native, events)
    by_op = defaultdict(list)
    for event in events:
        by_op[event['operation']].append(event)
    gu_ready = {ch: max((e['clock']+MAC_CK for e in native if e['Channel'] == ch and
                        e['command'] == 'MACAB' and GU_BASE <= offset(e) < DOWN_BASE), default=0)
                for ch in range(CHANNELS)}
    port_checks = audit_stream_ports(by_op['PIM_Z_GATHER'], by_op['PIM_Z_RETURN'],
        by_op['PIM_Z_LOAD_PORT'], by_op['PIM_Z_STORE'], by_op['PIM_Z_LOAD'],
        local_ck=local_ck, read_ck=read_ck, write_ck=write_ck,
        write_consume_ck=write_consume_ck, producer_ready=gu_ready, horizon=horizon)
    fragments_checked = audit_batches(events, native, horizon)
    if case == 'combined_timing_chain':
        written = {location(e): e['ready_at'] for e in by_op['PIM_Z_STORE']}
        for event in by_op['PIM_Z_LOAD']:
            require(location(event) in written and event['clock'] >= written[location(event)],
                    'Down reads Z before corresponding DRAM store completes')

    barriers = {e['marker']: e['frontend_tick'] for e in read_csv(simulation/'barriers.csv')}
    trace_barriers = {e[1]: i for i, e in enumerate(trace) if e[0] == 'BARRIER'}
    require(list(barriers) == list(trace_barriers) == list(range(len(barriers)))
            and len(trace_barriers) == counts['BARRIER'], 'Barrier coverage mismatch')
    flow, phases, previous_stop, down_tile = [], [], None, 0
    last_producers = {}
    for segment in manifest['segments']:
        start, stop = barriers[segment['start']], barriers[segment['end']]
        require(start < stop and (previous_stop is None or start == previous_stop), 'Noncontiguous phase bounds')
        previous_stop = stop
        body = trace[trace_barriers[segment['start']]+1:trace_barriers[segment['end']]+1]
        body_counts = Counter(e[0] for e in body)
        require(body_counts == Counter(segment['counts']), 'Segment counts differ from source trace')
        selected = [e for e in native if start <= tick(e['clock'], ratio) < stop]
        rich = [e for e in events if start <= tick(e['clock'], ratio) < stop]
        issued = Counter(NATIVE_OPS[e['command']] for e in selected if e['command'] in NATIVE_OPS)
        require(issued == Counter({op: body_counts[op] for op in WORK_OPS if body_counts[op]}),
                'Native work crosses declared phase boundary')
        waits = [e[1] for e in body if e[0] == 'WAIT']
        if body_counts['PIM_WR_GB']:
            require(issued == Counter(PIM_WR_GB=1024*horizon, PIM_MAC_AB=32768) and waits == [horizon],
                    'GU compute must preserve original full weight work and one MAC tail')
            for channel in range(CHANNELS):
                macs = [e for e in selected if e['Channel'] == channel and e['command'] == 'MACAB']
                writes = [e for e in selected if e['Channel'] == channel and e['command'] == 'WRGB']
                audit_gu_dependencies(writes, macs, horizon=horizon, burst_ck=burst_ck)
                require(stop >= tick(macs[-1]['clock']+MAC_CK, ratio), 'GU phase closes before MAC completion')
            flow.append('gu_compute')
        elif body_counts['PIM_Z_STORE']:
            require(issued == Counter(PIM_Z_STORE=4096*blocks) and not waits,
                    'Z store phase missing work or retains artificial bulk-gather delay')
            require(len([e for e in rich if e['operation'] == 'PIM_Z_GATHER']) == 4096*blocks and
                    len([e for e in rich if e['operation'] == 'PIM_Z_RETURN']) == 4096*blocks,
                    'Streamed gather/return crosses materialization phase boundary')
            require(stop >= max(tick(e['ready_at'], ratio) for e in rich if e['operation'] == 'PIM_Z_STORE'),
                    'Z phase closes before local DRAM writes complete')
            flow.append('z_store')
        elif body_counts['PIM_Z_LOAD']:
            require(issued == Counter(PIM_Z_LOAD=4096*blocks, PIM_MAC_AB=4096) and waits == [horizon],
                    'Down tile must retain original weight work and all128KiB Z reads')
            for channel in range(CHANNELS):
                macs = [e for e in selected if e['Channel'] == channel and e['command'] == 'MACAB']
                require(stop >= tick(macs[-1]['clock']+MAC_CK, ratio), 'Down phase closes before MAC completion')
                last_producers[channel] = macs
            down_tile += 1; flow.append('down_compute')
        elif body_counts['PIM_RD_RES']:
            require(issued == Counter(PIM_RD_RES=512*horizon) and waits == [4],
                    'Down export missing full rank partials or movement tail')
            for channel in range(CHANNELS):
                exports = [e for e in selected if e['Channel'] == channel and e['command'] == 'RDRES']
                audit_export_dependency(last_producers.get(channel, []), exports, start, stop, ratio, burst_ck, horizon)
            flow.append('export')
        else:
            raise AssertionError('Unrecognized/empty phase or obsolete whole-rank gather')
        phases.append(dict(name=segment['name'], start_tick=start, stop_tick=stop,
                           latency_us=(stop-start)*result['tCK_ns']/(ratio*1000)))
    expected_flow = ([] if case == 'down_tiled_dram' else ['gu_compute', 'z_store'])
    if case != 'gate_up_z_store':
        expected_flow += ['down_compute', 'export']*4
    require(flow == expected_flow, 'Producer/materialization/Down phase order differs from contract')
    require(phases and phases[0]['start_tick'] == barriers[0] and
            phases[-1]['stop_tick'] == barriers[max(barriers)], 'Unaudited leading/trailing phase')
    boundary = result['memory_cycles']*ratio-sum(p['stop_tick']-p['start_tick'] for p in phases)
    require(0 <= boundary <= 2*ratio, 'Unexpected unaccounted outer boundary time')
    return dict(status='passed', case=case, horizon=horizon, active_rank=1, active_banks=256,
        independent_weight_and_z_address_coverage=True, original_ffn_weight_mapping_preserved=True,
        native_address_multisets_match_trace=True, native_timelines_checked=16,
        bank_array_accesses_checked=row_checks, rank_local_events_checked=port_checks,
        channel_down_fragments_checked=fragments_checked, native_counts=dict(native_counts),
        native_external_dq_bytes=dq, native_internal_z_bytes=internal,
        weight_array_bytes=native_counts['MACAB']*16*BURST, z_external_dq_bytes=0,
        z_bytes_per_bank=512*blocks, bank_z_staging_bytes=32, rank_nonlinear_staging_bytes=160,
        rank_input_double_buffer_bytes=64*horizon, rank_full_z_sram_bytes=0,
        GU_accumulator_bytes_per_bank=128*horizon, Down_accumulator_bytes_per_bank=64*horizon,
        native_internal_payload_and_padding=expected_internal,
        rank_full_gu_sram_bytes=0, local_port_bytes=32, local_port_cycles=local_ck,
        trace_sha256=digest(trace_path), native_sha256=hashes, phases=phases, boundary_ticks=boundary,
        nonlinear_arithmetic_latency_ck=None, numerical_compute=False,
        unmeasured='GELU/multiply/BF16 cast and final GPU reduction; only movement and native array service timed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run)
    (args.run/'native-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

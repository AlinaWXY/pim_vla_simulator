"""Independent all-channel native event, ownership and completion verifier.

Run on one case directory or a run root containing case directories. No source
or result timing is modified; an independent-native-audit.json is produced.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path

NEW = ('PIM_WR_BG','PIM_WR_Q','PIM_MAC_K','PIM_MAC_QK','PIM_RD_BG','PIM_LOCAL_WR')
INPUTS = NEW[:2]
MACS = NEW[2:4]
TARGETS = dict(zip(NEW,(4,8,32,32,4,1)))
GROUPS = {'WRGB':('PIM_WR_GB','PIM_WR_BG','PIM_WR_Q'),
          'RDRES':('PIM_RD_RES','PIM_RD_BG'), 'MACAB':('PIM_MAC_AB',),
          'MACK':('PIM_MAC_K',),'MACQK':('PIM_MAC_QK',),'RD':('LD',),'WR':('ST',)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(address):
    return (address>>5&15,address>>15&1,address>>16&3,address>>18&3,
            address>>20,address>>9&63)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='',encoding='utf-8') as stream:
        return [{key: value if key in ('operation','command') else int(value)
                 for key,value in row.items()} for row in csv.DictReader(stream)]


def trace_records(path):
    result=[]
    for line,text in enumerate(path.read_text().splitlines(),1):
        fields=text.split()
        result.append(dict(line=line,operation=fields[0],address=int(fields[1],0),
                           buffer=int(fields[2]) if len(fields)>2 else -1))
    return result


def normalized_key(record):
    op=record['operation']
    if 'address' in record:
        c,r,g,b,row,col=decode(record['address'])
        if op in MACS:
            r=g=b=-1
        elif op in ('PIM_WR_BG','PIM_RD_BG'):
            b=-1
    else:
        c,r,g,b,row,col=(record[k] for k in ('Channel','Rank','BankGroup','Bank','Row','Column'))
    return op,c,r,g,b,row,col,record['buffer']


def target_banks(event):
    c,r,g,b,_,_=normalized_key(event)[1:7]
    op=event['operation']
    if op in MACS:
        return [(c,rr,gg,bb) for rr in range(2) for gg in range(4) for bb in range(4)]
    if op=='PIM_WR_Q':
        return [(c,rr,gg,b) for rr in range(2) for gg in range(4)]
    if op in ('PIM_WR_BG','PIM_RD_BG'):
        return [(c,r,g,bb) for bb in range(4)]
    return [(c,r,g,b)]


def audit_run(directory,manifest=None,result=None,write=True):
    directory=Path(directory).resolve()
    simulation=directory/'simulation'
    manifest=manifest or load_json(directory/'manifest.json')
    result=result or load_json(simulation/'result.json')
    records=trace_records(directory/'workload.trace')
    require(digest(directory/'workload.trace')==digest(simulation/'input.trace')==result['requests']['sha256'],
            'Native input and source trace hash mismatch')
    counts=Counter(r['operation'] for r in records)
    require(counts==Counter(manifest['counts']),'Manifest/trace request counts mismatch')
    cfg,timing=result['config'],result['resolved_timings_ck']
    require((cfg['channels'],cfg['ranks'])==(16,2),'Unsupported audit topology')
    ratio=cfg['frontend_ratio']//cfg['memory_ratio']
    require(ratio==32,'Unsupported frontend clock ratio')
    native_to_tick=lambda c:(c-1)*ratio+1
    barriers={r['marker']:r['frontend_tick'] for r in read_csv(simulation/'barriers.csv')}
    require(set(barriers)=={r['address'] for r in records if r['operation']=='BARRIER'},'Barrier coverage mismatch')
    phase_by_line={}
    for phase in manifest['segments']:
        for line in range(phase['line_start'],phase['line_end']+1):
            phase_by_line[line]=phase
    queues=defaultdict(deque)
    for record in records:
        if record['operation'] in NEW:
            queues[normalized_key(record)].append(record)
    events,native=[],[]
    hashes={}
    for c in range(16):
        ep=simulation/f'bgk_events.csv.ch{c}'
        np=simulation/f'timeline.csv.ch{c}'
        require(ep.exists() and np.exists(),f'Missing channel{c} traces')
        hashes[ep.name]=digest(ep);hashes[np.name]=digest(np)
        erows=read_csv(ep);nrows=read_csv(np)
        require(all(e['Channel']==c for e in erows+nrows),f'Channel{c} trace contains wrong channel')
        require(all(a['clock']<=b['clock'] for a,b in zip(erows,erows[1:])),f'Nonmonotonic event clock channel{c}')
        for event in erows:
            key=normalized_key(event)
            require(bool(queues[key]),f'Unexpected event {key}')
            source=queues[key].popleft()
            event['source_line']=source['line'];event['source_address']=source['address']
            require(event['target_banks']==TARGETS[event['operation']],f'Wrong multicast/compute fanout at {event}')
            require(event['bytes']==32,f'Wrong event transaction size at {event}')
            require(event['ready_at']>=event['clock'],f'Negative service at {event}')
            op=event['operation']
            service=(timing['nCCDK'] if op=='PIM_MAC_K' else timing['nCCDQK'] if op=='PIM_MAC_QK'
                     else timing['nLOCALWR'] if op=='PIM_LOCAL_WR' else timing['nBL_min'])
            require(event['service_cycles']==service,f'Wrong service cycles at {event}')
            tail=timing['nWR'] if op=='PIM_LOCAL_WR' else 0
            require(event['ready_at']==event['clock']+service+tail,f'Completion missing service/recovery at {event}')
            phase=phase_by_line[source['line']]
            require(barriers[phase['start']]<=native_to_tick(event['clock']),f'Event precedes phase start {event}')
            require(native_to_tick(event['ready_at'])<=barriers[phase['end']],f'Phase completion precedes event arrival {event}')
            events.append(event)
        native.extend(nrows)
    require(not any(queues.values()),'Native dropped requested new-operation event')
    require(Counter(e['operation'] for e in events)==Counter({k:counts[k] for k in NEW if counts[k]}),
            'Native event/request count mismatch')
    native_counts=Counter(e['command'] for e in native)
    for command,ops in GROUPS.items():
        require(native_counts[command]==sum(counts[op] for op in ops),f'Wrong native {command} count')
        require(native_counts[command]==result['issued_commands'].get(command,0),f'Result/native {command} mismatch')
    require(native_counts['LOCALWR']==0,'Local write was also charged to external CA')

    # Associate every input generation with source fragment and validate exact
    # vector identity, not merely a controller's written=true bit.
    fragments=manifest['input_fragments']
    fragment_by_line={}
    for fragment in fragments:
        for line in range(fragment['line_start'],fragment['line_end']+1):
            fragment_by_line[line]=fragment
    fills=defaultdict(list);macs=defaultdict(list)
    input_by_bank=defaultdict(list)
    generation_first_line={}
    k_base=0
    qk_base=68*2048
    for event in events:
        op=event['operation'];c=event['Channel']
        if op in INPUTS:
            fragment=fragment_by_line[event['source_line']]
            operator='k' if op=='PIM_WR_BG' else 'qk'
            generation=operator,fragment['fragment'],c
            event['generation']=generation
            fills[generation].append(event)
            generation_first_line[generation]=min(generation_first_line.get(generation,10**12),event['source_line'])
            for bank in target_banks(event):
                input_by_bank[generation,bank].append(event)
        elif op in MACS:
            offset=event['Row']*2048+event['Column']*32
            operator='k' if op=='PIM_MAC_K' else 'qk'
            fragment=(offset-k_base)//(64*32) if operator=='k' else ((offset-qk_base)//32)%4
            generation=operator,fragment,c
            event['generation']=generation
            macs[generation].append(event)
    for generation,consumers in macs.items():
        operator,fragment,c=generation
        require(generation in fills,f'MAC without input generation {generation}')
        ready=max(e['ready_at'] for e in fills[generation])
        require(min(e['clock'] for e in consumers)>=ready,f'MAC starts before whole input fragment ready {generation}')
        require(len(consumers)==(64 if operator=='k' else 62),f'Missing/duplicate MAC outputs for {generation}')
        require({e['buffer'] for e in consumers}=={e['buffer'] for e in fills[generation]},f'Wrong buffer {generation}')
        for r in range(2):
            for g in range(4):
                for b in range(4):
                    bank=(c,r,g,b)
                    delivered=input_by_bank[generation,bank]
                    offsets=[e['Row']*2048+e['Column']*32 for e in delivered]
                    if operator=='k':
                        queries=[q for q in range(20) if ((968+q)%2,(968+q)//2%2,(968+q)//4%4)==(c%2,r,g)]
                        expected=[32*q for q in range(len(queries))]
                    else:
                        expected=list(range(0,640,32))
                    require(sorted(offsets)==expected,f'Incomplete/wrong perbank query fill {generation} bank{bank}')
    for c in range(16):
        for buffer in range(2):
            generations=[gen for gen in fills if gen[2]==c and fills[gen][0]['buffer']==buffer]
            generations.sort(key=lambda gen:generation_first_line[gen])
            for earlier,later in zip(generations,generations[1:]):
                if earlier in macs:
                    used_until=max(e['ready_at'] for e in macs[earlier])
                    first_fill=min(e['clock'] for e in fills[later])
                    require(first_fill>=used_until,f'Input overwritten while previous generation in use {earlier}->{later}')

    # Exports cannot consume only a partial feature reduction. A 32B score
    # burst can straddle two 80B token score-vectors; check both tokens.
    qk_ready=defaultdict(dict)
    for gen,consumers in macs.items():
        if gen[0]!='qk':continue
        for event in consumers:
            slot=((event['Row']*2048+event['Column']*32)-qk_base)//128
            qk_ready[event['Channel'],slot][gen[1]]=event['ready_at']
    for event in events:
        if event['operation']=='PIM_RD_BG':
            offset=event['Row']*2048+event['Column']*32
            for slot in range(offset//80,(offset+31)//80+1):
                ready=qk_ready[event['Channel'],slot]
                require(set(ready)=={0,1,2,3},f'Export without all four feature fragments at {event}')
                require(event['clock']>=max(ready.values()),f'BG score exported before final MAC completion {event}')
        if event['operation']=='PIM_LOCAL_WR':
            final=macs.get(('k',63,event['Channel']))
            require(final and event['clock']>=max(e['ready_at'] for e in final),
                    f'K written before all1024 reduction features complete {event}')

    # Independently replay physical row state for new local stores. ACT/PRE
    # remain shared CA commands; local payload uses only the perbank port.
    local_writes=[e for e in events if e['operation']=='PIM_LOCAL_WR']
    row_checks=0
    for c in range(16):
        open_row={};last_write={};last_open={}
        entries=[(e['clock'],0,e) for e in native if e['Channel']==c]
        entries += [(e['clock'],1,e) for e in local_writes if e['Channel']==c]
        for _,kind,e in sorted(entries,key=lambda x:(x[0],x[1])):
            command=e['operation'] if kind else e['command']
            ranks=range(2) if e['Rank']<0 else [e['Rank']]
            groups=range(4) if e['BankGroup']<0 else [e['BankGroup']]
            banks=range(4) if e['Bank']<0 else [e['Bank']]
            targets=[(r,g,b) for r in ranks for g in groups for b in banks]
            if command in ('PREab','REFab','ACTAB'):
                targets=[(r,g,b) for r in ranks for g in range(4) for b in range(4)]
            if command in ('ACT2','ACTAB'):
                for bank in targets:
                    open_row[bank]=e['Row'];last_open[bank]=e['clock']
            elif command in ('PREpb','PREab','PREHB'):
                for bank in targets:
                    if bank in last_write:
                        require(e['clock']-last_write[bank]>=timing['nLOCALWR']+timing['nWR'],
                                f'Precharge before local write recovery ch{c} bank{bank}')
                    open_row.pop(bank,None)
            elif command=='PIM_LOCAL_WR':
                bank=(e['Rank'],e['BankGroup'],e['Bank'])
                require(open_row.get(bank)==e['Row'],f'Local write wrong/closed row ch{c} bank{bank}')
                require(e['clock']-last_open[bank]>=timing.get('nRCDWR',timing['nRCD']),
                        f'Local write before activation ready ch{c} bank{bank}')
                if bank in last_write:
                    require(e['clock']-last_write[bank]>=timing['nLOCALWR'],f'Local port oversubscribed ch{c} bank{bank}')
                last_write[bank]=e['clock'];row_checks+=1
    require(row_checks==counts['PIM_LOCAL_WR'],'Not every local store row audited')
    if manifest['case']=='chain_k_qk':
        for c in range(16):
            last=max(e['ready_at'] for e in local_writes if e['Channel']==c)
            first=min(e['clock'] for e in events if e['Channel']==c and e['operation']=='PIM_MAC_QK')
            require(first>=last,f'QK starts before suffix writes recover ch{c}')
    report=dict(status='passed',case=manifest['case'],schedule=manifest.get('schedule'),
        native_event_count=len(events),native_command_count=len(native),
        source_request_counts=dict(counts),new_event_counts=dict(Counter(e['operation'] for e in events)),
        input_generations_checked=len(fills),perbank_input_generations_checked=len(input_by_bank),
        local_store_row_checks=row_checks,channels_checked=16,
        checks=['exact trace/event identity and multiplicity','fanout destination geometry and full vector fills',
                'input arrival before allbank MAC','same-buffer overwrite after last consumer completes',
                'MAC service and local write recovery completion','all four QK feature fragments before BG export',
                'local store row/activation/port cadence/precharge recovery',
                'phase completion fences include custom request tails','native CA counts exclude local payload',
                'K local write->QK dependency when chained'],
        limitations=['Timing-only backend does not compute values','Target mask geometry is source-audited; log exposes count, not maskbits',
                     'K RoPE/cast and GPU work remain unmeasured'],
        trace_sha256=digest(directory/'workload.trace'),native_trace_hashes=hashes)
    if write:
        (directory/'independent-native-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    args=parser.parse_args()
    cases=[args.path] if (args.path/'manifest.json').exists() else sorted(p for p in args.path.iterdir()
        if p.is_dir() and (p/'manifest.json').exists() and (p/'simulation/result.json').exists())
    require(bool(cases),'No completed cases found')
    reports=[]
    for directory in cases:
        manifest=load_json(directory/'manifest.json')
        if manifest.get('case') not in ('k','v','qk','pv','chain_k_qk'):
            continue
        report=audit_run(directory,manifest)
        reports.append(dict(directory=str(directory),case=report['case'],status=report['status'],
                            events=report['native_event_count'],local_store_checks=report['local_store_row_checks']))
    require(bool(reports),'No attention cases audited')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()

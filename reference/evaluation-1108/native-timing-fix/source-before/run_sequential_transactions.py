"""Continuous rank0 addresses and explicit LPDDR5X row-state ablation."""
import argparse,csv,gzip,json
from pathlib import Path
from run_uniform_transactions import read,csvout
from sequential_transactions import SequentialEngine,U,CK

VARIANTS=dict(c='feature_gpu_before',d='token_gpu_before',e='token_local_before',f='token_gpu_after',g='token_local_after')

def templates(root,typed,programs,engine,variant,natural):
    raw=list(csv.DictReader((root/'corrected-overlap'/(variant+'-explicit_native_io-solo-schedule.csv')).open()))
    result=[];cache={};i=0
    while i<len(raw):
        row=raw[i]
        if row['kind']=='gpu':
            duration=round((float(row['end_us'])-float(row['begin_us']))*U)
            if row['source_gpu_ordinal']!='':
                duration=round(natural[int(row['source_gpu_ordinal'])]['step_mean_us'][int(row['step'])]*U)
            result.append(dict(kind='gpu',name=row['name'],duration=duration));i+=1;continue
        group=[]
        while i<len(raw) and raw[i]['kind']=='pim':group.append(raw[i]);i+=1
        key=tuple((x['source_case'],int(x['source_begin_tick']),int(x['source_end_tick'])) for x in group)
        if key not in cache:
            events=[];jobs=[];offset=0;phases={};ownership=[]
            for case,a,z in key:
                left=(a-1)*469;right=(z-1)*469
                source=typed[variant+'/'+case]['events']
                refresh_starts={(x[0],x[4]) for x in source if x[1]>=3}
                for ch,kind,rank,bg,x,y in source:
                    if kind==0 and x<right and y>left:phases.setdefault(ch,(x-left+offset)%CK)
                    keep=kind>=3 or (kind==0 and (ch,x) in refresh_starts)
                    if keep and x<right and (y>left or kind>=3):events.append((ch,kind,rank,bg,x-left+offset,y-left+offset))
                for j in programs[variant+'/'+case]['jobs']:
                    if left<=j[5]<right:
                        jobs.append([*j[:5],j[5]-left+offset,*j[6:]])
                for ch,begin,end in programs[variant+'/'+case].get('ownership',[]):
                    lo=max(begin,left);hi=min(end,right)
                    if lo<hi:ownership.append((ch,lo-left+offset,hi-left+offset))
                offset+=(z-a)*469
            for ch in range(16):
                merged=[]
                for _,lo,hi in sorted(x for x in ownership if x[0]==ch):
                    if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
                    else:merged.append([lo,hi])
                for lo,hi in merged:
                    jobs.append([ch,-2,-1,0,0,lo,0,0,0])
                    jobs.append([ch,-3,-1,0,0,hi,0,0,0])
            for ch,phase in phases.items():
                if not any(x[0]==ch and x[1]==0 for x in events):
                    events.append((ch,0,0,-1,phase-1000*CK,phase-999*CK))
            refresh=engine.add(offset,sorted(set(events)))
            mode=int(variant.endswith('after'))
            program=engine.program(offset,mode,refresh,jobs)
            classes={};representatives=[]
            for ch in range(16):
                identity=(tuple(tuple(j[1:]) for j in jobs if j[0]==ch),
                          tuple(sorted(tuple(e[1:]) for e in events if e[0]==ch)))
                representatives.append(classes.setdefault(identity,ch))
            solo=max(engine.drain(engine.begin(program,ch)).pim_end for ch in set(representatives))
            cache[key]=dict(program=program,duration=offset,solo=solo,representatives=representatives)
        result.append(dict(kind='pim',name=group[0]['name'],**cache[key]))
    return result

def simulate(seq,units,plans,engine):
    now=index=nominal=memory_stall=wait=pim_delay=0;placements=[];stages=[]
    row_totals={}
    solo=sum(b.get('solo',b['duration']) for b in seq)
    native_solo=sum(b['duration'] for b in seq)
    for bi,b in enumerate(seq):
        if bi%1000==0:print('BLOCK',bi,'VLM',index,flush=True)
        if b['kind']=='gpu':now+=b['duration'];continue
        origin=now;local=0
        initial={ch:engine.begin(b['program'],ch) for ch in set(b['representatives'])}
        contexts=[initial[ch] for ch in b['representatives']]
        def advance_all(until):
            unique={c:engine.advance(c,until) for c in set(contexts)}
            return [unique[c] for c in contexts]
        states=advance_all(0)
        while index<len(units) and max(s.pim_end for s in states)>local:
            u=units[index]
            # Exact state sharing only: fork before executing different plans.
            # No channel averaging and no request-count changes are performed.
            forks={};used=set()
            for c,p in zip(contexts,plans[index]):
                if (c,p) not in forks:
                    forks[c,p]=engine.clone(c) if c in used else c
                    used.add(c)
            contexts=[forks[c,p] for c,p in zip(contexts,plans[index])]
            answers={}
            for c,p in zip(contexts,plans[index]):
                if c not in answers:answers[c]=engine.execute(c,p,local)
            r=[answers[c] for c in contexts]
            finish=max(s.finish for s in r)
            assert finish>=local+u['duration']
            extra=finish-local-u['duration'];memory_stall+=extra;nominal+=u['duration']
            states=advance_all(finish)
            first=[s.first_issue for s in r if s.requests]
            placements.append(dict(ordinal=index,block=bi,start_us=(origin+local)/U,
                first_issue_us=(origin+min(first))/U if first else '',finish_us=(origin+finish)/U,
                model_reference_us=u['duration']/U,memory_stall_us=extra/U,
                read_bytes=u['read_sectors']*32,write_bytes=u['write_sectors']*32))
            assert sum(s.requests for s in r)==u['read_sectors']+u['write_sectors']
            index+=1;local=finish
        drained={c:engine.drain(c) for c in set(contexts)}
        states=[drained[c] for c in contexts]
        for context in contexts:
            for key,value in engine.stats(context).items():row_totals[key]=row_totals.get(key,0)+value
        pim_end=max(s.pim_end for s in states)
        assert pim_end>=b['solo']
        delay=pim_end-b['solo'];gpu_wait=max(0,local-pim_end)
        wait+=gpu_wait;pim_delay+=delay
        now+=pim_end+gpu_wait
        stages.append(dict(block=bi,name=b['name'],native_us=b['duration']/U,
            isolated_bus_model_us=b['solo']/U,PIM_with_GPU_us=pim_end/U,
            PIM_GPU_interference_us=delay/U,AE_GPU_wait_us=gpu_wait/U))
    vlm=sum(u['duration'] for u in units);remaining=vlm-nominal
    assert now==solo+pim_delay+wait
    assert [p['ordinal'] for p in placements]==list(range(index))
    return dict(AE_solo_ms=solo/U/1000,AE_native_schedule_ms=native_solo/U/1000,
        PIM_interference_ms=pim_delay/U/1000,AE_GPU_wait_ms=wait/U/1000,
        VLM_memory_stall_ms=memory_stall/U/1000,
        VLM_completed_model_ms=nominal/U/1000,VLM_remaining_model_ms=remaining/U/1000,
        raw_model_total_ms=(now+remaining)/U/1000,
        net_overlap_ms=(nominal-wait-pim_delay)/U/1000,completed_units=index,
        **{'GPU_'+key:value for key,value in row_totals.items()},
        placements=placements,stages=stages)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reference/evaluation-1108'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--row-hit-control',action='store_true')
    p.add_argument('--cases',default='cdefg')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    base=a.root/'online-arbitration-preparation'
    natural=read(base/'natural-ae-kernel-ledger.json')['units']
    units=read(a.root/'gpu/uniform-memory-demand.json')['units']
    warm=read(a.root/'gpu/warm-summary.json')['baseline']['phase_ms']
    measured=warm['vit']['mean']+warm['vlm']['mean']
    scale=measured*1000/sum(u['us'] for u in units)
    for u in units:u['duration']=round(u['us']*scale*U)
    units[-1]['duration']+=round(measured*1000*U)-sum(u['duration'] for u in units)
    engine=SequentialEngine(a.root/'sequential-rows/libsequential.so',rows=not a.row_hit_control)
    empty=engine.add(0,[],16)
    empty_program=engine.program(0,0,empty,[])
    plans=[];cache={};reference_counts={}
    for u in units:
        handles=[];durations=[]
        for ch in range(16):
            r=u['read_sectors']//16+int(ch<u['read_sectors']%16)
            w=u['write_sectors']//16+int(ch<u['write_sectors']%16)
            key=(u['duration'],r,w)
            if key not in cache:
                initial,_=engine.stream_plan(*key)
                h,ref=engine.reference(empty_program,initial)
                context=engine.begin(empty_program,0);check=engine.execute(context,h,0)
                assert check.finish==ref.finish and check.memory_stall==0
                assert ref.requests==check.requests==r+w;cache[key]=(h,ref.finish,engine.stats(context))
            h,d,counts=cache[key];handles.append(h);durations.append(d)
            for k,v in counts.items():reference_counts[k]=reference_counts.get(k,0)+v
        u['duration']=max(durations);plans.append(handles)
    model_vlm=sum(u['duration'] for u in units)/U/1000;bias=model_vlm-measured
    with gzip.open(a.root/'uniform-transactions/native-reservations.json.gz','rt') as f:typed=json.load(f)
    with gzip.open(base/'native-programs.json.gz','rt') as f:programs=json.load(f)
    rows=[];calibration=[]
    for case,v in VARIANTS.items():
        if case not in a.cases:continue
        print('BEGIN',case,flush=True)
        seq=templates(a.root,typed,programs,engine,v,natural)
        calibration.extend(dict(case=case,name=b['name'],native_us=b['duration']/U,isolated_us=b['solo']/U)
                           for b in seq if b['kind']=='pim')
        csvout(a.output/'PIM-isolated-check.csv',calibration)
        result=simulate(seq,units,plans,engine)
        csvout(a.output/(case+'-VLM-placement.csv'),result.pop('placements'))
        csvout(a.output/(case+'-PIM-and-AE-waits.csv'),result.pop('stages'))
        if case=='c':rows.append(dict(case='b',variant=v,total_ms=result['AE_solo_ms']+measured,
            AE_solo_ms=result['AE_solo_ms'],PIM_interference_ms=0,AE_GPU_wait_ms=0,net_overlap_ms=0))
        result['total_ms']=result['raw_model_total_ms']-bias
        rows.append(dict(case=case,variant=v,**result));csvout(a.output/'evaluation-b-g.csv',rows)
        print('RESULT',json.dumps(rows[-1]),flush=True)
    summary=dict(status='conditional sequential-address row-state estimate; not a measured Thor GPU trace',
        evaluation=rows,VLM_retained_ms=measured,VLM_model_reference_ms=model_vlm,VLM_reference_bias_ms=bias,
        GPU_row_state=not a.row_hit_control,GPU_solo_command_counts=reference_counts,
        address_assumption='One monotone stream of 32-byte sectors per kernel/channel, starting at sector0; actual read/write request order retained. RoBaRaCoCh restricted to GPU rank0: column=i%64, BG=(i//64)%4, bank=(i//256)%4, row=i//1024. Kernel footprints/reuse are unknown. Bank state persists within each contiguous PIM stage, resets at stage boundaries. This optimistic spatial approximation is not an actual physical-address trace.',
        policy=dict(before='PIM-first among released requests; committed GPU returns immutable; a new GPU request cannot prolong a pending PIM command wait',
                    after='Private rank commands excluded from external CA; complete input/output epochs own the external port exclusively, draining committed returns at ownership handoff; no GPU/PIM burst interleaving inside those epochs',
                    kernel='Ordered, launched when PIM is active without looking ahead at its end; may cross the PIM boundary; AE GPU waits until completion',
                    refresh='Known per-rank refresh calendar and periodic continuation; row-state mode reserves rank0 PREab at REF-nRPab and enforces RAS/read/write recovery before it, then closes all GPU rows. PREab and ACT1/ACT2 consume external CA.',
                    PIM_retiming='Preserve native causal gaps, shift remaining command program after a blocked command',
                    reference='Every solo GPU plan is normalized and replayed with the same causal arbiter; identical completion and zero added memory stall are required',
                    AE_GPU='Natural-cache profiled kernel durations; existing dedicated handoff microbenchmarks retained at logical-operation cold-input boundaries'),
        limitations=['The native controller has not been extended; this is an independent online bus model.',
            'Native PIM inter-command timing gaps remain a causal replay abstraction. GPU bank/row state is regenerated independently on disjoint rank0.',
            'Uniform temporal arrivals and synchronous L2-write proxy remain assumptions; write counters are not measured DRAM writebacks.',
            'Continuous addresses are an optimistic locality approximation; per-kernel traffic alone cannot reconstruct unique footprints, address reuse, or actual bursts.',
            'Same-direction ready-request ordering is retained; this does not model a full FR-FCFS scheduler or a calibrated GPU cache/warp pipeline.',
            'Request completion checkpoints are a GPU-progress abstraction, not a warp/SM simulator.',
            'Natural-cache AE timings are eager, profiled, and include small timestamp overlaps; sequential replay is not a measured concurrent runtime.',
            'No CPU dispatch latency, handoff signaling latency, or hardware queue capacity was invented; these omitted implementation costs prevent an implementation-ready latency claim.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()

"""Ordered VLM replay with measured-size uniform traffic and native PIM calendars."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from uniform_transactions import Engine, U


def read(p):
    return json.loads(Path(p).read_text())


def csvout(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def templates(root,typed,engine,variant):
    raw=list(csv.DictReader((root/(variant+'-explicit_native_io-solo-schedule.csv')).open()))
    sequence=[];cache={};i=0
    while i<len(raw):
        b=raw[i]
        if b['kind']=='gpu':
            sequence.append(dict(kind='gpu',name=b['name'],duration=round((float(b['end_us'])-float(b['begin_us']))*U)))
            i+=1;continue
        group=[]
        while i<len(raw) and raw[i]['kind']=='pim':
            group.append(raw[i]);i+=1
        key=tuple((b['source_case'],int(b['source_begin_tick']),int(b['source_end_tick'])) for b in group)
        if key not in cache:
            events=[];offset=0
            for case,a,z in key:
                left=(a-1)*469;right=(z-1)*469
                for ch,kind,rank,bg,x,y in typed[variant+'/'+case]['events']:
                    # Keep prior REF epochs even when their blackout ended
                    # before the slice; the continuation must retain phase.
                    if x<right and (y>left or kind>=3):
                        events.append((ch,kind,rank,bg,x-left+offset,y-left+offset))
                offset+=(z-a)*469
            cache[key]=(engine.add(offset,sorted(set(events))),offset)
        cal,duration=cache[key]
        sequence.append(dict(kind='pim',name=group[0]['name'],duration=duration,calendar=cal))
    return sequence


def simulate(sequence,units,plans,engine):
    now=index=wait=nominal_advanced=memory_extra=0
    places=[];waits=[];max_flight=0;bursts=0
    for bi,b in enumerate(sequence):
        if b['kind']=='gpu':now+=b['duration'];continue
        origin=now;local=0;previous=[None]*16
        while index<len(units) and local<b['duration']:
            u=units[index]
            equivalent={};r=[]
            for ch,p in enumerate(plans[index]):
                old=previous[ch]
                state=None if old is None else tuple(getattr(old,k) for k in
                     ('requests','last_ca','last_dq_a','last_dq_b','last_write','last_bg'))
                key=(engine.equivalence[b['calendar']][ch],p,state)
                if key not in equivalent:
                    equivalent[key]=engine.run(b['calendar'],p,local,ch,old)
                r.append(equivalent[key])
            first=[v.first_issue for v in r if v.requests]
            # A memory-using kernel is admitted only if a legal joint CA/DQ
            # transaction can start before PIM ends. Cache-only kernels may run.
            if first and min(first)>=b['duration']:break
            finish=max(v.finish for v in r)
            assert finish>=local+u['duration']
            memory_extra+=finish-local-u['duration']
            nominal_advanced+=u['duration']
            max_flight=max(max_flight,max(v.max_inflight for v in r))
            bursts+=sum(v.requests for v in r)
            places.append(dict(ordinal=index,kind=u['kind'],block=bi,start_us=(origin+local)/U,
                first_issue_us=(origin+min(first))/U if first else '',finish_us=(origin+finish)/U,
                model_reference_us=u['duration']/U,retained_us=u.get('measured_duration',u['duration'])/U,
                read_bytes=u['read_sectors']*32,write_bytes=u['write_sectors']*32,
                PIM_end_us=(origin+b['duration'])/U,extends_past_PIM=finish>b['duration'],
                memory_stall_us=(finish-local-u['duration'])/U,
                CA_delay_request_sum_us=sum(v.ca_delay for v in r)/U,
                DQ_delay_request_sum_us=sum(v.dq_delay for v in r)/U,
                refresh_delay_request_sum_us=sum(v.refresh_delay for v in r)/U))
            # Carry bus direction across consecutive VLM kernels.
            previous=[v if v.requests else old for v,old in zip(r,previous)]
            local=finish;index+=1
        delay=max(0,local-b['duration'])
        if delay:
            waits.append(dict(block=bi,VLM_ordinal=index-1,AE_GPU_ready_us=(origin+b['duration'])/U,
                              AE_GPU_resume_us=(origin+b['duration']+delay)/U,wait_us=delay/U))
        wait+=delay;now+=b['duration']+delay
    solo=sum(b['duration'] for b in sequence)
    vlm=sum(u['duration'] for u in units)
    remaining=vlm-nominal_advanced
    assert now==solo+wait
    assert all(x['finish_us']<=y['start_us']+1e-8 for x,y in zip(places,places[1:]))
    return dict(AE_solo_ms=solo/U/1000,VLM_ms=vlm/U/1000,AE_GPU_wait_ms=wait/U/1000,
        AE_with_interference_ms=now/U/1000,VLM_completed_model_reference_ms=nominal_advanced/U/1000,
        VLM_model_remaining_after_AE_ms=remaining/U/1000,total_ms=(now+remaining)/U/1000,
        saving_vs_serial_ms=(nominal_advanced-wait)/U/1000,
        VLM_memory_stall_ms=memory_extra/U/1000,completed_units=index,
        GPU_bursts=bursts,observed_max_inflight_per_channel=max_flight,
        placements=places,AE_waits=waits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--demand',type=Path,required=True)
    p.add_argument('--bus',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stream-reference',action='store_true',help='Keep fixed uniform arrivals and expose baseline-model bias')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    demand=read(a.demand);units=demand['units']
    warm=read(a.reference/'gpu/warm-summary.json')['baseline']['phase_ms']
    vlm_ms=warm['vit']['mean']+warm['vlm']['mean']
    scale=vlm_ms*1000/sum(u['us'] for u in units)
    for u in units:u['duration']=round(u['us']*scale*U)
    units[-1]['duration']+=round(vlm_ms*1000*U)-sum(u['duration'] for u in units)
    engine=Engine(a.library);empty=engine.add(0,[],16)
    plans=[];cache={};audit=[];failures=[]
    for u in units:
        channel_plans=[];reference_durations=[]
        measured=u['duration'];u['measured_duration']=measured
        for ch in range(16):
            r=u['read_sectors']//16+int(ch<u['read_sectors']%16)
            w=u['write_sectors']//16+int(ch<u['write_sectors']%16)
            key=(measured,r,w)
            if key not in cache:
                try:
                    handle,info=(engine.stream_plan(*key) if a.stream_reference else engine.plan(*key))
                    reference=engine.run(empty,handle)
                    assert reference.finish>=measured and reference.requests==r+w
                    if not a.stream_reference:assert reference.finish==measured
                    cache[key]=(handle,reference.finish)
                    audit.append(dict(ordinal=u['ordinal'],channel=ch,retained_us=measured/U,
                        reads=r,writes=w,minimum_us=info.solo_minimum/U,arrival_span_us=info.calibrated_span/U,
                        isolated_finish_us=reference.finish/U))
                except ValueError as exc:
                    failures.append(dict(ordinal=u['ordinal'],channel=ch,retained_us=u['duration']/U,
                                         reads=r,writes=w,error=str(exc)))
                    cache[key]=(None,measured)
            handle,duration=cache[key]
            channel_plans.append(handle);reference_durations.append(duration)
        u['duration']=max(reference_durations)
        plans.append(channel_plans)
        if u['ordinal']%200==0:print('CALIBRATE',u['ordinal'],'failures',len(failures),flush=True)
    csvout(a.output/'isolated-calibration.csv',audit)
    csvout(a.output/'incompatible-demand.csv',failures)
    if failures:
        (a.output/'STATUS.json').write_text(json.dumps(dict(status='traffic proxy incompatible with retained times',
            failures=len(failures),action='No bytes trimmed, no parallel latency published'),indent=2)+'\n')
        raise RuntimeError(f'{len(failures)} unique workloads fail isolated calibration; see incompatible-demand.csv')
    with gzip.open(a.bus,'rt') as f:typed=json.load(f)
    variants=dict(c='feature_gpu_before',d='token_gpu_before',e='token_local_before',
                  f='token_gpu_after',g='token_local_after')
    rows=[]
    model_vlm_ms=sum(u['duration'] for u in units)/U/1000
    reference_bias_ms=model_vlm_ms-vlm_ms
    for letter,v in variants.items():
        sequence=templates(a.reference/'corrected-overlap',typed,engine,v)
        print('REPLAY',letter,v,flush=True)
        result=simulate(sequence,units,plans,engine)
        csvout(a.output/(letter+'-VLM-placement.csv'),result.pop('placements'))
        csvout(a.output/(letter+'-AE-GPU-wait.csv'),result.pop('AE_waits'))
        if letter=='c':
            rows.append(dict(case='b',variant=v,AE_solo_ms=result['AE_solo_ms'],VLM_ms=vlm_ms,
                AE_GPU_wait_ms=0,AE_with_interference_ms=result['AE_solo_ms'],
                VLM_model_remaining_after_AE_ms=model_vlm_ms,total_ms=result['AE_solo_ms']+vlm_ms,saving_vs_serial_ms=0,
                raw_model_total_ms=result['AE_solo_ms']+model_vlm_ms,
                VLM_model_reference_ms=model_vlm_ms,VLM_reference_bias_ms=reference_bias_ms))
        result['raw_model_total_ms']=result['total_ms']
        result['VLM_model_reference_ms']=model_vlm_ms
        result['VLM_reference_bias_ms']=reference_bias_ms
        result['total_ms']-=reference_bias_ms
        result['VLM_ms']=vlm_ms
        result['total_basis']='Measured serial baseline minus model-predicted net overlap gain; raw model total also retained'
        rows.append(dict(case=letter,variant=v,**result))
        print('RESULT',json.dumps(rows[-1]),flush=True)
        csvout(a.output/'evaluation-b-g.csv',rows)
    summary=dict(status='completed conditional uniform-traffic bus model',evaluation=rows,
        VLM_retained_ms=vlm_ms,VLM_model_reference_ms=model_vlm_ms,VLM_reference_bias_ms=reference_bias_ms,
        duration_scale=scale,traffic=demand['assumptions'],
        policy=dict(PIM_priority=True,kernel_order='unchanged',kernel_cross_window=True,
            AE_GPU='wait until the active VLM kernel completes; no next VLM kernel until PIM resumes',
            progress='isolated request completion checkpoints; delayed returns shift later releases and completion',
            rank_refresh='recorded target rank blackout; one-CK shared CA pulse; periodic continuation after PIM',
            turnaround='native LPDDR direction and rank switching guards for every inserted GPU burst',
            arrival_span='max(0, retained kernel duration - (nCL+nBL_min)*tCK); read and write streams uniform separately',
            write_proxy='all measured L2 write sectors drained synchronously by the model, unlike real write-back caching',
            initial_GPU_bank_state='row-hit abstraction, not a reconstructed physical address trace',
            qualification='conditional model, not hardware end-to-end or an accuracy bound'),
        calibration=dict(unique_plans=len(cache),failures=0,byte_trimming=False,
            isolated_nominal_duration_preserved=not a.stream_reference,
            main_total='AE measured/native known terms + retained VLM - (raw model serial - raw model parallel)',
            limitation='Additive standalone model bias is assumed to cancel when transferring modeled overlap gain to measured baseline'))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()

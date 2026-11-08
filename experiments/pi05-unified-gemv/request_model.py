"""Continuous kernel scheduling over per-channel PIM reservation calendars."""
import ctypes as C
import csv,gzip,json,math
from pathlib import Path
U=16_000_000 # 1 us = 16,000,000 model ticks (1/16 ps each).
CK=15008
def ticks(us):return round(float(us)*U)
class Event(C.Structure):
    _fields_=[('channel',C.c_int32),('kind',C.c_int32),('a',C.c_int64),('b',C.c_int64)]
class Result(C.Structure):
    _fields_=[(k,C.c_int64) for k in ('finish','requests','ca_wait','dq_wait','refresh_wait','queue_wait','calibration_trim','max_inflight')]+[('channel_finish',C.c_int64*16)]
class Replay:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.replay_error.restype=C.c_char_p
        self.lib.replay_add.argtypes=[C.c_int64,C.c_int,C.POINTER(Event),C.c_int];self.lib.replay_add.restype=C.c_int
        self.lib.replay_kernel.argtypes=[C.c_int,C.c_int64,C.c_int64,C.c_double,C.c_int,C.c_int,C.c_int,C.POINTER(Result)]
        self.lib.replay_probe.argtypes=[C.c_int,C.c_int64,C.c_int64,C.c_int,C.POINTER(Result)]
    def check(self,value):
        if value<0:raise RuntimeError(self.lib.replay_error().decode())
        return value
    def add(self,end,events,channels=16):
        data=(Event*len(events))(*(Event(*e) for e in events));return self.check(self.lib.replay_add(end,channels,data,len(data)))
    def kernel(self,handle,start,duration,profile):
        r=Result();self.check(self.lib.replay_kernel(handle,start,duration,profile['pressure'],profile['outstanding'],profile['write_percent'],profile['guard_ck'],C.byref(r)));return r
    def probe(self,handle,start,duration,channel=0):
        r=Result();self.check(self.lib.replay_probe(handle,start,duration,channel,C.byref(r)));return r

def load_reference(root,replay):
    root=Path(root)
    with gzip.open(root/'native-bus-templates.json.gz','rt') as f:bus=json.load(f)
    refresh=json.loads((root/'request-model/refresh-events.json').read_text())
    prefix=json.loads((root/'gpu/prefix-kernels.json').read_text())['ordered_mean_units']
    old=json.loads((root/'summary.json').read_text());vlm=old['evaluation'][0]['VLM_ms']
    scale=vlm*1000/sum(k['us'] for k in prefix)
    kernels=[dict(k,duration=ticks(k['us']*scale)) for k in prefix]
    # Quantized sum is bound to the reused VLM phase mean, not a new measurement.
    kernels[-1]['duration']+=ticks(vlm*1000)-sum(k['duration'] for k in kernels)
    schedules={};templates={};template_audit=[]
    for model in old['models']:
        v=model['variant'];raw=list(csv.DictReader((root/(v+'-schedule.csv')).open()))
        sequence=[];i=0
        while i<len(raw):
            b=raw[i]
            if b['kind']=='gpu':
                sequence.append(dict(kind='gpu',name=b['name'],duration=ticks(float(b['end_us'])-float(b['begin_us']))));i+=1;continue
            group=[]
            while i<len(raw) and raw[i]['kind']=='pim':group.append(raw[i]);i+=1
            key=(v,tuple((b['source_case'],int(b['source_begin_tick']),int(b['source_end_tick'])) for b in group))
            if key not in templates:
                events=[];offset=0;pieces=[]
                for b in group:
                    stage=bus[v+'/'+b['source_case']];a,z=int(b['source_begin_tick']),int(b['source_end_tick'])
                    factor=round(stage['tick_us']*U);assert factor==469
                    duration=(z-a)*factor;pieces.append((b['source_case'],a,z))
                    for channel in stage['channels']:
                        ch=channel['channel']
                        name=f'simulation/rank_ca_events.csv.ch{ch}'
                        assert refresh['source_sha256'][stage['path']+'/'+name]==stage['source_sha256'][name]
                        for kind,label in ((0,'dq'),(1,'ca')):
                            for x,y in channel[label]:
                                left,right=max(x,a),min(y,z)
                                if left<right:events.append((ch,kind,offset+(left-a)*factor,offset+(right-a)*factor))
                        for rank,x,y in refresh['cases'][v+'/'+b['source_case']][str(ch)]:
                            if x<z and y>a:
                                # Preserve the original REFab start (possibly before the slice).
                                events.append((ch,2+rank,offset+(x-a)*factor,offset+(y-a)*factor))
                    offset+=duration
                handle=replay.add(offset,events);templates[key]=(handle,offset)
                template_audit.append(dict(variant=v,handle=handle,duration_us=offset/U,pieces=pieces,event_count=len(events)))
            handle,duration=templates[key]
            sequence.append(dict(kind='pim',name=group[0]['name'],handle=handle,duration=duration))
        assert math.isclose(sum(b['duration'] for b in sequence)/U/1000,model['AE_known_model_ms'],abs_tol=1e-8)
        schedules[v]=sequence
    return schedules,kernels,dict(VLM_ms=vlm,raw_prefix_ms=sum(k['us'] for k in prefix)/1000,
        prefix_duration_scale=scale,old_summary=old,templates=template_audit)

def simulate(sequence,kernels,replay,profile,collect=False):
    clock=0;index=0;ae_wait=0;memory_extra=0;dispatch=0;requests=0;trim=0;placed=[];delay_events=[]
    stats={k:0 for k in ('ca_wait','dq_wait','refresh_wait','queue_wait')}
    extra=ticks(profile.get('dispatch_us',0));solo=sum(x['duration'] for x in sequence)
    for block_index,b in enumerate(sequence):
        if b['kind']=='gpu':clock+=b['duration'];continue
        origin=clock;deadline=origin+b['duration'];local=0
        while index<len(kernels) and origin+local+extra<deadline:
            start=local+extra;k=kernels[index]
            r=replay.kernel(b['handle'],start,k['duration'],profile)
            assert r.finish>=start+k['duration']
            memory_extra+=r.finish-start-k['duration'];dispatch+=extra
            requests+=r.requests;trim+=r.calibration_trim
            for field in stats:stats[field]+=getattr(r,field)
            if collect:placed.append(dict(ordinal=index,kind=k['kind'],block_index=block_index,
                begin_us=(origin+start)/U,end_us=(origin+r.finish)/U,nominal_us=k['duration']/U,
                pim_deadline_us=deadline/U,crossed_AE_GPU_boundary=origin+r.finish>deadline,
                bursts=r.requests,memory_extra_us=(r.finish-start-k['duration'])/U))
            local=r.finish;index+=1
        wait=max(0,local-b['duration']);ae_wait+=wait
        if collect and wait:delay_events.append(dict(block_index=block_index,AE_ready_us=deadline/U,
            AE_resume_us=(deadline+wait)/U,AE_wait_us=wait/U,blocking_VLM_ordinal=index-1))
        clock=deadline+wait
    assert clock==solo+ae_wait
    progressed=sum(k['duration'] for k in kernels[:index]);total_vlm=sum(k['duration'] for k in kernels)
    finish=clock+total_vlm-progressed
    assert solo+total_vlm-finish==progressed-ae_wait
    return dict(AE_solo_ms=solo/U/1000,VLM_ms=total_vlm/U/1000,AE_parallel_ms=clock/U/1000,
        AE_wait_ms=ae_wait/U/1000,VLM_advanced_ms=progressed/U/1000,
        VLM_memory_extra_ms=memory_extra/U/1000,dispatch_extra_ms=dispatch/U/1000,
        effective_saving_ms=(progressed-ae_wait)/U/1000,total_ms=finish/U/1000,
        completed_VLM_units=index,completed_VLM_kernels=sum(k['kind']=='kernel' for k in kernels[:index]),
        generated_GPU_bytes=requests*32,calibration_trim_bursts=trim,
        **{k+'_request_sum_ms':v/U/1000 for k,v in stats.items()},placements=placed,AE_delays=delay_events)

"""Run the explicitly parameterized continuous-kernel sensitivity matrix."""
import argparse,json,time
from pathlib import Path
import analyze_evaluation as A
from request_model import Replay,load_reference,simulate

def profiles():
    base=dict(pressure=.3,outstanding=32,write_percent=20,guard_ck=2,dispatch_us=0)
    yield dict(base,name='compute_only',pressure=0)
    for rho,label in ((.1,'low'),(.3,'central'),(.6,'high')):yield dict(base,name=label,pressure=rho)
    for rho,label in ((.3,'central'),(.6,'high')):
        for q in (8,128):yield dict(base,name=f'{label}_q{q}',pressure=rho,outstanding=q)
        yield dict(base,name=f'{label}_dispatch1us',pressure=rho,dispatch_us=1)
        yield dict(base,name=f'{label}_guard8ck',pressure=rho,guard_ck=8)
        yield dict(base,name=f'{label}_read_only',pressure=rho,write_percent=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--profiles',nargs='*');a=ap.parse_args();out=a.reference/'request-model';out.mkdir(exist_ok=True)
    replay=Replay(a.library);seq,ks,meta=load_reference(a.reference,replay);A.dump(out/'template-audit.json',meta['templates'])
    rows=[];start=time.monotonic()
    for p in profiles():
        if a.profiles and p['name'] not in a.profiles:continue
        for v,blocks in seq.items():
            r=simulate(blocks,ks,replay,p,collect=p['name']=='central')
            places=r.pop('placements');delays=r.pop('AE_delays')
            if places:A.csvout(out/(v+'-kernel-execution.csv'),places)
            if delays:A.csvout(out/(v+'-AE-waits.csv'),delays)
            row=dict(profile=p['name'],variant=v,**{k:x for k,x in p.items() if k!='name'},**r);rows.append(row)
            print(json.dumps(dict(profile=p['name'],variant=v,total_ms=r['total_ms'],saving_ms=r['effective_saving_ms'],AE_wait_ms=r['AE_wait_ms'],elapsed_s=time.monotonic()-start)),flush=True)
    A.csvout(out/'sensitivity.csv',rows)
    central=[r for r in rows if r['profile']=='central']
    if central:
        table=[meta['old_summary']['evaluation'][0],meta['old_summary']['evaluation'][1]]
        for letter,v in zip('cdefg',seq):
            r=next(r for r in central if r['variant']==v)
            table.append(dict(case=letter,variant=v,AE_ms=r['AE_parallel_ms'],VLM_ms=r['VLM_ms'],
                overlap_ms=r['VLM_advanced_ms'],total_ms=r['total_ms'],kind='parameterized request reservation replay; NOT measured GPU memory trace'))
        A.csvout(out/'evaluation-a-g.csv',table)
    A.dump(out/'summary.json',dict(native_run=meta['old_summary']['native_run'],rows=rows,
        VLM_ms=meta['VLM_ms'],raw_prefix_ms=meta['raw_prefix_ms'],prefix_duration_scale=meta['prefix_duration_scale'],
        policy=dict(whole_kernel_fit=False,dependency_order=True,GPU='AE priority at kernel boundaries, no forced preemption; exclusive GPU execution',
            memory='32B rank0 requests per channel; recorded PIM CA/DQ reservations have priority; target-rank refresh blackout',
            arrivals='uniform over kernel duration minus RFC/latency tail; finite outstanding queue; backpressure',
            demand='pressure * eligible injection span / nBL_min bursts per channel; 64-read/16-write runs at 20 percent writes',
            calibration='trim synthetic burst count only if isolated requests do not fit nominal duration; preserve reused VLM aggregate latency',
            missing='GPU addresses/cache/arrival trace, DRAM row activation/precharge, real SM concurrency/preemption, host/coherence'),
        library_sha256=A.sha(a.library),source_sha256={p.name:A.sha(p) for p in (Path(__file__),Path(__file__).with_name('request_model.py'),Path(__file__).with_name('request_replay.cpp'))},
        input_sha256={p.name:A.sha(p) for p in (a.reference/'native-bus-templates.json.gz',out/'refresh-events.json',a.reference/'gpu/prefix-kernels.json')}))
if __name__=='__main__':main()

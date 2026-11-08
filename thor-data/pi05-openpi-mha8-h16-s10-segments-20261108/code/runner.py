"""Entry-evicted continuous GPU segments, native eager and graph separately."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/var/tmp/shiyuhuang-vla-baseline')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('probe','measure'),required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--probe')
    ap.add_argument('--samples',type=int,default=20)
    ns=ap.parse_args()
    if not ns.tag.replace('_','').isalnum() or not 10<=ns.samples<=40:
        raise ValueError('Invalid bounded arguments')
    out=ROOT/'runs'/ns.tag
    out.mkdir(exist_ok=False)
    report=dict(status='running',arguments=vars(ns),host=os.uname().nodename,pid=os.getpid(),
        client_date='2026-11-08',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        occupancy=[],cases=[],coarse=[],files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    def save():
        (out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
    def guard(label):
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True)
        others=[int(line.split(',')[0]) for line in raw.splitlines() if line.split(',')[0].strip().isdigit() and int(line.split(',')[0])!=os.getpid()]
        fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
        report['occupancy'].append(dict(label=label,raw=raw,others=others,free_disk_bytes=free))
        # This experiment writes small JSON only, no large profiler reports.
        if others or free<256*1024**2:
            save();raise RuntimeError('GPU occupied or <256 MiB disk reserve: '+str((others,free)))
    def log(text):print(time.strftime('%H:%M:%S'),text,flush=True)
    try:
        guard('before_import')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17)
        from timing import Eviction,TimedGraph,qualify_eviction,measure_case,equal,stats
        report['environment']=dict(python=sys.executable,torch=torch.__version__,threads=1,interop_threads=1,
            cpu_affinity=sorted(os.sched_getaffinity(0)),profiler=False,clock_change=False,
            graph_events='enable_timing=True, external=True, captured INSIDE the segment graph',
            graph_compile=False,segment_fusion=False)
        with torch.inference_mode():
            log('Prepare 256 MiB cache-eviction buffer')
            eviction=Eviction()
            if ns.mode=='probe':
                report['cache_probe']=qualify_eviction(eviction,guard)
                x=torch.arange(4096,device='cuda',dtype=torch.float32)
                case=dict(id='probe_two_kernel_segment',family='probe',suite='probe',fn=lambda:(x+1)*2,expected=(x+1)*2)
                report['cases'].append(measure_case(case,eviction,ns.samples,guard))
                empty=TimedGraph(lambda:None)
                values=[]
                for _ in range(20):empty.run();values.append(empty.milliseconds())
                report['empty_graph_event_ms']=stats(values)
                log('CACHE_PROBE '+json.dumps(report['cache_probe']['median_read_bytes']))
            else:
                if not ns.probe or not ns.probe.replace('_','').isalnum():raise ValueError('Verified probe tag required')
                probe_path=ROOT/'runs'/ns.probe/'capture.json'
                probe=json.loads(probe_path.read_text())
                if probe['status']!='complete' or not probe['cache_probe']['passed']:raise ValueError('Probe did not pass')
                report['cache_probe']=probe['cache_probe']
                report['empty_graph_event_ms']=probe['empty_graph_event_ms']
                report['probe_sha256']=hashlib.sha256(probe_path.read_bytes()).hexdigest()
                from builder import prepare
                model,cache,call,reference,metadata=prepare(log,guard)
                report.update(metadata)
                from native_support import Probe
                meter=Probe(model,torch)
                with cache.applied():
                    if not equal(call(),reference):raise RuntimeError('Cache changed full output')
                    log('20 complete native model warmups')
                    for i in range(20):
                        if i%5==0:guard('warmup_'+str(i))
                        call()
                    torch.cuda.synchronize()
                    def coarse(name,marked,count=30):
                        rows=[]
                        from contextlib import nullcontext
                        with meter.applied() if marked else nullcontext():
                            for i in range(count):
                                if i%5==0:guard(name+'_'+str(i))
                                meter.reset();torch.cuda.synchronize()
                                t=time.perf_counter_ns();meter.origin.record()
                                value=call();meter.end.record();meter.end.synchronize()
                                row=dict(wall_ms=(time.perf_counter_ns()-t)/1e6,gpu_ms=meter.origin.elapsed_time(meter.end))
                                if marked:row.update(meter.read(False))
                                if not equal(value,reference):raise RuntimeError('Native control output differs')
                                rows.append(row)
                        report['coarse'].append(dict(name=name,samples=rows));save()
                    coarse('clean_before',False)
                    coarse('phases_before',True)
                    from builder import build_cases
                    cases,audit=build_cases(model,cache,call,reference)
                    report['case_audit']=audit
                    log('Validate and measure '+str(len(cases))+' complete segments')
                    cache.active=True
                    try:
                        for i,case in enumerate(cases):
                            report['cases'].append(measure_case(case,eviction,ns.samples,guard))
                            if i%10==0:
                                save();log(f'SEGMENT {i+1}/{len(cases)} {case["id"]}')
                    finally:cache.active=False
                    for _ in range(10):call()
                    torch.cuda.synchronize()
                    coarse('phases_after',True)
                    coarse('clean_after',False)
                    report['cache_audit']=cache.metadata()
                report['reference_actions']=reference.cpu().tolist()
        guard('finished')
        report.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save();log('CAPTURE_COMPLETE '+ns.mode)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:(out/'exit.json').write_text(json.dumps(dict(status=report['status']))+'\n')


if __name__=='__main__':main()

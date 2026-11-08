"""Short supplementary capture, no large profiler traces or framework changes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/var/tmp/shiyuhuang-vla-baseline')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--probe',required=True)
    ns=ap.parse_args()
    if not all(s.replace('_','').isalnum() for s in (ns.tag,ns.probe)):raise ValueError('Invalid tags')
    out=ROOT/'runs'/ns.tag;out.mkdir(exist_ok=False)
    report=dict(status='running',client_date='2026-11-08',host=os.uname().nodename,pid=os.getpid(),cases=[],occupancy=[],
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplementary=True,
        pim_contract_commit='8764996de08741f248746b7094830360f666fbe6')
    def save():(out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
    def guard(label):
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True)
        others=[int(s.split(',')[0]) for s in raw.splitlines() if s.split(',')[0].strip().isdigit() and int(s.split(',')[0])!=os.getpid()]
        fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
        report['occupancy'].append(dict(label=label,raw=raw,others=others,free_disk_bytes=free))
        if others or free<256*1024**2:save();raise RuntimeError('Other GPU user or insufficient disk reserve')
    def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
    try:
        guard('before_import')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17)
        from builder import prepare
        from interfaces import build_interfaces
        from timing import Eviction,measure_case,equal
        p=ROOT/'runs'/ns.probe/'capture.json'
        probe=json.loads(p.read_text())
        if probe['status']!='complete' or not probe['cache_probe']['passed']:raise RuntimeError('No qualified cache probe')
        report['probe_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        with torch.inference_mode(False),torch.no_grad():model,cache,call,reference,metadata=prepare(log,guard)
        report.update(metadata)
        with torch.inference_mode(),cache.applied():
            eviction=Eviction()
            for i in range(20):
                if i%5==0:guard('warmup_'+str(i))
                value=call()
            torch.cuda.synchronize()
            if not equal(value,reference):raise RuntimeError('Full output mismatch')
            cases=build_interfaces(model,cache,call,reference)
            cache.active=True
            try:
                for i,c in enumerate(cases):
                    report['cases'].append(measure_case(c,eviction,20,guard))
                    if i%12==0:save();log(f'INTERFACE {i+1}/72')
            finally:cache.active=False
            if not equal(call(),reference):raise RuntimeError('Post-capture full output differs')
            report['cache_audit']=cache.metadata()
            report['reference_actions']=reference.cpu().tolist()
        guard('finished')
        report.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save();log('CAPTURE_COMPLETE interfaces')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:(out/'exit.json').write_text(json.dumps(dict(status=report['status']))+'\n')


if __name__=='__main__':main()

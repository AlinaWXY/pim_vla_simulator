"""Independently audit saved schedules with interval integrals, not replay()."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def close(x,y):
    assert math.isclose(x,y,rel_tol=1e-10,abs_tol=2e-6),(x,y)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    def rows(name):
        with (out/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    f=lambda r,k:float(r[k])
    with gzip.open(out/'native-bus-templates.json.gz','rt') as fp:stages=json.load(fp)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    units=rows('VLM-ordered-units.csv')
    checks=[]
    for r in rows('evaluation-b-g.csv')+rows('shared-tensor-io-accounting.csv'):
        if r['case']=='b':
            close(f(r,'total_ms'),f(r,'AE_solo_ms')+f(r,'VLM_ms'));continue
        variant=r['variant'];name=variant+'-'+r['accounting']
        solo=rows(name+'-solo-schedule.csv');actual=rows(name+'-execution.csv')
        placed=rows(name+'-VLM-placement.csv');waits=rows(name+'-AE-GPU-wait.csv')
        assert len(solo)==len(actual)
        start=[f(x,'start_us') for x in placed];end=[f(x,'end_us') for x in placed]
        assert [int(x['ordinal']) for x in placed]==list(range(len(placed)))
        assert all(x<=y+1e-7 for x,y in zip(end,start[1:]))
        integral=[0.0]*len(placed)
        for block,execution in zip(solo,actual):
            close(f(block,'end_us')-f(block,'begin_us'),f(execution,'end_us')-f(execution,'begin_us'))
            left,right=f(execution,'begin_us'),f(execution,'end_us')
            begin=bisect_right(end,left+1e-7);last=bisect_left(start,right-1e-7)
            if block['kind']=='gpu':
                assert begin>=last,(name,'VLM overlaps AE GPU',left,right)
                continue
            s=stages[variant+'/'+block['source_case']]
            base=int(block['source_begin_tick']);stop=int(block['source_end_tick'])
            # Independent area integral over intersections of actual kernel
            # residence and translated raw availability, including all stalls.
            for x,y in s['service_idle']:
                x,y=max(x,base),min(y,stop)
                if y<=x:continue
                lo=left+(x-base)*s['tick_us'];hi=left+(y-base)*s['tick_us']
                for i in range(begin,last):
                    integral[i]+=max(0.0,min(hi,end[i])-max(lo,start[i]))
        by_unit={int(w['vlm_ordinal']):w for w in waits}
        for i,x in enumerate(placed):
            close(integral[i],f(x,'progress_during_pim_us'))
            close(f(x,'solo_us'),f(units[i],'normalized_us'))
            tail=f(x,'tail_wait_us')
            close(tail,f(by_unit[i],'wait_us') if i in by_unit else 0)
            close(f(x,'solo_us'),integral[i]+tail)
        close(sum(integral)/1000,f(r,'VLM_progress_during_PIM_ms'))
        close(sum(f(w,'wait_us') for w in waits)/1000,f(r,'AE_gpu_wait_ms'))
        close(f(actual[-1],'end_us')/1000,f(r,'AE_with_interference_ms'))
        close(f(r,'total_ms'),f(r,'AE_with_interference_ms')+f(r,'VLM_remaining_after_AE_ms'))
        close(f(r,'total_ms'),f(r,'AE_solo_ms')+f(r,'VLM_ms')-sum(integral)/1000)
        packs=[b for b in solo if b['name'].endswith('_direct_PV_to_O_pack')]
        assert len(packs)==(180 if variant.endswith('before') else 0)
        assert all(stages[variant+'/'+c]['config']['pim_rank_controller']==variant.endswith('after')
                   for c in ['chain','pv','gate_up_gpu','down_gpu_input'])
        checks.append(dict(variant=variant,accounting=r['accounting'],status='passed',
                           VLM_units_verified=len(placed),AE_wait_events_verified=len(waits),
                           raw_idle_interval_integrals_match=True,no_AE_GPU_VLM_overlap=True))
    root=Path(__file__).resolve().parents[2]
    for original,digest in summary['source_sha256'].items():
        relative=original.split('/pim_vla_xinyao_1108_20261108/')[-1]
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
    result=dict(status='passed',checks=checks,source_hashes_verified=len(summary['source_sha256']))
    (out/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

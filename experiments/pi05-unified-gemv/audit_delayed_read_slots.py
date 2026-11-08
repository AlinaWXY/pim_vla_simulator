"""Necessary bus conditions for independent hypothetical row-hit RD probes.

No GPU traffic is generated. Each CK is an independent candidate, NOT a
scheduled stream, bandwidth estimate, or GPU service-time estimate. ACT/PRE,
bank state, rank switching, direction turnaround, request dependencies and
controller admission can further reject candidates classified as possible.
"""
import argparse
from bisect import bisect_right
import csv
import gzip
import hashlib
import json
from pathlib import Path


def overlap(intervals, ends, a, b):
    i=bisect_right(ends,a)
    return intervals[i] if i<len(intervals) and intervals[i][0]<b else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    a=ap.parse_args();out=a.reference/'delayed-read-audit';out.mkdir(exist_ok=True)
    source=a.reference/'corrected-overlap/native-bus-templates.json.gz'
    with gzip.open(source,'rt') as f:templates=json.load(f)
    rows=[];examples=[]
    for variant in ['token_local_before','token_local_after']:
        for case in ['chain','pv','gate_up_gpu','down_gpu_input']:
            s=templates[variant+'/'+case]
            r=json.loads((a.reference/'direct-before-corrected/native-provenance'/variant/case/'result.json').read_text())
            ratio=r['config']['frontend_ratio'];tm=r['resolved_timings_ck'];cl=tm['nCL']*ratio;bl=tm['nBL_min']*ratio
            # This diagnostic is per-channel. Verify that channel 0 represents
            # these retained synchronous traces; never silently average channels.
            for ch in s['channels']:assert ch['ca']==s['channels'][0]['ca'] and ch['dq']==s['channels'][0]['dq']
            ca,dq=s['channels'][0]['ca'],s['channels'][0]['dq']
            refs=s['gpu_rank_refresh']
            ce,de,re=([x[1] for x in intervals] for intervals in [ca,dq,refs])
            for phase in s['phases']:
                name=phase['name']
                if not ('compute' in name and 'release' not in name or name in
                        ['qkv_run_local_spills','pv_run_local_spill','o_run_two_tiles_with_direct_export',
                         'gate_up_run_with_direct_export','down_run_with_direct_export']):continue
                begin,end=phase['begin'],phase['end']
                count=now_idle=bad_future=passed=0;example=None
                first=((begin-1+ratio-1)//ratio)*ratio+1
                for t in range(first,end-cl-bl+1,ratio):
                    count+=1
                    ca_busy=overlap(ca,ce,t,t+ratio)
                    dq_now=overlap(dq,de,t,t+ratio)
                    dq_future=overlap(dq,de,t+cl,t+cl+bl)
                    ref_busy=overlap(refs,re,t,t+cl+bl)
                    if not ca_busy and not dq_future and not ref_busy:passed+=1
                    if ca_busy or dq_now:continue
                    now_idle+=1
                    if dq_future:
                        bad_future+=1
                        if example is None:
                            ck=lambda x:(x-1)/ratio+1
                            example=dict(variant=variant,case=case,phase=name,channel=0,
                                RD_CA_ck=[ck(t),ck(t+ratio)],DQ_required_ck=[ck(t+cl),ck(t+cl+bl)],
                                existing_PIM_DQ_ck=[ck(dq_future[0]),ck(dq_future[1])],
                                CA_and_DQ_now_idle=True,delayed_DQ_collision=True)
                rows.append(dict(variant=variant,case=case,phase=name,channel=0,
                    independent_CK_candidates=count,CA_and_DQ_now_idle_candidates=now_idle,
                    now_idle_but_return_DQ_collides=bad_future,
                    false_positive_percent=100*bad_future/now_idle if now_idle else 0,
                    passes_CA_future_DQ_refresh_necessary_check=passed,
                    CL_CK=tm['nCL'],burst_CK=tm['nBL_min'],tCK_ns=r['tCK_ns']))
                if example:examples.append(example)
    with (out/'independent-read-candidates.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='necessary-condition diagnostic only; no latency inference',
        source_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest(),
                       str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        definition=__doc__,examples=examples,rows=rows)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(examples[:2],indent=2))
    for row in rows:print(row['variant'],row['phase'],row['now_idle_but_return_DQ_collides'],
                          '/',row['CA_and_DQ_now_idle_candidates'],round(row['false_positive_percent'],3))


if __name__=='__main__':main()

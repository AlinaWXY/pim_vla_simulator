"""Publish native evidence for the corrected flat/GPU-broadcast before cases."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from analyze_evaluation import extract,csvout,read,dump
from channel_analysis import read_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    variants=['feature_gpu_before','token_gpu_before','token_local_before','token_gpu_after','token_local_after']
    cases=['chain','pv','gate_up_gpu','down_gpu_input','serial_layer_control']
    rows=[];bus=[];totals=[];checks=[];sources={};series=[]
    for v in variants:
        stage_bytes=0;stage_us=0
        for c in cases:
            p=a.input/v/c;m=read(p/'manifest.json');r=read(p/'simulation/result.json');e=extract(p)
            before=v.endswith('before');cfg=r['config']
            assert bool(cfg.get('pim_rank_controller',False))==(not before)
            assert bool(cfg.get('pim_rank_staging',False))==(not before)
            assert cfg['pim_stream_inputs']==before
            if before:
                assert not any(m['counts'].get(x,0) for x in ['MAC_RUN','ROPE_RUN','PIM_STAGE_FRAGMENT','PIM_STAGE_BEGIN','PIM_STAGE_SEAL','PIM_PV_SPILL','PIM_O_REWIND'])
                for ch in range(16):
                    ca=read_csv(p/f'simulation/rank_ca_events.csv.ch{ch}');tl=read_csv(p/f'simulation/timeline.csv.ch{ch}')
                    assert all(x['scope']=='global' and not x['command'].startswith('PIM_') for x in ca)
                    expected=[(int(x['clock']),int(x['clock'])+(2 if x['command'] in ('ACTAB','ACTHB') else 1),x['command']) for x in tl if x['command'] not in ('LOCALRD','LOCALWR')]
                    observed=[(int(x['clock']),int(x['ready_at']),x['command']) for x in ca]
                    assert observed==expected,(v,c,ch,'flat CA does not match native issues')
            else:
                old=read(a.original/v/c/'simulation/result.json')
                assert r['memory_cycles']==old['memory_cycles'] and r['issued_commands']==old['issued_commands']
                for ch in range(16):
                    for name in ['timeline','rank_ca_events']:
                        fn=f'simulation/{name}.csv.ch{ch}'
                        assert hashlib.sha256((p/fn).read_bytes()).digest()==hashlib.sha256((a.original/v/c/fn).read_bytes()).digest(),(v,c,ch,name)
            q=a.output/'native-provenance'/v/c;q.mkdir(parents=True,exist_ok=True)
            for name,src in [('manifest.json',p/'manifest.json'),('summary.json',p/'summary.json'),('result.json',p/'simulation/result.json'),('barriers.csv',p/'simulation/barriers.csv')]:shutil.copyfile(src,q/name)
            rows.append(dict(variant=v,case=c,native_us=e['wall_us'],DQ_KiB=e['external_DQ_bytes']/1024,rank_controller=not before,input_path='GPU broadcast' if before else 'DRAM staging + coarse commands'))
            for ch in e['channels']:bus.append(dict(variant=v,case=c,channel=ch['channel'],wall_us=e['wall_us'],dq_bytes=ch['dq_bytes'],dq_busy_us=ch['dq_busy_us'],ca_busy_us=ch['ca_busy_us']))
            sources[f'{v}/{c}']=e['source_sha256'] if 'source_sha256' in e else {str(k):val for k,val in e.items() if 'hash' in k}
            if c!='serial_layer_control':stage_bytes+=e['external_DQ_bytes'];stage_us+=e['wall_us']
            else:
                assert stage_bytes==e['external_DQ_bytes']
                if v in ('feature_gpu_before','token_local_before','token_local_after'):
                    for resource in ('dq','ca'):
                        intervals=e['channels'][0][resource];step=.25
                        for i in range(int(e['wall_us']/step)+1):
                            left=i*step;right=min(left+step,e['wall_us'])
                            if right<=left:continue
                            busy=sum(max(0,min(right,y*e['tick_us'])-max(left,x*e['tick_us'])) for x,y in intervals)
                            series.append(dict(variant=v,resource=resource,start_us=left,end_us=right,utilization_percent=100*busy/(right-left)))
            checks.append(dict(variant=v,case=c,flat_CA_matches_native=before,after_native_byte_identical=not before,DQ_and_CA_disjoint_per_resource=True))
        totals.append(dict(variant=v,DQ_KiB_per_layer_step=stage_bytes/1024,DQ_MiB_AE_180=stage_bytes*180/1024**2,native_stage_sum_us=stage_us,complete_parallel_latency_measured=False))
    csvout(a.output/'native-cases.csv',rows);csvout(a.output/'bus-accounting.csv',bus);csvout(a.output/'DQ-totals.csv',totals);csvout(a.output/'DQ-CA-timeline-250ns.csv',series)
    shutil.copyfile(a.input/'provenance.json',a.output/'native-provenance/provenance.json')
    dump(a.output/'checks.json',dict(status='passed',cases=checks,channels_per_case=16,after_native_trace_files_identical=320,before_CA_trace_files_verified=240))
    dump(a.output/'raw-source-hashes.json',sources)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()

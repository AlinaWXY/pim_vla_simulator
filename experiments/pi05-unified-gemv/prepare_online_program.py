"""Attach native DQ bursts to their causal issuing commands, preserving CA widths."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
from channel_analysis import dq_intervals

CK=15008
VARIANTS=['feature_gpu_before','token_gpu_before','token_local_before','token_gpu_after','token_local_after']
CASES=['chain','pv','gate_up_gpu','down_gpu_input']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result={}
    for v in VARIANTS:
        for case in CASES:
            root=a.runs/v/case;r=json.loads((root/'simulation/result.json').read_text())
            ratio=r['config']['frontend_ratio'];assert ratio==32
            jobs=[];hashes={};counts={};ownership=[]
            if v.endswith('after'):
                manifest=json.loads((root/'manifest.json').read_text())
                with (root/'simulation/barriers.csv').open() as f:marks={int(x['marker']):int(x['frontend_tick']) for x in csv.DictReader(f)}
                for s in manifest['segments']:
                    name=s['name'].split('/')[-1]
                    if name.endswith('gpu_input_stage') or name in ('q_export_to_gpu','q_remote_buffer_fill','qk_scores_dram_export','gpu_rope_export','gpu_rope_import'):
                        for ch in range(16):ownership.append([ch,(marks[s['start']]-1)*469,(marks[s['end']]-1)*469])
            for ch in range(16):
                paths=[root/f'simulation/{n}.csv.ch{ch}' for n in ['timeline','rank_ca_events']]
                for f in paths:hashes[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
                if v.endswith('after'):
                    runs=root/f'simulation/coarse_runs.csv.ch{ch}'
                    hashes[str(runs.relative_to(root))]=hashlib.sha256(runs.read_bytes()).hexdigest()
                    opened={}
                    with runs.open() as f:
                        for x in csv.DictReader(f):
                            key=(x['run_id'],x['tile'])
                            if x['event']=='TILE_READY':opened[key]=(int(x['clock'])-1)*CK
                            if x['event']=='TILE_RELEASE':
                                begin=opened.pop(key);ownership.append([ch,begin,(int(x['clock'])-1)*CK])
                    assert not opened
                with paths[0].open() as f:timeline=list(csv.DictReader(f))
                with paths[1].open() as f:ca=list(csv.DictReader(f))
                by_clock={int(x['clock']):x for x in timeline}
                bursts={x['clock']:x for x in dq_intervals(timeline,r['resolved_timings_ck'],ratio,r['config']['pim_data_path'])}
                consumed=set()
                for x in ca:
                    if x['scope']!='global' or x['command'] in ('REFab','REFpb'):continue
                    clock=int(x['clock']);command=x['command'];row=by_clock.get(clock)
                    burst=bursts.get(clock) if row is not None and row['command']==command else None
                    ready=(clock-1)*CK;length=(int(x['ready_at'])-clock)*CK
                    rank=int(x['Rank']);bg=int(row['BankGroup']) if row is not None else -1
                    write=int(command in ('WR','WRA','WRGB'))
                    array=int(not (command.startswith('PIM_') or command in ('WRGB','RDRES','MAC_RUN','ROPE_RUN')))
                    latency=(burst['begin']-1)*469-ready if burst else 0
                    dq=(burst['end']-burst['begin'])*469 if burst else 0
                    jobs.append([ch,rank,bg,write,array,ready,length,latency,dq])
                    counts[command]=counts.get(command,0)+1
                    if burst:consumed.add(clock)
                for clock,b in bursts.items():
                    if clock in consumed:continue
                    row=by_clock[clock];cmd=row['command'];ready=(clock-1)*CK
                    # Autonomous after-mode result transfers can use DQ without
                    # a new external CA command; they still request the I/O port.
                    jobs.append([ch,int(row['Rank']),int(row['BankGroup']),int(cmd in ('WR','WRA','WRGB')),
                                 int(cmd not in ('WRGB','RDRES')),ready,0,
                                 (b['begin']-1)*469-ready,(b['end']-b['begin'])*469])
            result[v+'/'+case]=dict(jobs=jobs,ownership=ownership,source_sha256=hashes,global_commands=counts,
                contract='Only external CA and external DQ are jobs. Private rank commands consume no external CA. Native inter-command gaps are preserved and shift after blocking; bank/row state is not regenerated.')
            print(v,case,len(jobs),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.output,'wt') as f:json.dump(result,f,separators=(',',':'))

if __name__=='__main__':main()

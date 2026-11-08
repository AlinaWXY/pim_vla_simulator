"""Native bus accounting plus retained GPU kernels and ordered whole-kernel fit."""
import argparse,csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'source'),str(HERE.parent/'pi05-bgk-gpuq/backend')]
from evaluation_workload import VARIANTS
from channel_analysis import read_csv,dq_intervals
from dq_ca_windows import ca_intervals

CASES=('chain','pv','gate_up_gpu','down_gpu_input')
REMOVED={'q_projection','k_projection','v_projection','attention_qkt','attention_sv',
         'ffn_gate','ffn_up','ffn_down','rope_apply','kv_concat','o_projection'}
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvout(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    opener=gzip.open(p,'wt',newline='') if str(p).endswith('.gz') else Path(p).open('w',newline='')
    with opener as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def merge(seq):
    out=[]
    for a,b in sorted(seq):
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
def gaps(seq,a,b):
    out=[];cursor=a
    for x,y in seq:
        if y<=a:continue
        if x>=b:break
        x=max(x,a);y=min(y,b)
        if x>cursor:out.append([cursor,x])
        cursor=max(cursor,y)
    if cursor<b:out.append([cursor,b])
    return out
def duration(x):return sum(b-a for a,b in x)

def extract(path):
    m=read(path/'manifest.json');r=read(path/'simulation/result.json');s=read(path/'summary.json')
    ratio=r['config']['frontend_ratio'];wall=r['memory_cycles']*ratio;tick=r['tCK_ns']/ratio/1000
    marks={int(x['marker']):int(x['frontend_tick']) for x in read_csv(path/'simulation/barriers.csv')}
    phases=[dict(name=x['name'],kind=x.get('kind',''),begin=marks[x['start']],end=marks[x['end']]) for x in m['segments']]
    channels=[];all_dq=[];all_ca=[];n=0;hashes={}
    for c in range(16):
        dqpath=path/f'simulation/timeline.csv.ch{c}';capath=path/f'simulation/rank_ca_events.csv.ch{c}'
        ds=dq_intervals(read_csv(dqpath),r['resolved_timings_ck'],ratio,r['config']['pim_data_path'])
        cs=ca_intervals(read_csv(capath),ratio,c)['global_ca']
        d=[[x['begin'],x['end']] for x in ds];ca=[[x['begin'],x['end']] for x in cs]
        assert all(0<=a<b<=wall for a,b in d+ca),(path,c,wall)
        assert duration(d)==duration(merge(d)),('DQ collision',path,c)
        n+=len(d);all_dq+=d;all_ca+=ca
        channels.append(dict(channel=c,dq_bytes=len(d)*32,dq_busy_us=duration(merge(d))*tick,
            ca_busy_us=duration(merge(ca))*tick,dq=merge(d),ca=merge(ca)))
        hashes[str(dqpath.relative_to(path))]=sha(dqpath);hashes[str(capath.relative_to(path))]=sha(capath)
    assert n*32==s['external_DQ_bytes']==m['expected_dq_bytes']['total']
    joint=merge(all_dq+all_ca);dq=merge(all_dq);ca=merge(all_ca)
    return dict(case=m['case'],wall_ticks=wall,tick_us=tick,wall_us=wall*tick,phases=phases,
        external_DQ_bytes=n*32,channels=channels,dq=dq,ca=ca,joint=joint,
        joint_idle=gaps(joint,0,wall),source_sha256=hashes,config=r['config'],path=str(path))

def kernels(gpu):
    ledger=read(gpu/'cold-ae-kernel-ledger.json');groups=[[x for x in ledger if x['repeat']==i] for i in range(3)]
    assert [len(x) for x in groups]==[1033]*3
    out=[]
    for i,ks in enumerate(zip(*groups)):
        assert len({(x['category'],x['operator'],str(x['shapes']),x['kernel']) for x in ks})==1
        out.append(dict(ordinal=i,category=ks[0]['category'],operator=ks[0]['operator'],shapes=ks[0]['shapes'],
                        us=sum(x['kernel_ns'] for x in ks)/3000))
    return out

def effective_pieces(stage,after):
    # Shared tensor stores belong to the retained GPU producer; bulk score reads
    # belong to the GPU scaling consumer. Physical DQ is kept in raw-native tables.
    excluded=[(p['begin'],p['end']) for p in stage['phases'] if
              p['name'].split('/')[-1] in ('gpu_rope_export','gpu_rope_import') or
              (after and (p['name'].split('/')[-1].endswith('_gpu_input_stage') or p['name'].split('/')[-1]=='qk_scores_dram_export'))]
    return gaps(merge(excluded),0,stage['wall_ticks']),sum(b-a for a,b in excluded)*stage['tick_us']

def make_schedule(v,stages,ks,handoffs):
    after=VARIANTS[v]['after'];gpu_rope=VARIANTS[v]['gpu_rope'];feature=VARIANTS[v]['feature']
    blocks=[];windows=[];cursor=0.;seen=set();layer=-1
    rms=[k['ordinal'] for k in ks if k['category']=='rms_pow'];assert len(rms)==37
    starts={rms[2*i]-1:i for i in range(18)};starts[rms[-1]-1]=18
    def add_gpu(name,us):
        nonlocal cursor
        blocks.append(dict(kind='gpu',name=name,begin_us=cursor,end_us=cursor+us,layer=layer));cursor+=us
    def add_pim(case,section='all'):
        nonlocal cursor
        s=stages[case];pieces,_=effective_pieces(s,after);split=s['wall_ticks']
        if case=='chain':
            names=('gpu_rope_export',) if gpu_rope else ('projection_gemv_release','qkv_run_local_spills')
            split=next(p['end'] for p in s['phases'] if p['name'].split('/')[-1] in names)
        left,right=(0,split) if section=='pre' else (split,s['wall_ticks']) if section=='post' else (0,s['wall_ticks'])
        for a,b in pieces:
            a=max(a,left);b=min(b,right)
            if a>=b:continue
            us=(b-a)*s['tick_us']
            blocks.append(dict(kind='pim',name=case+'_'+section,begin_us=cursor,end_us=cursor+us,layer=layer,
                               source_case=case,source_begin_tick=a,source_end_tick=b))
            for x,y in s['joint_idle']:
                x=max(x,a);y=min(y,b)
                if y>x:windows.append(dict(begin_us=cursor+(x-a)*s['tick_us'],end_us=cursor+(y-a)*s['tick_us'],
                    duration_us=(y-x)*s['tick_us'],case=case,layer=layer))
            cursor+=us
    for k in ks:
        cat=k['category']
        if k['ordinal'] in starts:layer=starts[k['ordinal']]
        trigger={'q_projection':('chain','pre'),'attention_qkt':('chain','post'),
                 'attention_sv':('pv','all'),'ffn_gate':('gate_up_gpu','all'),'ffn_down':('down_gpu_input','all')}.get(cat)
        if trigger and (layer,trigger) not in seen:
            seen.add((layer,trigger));add_pim(*trigger)
            if trigger==('chain','post') and feature:add_gpu('feature_score_rank_sum',handoffs['feature_score_rank_sum'])
            if trigger==('pv','all'):add_gpu('common_O_rank_sum_cast',handoffs['common_O_rank_sum_cast'])
            if trigger==('gate_up_gpu','all'):add_gpu('common_GateUp_fp32_to_bf16',handoffs['common_GateUp_fp32_to_bf16'])
            if trigger==('down_gpu_input','all'):add_gpu('common_Down_fp32_to_bf16',handoffs['common_Down_fp32_to_bf16'])
        keep=cat not in REMOVED or (gpu_rope and cat=='rope_apply')
        if cat=='copies_casts' and k['operator']=='aten::contiguous' and k['shapes']==[[1,16,8,256]]:keep=False
        if keep:add_gpu(cat,k['us'])
    assert len(seen)==18*5,(v,len(seen))
    expected=sum((s['wall_us']-effective_pieces(s,after)[1]) for s in stages.values())*18
    assert math.isclose(sum(b['end_us']-b['begin_us'] for b in blocks if b['kind']=='pim'),expected,abs_tol=1e-6)
    full=[];wins=[]
    for step in range(10):
        full.extend(dict(**{**b,'begin_us':b['begin_us']+step*cursor,'end_us':b['end_us']+step*cursor},step=step) for b in blocks)
        wins.extend(dict(**{**w,'begin_us':w['begin_us']+step*cursor,'end_us':w['end_us']+step*cursor},step=step) for w in windows)
    return full,wins,cursor/100

def fit(windows,units,launch=False,min_window_us=0):
    i=0;placements=[]
    for w in windows:
        if w['duration_us']<min_window_us:continue
        t=w['begin_us']
        while i<len(units):
            u=units[i];cost=u['us']+(u['launch_us'] if launch else 0)
            if t+cost>w['end_us']+1e-9:break
            placements.append(dict(ordinal=i,kind=u.get('kind','kernel'),begin_us=t,end_us=t+cost,kernel_us=u['us'] if u.get('kind','kernel')=='kernel' else 0,launch_us=u['launch_us'] if launch else 0,
                window_begin_us=w['begin_us'],window_end_us=w['end_us']))
            t+=cost;i+=1
    return dict(placed_kernels=sum(p['kind']=='kernel' for p in placements),placed_work_units=i,overlap_ms=sum(p['kernel_us'] for p in placements)/1000,
        next_kernel_us=units[i]['us'] if i<len(units) else None,next_launch_us=units[i]['launch_us'] if i<len(units) else None,
        placements=placements)

def plot_raw(out,raw):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,6),constrained_layout=True,sharex=True)
    maxwall=max(s['wall_us'] for s in raw.values());edges=np.arange(0,maxwall+.25,.25)
    rows=[]
    for ax,(v,s) in zip(axes,raw.items()):
        y=np.zeros(len(edges)-1);ca=np.zeros(len(y))
        for channel in s['channels']:
            for name,arr in (('dq',y),('ca',ca)):
                for a,b in channel[name]:
                    a*=s['tick_us'];b*=s['tick_us']
                    i=max(0,int(a/.25));j=min(len(arr)-1,int(b/.25))
                    for k in range(i,j+1):arr[k]+=max(0,min(b,edges[k+1])-max(a,edges[k]))/(.25*16)
        valid=edges[:-1]<s['wall_us'];ax.fill_between(edges[:-1][valid],y[valid]*100,step='post',alpha=.8,label='DQ active channel-time (%)')
        ax.plot(edges[:-1][valid],ca[valid]*100,lw=.65,color='#b3591b',label='Shared CA (%)')
        ax.set_title(v+' | raw continuous native layer (GPU arithmetic omitted)',loc='left',fontsize=10)
        ax.set_ylim(0,105);ax.set_ylabel('Occupancy (%)');ax.legend(loc='upper right',fontsize=8)
        for k in np.where(valid)[0]:rows.append(dict(variant=v,begin_us=float(edges[k]),end_us=float(min(edges[k+1],s['wall_us'])),dq_percent=float(y[k]*100),ca_percent=float(ca[k]*100)))
    axes[-1].set_xlabel('Time (us)');fig.savefig(out/'dq-timeline.png',dpi=180);fig.savefig(out/'dq-timeline.svg');plt.close(fig)
    csvout(out/'dq-timeline-250ns.csv',rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);gpu=out/'gpu'
    ks=kernels(gpu);hm=read(gpu/'handoff-measurements.json')
    handoffs={k:x['cold32MiB']['device_kernel_median_us'] for k,x in hm['operators'].items()}
    prefix=read(gpu/'prefix-kernels.json');units=prefix['ordered_mean_units']
    warm=read(gpu/'warm-summary.json')['baseline']['phase_ms'];vlm=warm['vit']['mean']+warm['vlm']['mean']
    allrows=[];traff=[];buses=[];validation=[];rawplots={};allstages={};result=[]
    for v in VARIANTS:
        stages={}
        for c in (*CASES,'serial_layer_control'):
            print('EXTRACT',v,c,flush=True);s=extract(a.runs/v/c)
            allstages[v+'/'+c]=s
            allrows.append(dict(variant=v,case=c,native_us=s['wall_us'],DQ_bytes=s['external_DQ_bytes']))
            for ch in s['channels']:
                buses.append(dict(variant=v,case=c,channel=ch['channel'],wall_us=s['wall_us'],
                    dq_bytes=ch['dq_bytes'],dq_busy_us=ch['dq_busy_us'],ca_busy_us=ch['ca_busy_us']))
            if c!='serial_layer_control':stages[c]=s
            elif v in ('feature_gpu_before','token_local_after'):rawplots[v]=s
        dq=sum(s['external_DQ_bytes'] for s in stages.values())
        assert dq==allstages[v+'/serial_layer_control']['external_DQ_bytes']
        traff.append(dict(variant=v,bytes_per_layer_step=dq,KiB_per_layer_step=dq/1024,
                          bytes_per_AE=dq*180,MiB_per_AE=dq*180/1024**2,
                          reduction_vs_feature_percent=(1-dq/6045696)*100))
        schedule,windows,ae=make_schedule(v,stages,ks,handoffs)
        csvout(out/(v+'-schedule.csv'),schedule)
        csvout(out/(v+'-joint-idle-windows.csv.gz'),windows)
        fitted=fit(windows,units);csvout(out/(v+'-VLM-placements.csv'),fitted['placements'])
        host_launch_fit=fit(windows,units,launch=True)
        sensitivity=[dict(min_window_us=t,**{k:x for k,x in fit(windows,units,min_window_us=t).items() if k!='placements'}) for t in (0,1,2,5,10,20)]
        csvout(out/(v+'-threshold-sensitivity.csv'),sensitivity)
        row=dict(variant=v,AE_known_model_ms=ae,VLM_measured_ms=vlm,serial_known_model_ms=ae+vlm,
                 overlap_whole_kernel_ms=fitted['overlap_ms'],parallel_known_model_ms=ae+vlm-fitted['overlap_ms'],
                 placed_VLM_kernels=fitted['placed_kernels'],next_kernel_us=fitted['next_kernel_us'],next_launch_us=fitted['next_launch_us'],
                 overlap_with_host_API_reserved_ms=host_launch_fit['overlap_ms'],
                 joint_idle_total_ms=sum(w['duration_us'] for w in windows)/1000,
                 longest_joint_window_us=max(w['duration_us'] for w in windows),
                 GPU_known_model_ms=sum(b['end_us']-b['begin_us'] for b in schedule if b['kind']=='gpu')/1000,
                 PIM_known_model_ms=sum(b['end_us']-b['begin_us'] for b in schedule if b['kind']=='pim')/1000)
        result.append(row)
        if v=='token_local_after':
            oldroot=Path('/fact_data/xinyaowang/pim_vla_xinyao_1107_20261107/final/after')
            for c in (*CASES,'serial_layer_control'):
                old=read(oldroot/c/'summary.json');new=read(a.runs/v/c/'summary.json')
                assert old['memory_cycles']==new['memory_cycles'],('old mode regression',c,old['memory_cycles'],new['memory_cycles'])
                assert old['external_DQ_bytes']==new['external_DQ_bytes']
            validation.append('All five unchanged token/local/after cases match 1107 native cycles and DQ bytes exactly')
    table=[dict(case='a',variant='pure_GPU_existing',AE_ms=warm['ae']['mean'],VLM_ms=vlm,overlap_ms=0,total_ms=warm['ae']['mean']+vlm,kind='existing measured phase means')]
    for letter,v,parallel in [('b','feature_gpu_before',False),('c','feature_gpu_before',True),('d','token_gpu_before',True),('e','token_local_before',True),('f','token_gpu_after',True),('g','token_local_after',True)]:
        r=next(x for x in result if x['variant']==v);overlap=r['overlap_whole_kernel_ms'] if parallel else 0
        table.append(dict(case=letter,variant=v,AE_ms=r['AE_known_model_ms'],VLM_ms=vlm,overlap_ms=overlap,total_ms=r['AE_known_model_ms']+vlm-overlap,kind='known-term hybrid model; not hardware E2E'))
    csvout(out/'evaluation-a-g.csv',table);csvout(out/'DQ-traffic.csv',traff);csvout(out/'native-latency.csv',allrows)
    csvout(out/'bus-accounting.csv',buses);csvout(out/'model-details.csv',result)
    plot_raw(out,rawplots)
    with gzip.open(out/'native-bus-templates.json.gz','wt') as f:json.dump(allstages,f,separators=(',',':'))
    summary=dict(branch='xinyao_1108',native_run=str(a.runs),evaluation=table,traffic=traff,models=result,
        policy=dict(VLM='vision + language prefix',GPU_baseline_rerun=False,VLM_baseline_rerun=False,
            parallel='AE + VLM - whole ordered kernels fitted inside all-16-channel DQ and shared-CA idle windows',
            preemption=False,kernel_reordering=False,dispatch='primary fits whole measured device kernels; prerequisite memory operations reserve windows but are not counted as compute savings; reserving host CUDA API duration is separate sensitivity',
            same_tensor_IO='GPU RoPE RD/WR, native input staging ST and bulk score RD assigned to existing GPU producer/consumer; physical raw counters retained',
            GPU_AE='retained NCU cold kernel means; added boundary CUDA device kernel medians',
            arithmetic_precision='timing and layout simulator, not tensor-executing numerical simulator'),
        unmeasured=['GPU kernel duration change for final rank-replicated tensor layout / padded data',
          'cache-coherence, visibility, fences and actual GPU/PIM scheduler/context-switch overhead',
          'step-external Euler/host overhead under the changed execution schedule',
          'actual GPU/PIM concurrent execution and interference'],
        validation=validation,source_native_provenance_sha256=sha(a.runs/'provenance.json'))
    dump(out/'summary.json',summary);print(json.dumps(table,indent=2),flush=True)

if __name__=='__main__':main()

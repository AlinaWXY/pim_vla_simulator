"""Native before/after CA and DQ plots; merge only genuinely contiguous idle time."""
import argparse,csv,gzip,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from channel_analysis import read_csv,read_json,dq_intervals,interval_accounting
from dq_ca_windows import ca_intervals
from continuous_dq import union_intervals
import reproduce as R

CASES=('chain','pv','gate_up_gpu','down_gpu_input','serial_layer_control')
LABELS=('QKV + local RoPE + QKT','PV + O projection','FFN Gate/Up','FFN Down','Continuous layer')
COLORS={'before':'#64748b','after':'#007f86'}


def analyze(directory):
    result=read_json(directory/'simulation/result.json');cfg=result['config']
    ratio=cfg['frontend_ratio'];end=result['memory_cycles']*ratio
    scale=result['tCK_ns']/ratio/1000
    resources={k:[] for k in ('dq','global_ca','rank1_ca','dq_global_ca')};stats={};raw_idle=[]
    all_intervals={k:[] for k in resources}
    for ch in range(16):
        timeline=read_csv(directory/f'simulation/timeline.csv.ch{ch}')
        dq=dq_intervals(timeline,result['resolved_timings_ck'],ratio,cfg['pim_data_path'])
        ca=ca_intervals(read_csv(directory/f'simulation/rank_ca_events.csv.ch{ch}'),ratio,ch)
        data=dict(dq=dq,global_ca=ca['global_ca'],rank1_ca=ca['rank1_ca'],
                  dq_global_ca=union_intervals(dq+ca['global_ca']))
        for resource,items in data.items():
            busy,idle=interval_accounting(items,0,end)
            resources[resource].append(dict(busy_us=busy*scale,idle_us=(end-busy)*scale,
                utilization_pct=100*busy/end,longest_idle_us=max((b-a for a,b in idle),default=0)*scale))
            all_intervals[resource].append(items)
            raw_idle.extend(dict(resource=resource,channel=ch,begin_us=a*scale,end_us=b*scale,duration_us=(b-a)*scale) for a,b in idle)
    for resource,channels in resources.items():
        merged=union_intervals([e for entries in all_intervals[resource] for e in entries])
        busy,idle=interval_accounting(merged,0,end)
        stats[resource]={key:float(np.mean([r[key] for r in channels])) for key in channels[0]}
        stats[resource].update(all_channels_idle_total_us=(end-busy)*scale,
            all_channels_longest_idle_us=max((b-a for a,b in idle),default=0)*scale)
        raw_idle.extend(dict(resource=resource,channel='all',begin_us=a*scale,end_us=b*scale,duration_us=(b-a)*scale) for a,b in idle)
    stats.update(latency_us=end*scale,external_DQ_bytes=read_json(directory/'summary.json')['external_DQ_bytes'])
    return stats,all_intervals,raw_idle,scale


def utilization(entries,end_us,scale,bin_us=.5):
    edges=np.arange(0,end_us+bin_us,bin_us);edges[-1]=end_us
    values=np.zeros(len(edges)-1)
    for channel in entries:
        for item in channel:
            a,b=item['begin']*scale,item['end']*scale
            first=max(0,int(a/bin_us));last=min(len(values)-1,int(b/bin_us))
            for i in range(first,last+1):values[i]+=max(0,min(b,edges[i+1])-max(a,edges[i]))
    return edges,100*values/(16*np.diff(edges))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--before',required=True);ap.add_argument('--after',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--skip-audit',action='store_true');a=ap.parse_args()
    roots=dict(before=Path(a.before),after=Path(a.after));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    R.setup(R.HERE.parent/'pi05-bgk-gpuq/backend')
    from audit_commands import audit_pair
    from audit_native import audit_run
    metrics={};series={};all_idle=[];audit={}
    for case in CASES:
        if not a.skip_audit:
            audit[case]={'before':audit_run(roots['before']/case),'after':audit_pair(roots['before']/case,roots['after']/case)}
            R.dump(out/'audit.json',audit)
        for mode,root in roots.items():
            stats,items,idle,scale=analyze(root/case)
            metrics[f'{mode}/{case}']=stats
            if case=='serial_layer_control':series[mode]=(items,idle,scale)
            all_idle.extend(dict(mode=mode,case=case,**r) for r in idle)
        print('analyzed',case,flush=True)
    with gzip.open(out/'idle_windows.csv.gz','wt',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(all_idle[0]));w.writeheader();w.writerows(all_idle)
    R.dump(out/'metrics.json',metrics)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5.8),layout='constrained')
    y=np.arange(len(CASES));h=.35
    for mode,delta in (('before',-h/2),('after',h/2)):
        vals=[metrics[f'{mode}/{c}'] for c in CASES]
        for ax,value in zip(axes,([v['latency_us'] for v in vals],[v['dq']['utilization_pct'] for v in vals],
                                 [v['global_ca']['idle_us'] for v in vals])):
            bars=ax.barh(y+delta,value,h,label=mode.title(),color=COLORS[mode]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
            ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            ax.margins(x=.19)
    axes[0].set_title('Native latency (us)');axes[1].set_title('DQ utilization (%)');axes[2].set_title('Global CA idle (us/channel)')
    axes[0].legend(loc='lower right');fig.suptitle('Same layout, local RoPE, score DRAM consolidation and PIM O projection\nModeled memory/compute work; GPU boundary arithmetic excluded',fontsize=13)
    fig.savefig(out/'comparison_overview.png',dpi=180);plt.close(fig)
    max_end=max(metrics[f'{m}/serial_layer_control']['latency_us'] for m in roots)
    for resource,filename,title in (('dq','dq_utilization_idle.png','External DQ utilization and continuous idle'),
                                     ('global_ca','ca_idle.png','Global CA utilization and continuous idle'),
                                     ('rank1_ca','rank_ca_idle.png','Rank 1 CA utilization and continuous idle')):
        fig,axes=plt.subplots(4,1,figsize=(15,9),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.4,1,1.4,1]})
        for i,mode in enumerate(roots):
            items,idle,scale=series[mode];stats=metrics[f'{mode}/serial_layer_control'];end=stats['latency_us']
            edges,val=utilization(items[resource],end,scale)
            ax=axes[2*i];ax.stairs(val,edges,fill=True,color=COLORS[mode],alpha=.9)
            gaps=[r for r in idle if r['resource']==resource and r['channel']=='all' and r['duration_us']>=.2]
            for r in gaps:ax.axvspan(r['begin_us'],r['end_us'],color='#efb54a',alpha=.3)
            s=stats[resource]
            ax.set_title(f"{mode.title()}: {end:.3f} us; utilization {s['utilization_pct']:.2f}%; mean idle {s['idle_us']:.3f} us/channel; longest simultaneous idle {s['all_channels_longest_idle_us']:.3f} us",loc='left',fontsize=10)
            ax.set_ylabel('Busy (%)');ax.set_ylim(0,102);ax.grid(axis='y',alpha=.15)
            ax=axes[2*i+1]
            for ch in range(16):
                spans=[(r['begin_us'],r['duration_us']) for r in idle if r['resource']==resource and r['channel']==ch and r['duration_us']>=.2]
                ax.broken_barh(spans,(ch-.4,.8),facecolors='#dfa331')
            ax.set_yticks([0,5,10,15]);ax.set_ylim(-.7,15.7);ax.set_ylabel('Channel');ax.axvline(end,color=COLORS[mode],lw=1)
            ax.set_title('Maximal continuous idle windows >= 0.2 us (all smaller gaps remain in totals/CSV)',loc='left',fontsize=9)
        axes[-1].set_xlim(0,max_end);axes[-1].set_xlabel('Time from continuous native run start (us)')
        fig.suptitle(title+'\n0.5 us utilization bins; yellow shading = all 16 channels idle',fontsize=14)
        fig.savefig(out/filename,dpi=180);plt.close(fig)
    # Bind the delivered tables/plots to every input and to the implemented source.
    hashes={}
    for mode,root in roots.items():
        for path in sorted(root.rglob('*')):
            if path.is_file():hashes[f'{mode}/{path.relative_to(root).as_posix()}']=R.digest(path)
    R.dump(out/'raw_sha256.json',hashes)


if __name__=='__main__':main()

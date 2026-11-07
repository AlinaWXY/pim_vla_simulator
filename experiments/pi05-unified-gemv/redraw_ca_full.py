"""Full-layer shared CA occupancy and exact long idle windows from native logs."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tarfile
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from redraw_command_figures import (
    BLUE, DARK_GREEN, GRAY, GREEN, MUTED, MODES, NAMES,
    digest, header, load, phase_bands, save, style,
)


def occupancy_bins(entry, width=1.0):
    """Integrate the exact idle union; normalize the final partial bin by its width."""
    duration=entry['summary']['latency_us']
    edges=[i*width for i in range(math.ceil(duration/width))]+[duration]
    widths=[b-a for a,b in zip(edges,edges[1:])]
    idle=[0.0]*len(widths)
    for a,b in entry['global_ca_idle']:
        for i in range(int(a/width),min(len(idle),math.ceil(b/width))):
            idle[i]+=max(0.0,min(b,edges[i+1])-max(a,edges[i]))
    busy=[w-t for w,t in zip(widths,idle)]
    assert all(-1e-9<=b<=w+1e-9 for b,w in zip(busy,widths))
    assert abs(sum(busy)-entry['summary']['global_ca']['busy_us'])<1e-8
    return dict(edges_us=edges,busy_us=busy,occupancy_pct=[100*b/w for b,w in zip(busy,widths)])


def full_figure(data,out):
    fig=plt.figure(figsize=(14,10))
    header(fig,'共享 CA 整层时间轴：从输入到最后结果导出',
        '覆盖约 210 μs；上下使用相同时间轴与纵轴。这里只统计控制器 → rank 的共享 CA。')
    before,after=(data[m]['summary']['global_ca'] for m in MODES)
    fig.text(.075,.844,
        f"全层平均占用：{before['utilization_pct']:.2f}% → {after['utilization_pct']:.2f}%"
        f"     总空闲：{before['idle_us']:.2f} → {after['idle_us']:.2f} μs"
        f"     最长一段：{before['longest_idle_us']:.3f} → {after['longest_idle_us']:.3f} μs",
        fontsize=13,weight='bold')
    fig.text(.075,.79,'① 蓝色越高，这 1 μs 内发命令占用的时间越多',fontsize=16,weight='bold',color=BLUE)
    maximum=max(d['summary']['latency_us'] for d in data.values())
    ticks=list(range(0,211,30));evidence={}
    for mode,bottom in (('before',.61),('after',.42)):
        d=data[mode];bins=occupancy_bins(d);evidence[mode]=dict(bins=bins)
        assert max(bins['occupancy_pct'])<=50, 'Increase the shared y-axis limit to avoid clipping'
        band=fig.add_axes([.12,bottom+.13,.83,.025])
        phase_bands(band,d['stages'],0,1)
        band.set_xlim(0,maximum);band.set_ylim(0,1);band.axis('off')
        ax=fig.add_axes([.12,bottom,.83,.12])
        ax.stairs(bins['occupancy_pct'],bins['edges_us'],fill=True,color=BLUE,alpha=.85,lw=.5)
        ax.set_xlim(0,maximum);ax.set_ylim(0,50)
        ax.set_yticks([0,25,50],['0%','25%','50%'])
        ax.set_ylabel(NAMES[mode],rotation=0,labelpad=25,fontsize=13,weight='bold',va='center')
        ax.set_xticks(ticks);ax.grid(axis='y',color='#dce3eb',lw=.6);ax.set_axisbelow(True)
        for start,_ in d['stages'][1:]:ax.axvline(start,color='#9baabd',ls=':',lw=.8)
        if mode=='before':ax.tick_params(axis='x',labelbottom=False)
        else:ax.set_xlabel('层内真实时间 / μs',labelpad=7)
    fig.text(.075,.314,'② 绿色是一整段完全空闲（仅显示 ≥ 1 μs 的区间）',fontsize=16,weight='bold',color=DARK_GREEN)
    ax=fig.add_axes([.12,.17,.83,.105])
    labels=[]
    for mode,y in (('before',1),('after',0)):
        d=data[mode];duration=d['summary']['latency_us']
        windows=[(a,b) for a,b in d['global_ca_idle'] if b-a>=1]
        evidence[mode]['long_idle_windows_us']=windows
        evidence[mode]['long_idle_count']=len(windows)
        evidence[mode]['long_idle_total_us']=sum(b-a for a,b in windows)
        labels.append(f'{NAMES[mode]}\n{len(windows)} 段')
        ax.add_patch(Rectangle((0,y-.21),duration,.42,facecolor=GRAY,lw=0))
        for a,b in windows:
            ax.add_patch(Rectangle((a,y-.21),b-a,.42,facecolor=GREEN,edgecolor='white',lw=.8))
        for start,_ in d['stages'][1:]:ax.vlines(start,y-.25,y+.25,color='#9baabd',ls=':',lw=.8)
    ax.set_xlim(0,maximum);ax.set_ylim(-.4,1.4);ax.set_yticks([1,0],labels,fontsize=11)
    ax.set_xticks(ticks);ax.set_xlabel('层内真实时间 / μs',labelpad=7)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.legend(handles=[Patch(facecolor=GREEN,label='连续完全空闲 ≥ 1 μs'),
        Patch(facecolor=GRAY,label='含命令或短于 1 μs 的空隙，并非整段忙碌')],
        loc='lower left',bbox_to_anchor=(.11,.064),ncol=2,frameon=False,fontsize=11)
    fig.text(.075,.042,'绿色小段之间仍有命令，不能连成一段；白线仅标出窗口边界。共享 CA 空闲时，rank 内部可以继续计算。',fontsize=11)
    fig.text(.075,.014,'图示 channel 0（本次16个通道忙闲相同）。上图逐1 μs积分，末尾不足1 μs按实际长度归一化；下图使用精确起止时间。',fontsize=10,color=MUTED)
    save(fig,out,'ca_full_readable')
    return evidence


def main():
    ap=argparse.ArgumentParser();source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path);source.add_argument('--archive',type=Path)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--font',type=Path);args=ap.parse_args()
    with ExitStack() as stack:
        data=args.data
        if args.archive:
            data=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='pim-ca-full-')))
            wanted={f'final/{m}/serial_layer_control/{n}' for m in MODES for n in
                ('manifest.json','simulation/barriers.csv','simulation/timeline.csv.ch0','simulation/rank_ca_events.csv.ch0')}
            with tarfile.open(args.archive,'r|gz') as tar:
                for member in tar:
                    if member.name in wanted:
                        dest=data/Path(member.name).relative_to('final');dest.parent.mkdir(parents=True,exist_ok=True)
                        dest.write_bytes(tar.extractfile(member).read())
        args.output.mkdir(parents=True,exist_ok=True);style(args.font)
        values,hashes=load(data,args.reference);series=full_figure(values,args.output)
        evidence=dict(native_rerun=False,source_sha256=hashes,metrics_sha256=digest(args.reference/'metrics.json'),
            renderer_sha256=digest(__file__),helper_sha256=digest(Path(__file__).with_name('redraw_command_figures.py')),
            plotted_channel=0,channel_schedule_equal=True,bin_width_us=1.0,idle_threshold_us=1.0,series=series)
        (args.output/'ca-full-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print('Native hashes and bin-integrated CA busy totals verified; generated full-layer PNG/SVG.')
        for mode,d in series.items():print(f"{mode}: {d['long_idle_count']} idle windows >= 1 us; total {d['long_idle_total_us']:.6f} us")


if __name__=='__main__':main()

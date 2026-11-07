"""Readable Chinese bus figures from immutable native logs; no simulation rerun.

Overview bars are explicitly totals, not chronological intervals. DQ uses
maximal idle windows, while CA uses event ticks with exact gap annotations.
The event tick linewidth never purports to represent command occupancy.
"""
import argparse
from contextlib import ExitStack
from collections import Counter
import hashlib,json,math,tarfile,tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch,Rectangle
from matplotlib.lines import Line2D
from channel_analysis import read_json,read_csv,dq_intervals,interval_accounting
from dq_ca_windows import ca_intervals

INK='#172b4d';MUTED='#526579';BLUE='#3866d6';GREEN='#b9e6ce';DARK_GREEN='#16794c'
GRAY='#e9edf2';ORANGE='#d77413';STAGES=('QKV / RoPE / QKᵀ','PV / O projection','FFN Gate/Up','FFN Down')
MODES=('before','after');NAMES={'before':'调整前','after':'调整后'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(data,reference):
    metrics=read_json(reference/'metrics.json');binding=read_json(reference/'raw_sha256.json')
    result={};hashes={}
    for mode in MODES:
        base=data/mode/'serial_layer_control';sim=base/'simulation'
        paths=[base/'manifest.json',sim/'barriers.csv',sim/'timeline.csv.ch0',sim/'rank_ca_events.csv.ch0']
        for p in paths:
            key=p.relative_to(data).as_posix();hashes[key]=digest(p)
            assert hashes[key]==binding[key],f'Native input hash changed: {key}'
        r=read_json(reference/'metadata'/mode/'serial_layer_control/result.json')
        ratio=r['config']['frontend_ratio'];scale=r['tCK_ns']/ratio/1000;end=r['memory_cycles']*ratio
        dq=dq_intervals(read_csv(paths[2]),r['resolved_timings_ck'],ratio,r['config']['pim_data_path'])
        ca=ca_intervals(read_csv(paths[3]),ratio,0)['global_ca']
        summary=metrics[f'{mode}/serial_layer_control']
        assert len(dq)*32*16==summary['external_DQ_bytes']
        entry=dict(scale=scale,end=end,summary=summary)
        for resource,items in (('dq',dq),('global_ca',ca)):
            busy,idle=interval_accounting(items,0,end)
            assert abs(busy*scale-summary[resource]['busy_us'])<1e-9
            assert abs(max(b-a for a,b in idle)*scale-summary[resource]['longest_idle_us'])<1e-9
            # Previous all-channel union and mean occupancy agree, so this
            # representative channel has the same busy/idle schedule as all 16.
            assert abs(summary[resource]['idle_us']-summary[resource]['all_channels_idle_total_us'])<1e-9
            entry[resource]=[dict(x,begin=x['begin']*scale,end=x['end']*scale) for x in items]
            entry[resource+'_idle']=[(a*scale,b*scale) for a,b in idle]
        manifest=read_json(paths[0]);markers={int(x['marker']):int(x['frontend_tick'])*scale for x in read_csv(paths[1])}
        starts=[markers[s['start']] for s in manifest['segments'] if 'gpu_input_stage' in s['name']]
        assert len(starts)==4
        starts[0]=0;entry['stages']=list(zip(starts,starts[1:]+[end*scale]))
        result[mode]=entry
    return result,hashes


def style(font):
    candidates=[font] if font else [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    chosen=next((p for p in candidates if p and p.is_file()),None)
    if not chosen:raise RuntimeError('Supply a Chinese-capable font with --font')
    font_manager.fontManager.addfont(str(chosen))
    plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=chosen).get_name(),'DejaVu Sans'],
        'font.size':12,'axes.unicode_minus':False,'text.color':INK,'axes.labelcolor':MUTED,
        'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':'#cbd4df','savefig.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False})


def save(fig,out,name):
    for suffix in ('png','svg'):
        path=out/f'{name}.{suffix}';fig.savefig(path,dpi=160)
        if suffix=='svg':
            # Matplotlib leaves spaces at path-line ends; newline separation
            # preserves SVG coordinates while keeping generated diffs clean.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)


def header(fig,title,subtitle):
    fig.text(.055,.95,title,fontsize=23,weight='bold',va='top')
    fig.text(.055,.889,subtitle,fontsize=12,color=MUTED,va='top')


def overview(data,out):
    fig=plt.figure(figsize=(14,8.3))
    header(fig,'先看结论：DQ 几乎不变，CA 连续空闲变长','同一层、同一 layout、本地 RoPE；以下均为每个 channel 的时间。')
    for col,(resource,title) in enumerate((('dq','DQ｜搬运数据'),('global_ca','共享 CA｜发送命令'))):
        x=.10+.48*col;ax=fig.add_axes([x,.56,.35,.19]);gapax=fig.add_axes([x,.17,.35,.20])
        vals=[data[m]['summary'][resource] for m in MODES]
        fig.text(x,.817,title,fontsize=18,weight='bold')
        fig.text(x,.775,f"忙碌 {vals[0]['busy_us']:.3f} → {vals[1]['busy_us']:.3f} μs",color=BLUE,fontsize=13)
        for y,(mode,v) in zip((1,0),zip(MODES,vals)):
            ax.barh(y,v['busy_us'],.48,color=BLUE)
            ax.barh(y,v['idle_us'],.48,left=v['busy_us'],color=GREEN)
            ax.text(v['busy_us']+v['idle_us']/2,y,f"空闲 {v['idle_us']:.3f} μs",ha='center',va='center',fontsize=12)
            gap=v['longest_idle_us'];gapax.barh(y,gap,.48,color=GREEN)
            gapax.text(gap+.015*max(w['longest_idle_us'] for w in vals),y,f'{gap:.3f} μs',va='center',color=DARK_GREEN,fontsize=13,weight='bold')
        for panel in (ax,gapax):
            panel.set_yticks([1,0],[NAMES[m] for m in MODES]);panel.set_ylim(-.65,1.65)
            panel.spines['left'].set_visible(False);panel.tick_params(axis='y',length=0);panel.grid(axis='x',alpha=.18);panel.set_axisbelow(True)
        ax.set_xlim(0,215);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('总量拆分 / μs（不表示时间顺序）',fontsize=11)
        gapax.set_xlim(0,max(v['longest_idle_us'] for v in vals)*1.3)
        gapax.set_title('最长一次连续空闲',loc='left',fontsize=15,pad=16)
        gapax.set_xlabel('连续空闲时长 / μs',fontsize=11)
        fig.text(x,.458,f"利用率：{vals[0]['utilization_pct']:.3f}% → {vals[1]['utilization_pct']:.3f}%",fontsize=13)
    fig.legend(handles=[Patch(facecolor=BLUE,label='忙碌'),Patch(facecolor=GREEN,label='空闲')],loc='lower left',bbox_to_anchor=(.05,.025),ncol=2,frameon=False)
    before,after=(data[m]['summary'] for m in MODES)
    assert before['external_DQ_bytes']==after['external_DQ_bytes']
    fig.text(.38,.055,f"层耗时 {before['latency_us']:.3f} → {after['latency_us']:.3f} μs；DQ 流量均为 {before['external_DQ_bytes']/1024:g} KiB（全部通道）。",fontsize=11,color=MUTED)
    save(fig,out,'bus_summary')


def phase_bands(ax,stages,y,height=.2):
    for (start,end),label in zip(stages,STAGES):
        ax.add_patch(Rectangle((start,y),end-start,height,facecolor='#f2f5f9',edgecolor='white',lw=2))
        ax.text((start+end)/2,y+height/2,label,ha='center',va='center',fontsize=11,color=MUTED)


def dq_figure(data,out):
    fig=plt.figure(figsize=(14,7.4));header(fig,'DQ：六段长空闲窗口，调整前后基本相同','DQ 负责搬数据。绿色标出整段没有任何 DQ 传输的窗口；这段时间 rank 内仍可计算。')
    before,after=(data[m]['summary']['dq'] for m in MODES)
    fig.text(.08,.79,f"忙碌时间：{before['busy_us']:.3f} → {after['busy_us']:.3f} μs",fontsize=17,color=BLUE,weight='bold')
    fig.text(.56,.79,f"最长连续空闲：{before['longest_idle_us']:.3f} → {after['longest_idle_us']:.3f} μs",fontsize=16,color=DARK_GREEN,weight='bold')
    ax=fig.add_axes([.10,.30,.85,.39]);maximum=max(v['summary']['latency_us'] for v in data.values())
    long_windows={}
    for y,mode in ((1.4,'before'),(0,'after')):
        d=data[mode];duration=d['summary']['latency_us'];phase_bands(ax,d['stages'],y+.57)
        ax.add_patch(Rectangle((0,y),duration,.48,facecolor=GRAY,lw=0))
        windows=[(a,b) for a,b in d['dq_idle'] if b-a>=1]
        assert len(windows)==6;long_windows[mode]=windows
        for a,b in windows:
            ax.add_patch(Rectangle((a,y),b-a,.48,facecolor=GREEN,lw=0))
            ax.text((a+b)/2,y+.24,f'{b-a:.2f}',ha='center',va='center',color=DARK_GREEN,fontsize=12,weight='bold')
        a,b=max(windows,key=lambda p:p[1]-p[0]);ax.add_patch(Rectangle((a,y),b-a,.48,fill=False,edgecolor=DARK_GREEN,lw=2))
    ax.set_yticks([1.64,.24],['调整前','调整后']);ax.set_ylim(-.15,2.30);ax.set_xlim(0,maximum+1)
    ax.set_xticks(range(0,211,25));ax.set_xlabel('层内真实时间 / μs',labelpad=12)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Patch(facecolor=GREEN,label='连续空闲 ≥ 1 μs（格内数字为时长 / μs）'),
        Patch(facecolor=GRAY,label='存在数据传输，也可能含更短的空隙')],loc='upper left',bbox_to_anchor=(0,-.27),frameon=False,ncol=2,fontsize=11)
    fig.text(.08,.08,'读法：从左到右看算子顺序；绿色越长，DQ 越久不用搬数据。最后一段 Down 最长，前后都约 47.6 μs。',fontsize=12)
    fig.text(.08,.035,'图示 channel 0；本次16个通道忙闲时间相同。短于1 μs的间隙不单独画出，仍计入总空闲统计。',fontsize=10,color=MUTED)
    save(fig,out,'dq_readable');return long_windows


def ca_figure(data,out):
    # Select an actual maximum idle gap in the common steady Down region,
    # rather than shifting or splicing separate portions of the two runs.
    eligible=[(a,b) for a,b in data['after']['global_ca_idle'] if a>180 and b<205]
    a,b=max(eligible,key=lambda p:p[1]-p[0]);left=math.floor(a*10)/10-.2;right=math.ceil(b*10)/10+.3
    fig=plt.figure(figsize=(14,8.4));header(fig,'共享 CA：空闲原本被小命令切碎，现在能连成一长段','CA 负责发命令。下面放大同一段 Down 时间，直接看命令在哪里把空闲打断。')
    before,after=(data[m]['summary']['global_ca'] for m in MODES)
    fig.text(.08,.796,f"总忙碌：{before['busy_us']:.3f} → {after['busy_us']:.3f} μs",fontsize=17,color=BLUE,weight='bold')
    fig.text(.56,.796,f"全层最长连续空闲：{before['longest_idle_us']:.3f} → {after['longest_idle_us']:.3f} μs",fontsize=16,color=DARK_GREEN,weight='bold')
    locator=fig.add_axes([.10,.686,.85,.048]);phase_bands(locator,data['after']['stages'],0,.9)
    locator.add_patch(Rectangle((left,0),right-left,.10,facecolor=ORANGE,lw=0))
    locator.set_xlim(0,211);locator.set_ylim(-.1,1);locator.axis('off')
    fig.text(.10,.65,f'橙色标记位置放大：层内 {left:.1f}–{right:.1f} μs（前后使用同一个时间范围）',fontsize=11,color=MUTED)
    ax=fig.add_axes([.10,.31,.85,.29]);counts={}
    for y,mode in ((1,'before'),(0,'after')):
        d=data[mode]
        for start,end in d['global_ca_idle']:
            l,r=max(start,left),min(end,right)
            if r>l:ax.add_patch(Rectangle((l,y-.19),r-l,.38,facecolor=GREEN,lw=0))
        events=[e for e in d['global_ca'] if left<=e['begin']<right]
        counts[mode]=dict(Counter(e['command'] for e in events))
        for e in events:
            control=e['command'].startswith('PIM_') or e['command'] in ('MAC_RUN','ROPE_RUN')
            ax.vlines(e['begin'],y-.24,y+.24,color=BLUE if control else ORANGE,lw=1.15)
    ax.annotate('',xy=(a,.35),xytext=(b,.35),arrowprops=dict(arrowstyle='|-|',color=DARK_GREEN,lw=1.6))
    ax.text((a+b)/2,.41,f'这一整段没有命令：{b-a:.3f} μs',ha='center',color=DARK_GREEN,fontsize=14,weight='bold')
    ax.set_xlim(left,right);ax.set_ylim(-.35,1.48)
    ax.set_yticks([1,0],[f'{NAMES[m]}\n{sum(counts[m].values())} 次发射' for m in MODES],fontsize=11)
    ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.set_xlabel('层内真实时间 / μs',labelpad=10)
    ax.legend(handles=[Line2D([],[],color=BLUE,marker='|',ls='None',markersize=15,label='计算控制命令'),
        Line2D([],[],color=ORANGE,marker='|',ls='None',markersize=15,label='维护命令（刷新、预充电）'),
        Patch(facecolor=GREEN,label='没有命令的空闲区间')],frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0,-.31),fontsize=11)
    assert not any(e['begin']<right and e['end']>left for mode in MODES for e in data[mode]['dq'])
    fig.text(.08,.135,'同一段时间里，DQ 前后都完全空闲；rank 内部仍在执行 Down。共享 CA 空闲不等于计算停止。',fontsize=12)
    fig.text(.08,.077,'蓝/橙刻线表示一次发射事件，线宽不表示命令持续时间；精确占用时长来自原始日志。',fontsize=10,color=MUTED)
    fig.text(.08,.035,'这里只画“控制器 → rank”的共享 CA；rank 内部 CA 的占用单独统计。图示 channel 0。',fontsize=10,color=MUTED)
    save(fig,out,'ca_readable');return dict(begin_us=left,end_us=right,long_gap_us=[a,b],command_counts=counts)


def main():
    ap=argparse.ArgumentParser();source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive',type=Path);source.add_argument('--data',type=Path)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--font',type=Path);args=ap.parse_args()
    with ExitStack() as stack:
        data=args.data
        if args.archive:
            data=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='pim-bus-figures-')))
            wanted={f'final/{m}/serial_layer_control/{n}' for m in MODES for n in
                ('manifest.json','simulation/barriers.csv','simulation/timeline.csv.ch0','simulation/rank_ca_events.csv.ch0')}
            with tarfile.open(args.archive,'r|gz') as tar:
                for member in tar:
                    if member.name in wanted:
                        dest=data/Path(member.name).relative_to('final');dest.parent.mkdir(parents=True,exist_ok=True)
                        dest.write_bytes(tar.extractfile(member).read())
        args.output.mkdir(parents=True,exist_ok=True);style(args.font)
        values,hashes=load(data,args.reference)
        overview(values,args.output);windows=dq_figure(values,args.output);zoom=ca_figure(values,args.output)
        evidence=dict(native_rerun=False,source_sha256=hashes,metrics_sha256=digest(args.reference/'metrics.json'),
            renderer_sha256=digest(__file__),dq_long_windows_us=windows,ca_zoom=zoom,
            plotted_channel=0,channel_schedule_equal=True,original_metrics_verified=True)
        (args.output/'plot-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(zoom,indent=2));print('Verified original hashes, busy totals and maximum gaps; generated three PNG/SVG pairs.')


if __name__=='__main__':main()

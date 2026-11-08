"""Independent checks and two actual DQ/CA trace excerpts for corrected case e."""
import argparse,bisect,csv,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
parser.add_argument('--data',type=Path)
args=parser.parse_args();ROOT=args.root.resolve()
OUT=args.data.resolve() if args.data else ROOT/'reference/evaluation-1108/e-dq-insertion-audit'
NATIVE=ROOT/'reference/evaluation-1108/e-dq-insertion-audit/native'
S=json.loads((OUT/'summary.json').read_text());CK=S['CK_ticks'];U=S['ticks_per_us']
TYPE={0:'RD',1:'WR',2:'WRGB',3:'RDRES',4:'control'}
CASES={41:('chain',0),46:('pv',0),48:('pv',390690),63:('gate_up_gpu',0),67:('down_gpu_input',0)}
def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
with gzip.open(ROOT/'reference/evaluation-1108/uniform-transactions/native-reservations.json.gz','rt') as f:native=json.load(f)
for case in {x[0] for x in CASES.values()}:
    p=NATIVE/case/'simulation/timeline.csv.ch0'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==native['token_local_before/'+case]['source_sha256']['simulation/timeline.csv.ch0']

traces={};gaps=[]
for m in S['trace_manifest']:
    rows=[{k:int(v) for k,v in x.items()} for x in readcsv(OUT/f'block-{m["block"]}-ch0-trace.csv')]
    traces[m['block']]=rows
    dq=sorted((x for x in rows if x['dq_a']>=0),key=lambda x:x['dq_a'])
    assert all(a['dq_b']<=b['dq_a'] for a,b in zip(dq,dq[1:]))
    pim=[x for x in dq if not x['gpu']];gpu=[x for x in dq if x['gpu']];starts=[x['dq_a'] for x in gpu]
    for a,b in zip(pim,pim[1:]):
        lo=a['native_issue']+a['dq_b']-a['issue'];hi=b['native_issue']+b['dq_a']-b['issue']
        selected=gpu[bisect.bisect_left(starts,a['dq_b']):bisect.bisect_left(starts,b['dq_a'])]
        if not selected:continue
        gaps.append(dict(block=m['block'],stage=m['name'],original_gap_CK=(hi-lo)/CK,
            actual_gap_CK=(b['dq_a']-a['dq_b'])/CK,previous=a,next=b,gpu=selected))
short=next(x for x in gaps if x['block']==63 and x['original_gap_CK']==2 and
           x['previous']['type']==x['next']['type']==2 and len(x['gpu'])==1 and
           x['gpu'][0]['type']==0 and x['actual_gap_CK']==7)
long=next(x for x in gaps if x['block']==63 and x['original_gap_CK']==x['actual_gap_CK']==513 and len(x['gpu'])>=20)
examples=[]
for x in (short,long):
    case,source_begin=CASES[x['block']];left=(source_begin-1)*469
    a=x['previous'];b=x['next']
    lo=a['native_issue']+a['dq_b']-a['issue']+left;hi=b['native_issue']+b['dq_a']-b['issue']+left
    commands=Counter(r['command'] for r in readcsv(NATIVE/case/'simulation/timeline.csv.ch0')
                     if lo<=(int(r['clock'])-1)*CK<hi)
    origin=x['gpu'][0]['issue'] if x is short else a['dq_b']
    x=dict(x,origin_ticks=origin,original_gap_commands=dict(commands))
    x['relative_CK']=dict(previous_DQ=[(a['dq_a']-origin)/CK,(a['dq_b']-origin)/CK],
        original_next_DQ=[(a['dq_b']-origin)/CK+x['original_gap_CK'],(a['dq_b']-origin)/CK+x['original_gap_CK']+2],
        actual_next_DQ=[(b['dq_a']-origin)/CK,(b['dq_b']-origin)/CK],
        GPU=[dict(command=TYPE[g['type']],issue=(g['issue']-origin)/CK,
                  DQ=[(g['dq_a']-origin)/CK,(g['dq_b']-origin)/CK]) for g in x['gpu']])
    examples.append(x)
(OUT/'verified-examples.json').write_text(json.dumps(examples,indent=2)+'\n')

available_fonts={font.name for font in font_manager.fontManager.ttflist}
font=next((name for name in ('Microsoft YaHei','Noto Sans CJK SC','SimHei') if name in available_fonts),'DejaVu Sans')
plt.rcParams.update({'font.family':font,
                    'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[1,1]})
colors={'pim':'#2e7d62','read':'#2575c4','write':'#e68b32','idle':'#f4dd87'}
for i,(ax,x) in enumerate(zip(axes,examples)):
    origin=x['origin_ticks'];a=x['previous'];b=x['next'];left=(a['dq_b']-origin)/CK
    right=(b['dq_a']-origin)/CK;native_right=left+x['original_gap_CK']
    limits=(-2,32) if i==0 else (-25,540)
    ax.broken_barh([(left,x['original_gap_CK'])],(2.7,.55),facecolors=colors['idle'],alpha=.65)
    anchor_shift=a['issue']-a['native_issue']
    for record in traces[x['block']]:
        if record['dq_a']<0:continue
        actual=(record['dq_a']-origin)/CK
        color=colors['pim'] if not record['gpu'] else colors['write' if record['type']==1 else 'read']
        if limits[0]-2<actual<limits[1]:
            ax.broken_barh([(actual,2)],(1.6,.55),facecolors=color)
        if not record['gpu']:
            nominal=(record['native_issue']+record['dq_a']-record['issue']+anchor_shift-origin)/CK
            if limits[0]-2<nominal<limits[1]:ax.broken_barh([(nominal,2)],(2.7,.55),facecolors=colors['pim'])
    for g in x['gpu']:
        color=colors['write' if g['type']==1 else 'read']
        issue=(g['issue']-origin)/CK
        ax.broken_barh([(issue,1)],(.5,.55),facecolors=color)
    ax.set_yticks([2.975,1.875,.775],['原 PIM DQ（对齐左侧）','实际共享 DQ','选中 GPU 请求的 CA 发射'])
    ax.set_ylim(.2,3.8);ax.set_xlabel('相对时间 / CK');ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right','left']].set_visible(False)
    if i==0:
        ax.set_xlim(-2,32);ax.set_xticks(range(0,33,2))
        ax.text(21,3.35,'原空档 2 CK',ha='center')
        ax.annotate('',xy=(right,2.38),xytext=(left,2.38),arrowprops=dict(arrowstyle='<->',color='#444'))
        ax.text((left+right)/2,2.43,'实际空档 7 CK',ha='center',fontsize=9)
        ax.annotate('RD@0 → 数据 [23,25)',xy=(24,1.85),xytext=(5,1.25),
                    arrowprops=dict(arrowstyle='->',color=colors['read']),color=colors['read'])
        ax.set_title('A  广播碎片：后续 WRGB 从 22 CK 推到 27 CK，才放下 GPU 数据',loc='left',weight='bold',pad=16)
    else:
        ax.set_xlim(-25,540)
        ax.text(256.5,3.35,'原空档 513 CK；实际仍为 513 CK',ha='center')
        ax.set_title(f'B  长计算窗口：{x["original_gap_commands"]["MACAB"]} 条 MACAB 期间插入 {len(x["gpu"])} 个 GPU burst，DQ 窗口未拉长',loc='left',weight='bold',pad=16)
fig.legend(handles=[Patch(color=colors['pim'],label='PIM DQ'),Patch(color=colors['read'],label='GPU RD / 返回'),
    Patch(color=colors['write'],label='GPU WR / 数据'),Patch(color=colors['idle'],label='原始 DQ 空档')],
    loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.56,.005))
fig.suptitle('e：修复后仲裁轨迹中的两种 GPU 插入位置（Gate/Up，block 63，channel 0）',y=.995,weight='bold')
fig.tight_layout(rect=(0,.055,1,.96),h_pad=2.8)
fig.savefig(OUT/'e-dq-insertion-examples.png',dpi=180)
fig.savefig(OUT/'e-dq-insertion-examples.svg')

# Human-readable tables retain the exact full-run counts and denominators.
gap_rows=readcsv(OUT/'gap-length-and-endpoints.csv');two=Counter();stage=defaultdict(Counter)
for r in gap_rows:
    n=int(r['reads'])+int(r['writes']);stage[r['stage']]['inside']+=n
    if r['kind']=='internal' and float(r['original_gap_CK'])==2:
        two[r['previous']+'→'+r['next']]+=n;stage[r['stage']]['in_2ck']+=n
    if r['kind']=='internal' and float(r['original_gap_CK'])>256:stage[r['stage']]['in_gt256ck']+=n
analysis=dict(full_run_job='1974893',original_replay_identical=True,
    post_PIM_percent=100*S['GPU_requests_DQ_after_PIM']/S['GPU_requests_entire_launched_prefix'],
    fits_original_guarded_DQ_geometry_percent=100*sum(x['fits_original_DQ_geometry_requests'] for x in S['categories'].values())/S['GPU_requests_DQ_fully_inside_PIM'],
    two_ck_by_endpoints=dict(two),by_stage=dict(stage),
    example_short={k:v for k,v in examples[0].items() if k not in ('previous','next','gpu')},
    example_long=dict(block=long['block'],original_gap_CK=513,actual_gap_CK=513,GPU_bursts=len(long['gpu']),
        original_gap_commands=examples[1]['original_gap_commands']),
    raw_native_ch0_hashes_verified=4,sampled_actual_DQ_nonoverlap_verified=True)
(OUT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(analysis,ensure_ascii=False,indent=2))

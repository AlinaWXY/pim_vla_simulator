"""Replay corrected e unchanged and classify every GPU burst across all 16 channels."""
import argparse,csv,ctypes as C,gzip,inspect,json,sys
from collections import defaultdict
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--reference-run',type=Path)
args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
sys.path.insert(0,str(root/'experiments/pi05-unified-gemv'))
from sequential_transactions import SequentialEngine,U,CK
import run_sequential_transactions as production

GAP_FIELDS='kind length previous next windows used_windows reads writes fit_reads fit_writes native_ticks actual_ticks extension_ticks'.split()
TRACE_FIELDS='issue native_issue shift ca dq_a dq_b rank type gpu cursor'.split()
class Gap(C.Structure):_fields_=[(k,C.c_int64) for k in GAP_FIELDS]
class Trace(C.Structure):_fields_=[(k,C.c_int64) for k in TRACE_FIELDS]
TYPES={-1:'boundary',0:'RD',1:'WR',2:'WRGB',3:'RDRES',4:'control'}
KINDS={0:'internal',1:'prefix',2:'tail',3:'reordered'}
def read(p):return json.loads(p.read_text())
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def category(g):
    if g['kind']:return KINDS[g['kind']]
    n=g['length']/CK
    if n<2:return 'internal_lt_2ck'
    if n==2:return 'internal_2ck'
    if n<=16:return 'internal_3_16ck'
    if n<=64:return 'internal_17_64ck'
    if n<=256:return 'internal_65_256ck'
    return 'internal_gt_256ck'

ref=root/'reference/evaluation-1108'
reference_run=args.reference_run or ref/'native-timing-fix/with-rows'
with (reference_run/'e-PIM-and-AE-waits.csv').open() as f:expected_stages=list(csv.DictReader(f))
expected_by_block={int(s['block']):s for s in expected_stages}
expected_summary=next(x for x in read(reference_run/'summary.json')['evaluation'] if x['case']=='e')

class DiagnosticEngine(SequentialEngine):
    def __init__(self):
        super().__init__(out/'libinsertiondiag.so')
        self.lib.audit_enable.argtypes=[C.c_int,C.c_int64,C.c_int]
        self.lib.audit_finalize.argtypes=[C.c_int]
        self.lib.audit_gaps.argtypes=[C.c_int,C.POINTER(Gap),C.c_int]
        self.lib.audit_extra.argtypes=[C.c_int,C.POINTER(C.c_int64)]
        self.lib.audit_trace.argtypes=[C.c_int,C.POINTER(Trace),C.c_int]
        self.block=None;self.traced_names=set();self.trace_written=set();self.cache={}
        self.aggregates=defaultdict(lambda:defaultdict(int));self.by_stage=defaultdict(lambda:defaultdict(int))
        self.extras=[0]*18;self.stage_extras=defaultdict(lambda:[0]*18)
        self.stage_records=[];self.trace_manifest=[]
    def start_block(self,bi,b):
        self.block=bi;self.name=b['name'];self.enable_trace=self.name not in self.traced_names
        self.traced_names.add(self.name)
    def begin(self,program,channel):
        c=super().begin(program,channel)
        if self.block is not None:
            end=round(float(expected_by_block[self.block]['PIM_with_GPU_us'])*U)
            self.check(self.lib.audit_enable(c,end,int(self.enable_trace and channel==0)))
        return c
    def stats(self,context):
        result=super().stats(context)
        if self.block is None:return result
        if context not in self.cache:
            self.check(self.lib.audit_finalize(context))
            n=self.check(self.lib.audit_gaps(context,None,0));data=(Gap*n)()
            self.check(self.lib.audit_gaps(context,data,n))
            gaps=[{k:getattr(x,k) for k in GAP_FIELDS} for x in data]
            extra=(C.c_int64*18)();self.check(self.lib.audit_extra(context,extra))
            self.cache[context]=(gaps,list(extra))
        gaps,extra=self.cache[context]
        counts=sum(g['reads']+g['writes'] for g in gaps)+sum(extra[:4])
        assert counts==result['reads']+result['writes'],(self.block,counts,result)
        assert counts==sum(extra[6:12])
        totals=defaultdict(int)
        for g in gaps:
            key=(self.name,g['kind'],g['length'],g['previous'],g['next'])
            for k in GAP_FIELDS[4:]:self.aggregates[key][k]+=g[k]
            c=category(g)
            for k in GAP_FIELDS[4:]:self.by_stage[self.name,c][k]+=g[k];totals[c+'_'+k]+=g[k]
        for i,n in enumerate(extra):self.extras[i]+=n;self.stage_extras[self.name][i]+=n
        self.stage_records.append(dict(block=self.block,name=self.name,**totals))
        if self.enable_trace and self.block not in self.trace_written:
            n=self.check(self.lib.audit_trace(context,None,0));data=(Trace*n)()
            self.check(self.lib.audit_trace(context,data,n))
            assert n>0
            rows=[{k:getattr(x,k) for k in TRACE_FIELDS} for x in data]
            csvout(out/f'block-{self.block}-ch0-trace.csv',rows)
            self.trace_manifest.append(dict(block=self.block,name=self.name,channel=0,records=n,
                PIM_end_ticks=round(float(expected_by_block[self.block]['PIM_with_GPU_us'])*U)))
            self.trace_written.add(self.block)
        return result

natural=read(ref/'online-arbitration-preparation/natural-ae-kernel-ledger.json')['units']
units=read(ref/'gpu/uniform-memory-demand.json')['units'];warm=read(ref/'gpu/warm-summary.json')['baseline']['phase_ms']
measured=warm['vit']['mean']+warm['vlm']['mean'];scale=measured*1000/sum(u['us'] for u in units)
for u in units:u['duration']=round(u['us']*scale*U)
units[-1]['duration']+=round(measured*1000*U)-sum(u['duration'] for u in units)
engine=DiagnosticEngine();empty=engine.add(0,[],16);empty_program=engine.program(0,0,empty,[])
plans=[];cache={}
for u in units:
    handles=[];durations=[]
    for ch in range(16):
        r=u['read_sectors']//16+int(ch<u['read_sectors']%16);w=u['write_sectors']//16+int(ch<u['write_sectors']%16)
        key=(u['duration'],r,w)
        if key not in cache:
            initial,_=engine.stream_plan(*key);h,answer=engine.reference(empty_program,initial)
            cache[key]=h,answer.finish
        h,d=cache[key];handles.append(h);durations.append(d)
    u['duration']=max(durations);plans.append(handles)
print('REFERENCE_READY',flush=True)
with gzip.open(ref/'uniform-transactions/native-reservations.json.gz','rt') as f:typed=json.load(f)
with gzip.open(ref/'online-arbitration-preparation/native-programs.json.gz','rt') as f:programs=json.load(f)
seq=production.templates(ref,typed,programs,engine,'token_local_before',natural)
# One observer callback before each block creates diagnostic state only.
source=inspect.getsource(production.simulate)
assert source.count('origin=now;local=0')==1
source=source.replace('origin=now;local=0','engine.start_block(bi,b)\n        origin=now;local=0')
namespace=dict(production.__dict__);exec(source,namespace)
result=namespace['simulate'](seq,units,plans,engine)
assert len(result['stages'])==len(expected_stages)
assert len(engine.stage_records)==16*len(expected_stages)
for row,old in zip(result['stages'],expected_stages):
    for k in ('PIM_with_GPU_us','PIM_GPU_interference_us','AE_GPU_wait_us'):
        assert abs(row[k]-float(old[k]))<1e-8,(row,old)
for k,v in result.items():
    if k in ('stages','placements'):continue
    assert abs(v-expected_summary[k])<1e-8,(k,v,expected_summary[k])
with (reference_run/'e-VLM-placement.csv').open() as f:old_placements=list(csv.DictReader(f))
assert len(old_placements)==len(result['placements'])
for row,old in zip(result['placements'],old_placements):
    for k,v in row.items():
        if v=='':assert old[k]==''
        else:assert abs(v-float(old[k]))<1e-7,(k,v,old[k])
rows=[]
for (name,kind,length,previous,next_),values in sorted(engine.aggregates.items()):
    rows.append(dict(stage=name,kind=KINDS[kind],original_gap_CK=length/CK,
        previous=TYPES[previous],next=TYPES[next_],**values))
csvout(out/'gap-length-and-endpoints.csv',rows)
rows=[];categories=defaultdict(lambda:defaultdict(int))
for (name,cat),values in sorted(engine.by_stage.items()):
    rows.append(dict(stage=name,category=cat,**values))
    for k,v in values.items():categories[cat][k]+=v
csvout(out/'insertion-by-stage.csv',rows)
total=sum(v['reads']+v['writes'] for v in categories.values())
assert total+sum(engine.extras[:4])==result['GPU_reads']+result['GPU_writes']
for cat,v in categories.items():
    v['requests']=v['reads']+v['writes'];v['percent_of_in_PIM_requests']=100*v['requests']/total
    v['fits_original_DQ_geometry_requests']=v['fit_reads']+v['fit_writes']
summary=dict(case='e',variant='token_local_before',CK_ticks=CK,ticks_per_us=U,burst_CK=2,
    scope='All 900 PIM blocks, 18 layers x 10 steps, all 16 channels, corrected timing model.',
    categories=dict(categories),GPU_requests_entire_launched_prefix=result['GPU_reads']+result['GPU_writes'],
    GPU_requests_DQ_fully_inside_PIM=total,GPU_requests_DQ_after_PIM=sum(engine.extras[:2]),
    GPU_requests_DQ_straddling_PIM_end=sum(engine.extras[2:4]),extra_counters=engine.extras,
    extra_counter_schema=['after_reads','after_writes','straddle_reads','straddle_writes',
        'straddle_inside_ticks','straddle_outside_ticks','all_CA_DQidle_reads','all_CA_DQidle_writes',
        'all_CA_PIM_DQbusy_reads','all_CA_PIM_DQbusy_writes','all_CA_GPU_DQbusy_reads','all_CA_GPU_DQbusy_writes',
        'inPIM_CA_DQidle_reads','inPIM_CA_DQidle_writes','inPIM_CA_PIM_DQbusy_reads','inPIM_CA_PIM_DQbusy_writes',
        'inPIM_CA_GPU_DQbusy_reads','inPIM_CA_GPU_DQbusy_writes'],
    stage_extra_counters=dict(engine.stage_extras),trace_manifest=engine.trace_manifest,
    original_results_and_placements_exactly_reproduced=True,
    original_metrics={k:v for k,v in result.items() if k not in ('stages','placements')},
    definitions=dict(category='GPU data burst is located between consecutive actual PIM DQ bursts; label that gap using the two corresponding original native bursts. Prefix/tail separate. PIM boundary is the maximum finish over 16 channels from the verified corrected replay, used only by instrumentation.',
        fit='Keep the left PIM DQ endpoint at its actual shifted time, restore the original gap length, and check whether this GPU burst and existing cross-rank DQ guards fit. This is a DQ geometry check, not proof of zero CA/row/dependency cost.',
        extension='Actual gap length minus original gap length. Counts CA/refresh/other delays as well; not a per-request causal attribution.',
        counts='Requests sum all channels; each is a 32-byte, 2 CK burst. Percentages of in-PIM requests exclude post-PIM kernel tails.'),
    limitations='Observational replay of the existing uniform-arrival sequential-address model; not a hardware trace. No GPU mover kernel/SM or cache residency model was added.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('trace_manifest','stage_extra_counters','original_metrics')},indent=2),flush=True)

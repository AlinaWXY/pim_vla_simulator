"""Read retained Thor measurements and export immutable, ordered GPU work units."""
import gzip,hashlib,json,shutil,sqlite3
from pathlib import Path

ROOT=Path('/fact_data/xinyaowang/pim_vla_xinyao_1108_20261108')
OLD=Path('/fact_data/xinyaowang/pim_vla_ae_corrected_20261105/thor-data')
WARM=OLD/'pi05-openpi-mha8-h16-s10-adarms-20261105'
COLD=OLD/'pi05-openpi-mha8-h16-s10-cold-kernel-20261105'
OUT=ROOT/'reference/evaluation-1108/gpu';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2)+'\n')
source=WARM/'raw/pi05_h16_latency_trace_20261105_01.sqlite.gz'
dbpath=ROOT/'results/retained-gpu.sqlite';dbpath.parent.mkdir(exist_ok=True)
with gzip.open(source,'rb') as src,dbpath.open('wb') as dst:shutil.copyfileobj(src,dst)
db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
strings={r['id']:r['value'] for r in db.execute('select * from StringIds')}
ranges=[dict(r,label=r['text'] or strings.get(r['textId'],'')) for r in db.execute('select * from NVTX_EVENTS') if r['end']]
runtime={r['correlationId']:dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
kernels=[dict(r,kernel=strings[r['demangledName']]) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
copies=[]
for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
    if table in tables:copies.extend(dict(r,kernel=table,kind='memory') for r in db.execute('select * from '+table))
requests=[r for r in ranges if r['label'].startswith('request/trace/')]
steps=[r for r in ranges if r['label']=='step/0']
prefix=[]
for request in requests:
    step=next(r for r in steps if r['globalTid']==request['globalTid'] and request['start']<r['start']<request['end'])
    items=[]
    for k in sorted([dict(k,kind='kernel') for k in kernels]+copies,key=lambda x:x['start']):
        launch=runtime.get(k['correlationId'])
        if launch and launch['globalTid']==request['globalTid'] and request['start']<=launch['start']<step['start']:
            items.append(dict(ordinal=len(items),kind=k['kind'],kernel=k['kernel'],us=(k['end']-k['start'])/1000,
                              gpu_begin_ns=k['start'],gpu_end_ns=k['end'],launch_ns=launch['start'],
                              launch_us=(launch['end']-launch['start'])/1000))
    prefix.append(dict(request=request['label'],kernels=items,kernel_sum_us=sum(k['us'] for k in items)))
assert len(prefix)==5 and all(len(p['kernels'])==len(prefix[0]['kernels']) for p in prefix)
units=[]
for i in range(len(prefix[0]['kernels'])):
    ks=[p['kernels'][i] for p in prefix]
    assert len({k['kernel'] for k in ks})==1
    units.append(dict(ordinal=i,kind=ks[0]['kind'],kernel=ks[0]['kernel'],us=sum(k['us'] for k in ks)/len(ks),
                      launch_us=sum(k['launch_us'] for k in ks)/len(ks),
                      repeats_us=[k['us'] for k in ks]))
summary=json.loads((WARM/'analysis/summary.json').read_text())
for src,name in ((WARM/'analysis/summary.json','warm-summary.json'),
                 (COLD/'analysis_verified/kernel_ledger.json','cold-ae-kernel-ledger.json'),
                 (COLD/'analysis_verified/operators.csv','cold-ae-operators.csv')):
    shutil.copyfile(src,OUT/name)
dump('prefix-kernels.json',dict(scope='Ordered GPU kernels and memory operations before AE step0; vision + language + prefix/setup',
     source=str(source),source_sha256=sha(source),requests=prefix,ordered_mean_units=units,
     kernel_sum_us=sum(k['us'] for k in units),gpu_rerun=False))
dump('sources.json',{str(p):sha(p) for p in (source,WARM/'analysis/summary.json',COLD/'analysis_verified/kernel_ledger.json')})
print(json.dumps(dict(prefix_kernels=len(units),prefix_kernel_sum_us=sum(k['us'] for k in units),
                     baseline=summary['baseline']['phase_ms'])),flush=True)

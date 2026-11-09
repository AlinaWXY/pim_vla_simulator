"""Compare completed single-layer outputs; no simulation or runner imports."""
from pathlib import Path
import gc,hashlib,json,math,sys
repo=Path('/fact_data/xinyaowang/pim_vla_sim2_20261109')
job=int(sys.argv[1]);new=repo/f'results/down-row-single-layer-{job}'
old=repo/'results/rank-refresh-single-layer-1980912'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-6),(a,b)
old_summary=read(old/'experiment/summary.json');new_summary=read(new/'experiment/summary.json')
rows=[]
for old_row,new_row in zip(old_summary['results'],new_summary['results']):
    variant=new_row['variant'];assert old_row['variant']==variant
    paths=[x/'experiment/native'/variant/'repeat-0' for x in (old,new)]
    raw=[read(p/'raw.json') for p in paths];prov=[read(p/'provenance.json') for p in paths]
    assert prov[0]['binary_sha256']==prov[1]['binary_sha256']
    assert prov[0]['runtime_libraries']['libramulator.so']['sha256']==prov[1]['runtime_libraries']['libramulator.so']['sha256']
    for p,pr in zip(paths,prov):assert sha(p/'raw.json')==pr['raw_sha256']
    assert raw[0]['refresh_policy']==raw[1]['refresh_policy']==dict(rank0='REFpb',rank1='REFab')
    before={s['name']:s for s in raw[0]['stages']};after={s['name']:s for s in raw[1]['stages']}
    assert before.keys()==after.keys()
    for name in before:
        if name!='down':assert before[name]==after[name],(variant,name)
    for key in ('GPU_us','GPU_measured_us','pim_read_bytes','pim_write_bytes'):
        close(old_row[key],new_row[key])
    difference=after['down']['duration_us']-before['down']['duration_us']
    close(new_row['PIM_us']-old_row['PIM_us'],difference)
    close(new_row['total_us']-old_row['total_us'],difference)
    rows.append(dict(variant=variant,old_down_us=before['down']['duration_us'],new_down_us=after['down']['duration_us'],down_change_us=difference,down_reduction_percent=-100*difference/before['down']['duration_us'],old_PIM_us=old_row['PIM_us'],new_PIM_us=new_row['PIM_us'],old_total_us=old_row['total_us'],new_total_us=new_row['total_us'],non_down_stage_timestamps_identical=True,GPU_and_DQ_unchanged=True,old_raw_sha256=sha(paths[0]/'raw.json'),new_raw_sha256=sha(paths[1]/'raw.json')))
result=dict(status='passed',baseline_job=1980912,new_job=job,native_binary_and_library_identical=True,refresh_policy=dict(rank0='REFpb',rank1='REFab'),results=rows)
initial=repo/'results/down-row-single-layer-1980945'
if job!=1980945:
    diagnostic=[]
    for row in rows:
        variant=row['variant']
        a=read(initial/'frozen/programs'/f'{variant}.json');b=read(new/'frozen/programs'/f'{variant}.json')
        for left,right in zip(a['stages'],b['stages']):
            if left['name']!='down' or variant.endswith('before'):assert left==right
            else:
                for old_ops,new_ops in zip(left['channels'],right['channels']):
                    assert [o for o in old_ops if o['op']!='WR']==[o for o in new_ops if o['op']!='WR']
                    def payloads(ops):
                        return sorted(json.dumps({k:v for k,v in o.items() if k!='id'},sort_keys=True) for o in ops if o['op']=='WR')
                    assert payloads(old_ops)==payloads(new_ops)
        first=read(initial/'experiment/native'/variant/'repeat-0/raw.json')
        first_prov=read(initial/'experiment/native'/variant/'repeat-0/provenance.json')
        assert first_prov['source_commit']=='7f7b6af9d5aefba0f31d5cc42904b6c7a508126f'
        assert sha(initial/'experiment/native'/variant/'repeat-0/raw.json')==first_prov['raw_sha256']
        first_down=next(s['duration_us'] for s in first['stages'] if s['name']=='down')
        diagnostic.append(dict(variant=variant,initial_colocation_down_us=first_down,final_down_us=row['new_down_us'],write_order_gain_us=first_down-row['new_down_us'],only_input_WR_order_changed=True))
        del a,b;gc.collect()
    result['scatter_order_diagnostic']=dict(job=1980945,source_commit='7f7b6af9d5aefba0f31d5cc42904b6c7a508126f',results=diagnostic)
target=new/'validation/down-layout-comparison.json'
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

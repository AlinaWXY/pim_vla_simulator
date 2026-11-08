"""Verify publication hashes and reconcile reproduced arbitration results."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def compare(actual,expected,where='root'):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),where
        for key in expected:compare(actual[key],expected[key],where+'/'+str(key))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),where
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,where+'/'+str(i))
    elif isinstance(expected,float):assert math.isclose(actual,expected,rel_tol=0,abs_tol=1e-8),(where,actual,expected)
    else:assert actual==expected,(where,actual,expected)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--results',type=Path);p.add_argument('--audit',type=Path)
    a=p.parse_args();root=a.root.resolve();ref=root/'reference/evaluation-1108';checks={}
    manifest=root/'reference/arbitrator-publication/SHA256SUMS.json'
    if manifest.exists():
        hashes=read(manifest)
        for name,expected in hashes.items():
            path=(root/name).resolve();assert path.is_relative_to(root),name
            assert path.is_file(),name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
        checks['publication_files_verified']=len(hashes)
    frozen=read(ref/'native-timing-fix/with-rows/summary.json')
    for row in frozen['evaluation']:
        if row['case']=='b':continue
        expected=row['AE_solo_ms']+row['PIM_interference_ms']+row['AE_GPU_wait_ms']+row['VLM_remaining_model_ms']
        assert abs(expected-row['raw_model_total_ms'])<1e-8
        assert abs(expected-frozen['VLM_reference_bias_ms']-row['total_ms'])<1e-8
    checks['frozen_latency_rows_reconciled']=5
    if a.results:
        new=read(a.results/'summary.json');indexed={r['case']:r for r in frozen['evaluation']}
        for row in new['evaluation']:
            compare(row,indexed[row['case']],'case/'+row['case'])
            if row['case']=='b':continue
            for suffix in ('PIM-and-AE-waits','VLM-placement'):
                filename=f'{row["case"]}-{suffix}.csv'
                def rows(path):
                    with path.open(newline='') as stream:return list(csv.DictReader(stream))
                compare(rows(a.results/filename),rows(ref/'native-timing-fix/with-rows'/filename),filename)
        for key in ('VLM_retained_ms','VLM_model_reference_ms','VLM_reference_bias_ms'):
            compare(new[key],frozen[key],key)
        checks['reproduced_cases']=[r['case'] for r in new['evaluation']]
    audit=read(ref/'e-dq-insertion-audit/summary.json')
    total=audit['GPU_requests_DQ_fully_inside_PIM']+audit['GPU_requests_DQ_after_PIM']+audit['GPU_requests_DQ_straddling_PIM_end']
    assert total==audit['GPU_requests_entire_launched_prefix']==312053120
    assert sum(x['requests'] for x in audit['categories'].values())==audit['GPU_requests_DQ_fully_inside_PIM']
    assert audit['categories']['internal_2ck']['fits_original_DQ_geometry_requests']==0
    if a.audit:
        actual=read(a.audit/'summary.json')
        for key in ('categories','extra_counters','stage_extra_counters','original_metrics',
                    'GPU_requests_DQ_fully_inside_PIM','GPU_requests_DQ_after_PIM','GPU_requests_DQ_straddling_PIM_end'):
            compare(actual[key],audit[key],'audit/'+key)
        checks['reproduced_e_audit_matches']=True
    checks['e_DQ_requests_reconciled']=total
    for name in ('README.md','experiments/pi05-unified-gemv/ARBITRATOR.zh-CN.md',
                 'reference/evaluation-1108/e-dq-insertion-audit/REPORT.zh-CN.md'):
        path=root/name
        if not path.exists():continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0]
            assert (path.parent/target).exists(),(name,target)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

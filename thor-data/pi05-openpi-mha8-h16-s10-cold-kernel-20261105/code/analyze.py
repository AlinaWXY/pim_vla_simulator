"""Summarize isolated cold-kernel times; never call their sum native latency."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
from warm_analysis import classify

TIME='gpu__time_duration.sum'
HIT='lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum'
MISS='lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum'
PASSES='profiler__replayer_passes'
MODEL='pi05_h16_cold_model_20261105_01'
PROBES={p:f'pi05_h16_cold_probe_{p}_20261105_01' for p in ('all','none')}


def classify_operation(op):
    name,shapes,roles=op['operator'],op['shapes'],op['roles']
    if roles and roles[0]=='attention' and name in ('aten::matmul','aten::bmm'):
        if len(shapes)<2 or min(len(shapes[0]),len(shapes[1]))<2:
            raise ValueError('Invalid attention matmul shape')
        if shapes[0][-1]==256 and shapes[1][-1]!=256:
            return 'attention_qkt','linear'
        if shapes[1][-1]==256 and shapes[0][-1]!=256:
            return 'attention_sv','linear'
        raise ValueError('Ambiguous attention matmul geometry')
    if roles and roles[0]=='rms_norm' and name=='aten::square':
        return 'rms_pow','nonlinear_core'
    return classify(name,shapes,roles)


def numeric(value):
    result=float(value.replace(',',''))
    if not math.isfinite(result) or result<0:
        raise ValueError('Invalid metric value')
    return result


def stats(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values),p50=statistics.median(values),
        minimum=min(values),maximum=max(values),stdev=statistics.stdev(values) if len(values)>1 else 0)


def read_csv(path):
    text=path.read_text()
    reader=csv.DictReader(io.StringIO(text[text.index('"ID",'):]))
    units=next(reader)
    if units[TIME]!='ns':
        raise ValueError('Timing metric is not ns')
    result=[]
    for row in reader:
        if not row.get('ID'):
            continue
        marker_text=row['Kernel Name']+' '+ ' '.join(value for key,value in row.items() if 'Push/Pop_Range' in key)
        owners={int(x) for x in re.findall(r'op_(\d+)(?=[:/])',marker_text)}
        if len(owners)!=1:
            raise ValueError('Kernel outside a dispatch/probe marker')
        # CUB pushes a nested NVTX range: recover its outer dispatch owner.
        parsed=dict(id=int(row['ID']),op_id=next(iter(owners)),kernel=row['Kernel Name'].split('/',1)[1],
            kernel_ns=numeric(row[TIME]),passes=int(numeric(row[PASSES])),
            grid=list(ast.literal_eval(row['Grid Size'])),block=list(ast.literal_eval(row['Block Size'])))
        for field in (HIT,MISS):
            if field in row:
                parsed[field]=numeric(row[field])
        result.append(parsed)
    if not result or len({r['id'] for r in result})!=len(result):
        raise ValueError('Missing kernels or duplicate IDs')
    return result


def capture(path):
    result=json.loads(path.read_text())
    if result['status']!='complete' or not result['output_bitwise_equal'] or any(r['others'] for r in result['occupancy']):
        raise ValueError('Incomplete, changed-output or occupied run')
    return result


def audit_probe(raw):
    results={}
    for policy,tag in PROBES.items():
        metadata=capture(raw/tag/'capture.json')
        rows=read_csv(raw/tag/'kernels.csv')
        if len(rows)!=4 or metadata['arguments']['cache_policy']!=policy:
            raise ValueError('Wrong probe policy/count')
        for row in rows:
            denominator=row[HIT]+row[MISS]
            row['l2_miss_fraction']=row[MISS]/denominator if denominator else None
            row['miss_requested_bytes']=row[MISS]*32
        results[policy]=dict(input_bytes=metadata['probe']['input_bytes'],l2_bytes=metadata['probe']['l2_bytes'],
            samples=rows,median_miss_sectors=statistics.median(r[MISS] for r in rows),
            median_miss_requested_bytes=statistics.median(r['miss_requested_bytes'] for r in rows),
            pass_counts=dict(Counter(r['passes'] for r in rows)))
    expected=results['all']['input_bytes']
    if not .95<=results['all']['median_miss_requested_bytes']/expected<=1.05:
        raise ValueError('Cold probe miss bytes differ from input footprint')
    if results['none']['median_miss_requested_bytes']>.1*expected:
        raise ValueError('Probe failed to distinguish cache regimes')
    return dict(passed=True,results=results,
        interpretation='L2 read-miss sectors validate this probe cache-policy contrast, not per-model-kernel DRAM bytes.',
        caveat='Probe uses multiple counter passes; its durations are NOT the model latency dataset. No L2 request means hit rate is undefined, not 100%.')


def csv_write(path,fields,rows):
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.root/'raw'
    probe=audit_probe(raw)
    meta=capture(raw/MODEL/'capture.json')
    rows=read_csv(raw/MODEL/'kernels.csv')
    if any(r['passes']!=1 for r in rows):
        raise ValueError('Model timing intended as single-pass; inspect unexpected replay counts')
    operations={r['id']:r for r in meta['operations']}
    combined=[]
    for row in rows:
        op=operations[row['op_id']]
        if op['roles'] and op['roles'][0]=='cached_adarms':
            raise ValueError('Cached AdaRMS dense launched a kernel')
        category,family=classify_operation(op)
        combined.append(dict(row,repeat=op['repeat'],operator=op['operator'],category=category,family=family,shapes=op['shapes'],
            capture_category=op['category'],capture_family=op['family']))
    repetitions=[]
    for rep in range(3):
        selected=[k for k in combined if k['repeat']==rep]
        category=Counter()
        family=Counter()
        for k in selected:
            category[k['category']]+=k['kernel_ns']/1e6
            family[k['family']]+=k['kernel_ns']/1e6
        repetitions.append(dict(repeat=rep,kernels=len(selected),sum_kernel_ms=sum(k['kernel_ns'] for k in selected)/1e6,
            category_ms=dict(category),family_ms=dict(family)))
        for expected in ('attention_qkt','attention_sv'):
            ids={k['op_id'] for k in selected if k['category']==expected}
            if len(ids)!=18:
                raise ValueError('Expected 18 attention matmuls: '+expected)
    def signature(k):
        return (k['kernel'],tuple(k['grid']),tuple(k['block']),k['category'])
    signatures=[[signature(k) for k in combined if k['repeat']==r] for r in range(3)]
    if not signatures[0] or any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('Kernel sequence/configuration differs across repetitions')
    warm=json.loads((args.warm/'analysis/summary.json').read_text())
    warm_capture=json.loads((args.warm/'raw/pi05_h16_latency_baseline_20261105_01/capture.json').read_text())
    if meta['reference_actions']!=warm_capture['reference_actions']:
        raise ValueError('Cold/warm final actions differ')
    native_labels=json.loads((args.warm/'analysis/operator_labels.json').read_text())
    checks=[]
    observed=[k for k in combined if k['repeat']==0]
    if len(observed)==len(native_labels):
        for i,(a,b) in enumerate(zip(observed,native_labels)):
            expected=b['signature']
            if (a['kernel'],a['grid'],a['block'])!=(expected[0],expected[1:4],expected[4:7]):
                checks.append(dict(index=i,cold_kernel=a['kernel'],warm_kernel=expected[0],cold_grid=a['grid'],warm_grid=expected[1:4],cold_block=a['block'],warm_block=expected[4:7]))
    else:
        checks.append(dict(reason='count mismatch',cold=len(observed),warm=len(native_labels)))
    category_keys=sorted({k for r in repetitions for k in r['category_ms']})
    family_keys=sorted({k for r in repetitions for k in r['family_ms']})
    summary=dict(configuration=meta['configuration'],dimensions=meta['dimensions'],environment=meta['environment'],
        cache_probe=probe,per_step_sum_kernel_ms=stats([r['sum_kernel_ms'] for r in repetitions]),
        category_ms={k:stats([r['category_ms'].get(k,0) for r in repetitions]) for k in category_keys},
        family_ms={k:stats([r['family_ms'].get(k,0) for r in repetitions]) for k in family_keys},repetitions=repetitions,
        kernel_count=len(rows),kernels_per_repeat=[r['kernels'] for r in repetitions],pass_counts=dict(Counter(k['passes'] for k in rows)),
        validation=dict(outputs_bitwise_equal=True,cached_dense_hits=meta['cache_audit']['last_request_dense_hits'],
            occupancy_checks=len(meta['occupancy']),other_gpu_pids_observed=False,repetitions_same_kernel_sequence=True,
            warm_native_kernel_count=len(native_labels),warm_signature_mismatches=checks),
        warm_reference=dict(source_commit='de6974fe0095098500191211c7bdaa231c895ab7',normal_api_wall_p50_ms=warm['baseline']['clean_wall_ms']['p50'],
            normal_step_mean_ms=warm['baseline']['step_per_request_mean_ms']['mean'],comparison_warning='Different regimes/tools; no causal cold/warm speedup or native total reconstructed'),
        boundary='Real fifth denoise_step, all 18 layers + action input/output projection. NCU CUDA kernel durations only; not CPU gaps/DMA/Euler/full step latency.',
        classification_notes='Final analysis normalizes dispatch matmul (including 4D operands) to QK/SV and RMS square to RMS core. Raw capture category fields were provisional and remain unchanged.',
        cache_regime='NCU cache-control all before each selected kernel pass; model and AdaRMS cache preloaded, 20 full warmups.',
        source_urls=['https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control',
                     'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay'],
        input_sha256={str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob('*/*')) if p.is_file()})
    total=summary['per_step_sum_kernel_ms']['mean']
    summary['category_share_pct']={k:v['mean']/total*100 for k,v in summary['category_ms'].items()}
    summary['family_share_pct']={k:v['mean']/total*100 for k,v in summary['family_ms'].items()}
    if not math.isclose(sum(summary['category_share_pct'].values()),100,abs_tol=1e-8):
        raise ValueError('Category totals do not reconcile')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'kernel_ledger.json').write_text(json.dumps(combined,indent=2)+'\n')
    csv_write(args.output/'operators.csv',['category','mean_ms','minimum_ms','maximum_ms','share_pct'],
        [dict(category=k,mean_ms=v['mean'],minimum_ms=v['minimum'],maximum_ms=v['maximum'],share_pct=summary['category_share_pct'][k]) for k,v in summary['category_ms'].items()])
    csv_write(args.output/'families.csv',['family','mean_ms','share_pct'],
        [dict(family=k,mean_ms=v['mean'],share_pct=summary['family_share_pct'][k]) for k,v in summary['family_ms'].items()])
    print('Cold sum per step:',summary['per_step_sum_kernel_ms'])
    print('Kernels:',summary['kernels_per_repeat'],'passes:',summary['pass_counts'])
    print('Families:',{k:v['mean'] for k,v in summary['family_ms'].items()})
    print('Warm signature differences:',len(checks))


if __name__=='__main__':
    main()

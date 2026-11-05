"""CPU-only audit: clean timing is never reconstructed from profiler fractions."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics

LINEAR = {'q_projection','k_projection','v_projection','o_projection','ffn_gate','ffn_up','ffn_down','action_in_proj','action_out_proj'}
COPY = {'aten::copy_','aten::_to_copy','aten::clone','aten::contiguous','aten::to'}
SIGNATURE = ('kernel','gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory')


def stats(values):
    x = sorted(values)
    def quantile(p):
        k = (len(x)-1)*p
        i = int(k)
        return x[i] + (x[min(i+1,len(x)-1)]-x[i])*(k-i)
    return dict(n=len(x),mean=statistics.mean(x),p50=quantile(.5),p95=quantile(.95),minimum=min(x),maximum=max(x),
                stdev=statistics.stdev(x) if len(x)>1 else 0)


def classify(operator, shapes, roles):
    role = roles[0] if roles else ''
    if operator in COPY:
        return 'copies_casts', 'data_movement'
    if role in LINEAR and operator in {'aten::mm','aten::matmul','aten::addmm','aten::bmm','aten::linear'}:
        return role, 'linear'
    if operator in ('aten::softmax','aten::_softmax'):
        return 'softmax', 'nonlinear_core'
    if operator in ('aten::gelu','aten::silu'):
        return operator[6:], 'nonlinear_core'
    if role == 'rms_norm':
        if operator in ('aten::mean','aten::sum','aten::rsqrt','aten::pow'):
            return 'rms_'+operator[6:], 'nonlinear_core'
        return 'norm_'+operator[6:], 'elementwise'
    if role == 'rope_generation':
        return 'rope_generation', 'rope'
    if role == 'rope_apply':
        return 'rope_apply', 'rope'
    if role == 'residual_gate':
        return 'residual_gate_'+operator[6:], 'elementwise'
    if operator == 'aten::bmm' and len(shapes)>=2 and len(shapes[0])==len(shapes[1])==3:
        if shapes[0][-1] == 256:
            return 'attention_qkt', 'linear'
        if shapes[1][-1] == 256:
            return 'attention_sv', 'linear'
    if role == 'attention':
        if operator == 'aten::cat':
            return 'kv_concat', 'data_movement'
        return 'attention_'+operator[6:], 'elementwise'
    if role == 'ffn' and operator == 'aten::mul':
        return 'ffn_gate_multiply', 'elementwise'
    return 'setup_'+operator.removeprefix('aten::'), 'setup'


def union(spans):
    result = []
    for a,b in sorted(spans):
        if b<=a:
            continue
        if result and a<=result[-1][1]:
            result[-1][1] = max(b,result[-1][1])
        else:
            result.append([a,b])
    return result


def exclusive(items, field):
    changes = defaultdict(Counter)
    for item in items:
        changes[item['start']][item[field]] += 1
        changes[item['end']][item[field]] -= 1
    previous = min(changes)
    active = Counter()
    totals = Counter()
    for stamp, delta in sorted(changes.items()):
        cats = [key for key,value in active.items() if value>0]
        if cats:
            totals[cats[0] if len(cats)==1 else 'overlap'] += (stamp-previous)/1e6
        active.update(delta)
        previous = stamp
    return dict(totals)


def validated(path):
    data = json.loads(path.read_text())
    if data['status']!='complete' or not data['output_bitwise_equal'] or any(r['others'] for r in data['occupancy']):
        raise ValueError('Incomplete, changed-output, or occupied GPU capture')
    if data['cache_audit']['last_request_dense_hits']!=370:
        raise ValueError('Cache coverage mismatch')
    return data


def summarize_baseline(data):
    blocks = {b['name']:b for b in data['blocks']}
    clean = blocks['clean_before']['samples'] + blocks['clean_after']['samples']
    phases = blocks['phases_before']['samples'] + blocks['phases_after']['samples']
    step = blocks['step_control']['samples']
    result = dict(clean_wall_ms=stats([r['wall_ms'] for r in clean]),
        phase_total_ms=stats([r['gpu_ms'] for r in phases]),
        phase_ms={key:stats([r[key+'_ms'] for r in phases]) for key in ('vit','vlm','ae','other')},
        step_ms=[stats([r['steps_ms'][i] for r in step]) for i in range(10)],
        step_per_request_mean_ms=stats([statistics.mean(r['steps_ms']) for r in step]),
        phase_samples=len(phases),clean_samples=len(clean),step_requests=len(step),
        blocks={key:{f:v for f,v in block.items() if f!='samples'} for key,block in blocks.items()})
    total = result['phase_total_ms']['mean']
    result['phase_share_pct'] = {k:100*v['mean']/total for k,v in result['phase_ms'].items()}
    if not math.isclose(sum(v['mean'] for v in result['phase_ms'].values()),total,abs_tol=1e-5):
        raise ValueError('Coarse phases do not reconcile')
    result['phase_marker_mean_overhead_pct'] = 100*(statistics.mean(r['wall_ms'] for r in phases)/result['clean_wall_ms']['mean']-1)
    result['clean_after_vs_before_p50_pct'] = 100*(blocks['clean_after']['wall_ms']['p50']/blocks['clean_before']['wall_ms']['p50']-1)
    result['step_event_mean_overhead_vs_phase_pct'] = 100*(statistics.mean(r['wall_ms'] for r in step)/statistics.mean(r['wall_ms'] for r in phases)-1)
    return result


def analyze_trace(path, capture, output):
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    strings = {r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
    ranges = [dict(r,label=r['text'] or strings.get(r['textId'],'')) for r in db.execute('SELECT * FROM NVTX_EVENTS')]
    ranges = [r for r in ranges if r['end'] is not None]
    runtime = {r['correlationId']:dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
    requests = [r for r in ranges if r['label'].startswith('request/')]
    steps = [r for r in ranges if r['label'].startswith('step/')]
    def container(rows, launch):
        if launch is None:
            return None
        return next((r for r in rows if r['globalTid']==launch['globalTid'] and r['start']<=launch['start']<r['end']),None)
    def tag(item):
        launch = runtime.get(item['correlationId'])
        request, step = container(requests,launch), container(steps,launch)
        return dict(item,launch=launch,request=request['label'] if request else None,
                    step=int(step['label'][5:]) if step else None,step_range_start=step['start'] if step else None)
    kernels = [tag(dict(r,kernel=strings[r['demangledName']],kind='kernel'))
               for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
    label_request = next(r for r in requests if r['label']=='request/attribution/0')
    label_step = next(r for r in steps if r['label']=='step/4' and label_request['start']<r['start']<label_request['end'])
    label_ranges = [r for r in ranges if label_step['start']<=r['start']<label_step['end']]
    labeled = [k for k in kernels if k['step_range_start']==label_step['start']]
    module_counts = Counter(r['label'][3:] for r in label_ranges if r['label'].startswith('op/'))
    for role in LINEAR-{'action_in_proj','action_out_proj'}:
        if module_counts[role]!=18:
            raise ValueError('Incomplete projection coverage: '+role)
    if module_counts['rms_norm']!=37 or module_counts['cached_adarms']!=37:
        raise ValueError('Incomplete normalization/cache labels')
    labels = []
    for kernel in labeled:
        launch = kernel['launch']
        parents = sorted([r for r in label_ranges if r['globalTid']==launch['globalTid'] and r['start']<=launch['start']<r['end']],key=lambda r:r['end']-r['start'])
        roles = [r['label'][3:] for r in parents if r['label'].startswith('op/')]
        aten = next((r['label'] for r in parents if r['label'].startswith('aten::')),None)
        if aten is None:
            raise ValueError('No ATen launch owner: '+kernel['kernel'])
        operator = aten.split(',')[0]
        match = re.search(r'sizes = (\[.*\]), input_op_ids',aten)
        shapes = ast.literal_eval(match[1]) if match else []
        category, family = classify(operator,shapes,roles)
        if roles and roles[0]=='cached_adarms':
            raise ValueError('Cached AdaRMS dense unexpectedly launched work')
        labels.append(dict(signature=[kernel[k] for k in SIGNATURE],operator=operator,shapes=shapes,roles=roles,category=category,family=family))
    copies = []
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:
            for r in db.execute('SELECT * FROM '+table+' ORDER BY start'):
                copies.append(tag(dict(r,kind='copy_engine',kernel=table,category='copy_engine',family='data_movement',operator=table)))
    block = next(b for b in capture['blocks'] if b['name']=='trace')
    samples, first_items = [], None
    for row in block['samples']:
        request = 'request/trace/'+str(row['iteration'])
        for step in range(10):
            selected = [k for k in kernels if k['request']==request and k['step']==step]
            if [[k[field] for field in SIGNATURE] for k in selected] != [k['signature'] for k in labels]:
                raise ValueError('Kernel sequence or geometry mismatch: '+request+' step '+str(step))
            items = [dict(k,**{f:label[f] for f in ('operator','shapes','roles','category','family')}) for k,label in zip(selected,labels,strict=True)]
            items += [k for k in copies if k['request']==request and k['step']==step]
            active = sum(b-a for a,b in union([(k['start'],k['end']) for k in items]))/1e6
            cat = exclusive(items,'category')
            family = exclusive(items,'family')
            span = row['steps_ms'][step]
            if active > span+.02 or not math.isclose(sum(cat.values()),active,abs_tol=1e-7):
                raise ValueError('Invalid exclusive accounting')
            samples.append(dict(request=row['iteration'],step=step,event_ms=span,active_ms=active,
                gap_ms=span-active,category_ms=cat,family_ms=family,kernel_count=len(selected),copy_count=len(items)-len(selected)))
            if first_items is None:
                first_items = [{k:item.get(k) for k in ('start','end','kernel','operator','shapes','roles','category','family','streamId','kind')} for item in items]
    if len(samples)!=50:
        raise ValueError('Expected 5 requests times 10 steps')
    def clustered(field, key=None):
        # Ten steps share one request; aggregate first, then report n=5 requests.
        return stats([statistics.mean(s[field].get(key,0) if key else s[field] for s in samples if s['request']==r['iteration']) for r in block['samples']])
    cat_keys = sorted({k for s in samples for k in s['category_ms']})
    family_keys = sorted({k for s in samples for k in s['family_ms']})
    result = dict(matched_steps=len(samples),kernels_per_step=len(labels),module_counts=dict(module_counts),
        per_step_event_ms=clustered('event_ms'),per_step_active_ms=clustered('active_ms'),per_step_gap_ms=clustered('gap_ms'),
        category_ms={k:clustered('category_ms',k) for k in cat_keys},family_ms={k:clustered('family_ms',k) for k in family_keys},
        step_index_ms=[stats([s['event_ms'] for s in samples if s['step']==i]) for i in range(10)],
        fifth_step_active_ms=stats([s['active_ms'] for s in samples if s['step']==4]),
        diagnostics=[r[0] for r in db.execute('SELECT text FROM DIAGNOSTIC_EVENT')],
        sample_definition='5 requests; average all 10 real denoising steps per request; 18 layers each',
        boundary='denoise_step includes action input/output projection and 18 layers; excludes Euler and outer setup',
        category_shares_denominator='exclusive GPU activity union, not unprofiled AE elapsed time')
    active = result['per_step_active_ms']['mean']
    result['category_active_share_pct'] = {k:100*v['mean']/active for k,v in result['category_ms'].items()}
    result['family_active_share_pct'] = {k:100*v['mean']/active for k,v in result['family_ms'].items()}
    blocks = {b['name']:b for b in capture['blocks']}
    result['trace_process_blocks'] = {k:{f:b[f] for f in ('wall_ms','phase_ms','steps_ms')} for k,b in blocks.items()}
    for position in ('before','after'):
        result['ae_p50_overhead_vs_'+position+'_pct'] = 100*(blocks['trace']['phase_ms']['ae']['p50']/blocks['trace_control_'+position]['phase_ms']['ae']['p50']-1)
    (output/'operator_labels.json').write_text(json.dumps(labels,indent=2)+'\n')
    (output/'first_step_kernels.json').write_text(json.dumps(first_items,indent=2)+'\n')
    (output/'step_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    with (output/'operators.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category','mean_per_step_ms','p50_request_mean_ms','p95_request_mean_ms','exclusive_active_share_pct'])
        for k,v in result['category_ms'].items():
            writer.writerow([k,v['mean'],v['p50'],v['p95'],result['category_active_share_pct'][k]])
    with (output/'steps.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['request','step_zero_based','event_ms','gpu_active_union_ms','instrumented_gap_ms','kernel_count','copy_count'])
        for s in samples:
            writer.writerow([s[k] for k in ('request','step','event_ms','active_ms','gap_ms','kernel_count','copy_count')])
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    bp = args.raw/'pi05_h16_latency_baseline_20261105_01/capture.json'
    tp = args.raw/'pi05_h16_latency_trace_20261105_01/capture.json'
    sp = args.raw/'pi05_h16_latency_trace_20261105_01.sqlite'
    base, trace = validated(bp), validated(tp)
    if base['configuration']!=trace['configuration'] or base['reference_actions']!=trace['reference_actions']:
        raise ValueError('Baseline/trace model or output differs')
    data = dict(configuration=base['configuration'],dimensions=base['dimensions'],environment=base['environment'],
        baseline=summarize_baseline(base),fine=analyze_trace(sp,trace,args.output),
        provenance={str(p.relative_to(args.raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (bp,tp,sp)},
        validation=dict(bitwise_equal=True,other_gpu_pids_detected=False,
            baseline_occupancy_checks=len(base['occupancy']),trace_occupancy_checks=len(trace['occupancy']),
            cache_dense_hits_per_request=370),
        cache_regime='warm steady-state, natural cache; no flush, no kernel replay, no DRAM-subtraction',
        limitations=['Synthetic physical MHA8 in BOTH VLM and AE, not a trained MHA policy',
            'Profiler time and fractions are not unbiased normal-inference time',
            'One-step kernel times include memory/cache access; they are not compute-only or PIM-offload latency',
            'Discrete PID checks cannot exclude every transient background SoC activity'])
    data['fine']['ae_mean_overhead_vs_independent_baseline_pct'] = 100*(data['fine']['trace_process_blocks']['trace']['phase_ms']['ae']['mean']/data['baseline']['phase_ms']['ae']['mean']-1)
    (args.output/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    with (args.output/'coarse.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['phase','mean_ms','p50_ms','p95_ms','same_pass_mean_share_pct'])
        for key,val in data['baseline']['phase_ms'].items():
            writer.writerow([key,val['mean'],val['p50'],val['p95'],data['baseline']['phase_share_pct'][key]])
    print(json.dumps({k:data[k] for k in ('configuration','validation')},indent=2))
    print('Clean wall:',data['baseline']['clean_wall_ms'])
    print('Phases:',{k:v['mean'] for k,v in data['baseline']['phase_ms'].items()})
    print('Fine event/active/gaps:',[data['fine'][k]['mean'] for k in ('per_step_event_ms','per_step_active_ms','per_step_gap_ms')])
    print('Fine families:',{k:v['mean'] for k,v in data['fine']['family_ms'].items()})


if __name__=='__main__':
    main()

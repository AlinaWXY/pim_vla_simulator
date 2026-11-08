"""Aggregate segment service times without adding overlapping benchmark suites."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES=('eager_reused','eager_entry_cold','graph_reused','graph_entry_cold')
RUN='pi05_h16_segments_20261108_02'
PROBE='pi05_h16_segment_probe_20261108_01'
INTERFACES='pi05_h16_interfaces_20261108_01'


def stats(values):
    x=sorted(values)
    def q(p):
        k=(len(x)-1)*p;i=int(k)
        return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(k-i)
    return dict(n=len(x),mean=statistics.mean(x),p50=q(.5),p95=q(.95),minimum=min(x),maximum=max(x))


def aggregate(cases):
    groups=defaultdict(list)
    for c in cases:
        if not c['bitwise_equal'] or c['internal_evictions']!=0 or c['evictions_per_cold_sample']!=1:
            raise ValueError('Invalid segment controls')
        for mode in MODES:
            samples=c['measurements'][mode]['samples_ms']
            if len(samples)!=20 or any(not math.isfinite(v) or v<0 for v in samples):
                raise ValueError('Incomplete or invalid samples')
            if not math.isclose(statistics.mean(samples),c['measurements'][mode]['summary_ms']['mean'],abs_tol=1e-10):
                raise ValueError('Recorded summary mismatch')
        groups[c['suite'],c['family']].append(c)
    rows=[]
    for (suite,family),members in sorted(groups.items()):
        row=dict(suite=suite,family=family,occurrences_per_step=len(members),case_ids=[c['id'] for c in members])
        for mode in MODES:
            means=[statistics.mean(c['measurements'][mode]['samples_ms']) for c in members]
            row[mode]=dict(sum_case_means_ms=sum(means),mean_occurrence_us=1000*statistics.mean(means),
                minimum_layer_mean_us=1000*min(means),maximum_layer_mean_us=1000*max(means))
        rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args()
    paths=[ns.root/'raw'/tag/'capture.json' for tag in (RUN,PROBE,INTERFACES)]
    data,probe,extra=[json.loads(p.read_text()) for p in paths]
    if any(d['status']!='complete' or any(x['others'] for x in d['occupancy']) for d in (data,probe,extra)):
        raise ValueError('Incomplete or occupied capture')
    if not probe['cache_probe']['passed'] or len(data['cases'])!=386:
        raise ValueError('Probe or case coverage failed')
    if data['cache_audit']['last_request_dense_hits']!=370:raise ValueError('Cache coverage differs')
    if len(extra['cases'])!=72 or extra['reference_actions']!=data['reference_actions'] or extra['cache_audit']['last_request_dense_hits']!=370:
        raise ValueError('Interface supplement coverage/output differs')
    primary=aggregate(data['cases'])
    groups={r['suite']+'/'+r['family']:r['occurrences_per_step'] for r in primary}
    if groups!=data['case_audit']['counts']:raise ValueError('Coverage changed')
    cases=data['cases']+extra['cases']
    rows=aggregate(cases)
    ids=[c['id'] for c in cases]
    if len(set(ids))!=458:raise ValueError('Duplicate cases')
    for d in (data,probe,extra):
        for name,expected in d['files_sha256'].items():
            if hashlib.sha256((ns.root/'code'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Captured source changed: '+name)
    blocks={b['name']:b['samples'] for b in data['coarse']}
    clean=blocks['clean_before']+blocks['clean_after']
    phase=blocks['phases_before']+blocks['phases_after']
    coarse=dict(clean_wall_ms=stats([r['wall_ms'] for r in clean]),
        phase_total_ms=stats([r['gpu_ms'] for r in phase]),
        phase_ms={key:stats([r[key+'_ms'] for r in phase]) for key in ('vit','vlm','ae','other')},
        clean_p50_drift_pct=100*(statistics.median(r['wall_ms'] for r in blocks['clean_after'])/statistics.median(r['wall_ms'] for r in blocks['clean_before'])-1))
    coarse['phase_share_pct']={k:100*v['mean']/coarse['phase_total_ms']['mean'] for k,v in coarse['phase_ms'].items()}
    if not math.isclose(sum(v['mean'] for v in coarse['phase_ms'].values()),coarse['phase_total_ms']['mean'],abs_tol=1e-6):
        raise ValueError('Coarse phase partition does not close')
    lookup={(r['suite'],r['family']):r for r in rows}
    roi={}
    for mode in MODES:
        roi[mode]=dict(qk_apply_18_layers_ms=lookup['native_blocks','rope_qk'][mode]['sum_case_means_ms'],
            position_generation_once_ms=lookup['native_blocks','rope_generation'][mode]['sum_case_means_ms'],
            fp32_interface_qk_apply_18_layers_ms=lookup['pim_interface_candidates','pim_rope_fp32_bf16'][mode]['sum_case_means_ms'])
        roi[mode]['native_rope_combined_sum_ms']=roi[mode]['qk_apply_18_layers_ms']+roi[mode]['position_generation_once_ms']
    summary=dict(client_date='2026-11-08',configuration=data['configuration'],dimensions=data['dimensions'],
        environment=data['environment'],cache_probe=probe['cache_probe'],empty_graph_event_ms=probe['empty_graph_event_ms'],
        coarse=coarse,aggregates=rows,rope=roi,
        validation=dict(measured_cases=458,primary_cases=386,interface_supplement_cases=72,samples_per_case_per_mode=20,modes=4,total_timed_segment_runs=458*20*4,
            all_outputs_bitwise_equal=True,source_hashes_verified=True,case_coverage_verified=True,
            other_gpu_pids_detected=False,occupancy_checks=len(data['occupancy'])+len(extra['occupancy']),
            failed_attempt_excluded='pi05_h16_segments_20261108_01: model parameters created inside inference_mode; no valid measurements'),
        suites=dict(native_blocks='Logical native operator blocks, isolated from the full pipeline; do not call their sum E2E.',
            handoff_composites='Continuous GPU combinations; replace overlapping native block rows, never add both.',
            pim_interface_candidates='Logical FP32 input variants; not physical PIM output/packing/coherency validation.'),
        pim_reference=dict(branch='xinyao_1104',commit=extra['pim_contract_commit'],
            files=['README.md','experiments/pi05-unified-gemv/TOKEN-PV.md'],
            note='RoPE is on PIM in that reference. GPU-RoPE cases are alternatives, not an additional cost for PIM-RoPE. Partial-PV sum and logical P parity packing are supplied as GPU candidates.'),
        modes=dict(eager_reused='Native function, external CUDA events; includes host submission gaps; fixed-address reuse control.',
            eager_entry_cold='One 256MiB .ca eviction before segment, external CUDA events; includes host submission gaps.',
            graph_reused='Same function captured, external timing events INSIDE CUDA graph; fixed-address reuse control.',
            graph_entry_cold='One entry eviction, timing INSIDE graph; no internal evictions, no CPU graph-launch/dispatch cost.'),
        limitations=['Eviction is an empirically qualified cache-state proxy, not an architectural cache invalidate or real PIM handoff.',
            'Entry eviction also affects GPU-resident constants/weights; their actual residency is not selectively reproduced.',
            'GPU is otherwise idle; concurrent ViT/VLM competition and preemption are not measured.',
            'No guarantee that output has drained to LPDDR or become PIM-visible at the recorded GPU event.',
            'No DMA, host/PIM dispatch, physical bank unpack/pack, or inter-device coherency cost included.',
            'Graph is a separately validated submission mode; original full-model baseline remains native eager.',
            'Each run captures the deterministic fifth step; twenty fixed-buffer repetitions are not twenty distinct model requests.',
            'A hybrid using graph submission needs an equivalently optimized GPU baseline before attributing any E2E speedup solely to PIM.',
            'Summed case means are not a measured denoising step or full AE end-to-end time.'],
        input_sha256={str(p.relative_to(ns.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_urls=['https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs',
            'https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html'])
    ns.output.mkdir(parents=True,exist_ok=False)
    (ns.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for suite in summary['suites']:
        with (ns.output/(suite+'.csv')).open('w',newline='') as f:
            fields=['family','occurrences_per_step']+[m+'_sum_case_means_ms' for m in MODES]+[m+'_mean_occurrence_us' for m in MODES]
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for row in rows:
                if row['suite']!=suite:continue
                flat={k:row[k] for k in ('family','occurrences_per_step')}
                flat.update({m+'_sum_case_means_ms':row[m]['sum_case_means_ms'] for m in MODES})
                flat.update({m+'_mean_occurrence_us':row[m]['mean_occurrence_us'] for m in MODES})
                writer.writerow(flat)
    with (ns.output/'cases.csv').open('w',newline='') as f:
        fields=['id','suite','family','layer','note']+[m+'_mean_ms' for m in MODES]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for c in cases:
            w.writerow({**{k:c[k] for k in ('id','suite','family','layer','note')},
                **{m+'_mean_ms':c['measurements'][m]['summary_ms']['mean'] for m in MODES}})
    print('RoPE:',json.dumps(roi,indent=2))
    print('Coarse:',{k:v['mean'] for k,v in coarse['phase_ms'].items()})
    print('Clean wall:',coarse['clean_wall_ms'])


if __name__=='__main__':main()

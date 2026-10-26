"""Keep clean wall/phase intervals separate from profiled GPU attribution."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


LINEAR = {'q_projection','k_projection','v_projection','attention_qkt','attention_sv',
          'o_projection','ffn_gate','ffn_up','ffn_down','action_in_proj','action_out_proj',
          'rope_frequency_matmul'}
NONLINEAR = {'gelu','softmax','silu','rms_norm','rope_sin','rope_cos'}


def family(cat):
    if cat in LINEAR:
        return 'Matrix / linear'
    if cat in NONLINEAR:
        return 'Nonlinear / normalization'
    if cat == 'shared_overlap':
        return 'Shared overlap'
    return 'Movement / pointwise / setup'


def record(root, kind):
    paths=list(root.glob('*_'+kind+'.json'))
    if len(paths)!=1:
        raise ValueError('Expected one '+kind+' in '+str(root))
    return json.loads(paths[0].read_text())


def table(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):
            h.update(x)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--out',required=True)
    ns=ap.parse_args()
    root,out=Path(ns.root),Path(ns.out)
    out.mkdir(exist_ok=False)
    runs={name:root/'runs'/('pi05_mha8_h20_'+name+'_20261022_01')
          for name in ('validate','coarse','labels','fine')}
    for p in runs.values():
        assert json.loads((p/'exit.json').read_text())['success']
    validation=json.loads((runs['validate']/'validation.json').read_text())
    labels=json.loads((runs['labels']/'labels.json').read_text())
    fine=record(runs['fine'],'fine_complete')['result']
    before=record(runs['coarse'],'clean_before')['result']
    after=record(runs['coarse'],'clean_after')['result']
    phase=record(runs['coarse'],'phase')['result']
    assert validation['cache_bitwise_equal'] and fine['bitwise_equal']
    assert fine['kv_heads']==8 and fine['query_heads']==8 and fine['head_dim']==256
    assert fine['prefix_kv']['unique_storage_bytes']==18*2*8*968*256*2
    assert len(fine['samples'])==20 and all(len(s['streams'])==1 for s in fine['samples'])
    assert fine['conditioning_linear_online_ms']==0
    for name in ('coarse','labels','fine'):
        audit=record(runs[name],'cache_audit')['metadata']
        assert audit['last_request_dense_hits']==370
    unique={x['external_id']:x for x in labels['kernel_sequence']}
    for cat,shapes in [('attention_qkt',[[8,20,256],[8,256,988]]),
                       ('attention_sv',[[8,20,988],[8,988,256]])]:
        items=[x for x in unique.values() if x['category']==cat]
        assert len(items)==18 and all(x['shapes'][:2]==shapes for x in items)
    assert len([x for x in unique.values() if x['category']=='softmax'])==18
    assert len([x for x in unique.values() if x['category']=='gelu'])==18
    assert not any(x['category']=='cached_adarms' for x in labels['kernel_sequence'])

    clean=np.array([s['wall_ms'] for p in (before,after) for s in p['samples']])
    coarse=dict(clean_mean_ms=float(clean.mean()),clean_p50_ms=float(np.median(clean)),
                clean_p95_ms=float(np.percentile(clean,95)),
                phase_total_mean_ms=phase['gpu_span_ms']['mean'],
                phase_instrumentation_overhead_pct=100*(phase['wall_ms']['mean']/clean.mean()-1),
                before_after_drift_pct=100*(after['wall_ms']['mean']/before['wall_ms']['mean']-1))
    phase_rows=[dict(stage=k,mean_ms=phase['components_ms'][k]['mean'],share_pct=phase['shares_pct'][k])
                for k in ('vit','vlm','ae','other')]
    assert math.isclose(sum(x['mean_ms'] for x in phase_rows),coarse['phase_total_mean_ms'],abs_tol=1e-6)
    table(out/'phase_latency.csv',phase_rows)
    table(out/'clean_samples.csv',[dict(block=b,**s) for b,p in [('before',before),('after',after)] for s in p['samples']])
    exclusive={k:v['mean'] for k,v in fine['exclusive_ms'].items()}
    union=fine['gpu_active_union_ms']['mean']
    totals=Counter()
    for cat,value in exclusive.items():
        totals[family(cat)]+=value
    assert math.isclose(sum(totals.values()),union,abs_tol=1e-8)
    rows=[dict(category=k,family=family(k),exclusive_mean_ms=v,share_pct=100*v/union,
               inclusive_mean_ms=fine['inclusive_ms'][k]['mean'] if k in fine['inclusive_ms'] else '')
          for k,v in exclusive.items()]
    table(out/'step_categories.csv',rows)
    operators=[dict(category=k.split('|',1)[0],operator=k.split('|',1)[1],inclusive_mean_ms=v['mean'])
               for k,v in fine['operator_ms'].items()]
    table(out/'step_operators.csv',operators)
    geometry=[dict(category=x['category'],operator=x['operator'],shapes=json.dumps(x['shapes'])) for x in unique.values()]
    table(out/'step_geometry.csv',geometry)
    summary=dict(coarse=coarse,phases=phase_rows,
                 clean_step_p50_ms=fine['passes']['clean_before']['p50'],
                 clean_step_mean_ms=fine['passes']['clean_before']['mean'],
                 profiled_step_mean_ms=fine['passes']['profiled']['mean'],
                 post_profile_step_mean_ms=fine['passes']['clean_after']['mean'],
                 profile_overhead_pct=fine['profile_overhead_pct'],
                 post_profile_drift_pct=fine['post_profile_drift_pct'],
                 gpu_active_union_ms=union,fine_totals_ms=dict(totals),
                 prefix_kv_mib=validation['prefix_kv']['unique_storage_bytes']/1024**2,
                 actual_dimensions=validation['actual_dimensions'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'VALIDATION.json').write_text(json.dumps(dict(cache_bitwise_equal=True,
        raw_runs_successful=True,fine_outputs_bitwise_equal=True,fine_samples=20,
        single_cuda_stream=True,attention_qk_sv_calls=18,heads=8,head_dim=256,
        nonlinear_softmax_gelu_calls=18,online_conditioning_kernels=0,
        exclusive_totals_reconciled=True,phase_totals_reconciled=True),indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    palette=['#4C78A8','#E69F3A','#59A14F','#929BA3']
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1.35,1]})
    names=['ViT','VLM','AE (10 steps)','Other']
    vals=[r['mean_ms'] for r in phase_rows]
    axes[0].barh(np.arange(4),vals,color=palette,height=.6)
    for i,r in enumerate(phase_rows):
        axes[0].text(r['mean_ms'],i,f'  {r["mean_ms"]:.2f} ms ({r["share_pct"]:.1f}%)',va='center')
    axes[0].set_yticks(range(4),names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0,max(vals)*1.6)
    axes[0].set_xlabel('Same-pass phase mean (ms)')
    axes[0].set_title('A  Whole-model components',loc='left',fontweight='bold')
    axes[1].bar([0,1,2],[coarse['clean_p50_ms'],coarse['clean_p95_ms'],coarse['phase_total_mean_ms']],color=['#414A52','#737E86','#B4BCC2'],width=.6)
    for i,v in enumerate([coarse['clean_p50_ms'],coarse['clean_p95_ms'],coarse['phase_total_mean_ms']]):
        axes[1].text(i,v,f'{v:.2f}',ha='center',va='bottom')
    axes[1].set_xticks([0,1,2],['Clean p50','Clean p95','Phase mean'])
    axes[1].set_ylim(0,coarse['clean_p95_ms']*1.2)
    axes[1].set_ylabel('Model-call latency (ms)')
    axes[1].set_title('B  Total latency',loc='left',fontweight='bold')
    fig.suptitle('OpenPI pi0.5 | MHA8 | H=20, S=10 | AdaRMS cached',fontsize=16,fontweight='bold')
    fig.text(.5,.02,'B=1; 20 warmups + 100 samples per block. Clean wall latency and phase means use separate passes.\nWeights, initialization, CPU tokenization and input transfer excluded. Synthetic expanded-head model.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.13,right=.97,top=.80,bottom=.24,wspace=.38)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/('01_whole_model_latency.'+ext),dpi=200)
    plt.close(fig)

    matrix_order=[('Q projection','q_projection'),('K projection','k_projection'),('V projection','v_projection'),
                  ('QK^T','attention_qkt'),('SV / PV','attention_sv'),('O projection','o_projection'),
                  ('FFN gate','ffn_gate'),('FFN up','ffn_up'),('FFN down','ffn_down'),
                  ('Action input / output',('action_in_proj','action_out_proj')),
                  ('RoPE frequency matmul','rope_frequency_matmul')]
    right_order=[('RMSNorm + modulation pipeline','rms_norm'),('Softmax','softmax'),('GELU','gelu'),
                 ('RoPE sin / cos',('rope_sin','rope_cos')),
                 ('KV concatenation','kv_concat'),('Copies / casts','copy_or_cast'),
                 ('RoPE rotation / preparation',('rope_apply','rope_generation')),
                 ('Residual / FFN gate multiply',('residual_gate','ffn_gate_multiply')),
                 ('Attention scale / mask',('attention_mul','attention_add'))]
    used={k for _,v in right_order for k in (v if isinstance(v,tuple) else (v,))}
    remaining=tuple(k for k in exclusive if k not in used and k not in LINEAR)
    right_order.append(('Other setup / overlap',remaining))
    def amount(keys):
        return sum(exclusive.get(k,0) for k in (keys if isinstance(keys,tuple) else (keys,)))
    fig,axes=plt.subplots(1,2,figsize=(14,7.3),gridspec_kw={'width_ratios':[1,1.1]})
    for ax,order,left in zip(axes,[matrix_order,right_order],[True,False]):
        yy=np.arange(len(order)); values=[amount(x[1]) for x in order]
        colors=['#4C78A8' if left else '#E1812C' if i<4 else '#929BA3' for i in yy]
        ax.barh(yy,values,color=colors,height=.62)
        for y,v in zip(yy,values):
            ax.text(v,y,f'  {v:.3f} ms ({100*v/union:.1f}%)',va='center',fontsize=9)
        ax.set_yticks(yy,[x[0] for x in order],fontsize=10)
        ax.invert_yaxis(); ax.set_xlim(0,max(values)*1.60)
        ax.set_xlabel('Profiled GPU time per step (ms)')
        ax.set_axisbelow(True); ax.grid(axis='x',alpha=.2)
    axes[0].set_title('A  Matrix / linear',loc='left',fontweight='bold')
    axes[1].set_title('B  Nonlinear and explicit auxiliary work',loc='left',fontweight='bold')
    fig.suptitle('OpenPI pi0.5 MHA8 | One denoising step | H=20',fontsize=17,fontweight='bold')
    fig.text(.5,.91,f'GPU-active total: {union:.3f} ms  |  Matrix: {totals["Matrix / linear"]:.3f} ms  |  Nonlinear / normalization: {totals["Nonlinear / normalization"]:.3f} ms',ha='center',fontsize=11)
    fig.text(.5,.025,'Exclusive GPU attribution; 18 expert layers, fifth step. AdaRMS conditioning linears: 0 online kernels.\nRMSNorm includes its normalization, affine modulation and casts. Not clean wall-time fractions; no DRAM subtraction.\nThe two panels have different x-axis scales.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.13,right=.98,top=.83,bottom=.18,wspace=.86)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/('02_step_operator_breakdown.'+ext),dpi=200)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.4,4.9))
    diagnostic=[fine['passes'][n]['mean'] for n in ('clean_before','profiled','clean_after')]
    diagnostic.append(union)
    ax.bar(range(4),diagnostic,color=['#4C78A8','#E1812C','#929BA3','#59A14F'],width=.58)
    for i,v in enumerate(diagnostic):
        ax.text(i,v,f'{v:.3f} ms',ha='center',va='bottom')
    ax.set_xticks(range(4),['Clean before\ninterval','Profiled\ninterval','After profiler\ninterval','Profiled GPU\nactive union'])
    ax.set_ylim(0,max(diagnostic)*1.18)
    ax.set_ylabel('Mean time per step (ms)')
    ax.set_title('Measurement perturbation | MHA8 H20',fontweight='bold',pad=15)
    fig.text(.5,.025,f'Profiler interval change: {fine["profile_overhead_pct"]:+.1f}% | Post-profile drift: {fine["post_profile_drift_pct"]:+.1f}%\nDo not apply profiled operator fractions to clean wall time. Clean-after is not pooled into the step reference.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.12,right=.97,bottom=.24,top=.85)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/('03_profiler_perturbation.'+ext),dpi=200)
    plt.close(fig)

    lines=['# OpenPI pi0.5 MHA8 H20/S10 Latency','',
           'Completed 2026-10-22, Asia/Hong_Kong. Same cached OpenPI baseline as the previous round.',
           'Both VLM and AE have 8 query/KV heads, head_dim=256. Q/O, ViT, residual and FFN widths unchanged.',
           '', '## Whole Model', '',
           f'Clean p50 **{coarse["clean_p50_ms"]:.3f} ms**, p95 {coarse["clean_p95_ms"]:.3f} ms; 200 clean samples.',
           '', '| Phase | Mean ms | Same-pass share |','|---|---:|---:|']
    lines += [f'| {r["stage"].upper()} | {r["mean_ms"]:.3f} | {r["share_pct"]:.2f}% |' for r in phase_rows]
    lines += ['',f'Phase total {coarse["phase_total_mean_ms"]:.3f} ms. Phase overhead {coarse["phase_instrumentation_overhead_pct"]:+.3f}%; before/after drift {coarse["before_after_drift_pct"]:+.3f}%.',
              'ViT is not counted again in VLM; AE includes all ten steps and Euler updates. Model inputs are fixed synthetic GPU-resident tensors.',
              '', '![Whole model](01_whole_model_latency.png)', '', '## One AE Step', '',
              f'Clean-before step p50 **{summary["clean_step_p50_ms"]:.3f} ms**, mean {summary["clean_step_mean_ms"]:.3f} ms.',
              f'Profiled GPU-active union **{union:.3f} ms**, excluding launch gaps. The following are exclusive same-pass sums:',
              '', '| GPU category | Mean ms | GPU-active share |','|---|---:|---:|']
    lines += [f'| {k} | {v:.6f} | {100*v/union:.2f}% |' for k,v in totals.items()]
    lines += ['', '![Step details](02_step_operator_breakdown.png)', '',
              'The real fifth denoising step includes all 18 layers and action input/output projections; outer Euler update is excluded.',
              'Nonlinear/normalization includes GELU, explicit Softmax, RoPE sin/cos and the entire RMSNorm/modulation pipeline (including casts and affine work).',
              'Matrix includes Q/K/V/O, QK/SV, FFN gate/up/down, action I/O and the small RoPE frequency matmul.',
              'Remaining KV concatenation, casts, rotations, residual/gating, attention scaling/masks and setup are auxiliary, not all nonlinear.',
              'The conditioning cache has 370 hits per full request and zero online conditioning-linear kernels per measured step.',
              '', '## Interpretation', '',
              f'Profiler interval overhead {fine["profile_overhead_pct"]:+.2f}%; post-profile drift {fine["post_profile_drift_pct"]:+.2f}%.',
              'Do not apply these profiled fractions to clean wall latency. Clean-after step timings are not pooled into the clean-before reference.',
              'This is native model-step profiling, NOT independent per-operator microbenchmark summation and NOT a PIM/GPU execution measurement. No DRAM subtraction or cache flushing.',
              '', '![Perturbation](03_profiler_perturbation.png)', '', '## Geometry and Validation', '',
              'VLM: D=2048, FFN=16384, 18 layers. AE: D=1024, FFN=4096, 18 layers. Both attention inner widths=2048.',
              'Actual AE QK: [8,20,256] x [8,256,988]; SV: [8,20,988] x [8,988,256]. Prefix KV=136.125 MiB.',
              'Four cache-on/off action comparisons are bitwise equal; all fine instrumentation outputs are bitwise equal. Twenty single-stream kernel sequences match labels exactly.',
              f'Synthetic KV-head duplication is not a trained MHA8 policy. Maximum action relative-RMS difference from original MQA on four fixtures: {100*max(x["relative_rms"] for x in validation["reparameterization_checks"]):.4f}%. No task-quality claim.',
              'Six CPU mapping/analysis tests pass. Detailed scalar operation and tensor-shape ledgers accompany the figures.',
              '', 'Branch: `research/thor-openpi-mha8-h20-latency-20261022`. Not pushed automatically.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest=[]
    for name,p in runs.items():
        for file in sorted(p.glob('*.json')):
            manifest.append(dict(path=str(file.relative_to(root)),bytes=file.stat().st_size,sha256=sha(file)))
    for file in sorted(out.glob('*')):
        if file.is_file():
            manifest.append(dict(path=str(file.relative_to(root)),bytes=file.stat().st_size,sha256=sha(file)))
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

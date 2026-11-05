"""Generate scientific figures and a local-only, static HTML report."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = dict(vit='#397ca8',vlm='#68a6c5',ae='#16816d',other='#889399',
              linear='#16816d',nonlinear_core='#b75b45',elementwise='#ad8426',
              rope='#647ca8',data_movement='#7d8790',setup='#a9b5b7',overlap='#c9c9c9')
LABELS = {
    'linear':'Linear matrix operations','nonlinear_core':'Nonlinear / RMS core',
    'elementwise':'Elementwise affine / gates','rope':'RoPE generation / application',
    'data_movement':'KV concat / copies / casts','setup':'Masks / small setup', 'overlap':'Cross-group overlap',
    'q_projection':'Q projection','k_projection':'K projection','v_projection':'V projection',
    'attention_qkt':'QK transpose','attention_sv':'Score-value product (SV)','o_projection':'O projection',
    'ffn_gate':'FFN gate','ffn_up':'FFN up','ffn_down':'FFN down',
    'action_in_proj':'Action input projection','action_out_proj':'Action output projection',
    'kv_concat':'K/V concatenation','copies_casts':'Copies / dtype casts','copy_engine':'CUDA copy engine',
    'softmax':'Softmax','gelu':'GELU','rms_pow':'RMS square','rms_mean':'RMS mean reduction','rms_rsqrt':'RMS rsqrt',
    'norm_add':'Norm / AdaRMS add','norm_mul':'Norm / AdaRMS multiply',
    'residual_gate_add':'Residual add','residual_gate_mul':'Residual gate multiply',
    'ffn_gate_multiply':'FFN gate x up','attention_mul':'Attention scaling','attention_add':'Attention mask addition',
    'rope_generation':'RoPE position generation','rope_apply':'RoPE application',
}
LINEAR_ORDER = ['q_projection','k_projection','v_projection','attention_qkt','attention_sv','o_projection',
                'ffn_gate','ffn_up','ffn_down','action_in_proj','action_out_proj']
REST_ORDER = ['softmax','gelu','rms_pow','rms_mean','rms_rsqrt','norm_add','norm_mul','residual_gate_mul',
              'residual_gate_add','ffn_gate_multiply','attention_mul','attention_add','rope_generation','rope_apply',
              'kv_concat','copies_casts','copy_engine']


def save(fig, folder, name):
    for ext in ('svg','png','pdf'):
        fig.savefig(folder/(name+'.'+ext),dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#cdd6d4')
    ax.grid(axis='x',color='#e6ecea',linewidth=.8)
    ax.set_axisbelow(True)


def barplot(data, keys, folder, name, title, subtitle, group_for):
    fine=data['fine']
    vals=[fine['category_ms'][k]['mean'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,max(4.5,.43*len(keys)+1.7)))
    y=np.arange(len(keys))
    ax.barh(y,vals,height=.65,color=[COLORS[group_for(k)] for k in keys])
    ax.set_yticks(y,[LABELS.get(k,k) for k in keys])
    ax.invert_yaxis()
    maximum=max(vals)
    for i,(key,v) in enumerate(zip(keys,vals)):
        ax.text(v+maximum*.025,i,f"{v:.4f} ms  ({fine['category_active_share_pct'][key]:.2f}%)",va='center',fontsize=10)
    ax.set_xlim(0,maximum*1.55)
    ax.set_xlabel('Measured GPU activity per denoising step (ms), profiler pass')
    ax.set_title(title,loc='left',fontweight='bold',pad=35,fontsize=17)
    ax.text(0,1.025,subtitle,transform=ax.transAxes,fontsize=10,color='#54645e')
    style(ax)
    fig.tight_layout()
    save(fig,folder,name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((args.root/'analysis/summary.json').read_text())
    b,f=data['baseline'],data['fine']
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(12,3.6))
    left=0
    for key,label in [('vit','ViT'),('vlm','VLM'),('ae','AE'),('other','Other')]:
        value=b['phase_ms'][key]['mean']
        ax.barh(0,value,left=left,color=COLORS[key],height=.52)
        if key!='other':
            ax.text(left+value/2,0,f"{label}\n{value:.2f} ms\n{b['phase_share_pct'][key]:.1f}%",ha='center',va='center',color='white',fontsize=12,fontweight='bold')
        left+=value
    ax.set_yticks([])
    ax.set_ylim(-.6,.6)
    ax.set_xlim(0,left*1.05)
    ax.set_xlabel('Same-pass GPU event interval (ms); sum of phase means')
    ax.set_title(f"Normal inference: phase total {left:.2f} ms",loc='left',fontweight='bold',pad=35,fontsize=17)
    ax.text(0,1.025,f"100 phase samples; separate clean API wall p50 {b['clean_wall_ms']['p50']:.2f} ms. Other = {b['phase_ms']['other']['mean']:.2f} ms.",transform=ax.transAxes,fontsize=10,color='#54645e')
    style(ax)
    fig.tight_layout()
    save(fig,out,'01_coarse')
    fig,ax=plt.subplots(figsize=(12,5.3))
    keys=['linear','nonlinear_core','elementwise','rope','data_movement','setup','overlap']
    vals=[f['family_ms'][k]['mean'] for k in keys]
    ax.barh(np.arange(len(keys)),vals,color=[COLORS[k] for k in keys],height=.62)
    ax.set_yticks(np.arange(len(keys)),[LABELS[k] for k in keys])
    ax.invert_yaxis()
    for i,(k,v) in enumerate(zip(keys,vals)):
        ax.text(v+.06,i,f"{v:.4f} ms  ({f['family_active_share_pct'][k]:.2f}%)",va='center',fontsize=10)
    ax.set_xlim(0,max(vals)*1.48)
    ax.set_xlabel('Exclusive GPU activity per step (ms)')
    ax.set_title(f"One step: {f['per_step_active_ms']['mean']:.3f} ms of recorded GPU activity",loc='left',fontweight='bold',pad=35,fontsize=17)
    ax.text(0,1.025,'Profiler pass: 5 requests x 10 steps. Percentages use GPU-active union, NOT clean elapsed time.',transform=ax.transAxes,fontsize=10,color='#54645e')
    style(ax)
    fig.tight_layout()
    save(fig,out,'02_families')
    barplot(data,LINEAR_ORDER,out,'03_linear','Linear matrix operations','All 18 layers; physical Q8/K8/V8, head_dim=256, H16.',lambda k:'linear')
    def family(k):
        if k in {'softmax','gelu','rms_pow','rms_mean','rms_rsqrt'}:return 'nonlinear_core'
        if k.startswith('rope'):return 'rope'
        if k in {'kv_concat','copies_casts','copy_engine'}:return 'data_movement'
        return 'elementwise'
    barplot(data,REST_ORDER,out,'04_nonmatrix','Nonlinear, elementwise and data movement','RMS core is separated from casts / affine work. AdaRMS conditioning dense: cached, 0 online kernels.',family)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.35,1]})
    ax=axs[0]
    x=np.arange(1,11)
    means=[v['mean'] for v in b['step_ms']]
    ax.plot(x,means,color=COLORS['ae'],marker='o',label='No profiler: mean')
    ax.plot(x,[v['p50'] for v in b['step_ms']],color=COLORS['vit'],marker='.',label='No profiler: p50')
    ax.fill_between(x,[v['minimum'] for v in b['step_ms']],[v['p95'] for v in b['step_ms']],color=COLORS['ae'],alpha=.12,label='Minimum to p95')
    ax.set_xticks(x)
    ax.set_xlabel('Denoising step index (1-based)')
    ax.set_ylabel('Native denoise_step event interval (ms)')
    ax.set_title('Clean one-step elapsed time',loc='left',fontweight='bold')
    ax.legend(fontsize=9,frameon=False)
    style(ax)
    ax=axs[1]
    active=f['per_step_active_ms']['mean']
    gaps=f['per_step_gap_ms']['mean']
    ax.barh(0,active,color=COLORS['ae'],height=.4,label='GPU activity union')
    ax.barh(0,gaps,left=active,color='#d8dddb',height=.4,label='Instrumented gaps')
    ax.barh(1,b['step_per_request_mean_ms']['mean'],color=COLORS['vit'],height=.4,label='No-profiler event interval')
    ax.set_yticks([0,1],['Trace','Normal'])
    ax.invert_yaxis()
    ax.text(active/2,0,f'{active:.2f}',ha='center',va='center',color='white')
    ax.text(active+gaps/2,0,f'{gaps:.2f}',ha='center',va='center')
    ax.text(b['step_per_request_mean_ms']['mean']/2,1,f"{b['step_per_request_mean_ms']['mean']:.2f}",ha='center',va='center',color='white')
    ax.set_xlabel('ms / step (different measurement passes)')
    ax.set_title('Profiler perturbation is retained',loc='left',fontweight='bold')
    ax.legend(fontsize=9,frameon=False,loc='lower right')
    ax.set_ylim(1.65,-.55)
    style(ax)
    fig.tight_layout()
    save(fig,out,'05_method_audit')
    rows=''.join(f"<tr><td>{label}</td><td>{b['phase_ms'][key]['mean']:.3f}</td><td>{b['phase_ms'][key]['p50']:.3f}</td><td>{b['phase_ms'][key]['p95']:.3f}</td><td>{b['phase_share_pct'][key]:.2f}%</td></tr>" for key,label in [('vit','ViT（3 路图像合计）'),('vlm','VLM prefix'),('ae','AE（10 steps + 外围采样工作）'),('other','Other')])
    ops=''.join(f"<tr><td>{html.escape(LABELS.get(k,k))}</td><td>{v['mean']:.6f}</td><td>{f['category_active_share_pct'][k]:.3f}%</td></tr>" for k,v in f['category_ms'].items())
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenPI pi0.5 H16 MHA8 · Thor latency</title>
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#fff;color:#20312c;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}main{{max-width:1240px;margin:auto;padding:32px 28px 70px}}header{{border-bottom:2px solid #16816d;padding-bottom:24px}}h1{{font-size:32px;line-height:1.25;margin:8px 0 14px}}h2{{font-size:24px;margin:0 0 14px}}h3{{font-size:19px}}p{{margin:10px 0}}.meta{{color:#5e7068}}nav{{display:flex;gap:22px;flex-wrap:wrap;margin-top:16px}}a{{color:#176e61;text-underline-offset:3px}}section{{padding:30px 0;border-bottom:1px solid #dce4e1}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px;margin:24px 0}}.metrics strong{{display:block;font-size:32px;line-height:1.2}}.metrics span{{font-size:14px;color:#5e7068}}.note{{border-left:4px solid #ad8426;background:#faf7ee;padding:14px 18px}}.figure{{width:100%;height:auto;display:block;margin:14px 0}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:9px 12px;border-bottom:1px solid #e2e8e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff4f1}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere;font-size:14px}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:600}}.small{{font-size:14px;color:#5e7068}}li{{margin:8px 0}}@media(max-width:720px){{main{{padding:22px 16px}}h1{{font-size:27px}}h2{{font-size:22px}}.metrics{{grid-template-columns:1fr;gap:16px}}.columns{{grid-template-columns:1fr}}th,td{{padding:7px;font-size:13px}}}}
</style><main><header><div class="meta">2026-11-05 · NVIDIA Thor · 新实测 · 非 H20 外推</div>
<h1>OpenPI π0.5 · H16 · MHA8</h1><p>Batch 1 · step 10 · Q/K/V 各 8 heads · head_dim 256 · AdaRMS 预计算缓存</p>
<nav><a href="#coarse">正常推理</a><a href="#fine">算子拆分</a><a href="#method">测量口径</a><a href="#source">原始证据</a></nav></header>
<div class="metrics"><div><strong>{b['clean_wall_ms']['p50']:.2f} ms</strong><span>完整模型 API wall p50，无 profiler，100 次</span></div><div><strong>{b['phase_share_pct']['ae']:.1f}%</strong><span>AE 占同一轮 phase 均值总和</span></div><div><strong>{b['step_per_request_mean_ms']['mean']:.2f} ms</strong><span>正常 denoise_step 平均，无 profiler</span></div></div>
<p class="note"><b>热启动，不等于数据全在缓存。</b> 权重先加载，20 次完整 warmup 后计时；不 flush 缓存、不强制冷读、不做 kernel replay。实际缓存未命中的访问仍会读 DRAM，测得时间包含这些等待；没有扣除 DRAM 时间。</p>
<section id="coarse"><h2>01 / 正常推理：ViT、VLM、AE</h2><p>正常计时进程没有 Nsight 注入。CUDA Events 分段，末尾统一等待完成；没有逐算子同步。粗粒度占比采用 100 次同一轮 phase 均值，合计 <b>{b['phase_total_ms']['mean']:.3f} ms</b>。</p>
<img class="figure" src="01_coarse.svg" alt="正常推理阶段延迟与占比">
<div class="scroll"><table><thead><tr><th>部分</th><th>均值 / ms</th><th>p50 / ms</th><th>p95 / ms</th><th>均值占比</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="small">API wall：均值 {b['clean_wall_ms']['mean']:.3f} ms，p50 {b['clean_wall_ms']['p50']:.3f} ms，p95 {b['clean_wall_ms']['p95']:.3f} ms。100 次中最大值 {b['clean_wall_ms']['maximum']:.3f} ms，保留全部样本，不剔除长尾。API wall 与 phase 是相邻但独立的控制组，不能拼接分量中位数。</p>
<p>AE 边界沿用之前的口径：prefix VLM 完成到动作生成结束，含 10 次 denoising、动作投影及外围 Euler/采样工作；ViT 是 3 次图像编码合计，VLM 不重复包含 ViT。</p></section>
<section id="fine"><h2>02 / 单个 denoising step 的算子级拆分</h2>
<p>从 <b>5 个真实完整请求 × 10 steps</b> 的 kernel 时间线测量，先对每个请求内 10 steps 求平均，再统计 5 个请求。每 step 含 18 层及 action input/output projection，不含外部 Euler 更新。单独标注第 5 步，通过 CUDA launch correlation 归因，全部 <b>{f['matched_steps']} steps × {f['kernels_per_step']} kernels</b> 的名称、顺序和 launch geometry 一致。</p>
<p class="note"><b>细粒度为带采集扰动的实测，不是正常总延迟的精确分摊。</b> 记录的 GPU activity union 为 <b>{f['per_step_active_ms']['mean']:.3f} ms/step</b>，采集时 event interval 为 {f['per_step_event_ms']['mean']:.3f} ms/step。下图百分比以 GPU activity union 为分母，不以正常的 {b['step_per_request_mean_ms']['mean']:.3f} ms 为分母。</p>
<img class="figure" src="02_families.svg" alt="单步 GPU activity 分类汇总">
<img class="figure" src="03_linear.svg" alt="QKV、QK、SV、O projection 和 FFN 线性算子实测延迟">
<img class="figure" src="04_nonmatrix.svg" alt="非线性核心、逐元素运算、RoPE 和拷贝延迟">
<p><b>分类说明：</b> Nonlinear / RMS core 包含 softmax、GELU、RMS square/mean/rsqrt；norm affine、residual gate、FFN gate × up 单列逐元素操作。K/V concat、dtype cast/copy 不是非线性核心。RoPE 独立列出。AdaRMS 只缓存时间条件和 dense modulation，归一化、调制乘加及 residual 仍在线执行；每请求命中 370 次 cached dense，<b>在线 conditioning dense 无 kernel</b>。</p>
<p class="small">Kernel 时长包含真实内存/缓存访问，不是算术单元纯计算时间，也不是未来 GPU/PIM offload 的端到端代价。组内重叠只计一次，跨组重叠另列；细分类的 overlap 与大类的 overlap 会有微小差别。</p>
<details><summary>完整算子表（ms/step）</summary><div class="scroll"><table><thead><tr><th>类别</th><th>均值 / ms</th><th>GPU-active 占比</th></tr></thead><tbody>{ops}</tbody></table></div></details></section>
<section id="method"><h2>03 / 正常计时和 profiler 时间为什么不同</h2><img class="figure" src="05_method_audit.svg" alt="正常十个步骤的耗时与采集扰动对照">
<div class="columns"><div><h3>正常延迟如何测</h3><ul><li>独立 Python 进程，不由 Nsight 启动；20 次完整 warmup。</li><li>50 clean → 50 phases → 50 step-control → 50 phases → 50 clean。clean/phase 各 100 次；step-control 为 50 个请求的全部 10 步。</li><li>每请求起止同步，段内仅 record CUDA Event，不逐算子 synchronize。输入已在 GPU；不包括加载、cache 构建、传感器/网络/机器人工作。</li><li>Phase marker 均值差 {b['phase_marker_mean_overhead_pct']:.2f}%；clean 前后 p50 漂移 {b['clean_after_vs_before_p50_pct']:.2f}%。每次推理后检查动作 bitwise 相同，检查不计入延迟。</li></ul></div>
<div><h3>细粒度如何测</h3><ul><li>独立 Nsight Systems 2025.6.3 进程，只开 CUDA/NVTX，无 NCU、无 PMU、无 GPU metrics、无 CPU sampling。</li><li>20 warmup，20 before-control，2 trace burn-in，5 trace 请求，1 单独 attribution 请求，10 恢复请求，20 after-control。</li><li>Thor 不支持该工具的 CUDA hardware tracing，实际为软件 trace。AE p50 比 trace 进程 before-control 增加 {f['ae_p50_overhead_vs_before_pct']:.1f}%，比 after-control 增加 {f['ae_p50_overhead_vs_after_pct']:.1f}%；相对独立正常 AE 均值增加 {f['ae_mean_overhead_vs_independent_baseline_pct']:.1f}%。</li><li>所以 {f['per_step_gap_ms']['mean']:.3f} ms 的 traced gap 不能称为正常空闲时间；不按比例缩放到正常 AE，也不把两轮差值解释为纯 profiler 开销。</li></ul></div></div>
<p class="small">Nsight 原始诊断包含“Not all NVTX/CUDA events might have been collected”。我们验证了全部选中 step 的 kernel 序列、launch geometry 和 18 层投影覆盖，并保留诊断；这不等于证明所有后台事件均被完整记录。</p></section>
<section id="source"><h2>04 / 配置、验证与原始证据</h2><p>官方 OpenPI 原生 PyTorch 路径，selective BF16/FP32，Torch 2.11.0+cu130 / Transformers 4.53.2，CPU/interop 各 1 线程；不启用 compile 或推理 CUDA Graph，不改时钟或共享环境。</p>
<p>沿用已确认的 OpenPI <b>eager、未融合 attention</b> 路径，所以 QK / softmax / SV 可分别计时。原始 JSON 的 <code>attention_implementation=sdpa</code> 是首次推理前的构造时快照；OpenPI 在采样函数内切为 eager，不代表实际使用了 SDPA。<a href="../ATTENTION_BACKEND_AUDIT.md">源码核对说明</a>。</p>
<p>MHA 是之前确认的 <b>VLM 和 AE 均物理扩展 K/V 至 8 heads</b>；Q/O、ViT、residual 和 FFN 宽度未改。AE D=1024/F=4096；VLM D=2048/F=16384；均 18 层，prefix 968 tokens，AE attention KV 长度 984。该修改仅用于结构延迟研究，不声明训练后 MHA 策略精度。</p>
<p>权重复用 Thor 现有文件，sequential readinto 加载；本次未下载模型。缓存开关及两个进程输出 bitwise 一致。正常轮 {data['validation']['baseline_occupancy_checks']} 次、trace 轮 {data['validation']['trace_occupancy_checks']} 次占用检查均未发现其他 GPU PID；离散检查不能排除所有瞬态 SoC 背景活动。任务结束 GPU 进程列表为空，利用率 0%。</p>
<p><a href="../analysis/summary.json">完整统计 JSON</a> · <a href="../analysis/coarse.csv">阶段 CSV</a> · <a href="../analysis/operators.csv">算子 CSV</a> · <a href="../analysis/steps.csv">逐 step CSV</a> · <a href="../README.md">方法与复现</a></p>
<p class="small">GitHub：AlinaWXY/pim_vla_simulator<br>分支：<code>thor-data/pi05-openpi-mha8-h16-s10-adarms-latency-20261105</code><br>路径：<code>thor-data/pi05-openpi-mha8-h16-s10-adarms-20261105/</code></p></section></main></html>'''
    (out/'index.html').write_text(doc)
    print(out/'index.html')


if __name__=='__main__':
    main()

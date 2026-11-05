"""Publication figures for the cold-kernel dataset, separate from warm latency."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'linear':'#16816d','nonlinear_core':'#b75b45','elementwise':'#ad8426','rope':'#647ca8','data_movement':'#7d8790','setup':'#a9b5b7'}
FAMILIES={'linear':'Linear matrix operations','nonlinear_core':'Nonlinear / RMS core','elementwise':'Elementwise affine / gates',
          'rope':'RoPE generation / application','data_movement':'KV concat / copies / casts','setup':'Masks / small setup'}
LABELS={'q_projection':'Q projection','k_projection':'K projection','v_projection':'V projection','o_projection':'O projection',
    'attention_qkt':'QK transpose','attention_sv':'Score-value product (SV)','ffn_gate':'FFN gate','ffn_up':'FFN up','ffn_down':'FFN down',
    'action_in_proj':'Action input projection','action_out_proj':'Action output projection','softmax':'Softmax','gelu':'GELU',
    'rms_pow':'RMS square','rms_mean':'RMS mean reduction','rms_rsqrt':'RMS rsqrt','norm_add':'Norm / AdaRMS add','norm_mul':'Norm / AdaRMS multiply',
    'residual_gate_add':'Residual add','residual_gate_mul':'Residual gate multiply','ffn_gate_multiply':'FFN gate x up',
    'attention_mul':'Attention scaling','attention_add':'Attention mask addition','rope_generation':'RoPE position generation','rope_apply':'RoPE application',
    'kv_concat':'K/V concatenation','copies_casts':'Copies / dtype casts'}
LINEAR=['q_projection','k_projection','v_projection','attention_qkt','attention_sv','o_projection','ffn_gate','ffn_up','ffn_down','action_in_proj','action_out_proj']
REST=['softmax','gelu','rms_pow','rms_mean','rms_rsqrt','norm_add','norm_mul','residual_gate_mul','residual_gate_add','ffn_gate_multiply',
      'attention_mul','attention_add','rope_generation','rope_apply','kv_concat','copies_casts']


def save(fig,out,name):
    for ext in ('png','svg','pdf'):
        fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#cdd6d4')
    ax.grid(axis='x',color='#e6ecea')
    ax.set_axisbelow(True)


def group(k):
    if k in LINEAR:return 'linear'
    if k in ('softmax','gelu','rms_pow','rms_mean','rms_rsqrt'):return 'nonlinear_core'
    if k.startswith('rope'):return 'rope'
    if k in ('kv_concat','copies_casts'):return 'data_movement'
    return 'elementwise'


def detail(d,out,keys,name,title):
    values=[d['category_ms'][k]['mean'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,.43*len(keys)+1.7))
    ax.barh(np.arange(len(keys)),values,height=.65,color=[COLORS[group(k)] for k in keys])
    ax.set_yticks(np.arange(len(keys)),[LABELS[k] for k in keys])
    ax.invert_yaxis()
    m=max(values)
    for i,(k,v) in enumerate(zip(keys,values)):
        ax.text(v+m*.025,i,f"{v:.4f} ms  ({d['category_share_pct'][k]:.2f}%)",va='center',fontsize=10)
    ax.set_xlim(0,m*1.5)
    ax.set_xlabel('Cold-kernel duration sum per selected step (ms), NOT step wall latency')
    ax.set_title(title,loc='left',fontweight='bold',fontsize=17,pad=35)
    ax.text(0,1.025,f"18 layers; mean of 3 real fifth steps. Percentages use {d['per_step_sum_kernel_ms']['mean']:.3f} ms cold-kernel sum.",transform=ax.transAxes,fontsize=10,color='#54645e')
    axis(ax)
    fig.tight_layout()
    save(fig,out,name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    d=json.loads((args.root/'analysis_verified/summary.json').read_text())
    if d['validation']['warm_signature_mismatches'] or d['kernels_per_repeat']!=[1033,1033,1033]:
        raise ValueError('Figure labels require verified kernel coverage')
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    total=d['per_step_sum_kernel_ms']['mean']
    keys=list(FAMILIES)
    values=[d['family_ms'][k]['mean'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,5.4))
    ax.barh(np.arange(len(keys)),values,color=[COLORS[k] for k in keys],height=.62)
    ax.set_yticks(np.arange(len(keys)),[FAMILIES[k] for k in keys])
    ax.invert_yaxis()
    for i,(k,v) in enumerate(zip(keys,values)):
        ax.text(v+.08,i,f"{v:.4f} ms ({d['family_share_pct'][k]:.2f}%)",va='center',fontsize=10)
    ax.set_xlim(0,max(values)*1.5)
    ax.set_xlabel('NCU isolated cold-kernel duration sum (ms / selected step)')
    ax.set_title(f'Cold-cache AE kernels: {total:.3f} ms summed per step',loc='left',fontweight='bold',fontsize=17,pad=35)
    ax.text(0,1.025,'H16 / MHA8 / S10 / AdaRMS precomputed. Kernel-cache flush ON. This is NOT native step latency.',transform=ax.transAxes,fontsize=10,color='#54645e')
    axis(ax)
    fig.tight_layout()
    save(fig,out,'01_cold_families')
    detail(d,out,LINEAR,'02_cold_linear','Cold-cache linear matrix operations')
    detail(d,out,REST,'03_cold_nonmatrix','Cold-cache nonlinear, elementwise and movement')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    probe=d['cache_probe']['results']
    ax=axs[0]
    ax.bar(['No flush','Flush all'],[probe[p]['median_miss_requested_bytes']/1024**2 for p in ('none','all')],color=['#397ca8','#b75b45'],width=.55)
    ax.set_ylabel('L2 read-miss requested bytes (MiB)')
    ax.set_ylim(0,1.25)
    ax.text(0,.025,'0',ha='center')
    ax.text(1,1.025,'1.000 MiB',ha='center')
    ax.set_title('Cache-control probe: 1 MiB input',loc='left',fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax=axs[1]
    times=[r['sum_kernel_ms'] for r in d['repetitions']]
    ax.bar([1,2,3],times,color=COLORS['linear'],width=.55)
    for i,v in enumerate(times,1):ax.text(i,v+.12,f'{v:.3f}',ha='center')
    ax.set_xticks([1,2,3],['Request 1','Request 2','Request 3'])
    ax.set_ylim(0,max(times)*1.2)
    ax.set_ylabel('Cold-kernel sum (ms / fifth step)')
    ax.set_title('Independent requests; one NCU pass/kernel',loc='left',fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    save(fig,out,'04_cache_probe_and_repeats')
    family_rows=''.join(f"<tr><td>{FAMILIES[k]}</td><td>{d['family_ms'][k]['mean']:.4f}</td><td>{d['family_share_pct'][k]:.2f}%</td></tr>" for k in keys)
    op_rows=''.join(f"<tr><td>{html.escape(LABELS.get(k,k))}</td><td>{v['mean']:.6f}</td><td>{d['category_share_pct'][k]:.3f}%</td></tr>" for k,v in d['category_ms'].items())
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thor π0.5 H16 MHA8 · 冷缓存 kernel 计时</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:#20312c;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}main{{max-width:1240px;margin:auto;padding:32px 28px 70px}}header{{border-bottom:2px solid #b75b45;padding-bottom:24px}}h1{{font-size:32px;line-height:1.25;margin:8px 0 14px}}h2{{font-size:24px;margin:0 0 14px}}p{{margin:10px 0}}section{{padding:30px 0;border-bottom:1px solid #dce4e1}}.meta,.small{{color:#5e7068}}.small{{font-size:14px}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px;margin:24px 0}}.metrics strong{{display:block;font-size:31px}}.metrics span{{font-size:14px;color:#5e7068}}.note{{border-left:4px solid #b75b45;background:#fbf1ee;padding:14px 18px}}img{{width:100%;height:auto;display:block;margin:16px 0}}a{{color:#176e61;text-underline-offset:3px}}nav{{display:flex;gap:20px;flex-wrap:wrap;margin-top:14px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:9px 12px;border-bottom:1px solid #e2e8e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff4f1}}.scroll{{overflow-x:auto}}code{{font-size:14px;overflow-wrap:anywhere}}li{{margin:8px 0}}summary{{cursor:pointer;font-weight:600}}@media(max-width:720px){{main{{padding:20px 16px}}h1{{font-size:26px}}h2{{font-size:22px}}.metrics{{grid-template-columns:1fr;gap:12px}}th,td{{padding:7px;font-size:13px}}}}</style>
<main><header><div class="meta">2026-11-05 · NVIDIA Thor · 独立冷缓存数据批次</div><h1>π0.5 · H16 · MHA8<br>逐 kernel 冷缓存计时</h1><p>OpenPI · step 10 · Q/K/V 各 8 heads · head_dim 256 · B1 · AdaRMS 预计算保留</p><nav><a href="#sum">分类汇总</a><a href="#ops">算子细分</a><a href="#audit">缓存验证</a><a href="#method">测量口径</a></nav></header>
<div class="metrics"><div><strong>{total:.3f} ms</strong><span>单个选中 step 的冷缓存 kernel 耗时之和</span></div><div><strong>3 × 1,033</strong><span>3 个完整请求，各采真实第 5 步</span></div><div><strong>1 pass / kernel</strong><span>正式模型只收 duration，无多指标回放</span></div></div>
<p class="note"><b>这不是单步端到端 latency，也不是模型加载时间。</b> 权重和 AdaRMS 预计算张量先放在内存中，完整 warmup 20 次；之后由 NCU 在每个被测 kernel 前清硬件缓存。只汇总 kernel GPU duration，清缓存、工具同步、CPU 提交、DMA memcpy 和 kernel 间隙不在这个合计中。</p>
<section id="sum"><h2>01 / 冷缓存耗时分布</h2><img src="01_cold_families.svg" alt="冷缓存单步 kernel 耗时分类与占比"><div class="scroll"><table><thead><tr><th>类别</th><th>均值 ms/step</th><th>冷缓存 kernel 合计占比</th></tr></thead><tbody>{family_rows}</tbody></table></div><p class="small">范围 {d['per_step_sum_kernel_ms']['minimum']:.3f}–{d['per_step_sum_kernel_ms']['maximum']:.3f} ms，标准差 {d['per_step_sum_kernel_ms']['stdev']:.3f} ms；统计单位是 3 个不同请求中对应的第 5 步，而不是 3,099 个独立请求。</p></section>
<section id="ops"><h2>02 / 线性与非线性算子细分</h2><img src="02_cold_linear.svg" alt="QKV、QK、SV、O 和 FFN 的冷缓存 kernel 耗时"><img src="03_cold_nonmatrix.svg" alt="softmax、GELU、RMS、逐元素操作、RoPE、拷贝的冷缓存 kernel 耗时"><p>Nonlinear / RMS core 为 softmax、GELU、RMS square/mean/rsqrt；norm 调制、residual gate、FFN gate × up 等另列逐元素操作。RoPE 独立列出，K/V concat 与 cast/copy 不是非线性核心。</p><p><b>AdaRMS conditioning dense 仍为 0 个在线 kernel。</b> 清空 L1/L2 等硬件缓存，不会删除存在设备内存中的预计算张量；这些条件值依旧直接复用，只是读取时可能需要经过内存层级。</p><details><summary>完整算子表</summary><div class="scroll"><table><thead><tr><th>类别</th><th>均值 / ms</th><th>占比</th></tr></thead><tbody>{op_rows}</tbody></table></div></details></section>
<section id="audit"><h2>03 / 清缓存是否生效</h2><img src="04_cache_probe_and_repeats.svg" alt="清缓存前后 L2 miss 对照及三次模型测量"><p>同一 1 MiB 输入做固定加法，50 次预热后比较 NCU <code>--cache-control none</code> 和 <code>all</code>。不清缓存的 4 个样本 L2 read-miss sectors 均为 0；清缓存的 4 个样本均为 32,768，对应 1 MiB 的 miss 请求量，验证本探针的清缓存效果。</p><p class="small">未清缓存时 L2 hit 和 miss 计数都为 0，因此不能把它写成“L2 命中率 100%”；这与请求由更近的缓存服务相符。探针的 hit/miss 指标需要 3 个计数器 pass，<b>不用于模型延迟统计</b>。L2 miss 不是本次所有模型算子的 DRAM 字节测量。</p></section>
<section id="method"><h2>04 / 方法、边界与限制</h2><ul>
<li>Nsight Compute 2026.1.1：<code>--metrics gpu__time_duration.sum --cache-control all --replay-mode kernel --clock-control none --import-sass off</code>；用 NVTX 仅选真实第 5 个 denoise_step。</li>
<li>每个请求仍执行 ViT、VLM 及全部 10 steps，仅记录其中第 5 步的 kernel。重复 3 次，包含 18 层以及动作输入/输出投影，不包含外围 Euler 更新。所有 3,099 kernels 均为 1 pass。</li>
<li>使用实际 ATen dispatch 的 NVTX 标识映射算子；CUB 自己的嵌套 NVTX 名字通过外层完整 stack 还原归属，没有丢弃这些 kernel。</li>
<li>每次均有 1,033 kernels。三次名称、顺序、grid/block 一致，也与之前自然缓存 trace 的 1,033 个 kernel 一致。最终动作与之前正常推理逐位相同；370 次 dense cache 命中/请求。</li>
<li>保留此前 OpenPI 原生 eager attention、selective BF16/FP32、1 CPU/interop 线程，VLM 与 AE 均为物理 MHA8；不是训练过的 MHA 策略精度验证。</li>
<li>不锁系统时钟。NCU 保留默认 stable Tensor Core pipeline boost 设置；工具自身的串行化、计时口径与运行时频率差异都可能影响结果。</li>
<li>一个逻辑算子可能包含多个 kernel。逐 kernel 清缓存会破坏其内部跨 kernel 复用，因此这组数据不是一次逻辑算子从 DRAM 起步但内部正常复用的时间，更不是实际 GPU/PIM offload 的总成本。</li>
<li>现有正常 step 平均 {d['warm_reference']['normal_step_mean_ms']:.3f} ms、完整请求 p50 {d['warm_reference']['normal_api_wall_p50_ms']:.3f} ms 仍保留。不同工具/缓存/同步口径，<b>不计算冷/热加速比，不把本页 kernel 和当成正常 AE 延迟，不乘 10 冒充完整 AE 实测</b>。</li></ul>
<p><a href="../README.md">完整说明与复现</a> · <a href="../analysis_verified/summary.json">统计 JSON</a> · <a href="../analysis_verified/operators.csv">算子 CSV</a> · <a href="../analysis_verified/families.csv">分类 CSV</a> · <a href="../RESOURCE_STATUS.txt">资源释放记录</a></p><p class="small">上传仓库：AlinaWXY/pim_vla_simulator<br>分支：<code>thor-data/pi05-openpi-mha8-h16-s10-cold-kernel-20261105</code><br>数据目录：<code>thor-data/pi05-openpi-mha8-h16-s10-cold-kernel-20261105/</code></p></section></main></html>'''
    (out/'index.html').write_text(text)
    print(out/'index.html')


if __name__=='__main__':
    main()

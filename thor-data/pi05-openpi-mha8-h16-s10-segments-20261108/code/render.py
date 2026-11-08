"""Static scientific figures and Chinese report for complete GPU segments."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS={
 'action_in':'Action input projection','action_embedding_prepare':'Embedding concat / cast',
 'mask_position_setup':'Mask / position setup','rope_generation':'RoPE tables (once)',
 'norm_input':'Input AdaRMS (18 calls)','norm_ffn':'FFN AdaRMS (18 calls)','norm_final':'Final AdaRMS (once)',
 'qkv_projection':'Q / K / V projections together','rope_qk':'Q / K RoPE together',
 'kv_concat_pair':'K and V concat together','qkt':'QK transpose','scale_mask_softmax_cast':'Scale / mask / softmax / cast',
 'sv':'Score-value matmul','sv_output_pack':'SV output layout','o_projection':'O projection',
 'residual_attention':'Attention gated residual','ffn_gate_up':'FFN gate / up projections',
 'gelu_multiply':'GELU and gate x up','ffn_down':'FFN down projection','residual_ffn':'FFN gated residual',
 'action_output_cast':'Action output cast','action_out':'Action output projection','euler_update':'Euler update',
 'post_attention_residual_norm':'Attention residual + FFN norm',
 'post_ffn_residual_nextnorm':'FFN residual + next/final norm',
 'sv_pack_o_residual_norm':'SV layout + O + residual + norm',
 'pim_gelu_multiply_bf16':'FP32 G/U -> GELU x up -> BF16',
 'pim_rope_fp32_bf16':'FP32 Q/K -> RoPE -> BF16',
 'pim_scale_mask_softmax_bf16':'FP32 scores -> softmax -> BF16',
 'pim_o_fp32_cast_resnorm':'FP32 O -> cast / residual / norm',
 'pim_down_fp32_cast_resnorm':'FP32 Down -> cast / residual / next norm',
 'pim_pv_sum_cast_pack_o_resnorm':'Two FP32 PV partials -> sum / cast / O / norm',
 'pim_softmax_bf16_parity_pack':'Scores -> softmax / BF16 / parity packing',
}
CN={
 'rope_qk':'完整 Q/K RoPE','norm_input':'层入口 AdaRMS','norm_ffn':'FFN 前 AdaRMS','norm_final':'最终 AdaRMS',
 'scale_mask_softmax_cast':'缩放 + mask + softmax + cast','gelu_multiply':'GELU + gate×up',
 'post_attention_residual_norm':'Attention 残差 + FFN 前归一化（O 在 PIM）',
 'post_ffn_residual_nextnorm':'FFN 残差 + 下一层/最终归一化',
 'sv_pack_o_residual_norm':'SV 排布 + O 投影 + 残差 + 归一化（O 在 GPU）',
 'pim_rope_fp32_bf16':'FP32 Q/K → 完整 RoPE → BF16',
 'pim_gelu_multiply_bf16':'FP32 Gate/Up → GELU×Up → BF16',
 'pim_scale_mask_softmax_bf16':'FP32 score → scale/mask/softmax → BF16',
 'pim_o_fp32_cast_resnorm':'FP32 O → BF16 → 残差 + 归一化',
 'pim_down_fp32_cast_resnorm':'FP32 Down → BF16 → 残差 + 下一次归一化',
 'pim_pv_sum_cast_pack_o_resnorm':'两份 FP32 PV 部分和 → 求和/cast/排布/O/残差/归一化',
 'pim_softmax_bf16_parity_pack':'FP32 score → softmax/BF16/补齐/奇偶 KV 分片',
}
BLUE='#397ca8';GREEN='#16816d';RED='#b75b45';GRAY='#7d8790'


def save(fig,out,name):
    for ext in ('svg','png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#cdd6d4')
    ax.grid(axis='x',color='#e6ecea');ax.set_axisbelow(True)


def paired(rows,out,name,title,unit='sum_case_means_ms'):
    fig,ax=plt.subplots(figsize=(13,max(4.5,.62*len(rows)+1.9)))
    ys=np.arange(len(rows));mx=0
    for offset,mode,color,label in [(-.17,'eager_entry_cold',BLUE,'Eager event span (CPU feed included)'),(.17,'graph_entry_cold',GREEN,'Inside-graph GPU service interval')]:
        values=[r[mode][unit] for r in rows];mx=max(mx,max(values))
        ax.barh(ys+offset,values,height=.29,color=color,label=label)
        for y,v in zip(ys+offset,values):ax.text(v,y,f'  {v:.3f}',va='center',fontsize=9)
    ax.set_yticks(ys,[LABELS.get(r['family'],r['family']) for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0,mx*1.24)
    ax.set_xlabel('Sum of measured case means across their calls in one step (ms)' if unit=='sum_case_means_ms' else 'Mean complete segment per occurrence (us)')
    ax.set_title(title,loc='left',fontweight='bold',fontsize=17,pad=48)
    ax.text(0,1.025,'One entry eviction, none inside. Same math; distinct submission/timing boundaries. NOT E2E.',transform=ax.transAxes,fontsize=10,color='#54645e')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,loc='lower center',ncol=2)
    style(ax);fig.tight_layout(rect=(0,.08,1,1))
    save(fig,out,name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args();d=json.loads((ns.root/'analysis_verified/summary.json').read_text())
    ns.output.mkdir(parents=True,exist_ok=False);out=ns.output
    rows=d['aggregates'];lookup={(r['suite'],r['family']):r for r in rows}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(13,4.7))
    r=d['rope'];ax=axs[0]
    vals=[r[m]['qk_apply_18_layers_ms'] for m in ('graph_reused','graph_entry_cold','eager_reused','eager_entry_cold')]
    ax.barh(range(4),vals,color=[GREEN,GREEN,BLUE,BLUE],height=.6)
    ax.set_yticks(range(4),['Graph: reused control','Graph: one entry eviction','Eager: reused control','Eager: one entry eviction']);ax.invert_yaxis()
    for i,v in enumerate(vals):ax.text(v+.02,i,f'{v:.3f}',va='center')
    ax.set_xlim(0,max(vals)*1.2);ax.set_xlabel('Q/K RoPE, 18 layer calls summed (ms)')
    ax.set_title('RoPE is now one complete Q/K segment',loc='left',fontweight='bold');style(ax)
    ax=axs[1];p=d['cache_probe']['median_read_bytes']
    values=[p[k]/1024**2 for k in ('reused','entry_once','per_kernel')]
    ax.bar(['Reused','Entry once','Each kernel'],values,color=[BLUE,GREEN,RED],width=.58)
    for i,v in enumerate(values):ax.text(i,v+.04,f'{v:.4f}',ha='center')
    ax.set_ylim(0,max(values)*1.24);ax.set_ylabel('Probe DRAM reads (MiB), 1 MiB input')
    ax.set_title('Intermediate cache reuse is preserved',loc='left',fontweight='bold')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();save(fig,out,'01_rope_and_cache_scope')
    families=['qkv_projection','rope_qk','kv_concat_pair','qkt','scale_mask_softmax_cast','sv','o_projection',
        'norm_input','norm_ffn','ffn_gate_up','gelu_multiply','ffn_down','residual_attention','residual_ffn']
    paired([lookup['native_blocks',k] for k in families],out,'02_native_blocks','Native logical blocks (18 occurrences per row)')
    paired([r for r in rows if r['suite']=='handoff_composites'],out,'03_composite_boundaries','Continuous GPU boundaries: choose alternatives, do not sum with components','mean_occurrence_us')
    paired([r for r in rows if r['suite']=='pim_interface_candidates'],out,'04_fp32_interface_candidates','Logical FP32 PIM interfaces: physical transport / visibility excluded','mean_occurrence_us')
    c=d['coarse'];fig,ax=plt.subplots(figsize=(12,3.5));left=0
    for key,label,color in [('vit','ViT',BLUE),('vlm','VLM','#68a6c5'),('ae','AE',GREEN),('other','Other',GRAY)]:
        v=c['phase_ms'][key]['mean'];ax.barh(0,v,left=left,height=.5,color=color)
        if key!='other':ax.text(left+v/2,0,f'{label}\n{v:.2f} ms\n{c["phase_share_pct"][key]:.1f}%',ha='center',va='center',color='white',fontweight='bold')
        left+=v
    ax.set_yticks([]);ax.set_xlabel('Normal full-model phase means, no profiler / graph / eviction inside model (ms)')
    ax.set_title(f'2026-11-08 normal OpenPI control: {left:.2f} ms phase total',loc='left',fontweight='bold',pad=35)
    ax.text(0,1.04,f'60 phase samples. Independent clean API-wall p50: {c["clean_wall_ms"]["p50"]:.2f} ms (60 requests).',transform=ax.transAxes,fontsize=10,color='#54645e')
    style(ax);fig.tight_layout();save(fig,out,'05_normal_model_control')
    def table(suite):
        body=''.join(f'<tr><td>{html.escape(CN.get(r["family"],LABELS.get(r["family"],r["family"])))}</td><td>{r["occurrences_per_step"]}</td><td>{r["eager_entry_cold"]["sum_case_means_ms"]:.4f}</td><td>{r["graph_entry_cold"]["sum_case_means_ms"]:.4f}</td><td>{r["graph_entry_cold"]["mean_occurrence_us"]:.3f}</td></tr>' for r in rows if r['suite']==suite)
        return '<div class="scroll"><table><thead><tr><th>完整计算段</th><th>次数/step</th><th>Eager 合计 ms</th><th>Graph 合计 ms</th><th>Graph 单次均值 µs</th></tr></thead><tbody>'+body+'</tbody></table></div>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-08 π0.5 H16 连续 GPU 段延迟</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:#20312c;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}main{{max-width:1280px;margin:auto;padding:30px 28px 70px}}header{{border-bottom:2px solid #16816d;padding-bottom:24px}}h1{{font-size:32px;line-height:1.25;margin:8px 0 14px}}h2{{font-size:23px;margin:0 0 14px}}h3{{font-size:18px}}section{{padding:28px 0;border-bottom:1px solid #dce4e1}}p{{margin:10px 0}}nav{{display:flex;gap:22px;flex-wrap:wrap;margin-top:14px}}a{{color:#176e61;text-underline-offset:3px}}.meta,.small{{color:#5e7068}}.small{{font-size:14px}}.note{{border-left:4px solid #ad8426;background:#faf7ee;padding:14px 18px}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin:24px 0}}.metrics strong{{display:block;font-size:30px}}.metrics span{{font-size:14px;color:#5e7068}}img{{display:block;width:100%;height:auto;margin:16px 0}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:9px 11px;border-bottom:1px solid #e2e8e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff4f1}}code{{font-size:14px;overflow-wrap:anywhere}}li{{margin:8px 0}}summary{{cursor:pointer;font-weight:600}}@media(max-width:720px){{main{{padding:20px 15px}}h1{{font-size:26px}}h2{{font-size:21px}}.metrics{{grid-template-columns:1fr;gap:12px}}th,td{{padding:7px;font-size:13px}}}}</style>
<main><header><div class="meta">2026-11-08 · Thor 实测 · 替换逐 kernel 冷缓存口径</div><h1>π0.5 H16 连续 GPU 段延迟</h1><p>MHA8 · head_dim256 · step10 · B1 · AdaRMS 条件预计算缓存 · OpenPI 原生算术</p><nav><a href="#rope">RoPE 与缓存边界</a><a href="#blocks">原生分块</a><a href="#interfaces">交接段</a><a href="#method">计时与 E2E</a></nav></header>
<div class="metrics"><div><strong>{r['graph_entry_cold']['qk_apply_18_layers_ms']:.3f} ms</strong><span>18 层完整 Q/K RoPE，Graph 段内计时合计</span></div><div><strong>{r['eager_entry_cold']['qk_apply_18_layers_ms']:.3f} ms</strong><span>相同 RoPE 的 Eager 段计时，含 CPU 提交间隙</span></div><div><strong>458 段 × 4 口径</strong><span>每段每口径 20 次，输出精确一致</span></div></div>
<p class="note"><b>修正的是边界，不是把旧时间除以 kernel 数。</b> 每个完整段入口执行一次 256 MiB 缓存驱逐，之后连续计算，段内没有驱逐；驱逐本身不计时。它是经过探针验证的缓存状态近似，<b>不是已接通 PIM 的物理交接或完整 E2E 实测</b>。</p>
<section id="rope"><h2>01 / RoPE 的小 kernel 连续运行</h2><p>一次 Q/K RoPE 段完整包含两组 mul、neg、cat、mul、add。18 层分别实测，不再让 180 个应用 kernel 各自从冷状态开始。位置表生成仍单列一次：Graph 入口驱逐下为 <b>{r['graph_entry_cold']['position_generation_once_ms']:.4f} ms</b>；Q/K 应用加位置表的合计为 <b>{r['graph_entry_cold']['native_rope_combined_sum_ms']:.4f} ms</b>。这是各段均值合计，不是整段 AE 时间。</p><img src="01_rope_and_cache_scope.svg" alt="整段 RoPE 计时与入口一次驱逐的 DRAM 读取验证"><p>两 kernel 探针的输入为 1 MiB。地址复用、入口一次驱逐、每 kernel 驱逐的 DRAM 读取中位数分别为 <b>{p['reused']:,} / {p['entry_once']:,} / {p['per_kernel']:,} bytes</b>，支持段内中间结果得到缓存复用。只验证这类驱逐近似，不表示每个模型张量的全部字节都必然来自 DRAM。</p><p class="small">旧数据使用 NCU、新数据不使用 profiler，提交/计时方式也不同。因此不能把新旧差额全部解释成缓存收益，更不能由此直接宣称 PIM 加速比。仍未做 RoPE kernel 融合，Graph 只改变提交方式。</p></section>
<section id="blocks"><h2>02 / 原生逻辑块逐层测量</h2><p>保留原生精度、形状与实际张量 stride。Q/K/V 一组，Gate/Up 一组，RMSNorm 含全部归约/调制/cast，softmax 流程含缩放/mask/cast，GELU 与乘法一起测。矩阵、布局和外围算子也单列。</p><img src="02_native_blocks.svg" alt="原生逻辑块的 Eager 与 Graph 入口驱逐计时"><details><summary>完整基础分块表</summary>{table('native_blocks')}</details><p class="small">表中合计已经乘入实际调用次数；不能再乘 18，也不能把隔离测量的各块之和当作真实一步延迟。静态 suffix 掩码已预分配，CPU/H2D 创建不在分段计时内。</p></section>
<section id="interfaces"><h2>03 / 按真正的连续计算边界组合</h2><p>残差之后没有 PIM 运算打断时，把残差与下一次归一化连续执行。O 投影留 GPU 和放 PIM 是两个备选出口，不能同时收费。</p><img src="03_composite_boundaries.svg" alt="残差归一化和 GPU O 投影连续段"><p class="note"><b>以下组合替换重叠的基础行，不能重复相加。</b> 例如“FFN 残差 + 下一次归一化”覆盖下一层入口 norm；最后一层覆盖 final norm。入口第 0 层 norm 仍需单独一次。</p>{table('handoff_composites')}
<h3>FP32 逻辑接口候选</h3><p>另外提供 PIM FP32 输出转换、PV 两部分求和、以及 P 补齐到 992 列后按奇偶 KV 分片的连续段。每个 head/rank 的 P 输出为 [16,496] BF16。FP32 数值由原生输出构造，只验证 GPU 函数与 Graph 结果一致，<b>不是数值 PIM 仿真或物理 bank 布局校验</b>。</p><img src="04_fp32_interface_candidates.svg" alt="FP32 交接候选连续 GPU 计算段">{table('pim_interface_candidates')}
<p class="small">同学仓库 xinyao_1104@8764996 的参考方案已经把 RoPE 放在 PIM，不能再计 GPU RoPE；本页 RoPE 行供 GPU-RoPE 替代分工使用。P 分片包含逻辑 pad/gather/cast，不含实际 DMA 或 bank 地址写入。PV 求和假设两个已解包的稠密 FP32 buffer；物理输入解包仍未测。</p></section>
<section id="method"><h2>04 / 两种计时边界不能混用</h2><ul>
<li><b>Eager：</b>CUDA events 包围整个原生函数，保留 Python/CUDA 提交导致的 GPU 执行间隙；不是纯 kernel busy time。</li>
<li><b>Graph：</b>同一函数经过预热和输出校验后捕获，计时 event 位于 graph 内部（external=True），记录从段内首 event 到末 event 的 GPU 服务区间。CPU graph launch、graph 构建和外部同步不在其中；没有 torch.compile 或算子融合。</li>
<li><b>入口驱逐：</b>每次只有一遍 256 MiB .ca 读流，且在计时前完成。连 GPU 常量/权重缓存也会被扰动，不能称为选择性冷输入或真实 PIM 交接。地址复用列只作固定缓冲区控制，不代表完整模型自然缓存状态。</li>
<li>输入/权重读取、段内中间读写/cast、输出 store 的实际执行成本已包含；未测输出全部排空到 LPDDR、PIM 可见性、外部 DMA、dispatch、bank 解包/写入或与后台 VLM 的争用。</li>
<li>空 graph 的计时均值为 {d['empty_graph_event_ms']['mean']*1000:.3f} µs，作为参考保留，未做减法修正。非常小的段需考虑计时节点/调度粒度。</li>
<li>实际部署需要固定 buffer/每 step 的 graph 或参数更新；这些管理成本未计入图内服务时间。若混合系统使用 Graph，纯 GPU baseline 也应采用可比优化后再归因 PIM 收益。</li></ul>
<h3>今天重新测的正常模型控制组</h3><img src="05_normal_model_control.svg" alt="今天无 profiler 的完整模型 ViT VLM AE 分段"><p>完整模型仍是原生 eager、自然缓存，不启用模型级 Graph。clean wall p50 <b>{c['clean_wall_ms']['p50']:.3f} ms</b>；60 次 phase 的均值合计 {c['phase_total_ms']['mean']:.3f} ms，AE 均值 {c['phase_ms']['ae']['mean']:.3f} ms。不能把图内分块数字直接替换这一完整执行时间。</p>
<p>386 个主段加 72 个补充接口段，共 36,640 次计时，输出精确相等（torch.equal）。初始化失败的 01 记录保留并排除；有效主采集为 02，接口补充为 interfaces_01。没有修改共享运行时、权重、系统时钟或同学的仿真代码。</p>
<p><a href="../README.md">中文说明与复现</a> · <a href="../LATENCY_CN.md">E2E 统计简表</a> · <a href="../analysis_verified/summary.json">完整 JSON</a> · <a href="../analysis_verified/native_blocks.csv">基础块 CSV</a> · <a href="../analysis_verified/handoff_composites.csv">连续段 CSV</a> · <a href="../analysis_verified/pim_interface_candidates.csv">接口候选 CSV</a></p><p class="small">仓库：AlinaWXY/pim_vla_simulator<br>分支：<code>thor-data/pi05-openpi-mha8-h16-s10-segments-20261108</code><br>旧的 2026-11-05 数据保留，本批另存，不覆盖。</p></section></main></html>'''
    (out/'index.html').write_text(text)
    print(out/'index.html')


if __name__=='__main__':main()

"""Render the corrected conditional overlap evaluation and its audit trail."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'b':'feature / GPU RoPE / before / serial',
          'c':'feature / GPU RoPE / before', 'd':'token / GPU RoPE / before',
          'e':'token / local RoPE / before', 'f':'token / GPU RoPE / after',
          'g':'token / local RoPE / after'}
CHINESE = {'b':'feature + GPU RoPE，融合前串行', 'c':'feature + GPU RoPE，融合前并行',
           'd':'token + GPU RoPE，融合前并行', 'e':'token + 本地 RoPE，融合前并行',
           'f':'token + GPU RoPE，融合后并行', 'g':'token + 本地 RoPE，融合后并行'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    out = a.output
    def rows(name):
        with (out/name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    mainrows = rows('evaluation-b-g.csv')
    alt = {r['case']:r for r in rows('shared-tensor-io-accounting.csv')}
    data = {r['case']:r for r in mainrows}
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    f = lambda r,k:float(r[k])
    table = '\n'.join('| '+r['case']+' | '+CHINESE[r['case']]+' | '+
                      ' | '.join(f'{f(r,k):.6f}' for k in ['AE_solo_ms','AE_gpu_wait_ms',
                         'AE_with_interference_ms','VLM_progress_during_PIM_ms','total_ms'])+' |' for r in mainrows)
    alternate = '\n'.join(f"| {r['case']} | {f(r,'total_ms'):.6f} | {f(alt[r['case']],'total_ms'):.6f} |" for r in mainrows)
    e,g=data['e'],data['g']
    delta=f(e,'total_ms')-f(g,'total_ms')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),gridspec_kw={'width_ratios':[1.2,1]})
    y=np.arange(6)
    axes[0].barh(y,[f(r,'total_ms') for r in mainrows],color=['#8497a8','#e5ab53','#e5ab53','#d58d38','#4f9c87','#2b806d'])
    for i,r in enumerate(mainrows):
        axes[0].text(f(r,'total_ms')+1,i,f"{f(r,'total_ms'):.3f}",va='center',fontsize=9)
    axes[0].set_yticks(y,[r['case']+'  '+LABELS[r['case']] for r in mainrows],fontsize=9)
    axes[0].invert_yaxis();axes[0].set_xlim(0,max(f(r,'total_ms') for r in mainrows)*1.14)
    axes[0].set_xlabel('VLM + AE completion (ms)')
    axes[0].set_title('Corrected baseline; explicit native transfers',loc='left',fontsize=11)
    solo=np.array([f(r,'AE_solo_ms') for r in mainrows]);wait=np.array([f(r,'AE_gpu_wait_ms') for r in mainrows])
    axes[1].barh(y,solo,color='#628bad',label='AE alone')
    axes[1].barh(y,wait,left=solo,color='#ba7695',label='AE waits for active VLM kernel')
    axes[1].set_yticks(y,[r['case'] for r in mainrows]);axes[1].invert_yaxis()
    for i,value in enumerate(solo+wait):axes[1].text(value+1,i,f'{value:.3f}',va='center',fontsize=9)
    axes[1].set_xlim(0,max(solo+wait)*1.16);axes[1].set_xlabel('AE completion (ms)')
    axes[1].set_title('GPU interference is charged to AE',loc='left',fontsize=11)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8,ncol=1)
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('RETRACTED as performance evaluation — historical idle-service calculation',fontsize=12)
    fig.tight_layout();fig.savefig(out/'evaluation-b-g.png',dpi=180);fig.savefig(out/'evaluation-b-g.svg');plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(13.5,5.4),sharex=True)
    for ax,case in zip(axes,['e','g']):
        v=data[case]['variant'];prefix=v+'-explicit_native_io-'
        execution=rows(prefix+'execution.csv');placements=rows(prefix+'VLM-placement.csv');waits=rows(prefix+'AE-GPU-wait.csv')
        limit=500.0
        for r in execution:
            start,end=f(r,'begin_us'),f(r,'end_us')
            if start>=limit:break
            color='#628bad' if r['kind']=='gpu' else '#dfaa58'
            ax.broken_barh([(start,min(end,limit)-start)],(1.2,.55),facecolors=color)
        for r in waits:
            start,end=f(r,'ready_us'),f(r,'gpu_begin_us')
            if start>=limit:break
            ax.broken_barh([(start,min(end,limit)-start)],(1.2,.55),facecolors='#ba7695')
        for r in placements:
            start,end=f(r,'start_us'),f(r,'end_us')
            if start>=limit:break
            ax.broken_barh([(start,min(end,limit)-start)],(.25,.55),facecolors='#89b3a7',edgecolors='white',linewidth=.3)
            if min(end,limit)-start>18:
                ax.text((start+min(end,limit))/2,.525,'K'+r['ordinal'],ha='center',va='center',fontsize=7)
        ax.set_yticks([.525,1.475],['VLM in flight\n(includes bus stalls)','AE/PIM path'],fontsize=9)
        ax.set_ylim(0,2);ax.set_title(case+'  '+LABELS[case],loc='left',fontsize=10);ax.grid(axis='x',alpha=.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#628bad',label='AE GPU'),Patch(color='#dfaa58',label='Native PIM'),
                        Patch(color='#ba7695',label='AE waiting for active VLM')],loc='lower center',ncol=3,fontsize=8)
    axes[-1].set_xlim(0,500);axes[-1].set_xlabel('Time from AE start (us)')
    fig.suptitle('Historical scheduler trace — GPU progress model lacks delayed DQ reservations',fontsize=12)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'e-g-first-500us.png',dpi=180);fig.savefig(out/'e-g-first-500us.svg');plt.close(fig)
    report=f'''# b–g：修正基线上的跨窗口 VLM+AE 延迟

**撤回说明：以下 c–g 并行 latency、overlap 及 e/g 差值只保留为历史算法输出，不再作为有效 Evaluation。** “任何正长 DQ/CA joint idle 都贡献 GPU 进度”没有检查 CA 发出 RD 后、CL 延迟处的完整 DQ burst 是否与 PIM 冲突。数值守恒、kernel 顺序和 AE 等待检查通过，并不证明 DRAM 请求可以被调度。串行 b 及修正后的原生 PIM/DQ/CA 数据不因这一缺陷失效，其原有 I/O 计费限制仍保留。

已在 e 的实际 QKV trace（channel 0）确认反例：CK 3–4 的 CA/DQ 当前均空闲；若 CK 3 发出一个 row-hit RD，nCL=23 CK、nBL_min=2 CK 要求 CK 26–28 的 DQ，但 PIM WRGB 在 CK 25–27 已占用 DQ。该 RD 在保持 PIM 原排程时不能按 CK 3 发射。诊断见 [audit.json](../delayed-read-audit/audit.json)，逐阶段计数见 [CSV](../delayed-read-audit/independent-read-candidates.csv)。各 CK 是独立假想候选，不是实际 GPU 请求序列；通过该检查也不代表满足 ACT/PRE、bank/rank、读写转向及控制器准入约束。

后续必须联合检查命令时段与未来数据 burst，按依赖推进 kernel；不能把当前 joint idle 面积直接当作 GPU 计算量，也不能改用任意窗口阈值或恢复整 kernel 单窗口限制。CA 与 DQ 可以分处两个不同空隙，二者之间不必整段同时空闲。以下为撤回前的完整记录。

**按用户确认的调度近似完成重算。没有使用 ρ、在途请求数、读写比，也没有整 kernel 单窗口容纳限制。** 主表保留全部原生传输时间。e 为 **{f(e,'total_ms'):.6f} ms**，g 为 **{f(g,'total_ms'):.6f} ms**，g 减少 **{delta:.6f} ms（{delta/f(e,'total_ms')*100:.3f}%）**。这是给定窗口推进规则的计算结果，不是硬件端到端测量或逐请求仲裁仿真。

原生输入来自最终正确作业 **1971385**；本次重放作业 **{a.job}**。before 的 b/c/d/e 没有 rank controller、没有输入 DRAM 暂存，计算期间通过 WRGB 持续补给输入；f/g 包含 DRAM 暂存、本地读取、rank controller 和粗命令。a 和已有 VLM 不重跑。

## 主表

VLM 固定沿用已有 ViT + language 测量：**{summary['VLM_measured_ms']:.9f} ms**。单位全部为 ms。

| 项 | 配置 | AE 独立执行 | 等待在途 VLM kernel | 含干扰的 AE 完成 | PIM 期间 VLM 推进 S | VLM+AE 总完成 |
|---|---|---:|---:|---:|---:|---:|
{table}

![总延迟与AE等待](evaluation-b-g.png)

“AE 独立执行”由 18 层 × 10 步原生阶段与保留的 GPU kernel ledger 拼接，不是实机 AE 墙钟时延。b 为串行，因此没有 VLM overlap 或 AE 等待。

## 调度与计费

保留 VLM 的 2357 个有序 work units（2353 个 kernel、4 个已有 copy），不越过前驱。第一次出现正长可用窗口就启动当前 work unit；不要求整个 kernel 放进窗口，不设置最小窗口长度。PIM 期间，仅在全部 16 channel 的外部 DQ/CA 同时空闲、且 GPU rank 0 未处于刷新阻塞时，按原 kernel 速度累计有效进度。占用期间保留当前 kernel 的剩余工作，下一窗口继续。

遇到 AE GPU 操作，如果当前 VLM kernel 尚未完成，先完成其剩余部分，再执行 AE GPU 操作；AE 连续 GPU 操作之间不插入新的 VLM kernel。PIM 片段结束但后续仍是 PIM 时，不强制排空 VLM kernel。AE 的源 kernel 序号和 VLM 顺序均检查单调；所有实际开始时间随等待顺延。

设 AE 独立时间为 A，VLM 独立时间为 V，PIM 期间推进为 S，AE 因在途 VLM 而等待为 W，则：

`AE_with_interference = A + W`

`VLM_progress_during_AE = S + W`

`T = (A + W) + V − (S + W) = A + V − S`

W 已计入 AE 完成时间，同时这段等待也实际推进 VLM，不能在上述总延迟上再加一次 W。本次所有并行组的 VLM 在 AE 完成时仍未结束，正长窗口均被利用；因此顺序和尾部等待改变 AE 完成位置，但在本次规则下，总完成时间满足上式。逐 kernel 的跨窗口记录、逐 AE 阻塞记录与移动后的完整时间线均保存为 CSV。

e：`{f(e,'AE_with_interference_ms'):.6f} + {f(e,'VLM_ms'):.6f} − {f(e,'VLM_progress_during_AE_ms'):.6f} = {f(e,'total_ms'):.6f}`。

g：`{f(g,'AE_with_interference_ms'):.6f} + {f(g,'VLM_ms'):.6f} − {f(g,'VLM_progress_during_AE_ms'):.6f} = {f(g,'total_ms'):.6f}`。

![首500us中的跨窗口kernel和AE等待](e-g-first-500us.png)

绿色条表示 kernel 在途区间，**包含被总线阻塞的时间**，不能把整条长度视为有效 GPU 工作量。

## e/g 为什么仍有较多抵消

g 比 e 的独立 AE 时间增加 {f(g,'AE_solo_ms')-f(e,'AE_solo_ms'):.6f} ms；PIM 期间 VLM 可推进量增加 {f(g,'VLM_progress_during_PIM_ms')-f(e,'VLM_progress_during_PIM_ms'):.6f} ms，因此总延迟净减少 {delta:.6f} ms。该近似允许 before 的每个细碎空隙以零恢复成本贡献有效 GPU 进度，所以它没有量化碎片化导致的真实请求停顿、带宽效率和指令恢复成本。不能据此断言真实硬件上 g 仅改善这么多，也不能人为把 e 的 CA/DQ 改为全时占满来得到期望的差距。

## 传输口径与新增 GPU 边界

主表完整保留 native 的 GPU RoPE 导出/导入、after 输入 DRAM 写入和 score 读取阶段，避免把实际阶段静默设为零。既有 GPU kernel 时长自身也包含其访存，边界共享张量的读写可能部分重复计费。为与旧口径可比，另给出此前“同一 tensor 的 ST/RD 归属 GPU producer/consumer、从 native latency 去重”的独立附表；物理原始流量不会随去重改变：

| 项 | 主表：原生传输全部保留 | 共享张量 I/O 去重附表 |
|---|---:|---:|
{alternate}

两表是明确的计费选择，不是统计置信区间，也不保证包住真实硬件延迟。主表与附表不得混列比较。

修正 before 的 PV→O 路径需先将 PV 结果导出 GPU，再按 O 输入顺序广播两次。本次提供显式 GPU `transpose(0,1).contiguous()` 实现，将 query/feature-packet 顺序互换、保持 channel 和 BF16 lane，生成的缓冲供两次 O 扫描复用。Thor 实测 feature 为 **{summary['handoff_us']['feature_direct_PV_to_O_pack']:.6f} µs**，token 为 **{summary['handoff_us']['token_direct_PV_to_O_pack']:.6f} µs**，每层每步一次；在 PV 导出完成与第一个 O 广播之间插入独立 GPU 依赖节点。该显式重排实现不代表所有驱动必然需要一次 copy。保留已有 O 求和及 Gate/Up、Down 类型转换测量。

新测量仅涉及这两个边界算子。每项 60 次 device-kernel profiling，使用与已有 supplemental handoff 相同的 32 MiB 清缓存流程，取中位数，并检查全部元素的重排正确性。不是重新运行纯 GPU 或 VLM baseline。

## 来源与验证

VLM 仍使用已有 phase mean；原始 prefix/setup 有序 trace 总和为 105.2205782 ms，统一乘 **{summary['VLM_profile_scale']:.12f}** 使总服务量匹配 {summary['VLM_measured_ms']:.9f} ms。这个比例仅用于对齐两次既有计时口径，不生成访存量。未归一化结果单列在 [profile-duration-check.csv](profile-duration-check.csv)。原 trace 未单独标记严格 VLM-only kernel 区间，因此该对应关系仍是现有 profiling 的限制。

刷新掩码来自同一次 native CA 日志：REFab 外部 CA 命令与目标 rank 的 nRFC 阻塞分开处理。只有 rank 0 的刷新额外阻塞 GPU；rank 1 的内部刷新不直接全程占据外部 CA。nRFC 为 224 CK，tCK 为 0.938 ns。

8 个调度行为测试覆盖：短窗口累计、长 kernel 阻塞后续 AE、禁止后发短 kernel 超越、连续 PIM 片段继续执行、无窗口不发射、VLM 提前完成、极短正窗口和最终 PIM 尾部。20 个原生阶段 × 2 种 I/O 口径的重放检查已通过，核对物理时间守恒、GPU 顺序、等待与工作量守恒；native 原始配置/流量已有 [修正基线校验](../direct-before-corrected/checks.json)。

该模型固定 PIM 优先级并复用独立原生阶段，GPU 不反过来改变 PIM 命令排程；插入 GPU 操作后的全局 refresh 相位不重新仿真。没有模拟 host launch gap、上下文切换或真实地址请求。由于这些限制，本表明确作为用户选定规则下的**窗口推进近似**，不恢复已撤回的合成请求结果。

另用独立区间积分程序复核保存结果：从原始可用窗口与每个 kernel 的实际在途区间取交集，逐项核对有效推进量、尾部等待、AE GPU 与 VLM 不重叠和 7 个代码/输入哈希。10 组重放均通过，见 [独立核验](independent-audit.json)。

主要数据：[主表](evaluation-b-g.csv)、[I/O 附表](shared-tensor-io-accounting.csv)、[模型与哈希](summary.json)、[测试记录](unit-tests.log)、[校验](checks.json)。所有新结果保存在 dev0：`/fact_data/xinyaowang/pim_vla_xinyao_1108_20261108/reference/evaluation-1108/corrected-overlap`。

复现：在服务器仓库根目录提交 `sbatch experiments/pi05-unified-gemv/corrected-overlap.sbatch`；绘图和报告运行 `python experiments/pi05-unified-gemv/publish_corrected_overlap.py --output reference/evaluation-1108/corrected-overlap --job <job-id>`。
'''
    (out/'REPORT.zh-CN.md').write_text(report,encoding='utf-8',newline='\n')
    hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(table)


if __name__=='__main__':main()

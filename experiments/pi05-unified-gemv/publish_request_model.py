"""Publish conditional request-replay estimates and the explicitly assumed model."""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'feature_gpu_before':'Feature / GPU RoPE / before','token_gpu_before':'Token / GPU RoPE / before',
        'token_local_before':'Token / local RoPE / before','token_gpu_after':'Token / GPU RoPE / after','token_local_after':'Token / local RoPE / after'}
def read_csv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args();out=a.reference/'request-model'
    s=json.loads((out/'summary.json').read_text());rows=s['rows'];central=[r for r in rows if r['profile']=='central']
    assert len(rows)==70 and len(central)==5
    summaries=[]
    for c in central:
        rs=[r for r in rows if r['variant']==c['variant'] and r['pressure']>0]
        lo=min(rs,key=lambda r:r['total_ms']);hi=max(rs,key=lambda r:r['total_ms'])
        summaries.append(dict(variant=c['variant'],central_ms=c['total_ms'],min_ms=lo['total_ms'],max_ms=hi['total_ms'],min_profile=lo['profile'],max_profile=hi['profile']))
        for r in rs:
            assert abs(r['total_ms']-(r['AE_solo_ms']+r['VLM_ms']-r['effective_saving_ms']))<1e-8
            assert abs(r['effective_saving_ms']-(r['VLM_advanced_ms']-r['AE_wait_ms']))<1e-8
    with (out/'sensitivity-range.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    for i,r in enumerate(summaries):
        ax.plot([r['min_ms'],r['max_ms']],[i,i],lw=7,color='#d4e2ec',solid_capstyle='round')
        ax.plot(r['central_ms'],i,'o',color='#176391',ms=8)
        ax.text(r['max_ms']+.35,i,f"{r['central_ms']:.3f}  [{r['min_ms']:.3f}, {r['max_ms']:.3f}]",va='center',fontsize=9)
    ax.set_yticks(range(5),[f'{letter}  '+LABELS[r['variant']] for letter,r in zip('cdefg',summaries)])
    ax.invert_yaxis();ax.set_xlim(143,179);ax.set_xlabel('VLM + AE latency / ms')
    ax.set_title('Parameterized request replay: central value and scanned range\nConditional model estimates, not hardware measurements or confidence intervals',loc='left',fontsize=11)
    ax.grid(axis='x',alpha=.2);fig.savefig(out/'request-latency.png',dpi=190);fig.savefig(out/'request-latency.svg');plt.close(fig)
    # Reconstruct actual AE/PIM wall times using the recorded AE wait at every PIM boundary.
    fig,axes=plt.subplots(2,1,figsize=(12,5.5),layout='constrained',sharex=True)
    for ax,v in zip(axes,('feature_gpu_before','token_local_after')):
        raw=read_csv(a.reference/(v+'-schedule.csv'));seq=[]
        for r in raw:
            d=float(r['end_us'])-float(r['begin_us'])
            if r['kind']=='pim' and seq and seq[-1][0]=='pim':seq[-1][1]+=d
            else:seq.append([r['kind'],d])
        waits={int(r['block_index']):float(r['AE_wait_us']) for r in read_csv(out/(v+'-AE-waits.csv'))}
        clock=0;tracks={0:[],1:[],2:[],3:[]}
        def add(track,a,b):
            if a<600 and b>0:tracks[track].append((max(0,a),min(600,b)-max(0,a)))
        for i,(kind,d) in enumerate(seq):
            add(0 if kind=='gpu' else 1,clock,clock+d);clock+=d
            delay=waits.get(i,0);add(3,clock,clock+delay);clock+=delay
        for r in read_csv(out/(v+'-kernel-execution.csv')):add(2,float(r['begin_us']),float(r['end_us']))
        for y,color in enumerate(('#707b91','#488d73','#427ba6','#d7954e')):
            ax.broken_barh(tracks[y],(y-.32,.64),facecolors=color)
        ax.set_yticks(range(4),['AE on GPU','PIM','VLM on GPU','AE waiting']);ax.invert_yaxis();ax.set_xlim(0,600)
        ax.set_title(LABELS[v]+' | central model, first 600 us',loc='left',fontsize=10)
    axes[-1].set_xlabel('Simulated wall time / us; VLM may extend beyond PIM completion')
    fig.savefig(out/'continuous-kernel-timeline.png',dpi=190);fig.savefig(out/'continuous-kernel-timeline.svg');plt.close(fig)
    e=read_csv(out/'evaluation-a-g.csv')
    evaltable=table(['项','配置','VLM+AE / ms','来源'],[[r['case'],'纯 GPU' if r['case']=='a' else LABELS.get(r['variant'],r['variant'])+(' / serial' if r['case']=='b' else ''),f"{float(r['total_ms']):.6f}",'已有 GPU 测量' if r['case']=='a' else '已有串行已知项模型' if r['case']=='b' else '本次参数化请求模型'] for r in e])
    parts=table(['配置','AE单独 / ms','AE等待 / ms','提前完成VLM工作 / ms','净节省 / ms','合计 / ms'],[[LABELS[r['variant']],*[f"{r[k]:.6f}" for k in ('AE_solo_ms','AE_wait_ms','VLM_advanced_ms','effective_saving_ms','total_ms')]] for r in central])
    ranges=table(['配置','中间配置 / ms','扫描范围 / ms','最小值配置','最大值配置'],[[LABELS[r['variant']],f"{r['central_ms']:.6f}",f"{r['min_ms']:.6f}–{r['max_ms']:.6f}",r['min_profile'],r['max_profile']] for r in summaries])
    c={r['variant']:r for r in central};h={r['variant']:r for r in rows if r['profile']=='high'}
    report=f'''# xinyao_1108：允许 kernel 跨窗口推进的参数化请求仲裁

日期：2026-11-08。用户指定取消整kernel装入单窗口的限制，并接受明确假设下的参数化访存敏感性分析。最终运行job **1971324**；复用native job1971246及已有GPU/VLM测量，没有重测纯GPU或VLM baseline。本报告替代此前整kernel窗口装箱的并行主表；原始DQ流量与native时间线不变。

**c–g是条件性模型估计，不是采集到的真实GPU访存trace或硬件端到端测量。扫描范围也不是置信区间或真实性能上下界。** 原有Nsys记录只有kernel顺序和时长；cold AE NCU记录没有DRAM请求量；既有Thor探测未暴露DRAM读写计数器。没有将L2流量冒充DRAM流量。

## 新的a–g主表

{evaltable}

中间配置采用请求压力ρ=0.30、每channel最多32个在途请求、读写80/20、GPU与PIM DQ切换保护2CK、额外门控开销0µs。此配置只是参数扫描中的参考点，**ρ不是测得的GPU带宽利用率**。

![参数化延迟及范围](request-latency.png)

## 对后续AE GPU算子的干扰已计入

AE GPU算子就绪后优先调度，但不抢占正在运行的VLM kernel。VLM kernel可以跨多个DQ/CA忙闲片段，也可以跨过PIM阶段结束点；后一种情况会让AE GPU算子等待。AE与VLM不同时占用SM，这是明确的调度策略，不是宣称硬件不支持并发kernel。

使用下式核算，不能把VLM完成的全部工作都当成免费重叠：

```text
AE_parallel = AE_solo + AE_wait
T_total = AE_parallel + (VLM_baseline - VLM_advanced)
net_saving = VLM_advanced - AE_wait
```

`VLM_advanced`是AE完成前已执行VLM工作对应的基线等效时长，包含AE等待期间的推进；`AE_wait`必须扣回来。所有70组均验证上述等式。各kernel的实际执行区间、跨PIM边界情况和每次AE等待分别在`*-kernel-execution.csv`、`*-AE-waits.csv`。

{parts}

例如g：AE等待{c['token_local_after']['AE_wait_ms']:.6f}ms，VLM提前完成{c['token_local_after']['VLM_advanced_ms']:.6f}ms工作，净节省{c['token_local_after']['effective_saving_ms']:.6f}ms，总延迟{c['token_local_after']['total_ms']:.6f}ms。因此单独AE响应时间变长，而本模型中的VLM+AE总完成时间缩短；这两项不能混为一谈。

![持续执行及AE等待](continuous-kernel-timeline.png)

## 请求与总线如何推进

1. 保持已有2357个VLM工作单元的严格顺序（2353个kernel及4个内存操作），不跳过依赖。原始prefix时长和为{s['raw_prefix_ms']:.9f}ms；为与已有VLM phase mean 104.950496ms一致，统一乘以{s['prefix_duration_scale']:.12f}。该缩放只做总量校准，不是重新测量kernel。
2. 按每个channel生成32B、目标rank0的GPU请求。每个channel独立查找可用CA与DQ时段，没有“全部16 channels必须同时空闲才能启动”的判定。完全相同的channel日历可共享计算结果，等价性逐事件比较，不假设所有channel天然相同。
3. 每个GPU读请求占1CK CA，并在nCL=23CK后占2CK DQ；写请求使用nCWL=12CK。请求的CA和数据不需要同时空闲。GPU请求不覆盖native PIM的既有预约，PIM优先；GPU侧保留请求队列、在途数限制和背压。
4. 读取全部320份native CA日志的REFab事件，按目标rank生成nRFC=224CK blackout。rank1 refresh仅通过其1CK共享CA脉冲影响GPU，不能把rank1的224CK作为rank0的封锁。rank0 refresh期间GPU请求等待。模板结束后按nREFI=4164CK继续产生refresh；已有日志和bus模板哈希逐文件一致。
5. kernel完成需要同时满足名义kernel时长已推进、其全部请求已完成。没有单窗口长度门槛；仍保留每个32B burst的完整预约。模型kernel可以因访存竞争变长。

这是一套**固定PIM优先预约的重放模型**。PIM阶段被AE依赖推迟时，其native相对时序整体平移。它不是GPU与PIM联合重新运行完整Ramulator控制器；各native阶段本来独立运行，阶段间的完整DRAM状态及全局refresh相位未重建。

## 参数化访存与敏感性

每kernel、每channel的初始burst数为`floor(ρ × eligible_span / nBL_min)`，其中`eligible_span=max(0,Tkernel−(nRFC+nCL+nBL+16CK))`。请求在该区间均匀到达；20%写采用64读/16写一组。因没有真实地址、缓存命中和请求时间，均匀流和读写比例都是模型假设。最末端的保护区间是校准假设，不是测得的kernel尾部结构。

如果合成请求在固定单独运行参考相位下不能于名义kernel时间内完成，降低burst数。**该校准仅按kernel时长和访存参数缓存一次，同一kernel在所有layout中携带完全相同的需求；不会因某个layout竞争更强而减少其请求数。** `calibration_trim_bursts`保存修正量。kernel基线时长作为服务下限，请求可与其重叠；未重建kernel内部计算—访存依赖，因此本模型可能低估中途访存等待对计算关键路径的影响。

扫描14套参数×5种layout/RoPE组合，共70组：ρ为0、0.10、0.30、0.60；另对0.30/0.60分别扫描在途数8/128（默认32）、每次额外门控1µs（默认0）、DQ切换保护8CK（默认2）、全读（默认80/20）。这是单因素扩展，**不是所有参数的笛卡尔积**。ρ=0仅作为无访存竞争诊断对照，不纳入下面范围。

{ranges}

中间配置下，融合后相对融合前：GPU RoPE缩短{c['token_gpu_before']['total_ms']-c['token_gpu_after']['total_ms']:.6f}ms，本地RoPE缩短{c['token_local_before']['total_ms']-c['token_local_after']['total_ms']:.6f}ms。ρ=0.60、其余默认时，对应缩短{h['token_gpu_before']['total_ms']-h['token_gpu_after']['total_ms']:.6f}ms、{h['token_local_before']['total_ms']-h['token_local_after']['total_ms']:.6f}ms。**收益明显依赖GPU访存假设，不能把单一中间配置当成已证实的硬件收益。**

`generated_GPU_bytes`是已推进VLM部分的合成GPU请求量，不是实测DQ；推进的kernel数量不同，累计字节数自然不同。原布局对比5904/4968/4776KiB仍只统计PIM接口物理流量，不把新合成GPU流量混入。各`*_wait_request_sum_ms`是逐请求等待之和，可在channel和在途请求之间重叠，不能当成kernel或端到端latency相加。

## 能说明什么、还缺什么

本次落实了kernel顺序、逐channel/请求/burst的资源预约、目标rank refresh、kernel跨窗口执行，以及对AE GPU算子的非抢占等待。它说明旧的“整kernel装入窗口”导致的近零收益不是这套请求仲裁策略的结果。

未测量或未建模：真实GPU请求地址及到达时序、缓存行为、kernel内部访存/计算依赖、ACT/PRE和完整行状态、真实SM资源共享/抢占、跨设备coherence与fence、host关键路径、原AE最终rank复制布局对GPUkernel的影响。参数范围只覆盖列出的假设。不能据此声称精确的硬件E2E，也不能把同一输入的VLM→AE依赖自动视为可并行；本报告延续用户指定的VLM/AE重叠核算范围。

## 验证与复现

10项测试通过，包括按channel独立仲裁、CA与DQ分时服务、rank0/rank1 refresh区分、请求队列限制、单独运行校准、跨PIM边界导致AE等待、kernel顺序、跨layout请求量一致。每个合成burst在执行时断言不覆盖PIM预约、不跨目标rank refresh、队列不超限、同channel数据不重叠。70组完整运行成功；原native/GPU原始数据未变。

```bash
sbatch experiments/pi05-unified-gemv/request-model.sbatch
python experiments/pi05-unified-gemv/publish_request_model.py --reference reference/evaluation-1108
```

服务器工作树：`/fact_data/xinyaowang/pim_vla_xinyao_1108_20261108`。共享库在`results/request-model/librequest_replay.so`；其哈希、源码/输入哈希在[summary.json](summary.json)。[a–g主表](evaluation-a-g.csv)、[70组全精度数据](sensitivity.csv)、[范围及对应参数](sensitivity-range.csv)、[测试](tests.log)。
'''
    (out/'REPORT.zh-CN.md').write_text(report,encoding='utf-8')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
if __name__=='__main__':main()

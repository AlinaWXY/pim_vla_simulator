# xinyao_1103：LOCALWR ACT1/ACT2 调度修正

主五阶段 latency：**187.376756 → 187.293274 µs**，变化 **-0.083482 µs**。本轮完整 ACT1/ACT2 审计通过，前版 projection spill 的违规已消除。结果来自修改前后两套独立 backend 的重新运行。

## 修复范围与保留条件

在 `a987650` 的 staging 修复之上，共用内部激活管理 `pim_activation_extension.inc`：staging、projection spill/RoPE 的 LOCALRD/LOCALWR、BGK 本地写的行准备及 FFN Z 内部访问均接入相同的 ordinary lookahead、bank owner 与 deadline。ACT1 只有在 ACT2 能及时跟上时发出；ACT2 pending 时阻止其它本地扫描抢占 CA，维护请求继续排队。ACT2 完成不提前结束数据请求，原读写 service、写恢复、子请求和父描述符完成依赖保留。

SRF仍为每bank 2×512 B；输入暂存仍为BG0/B0与BG1/B0、各64 KiB；trace、地址、计算/数据量和所有生产DRAM时序参数均不变。rank CA、外层CA、DQ和bank阵列约束继续执行。新增guard使用更频繁刷新仅用于对抗测试，不改变七个工作负载的nREFI。

## 各阶段 latency

单位µs/层/denoising step，H16、batch1、16 channel。

| 阶段 | 修正前 a987650 | 修正后 | 变化 |
|---|---:|---:|---:|
| Q/K + RoPE + QKT | 41.969872 | 41.886390 | -0.083482 |
| V | 20.592852 | 20.592852 | +0.000000 |
| PV | 18.994500 | 18.994500 | +0.000000 |
| Gate/Up | 47.127934 | 47.127934 | +0.000000 |
| Down | 58.691598 | 58.691598 | +0.000000 |
| **主五阶段合计** | **187.376756** | **187.293274** | **-0.083482** |

combined FFN控制：105.910518 → 105.910518 µs；单层连续控制：188.328826 → 187.887966 µs。两个控制实验不重复计入主五阶段。

## Q/K链的子阶段

| 子阶段 | 修正前 | 修正后 | 变化 |
|---|---:|---:|---:|
| projection_gpu_input_stage | 1.245693 | 1.245693 | +0.000000 |
| projection_local_input_compute | 22.533574 | 22.533574 | +0.000000 |
| projection_local_spill | 0.385518 | 0.347998 | -0.037520 |
| projection_gemv_release | 0.002814 | 0.002814 | +0.000000 |
| rope_local | 0.885472 | 0.885472 | +0.000000 |
| q_export_to_gpu | 4.029648 | 4.012764 | -0.016884 |
| q_remote_buffer_fill | 0.481194 | 0.481194 | +0.000000 |
| qk_tile_0_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_0_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_1_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_1_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_2_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_2_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_3_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_3_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_4_compute | 0.530908 | 0.528094 | -0.002814 |
| qk_tile_4_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_5_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_5_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_6_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_6_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_7_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_7_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_8_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_8_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_9_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_9_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_10_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_10_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_11_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_11_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_12_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_12_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_13_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_13_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_14_compute | 0.490574 | 0.487760 | -0.002814 |
| qk_tile_14_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_15_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_15_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_16_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_16_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_17_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_17_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_18_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_18_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_19_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_19_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_20_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_20_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_21_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_21_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_22_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_22_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_23_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_23_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_24_compute | 0.559048 | 0.535598 | -0.023450 |
| qk_tile_24_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_25_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_25_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_26_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_26_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_27_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_27_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_28_compute | 0.285152 | 0.285152 | +0.000000 |
| qk_tile_28_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_29_compute | 0.244818 | 0.244818 | +0.000000 |
| qk_tile_29_export | 0.119126 | 0.119126 | +0.000000 |
| qk_tile_30_compute | 0.277648 | 0.277648 | +0.000000 |
| qk_tile_30_export | 0.119126 | 0.119126 | +0.000000 |
| qk_gemv_release | 0.001876 | 0.001876 | +0.000000 |

其余阶段及边界时间见 [完整phase对照](reference/localwr-act2/phase-latency.csv)。子阶段取barrier之间的墙钟时间，不把重叠请求延迟相加；变化包括调度、行状态及refresh相位的相互影响。

本次Q/K链共缩短89 CK，其中spill本身缩短40 CK，下游Q export/QKT合计缩短49 CK。旧日志中，最后一个bank的ACT2 pending期间，channel0其它bank的LOCALWR连续占用rank CA 193 CK（[25460, 25653)）；修复后由已登记的owner优先发ACT2。这个区间证明了CA竞争，但其中仍可能有DRAM恢复等待，不能把整段都当作ACT2已ready后的stall，也不能把下游变化全归因于ACT开销。[逐bank调度证据及因果边界](reference/localwr-act2/spill-scheduling.json)。

## DQ 空闲时间 breakdown

下表为channel0，全部16channel均独立重算并闭合；完整无传输窗口已扣除边界数据尾部。

| 阶段 | DQ占用 | 修正前DQ空闲 | 修正后DQ空闲 | 修正后完整无传输窗口 | 修正后其它DQ空闲 |
|---|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 3.301760 | 38.668112 | 38.584630 | 32.448234 | 6.136396 |
| V | 3.121664 | 17.471188 | 17.471188 | 16.065126 | 1.406062 |
| PV | 2.341248 | 16.653252 | 16.653252 | 15.588622 | 1.064630 |
| Gate/Up | 3.842048 | 43.285886 | 43.285886 | 40.757038 | 2.528848 |
| Down | 7.924224 | 50.767374 | 50.767374 | 47.638206 | 3.129168 |
| **合计** | **20.530944** | **166.845812** | **166.762330** | **152.497226** | **14.265104** |

外部流量仍为5472 KiB/层/步（16channel合计）。[逐channel DQ/CA前后对照](reference/localwr-act2/channel-dq-comparison.csv)、[按阶段类别分解](reference/localwr-act2/phase-kind-dq-comparison.csv)、[完整JSON](reference/localwr-act2/localwr-act2-comparison.json)。DQ空闲是模型中的资源窗口，尚未注入并发GPU请求。

完整无传输窗口合计152.563824 → 152.497226 µs；其它DQ空闲14.281988 → 14.265104 µs。两者之和为DQ空闲总时间。

另按总CA状态拆分：DQ空闲且总CA也空闲的时间为158.674894 → 158.591412 µs；DQ空闲但总CA忙的时间为8.170918 → 8.170918 µs。这是另一种分区方式，不与上述两项叠加；即使DQ/总CA同时空闲，GPU实际可服务性仍受目标rank/bank时序和请求内容约束。

## 完整协议审计与控制实验

主五阶段全部16channel共10992对ACT1/ACT2通过严格检查，零超期、孤立、取消或末尾未完成owner。两个控制案例也通过相同检查。检查范围为本轮测量日志中的ACT1/ACT2 owner、deadline及干扰，不代表全部DRAM物理规格验证。旧主流程违规为 `{"chain": {"ACT1_while_channel_ACT2_pending": 240, "late_ACT2": 256}}`，新结果对应计数均为0。[完整协议前后证据](reference/localwr-act2/localwr-act2-protocol-audit.json)。

新增 `Local_ACT2_spill_rope_refresh` guard 对每个bank强制spill和RoPE STORE换行，通过真实REF和16bank细请求检验CA互斥、完整LOCALWR数量、owner及deadline。原生job 1903036通过 **104项Python测试、39项native guards、21个完整案例**（before/corrected/direct各7个）。修正前重跑与归档a987650的周期、phase、trace、字节、原始事件CSV哈希和channel分区完全一致。

共同内部路径的修正也适用于direct-input模式；它不是保留旧非法调度的开关。该模式主五阶段 109.762884 → 109.762884 µs，工作量不变且完整ACT检查通过：[direct控制对比](reference/localwr-act2/direct-input-control.json)。需要复现历史1101或a987650日志时必须使用对应历史checkout/backend。

## 备份与复现

修正前完整Git bundle、source.zip及job1902838原始结果备份于本地 `backups/xinyao_1103_before_localwr_act2_a987650/`，附SHA256；紧凑旧结果位于 [previous-localwr-act2](reference/previous-localwr-act2/summary.json)。[修改前复现校验](reference/localwr-act2/before-regression.json)、[验证记录](reference/validation.json)、[源码字节校验](reference/publish-byte-verification.json)。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh
python verify_reference.py results/fresh
python compare_localwr_act2.py --before /path/to/a987650-run --after results/fresh
```

GPU softmax、GeLU/multiply、packing、O projection、norm/residual与背景VLM竞争未计时；模型不执行BF16数值计算，主阶段合计不等于完整AE端到端latency。模型说明见 [MODEL.md](experiments/pi05-unified-gemv/MODEL.md)。此前staging-only的零latency变化结果保留在 [act2-deadline](reference/act2-deadline/act2-deadline-comparison.json)。

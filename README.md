# xinyao_1103：内部 staging ACT1/ACT2 deadline 修正

本次主五阶段 latency 为 **187.376756 → 187.376756 µs**，变化 **+0.000000 µs**；channel0 DQ 空闲为 **166.845812 → 166.845812 µs**。这是两套backend重新模拟的结果。

在 `14fb3bf` 上修正 `PIM_STAGE_LOAD/FRAGMENT` 的内部 split-activation 调度。SRF 仍为每 bank 2×512 B，源 bank 仍为 rank1/BG0/B0 与 BG1/B0，各 64 KiB；输入布局、trace、计算量、时序参数、CA/DQ 竞争和 barrier 均保留。

ACT1 发出前复用 ordinary RD/WR 的 `can_start_act1` lookahead；发出后登记 bank owner 与 `ACT1_clock+nAAD` deadline，并优先完成 ACT2，防止 PRE/refresh 取消未完成激活。ACT2 仍检查真实 DRAM/rank CA 时序，超时或 owner 丢失会报错。nRP、nRFC、nRCD 等等待未删除。

## 修正前后 latency

两套独立 backend 均重新运行。单位 µs/层/denoising step，H16、batch1、16 channel、每 channel 两 rank。

| 阶段 | 修正前 14fb3bf | 修正后 | 变化 |
|---|---:|---:|---:|
| Q/K + RoPE + QKT | 41.969872 | 41.969872 | +0.000000 |
| V | 20.592852 | 20.592852 | +0.000000 |
| PV | 18.994500 | 18.994500 | +0.000000 |
| Gate/Up | 47.127934 | 47.127934 | +0.000000 |
| Down | 58.691598 | 58.691598 | +0.000000 |
| **主五阶段合计** | **187.376756** | **187.376756** | **+0.000000** |

combined FFN 控制：105.910518 → 105.910518 µs；单层连续控制：188.328826 → 188.328826 µs。这两个控制实验不重复计入主五阶段合计。完整各子阶段见 [phase-latency.csv](reference/act2-deadline/phase-latency.csv)。

## DQ 空闲时间 breakdown

下表为 channel0；全部16个channel均独立重算并检查时间闭合。完整无外部传输窗口已扣除边界数据尾部，其它空闲包括传输阶段内部间隙。

| 阶段 | DQ占用 | 修正前DQ空闲 | 修正后DQ空闲 | 修正后完整无传输窗口 | 修正后其它空闲 |
|---|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 3.301760 | 38.668112 | 38.668112 | 32.514832 | 6.153280 |
| V | 3.121664 | 17.471188 | 17.471188 | 16.065126 | 1.406062 |
| PV | 2.341248 | 16.653252 | 16.653252 | 15.588622 | 1.064630 |
| Gate/Up | 3.842048 | 43.285886 | 43.285886 | 40.757038 | 2.528848 |
| Down | 7.924224 | 50.767374 | 50.767374 | 47.638206 | 3.129168 |
| **合计** | **20.530944** | **166.845812** | **166.845812** | **152.563824** | **14.281988** |

外部流量仍为 5472 KiB/层/步（16channel合计）。[逐channel DQ与CA对比](reference/act2-deadline/channel-dq-comparison.csv)、[按阶段类别的DQ分解](reference/act2-deadline/phase-kind-dq-comparison.csv)、[全部结果JSON](reference/act2-deadline/act2-deadline-comparison.json)。DQ空闲窗口未注入真实GPU竞争，不能直接当作GPU已测得的可用带宽收益。

## 协议修复与性能影响

主五阶段、全部16channel的 8272 对 staging ACT1/ACT2 通过专项审计；旧日志中 staging 超过 nAAD=8 CK 的 8160 对已消除。普通请求与 staging 的列时序、数据量和依赖保持不变。详细间隔分布、取消次数与分类见 [protocol-audit.json](reference/act2-deadline/protocol-audit.json)。

修正把过早的 ACT1 推迟到合法窗口，保留 ACT2 之前的预充电或刷新恢复等待。不能把旧 ACT1→ACT2 超出的间隔从 latency 中直接扣除。前后原始命令差异见 [native-command-comparison.csv](reference/act2-deadline/native-command-comparison.csv)。

各case去除ACT1后的完整命令序列是否逐事件一致：`{"chain": true, "v": true, "pv": true, "gate_up_gpu": true, "down_gpu_input": true, "combined_timing_chain": true, "serial_layer_control": true}`。Down channel0 的刷新与激活示例已保存于同一协议审计JSON，直接展示ACT1后移是否改变ACT2与LOCALRD时刻。

本次只修复 staging。专项审计另发现 projection spill 的内部 LOCALWR 路径仍有 split-activation 违规，保留原调度以隔离本轮影响。修正后全协议审计仍未通过的案例及计数：`{"chain": {"ACT1_while_channel_ACT2_pending": 240, "late_ACT2": 256}, "serial_layer_control": {"ACT1_while_channel_ACT2_pending": 240, "late_ACT2": 256}}`。这是明确保留的模型限制；staging专项通过不代表全部内部路径合规。旧日志中这些非staging违规位于projection spill，RoPE阶段没有该问题。

## 备份、验证与复现

修正前完整 Git bundle、source.zip 和原始 job1902756 证据位于本地 `backups/xinyao_1103_before_act2_deadline_14fb3bf/`，附SHA256清单；紧凑旧结果见 [previous-act2-deadline](reference/previous-act2-deadline/summary.json)。

原生 job 1902838：100 项 Python 测试、38 项原生 guards、21个完整案例（修正前7、修正后7、1101直接输入7）。修正前重跑与归档结果的周期、流量、trace、各phase、原始事件CSV哈希及channel分区完全一致；1101直接输入回归也一致。[修正前回归证明](reference/act2-deadline/before-regression.json)、[完整验证记录](reference/validation.json)。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh
python verify_reference.py results/fresh
python compare_act2_deadline.py --before /path/to/14fb3bf-run --after results/fresh
```

对比必须使用修正前独立 checkout/backend 或归档原始结果。`--legacy-rank-dispatch`只恢复旧派发策略，仍执行本版staging deadline修正，不能替代旧backend。完整模型见 [MODEL.md](experiments/pi05-unified-gemv/MODEL.md)。

GPU softmax、GeLU/multiply、packing、O projection、norm/residual及背景VLM竞争未计时；模型不执行BF16数值计算。主阶段合计不等于完整AE端到端latency。历史channel派发对照保留在 [rank-dispatch-comparison.json](reference/rank-dispatch-comparison.json)。

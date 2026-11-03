# xinyao_1102：单 bank 输入暂存与 rank controller

从 `xinyao_1101` Full DQ 的提交 `4073452` 复制，保留 GPU GeLU/multiply 与按输出维分片的 Down。GPU 的广播输入先整批写到 rank1 的 BG0/bank0 DRAM，完成后再通过 `LOCALRD → bank0 SRF → 同 rank 其它 SRF` 装载。读取、广播和原有 PIM 计算共用 rank 命令总线；总 controller 下发描述符。[模型、指令和地址布局](experiments/pi05-unified-gemv/MODEL.md)。

GPU 输入装载现在以两个 SRF slot 为一批：先连续读取并广播两个 fragment，再切换到权重行计算。每 bank 的 SRAM 仍为 2×512 B，没有扩容。

## 1102 备份时的已知问题

本分支保留单 bank 暂存版本及其原生结果（job1902586），作为后续 `xinyao_1103` 的对照。

1. **集中写入一个 bank 导致 DQ 利用率低。** 当前同 bank group 列间隔为4CK，32B DQ burst只占2CK；再叠加换行、写恢复和refresh，Down整批写入阶段的DQ利用率约36.02%，128KiB/rank耗时21.332025µs。
2. **引入 rank controller 后出现命令堆积和派发阻塞。** 当前每条单bank RoPE细指令都经过总CA准入；channel优先排列的单条trace遇到发送失败便停住，也挡住其它channel的请求，造成跨channel队头阻塞。RoPE从0.529032µs放大到12.652682µs，而channel0的rank CA只忙0.966140µs。这不是新增广播工作量，也不能全部归因于硬件命令总线竞争。

因此本分支的214.354574µs是当前实现的实测值，包含上述已知问题，不能当作该架构不可避免的latency。1103首先仅实验两个不同bank group的bank交错写入；命令堆积问题应单独修正和验证，避免混淆两项改动的收益。

## 新版原生结果

H=16，batch=1，8 attention heads，head_dim=256，hidden=1024，FFN=4096；16 个 LPDDR5X-8533 channel，每个 channel 的 rank0 预留给 GPU、rank1 的16个 bank 执行 PIM。以下为 **µs/层/denoising step**；DQ 与窗口列为单 channel，16个 channel 均已分别验证且相同。

| 阶段 | 1101 latency | 1102 latency | DQ 占用 | DQ 空闲 | 完整无外部传输窗口 |
|---|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 35.273490 | 58.708482 | 3.301760 | 55.406722 | 46.370968 |
| V projection/cache | 10.283294 | 21.992348 | 3.121664 | 18.870684 | 14.905758 |
| PV | 9.190524 | 20.364918 | 2.341248 | 18.023670 | 14.228522 |
| Gate/Up + GPU 回传 | 37.620366 | 48.579958 | 3.842048 | 44.737910 | 39.443838 |
| GPU Z + Down + 输出 | 17.395210 | 64.708868 | 7.924224 | 56.784644 | 42.893802 |
| **五阶段合计** | **109.762884** | **214.354574** | **20.530944** | **193.823630** | **157.842888** |

- 相对1101，已建模 latency 变化 **+95.29%**。按18层×10步折算为 **38.583823 ms**，基线为 **19.757319 ms**。
- DQ 空闲比例 **90.42%**；完整工作窗口比例 **73.64%**。
- 时间闭合：`214.354574 = 20.530944 + 157.842888 + 35.980742`，分别是 DQ 占用、完整无传输工作窗口、其它 DQ 空隙。
- 外部流量仍为 **5472 KiB/层/步（全部16个 channel）**，累计 DQ 占用仍为 **20.530944 µs/channel**。集中暂存改变空闲时间的分布，没有压缩原输入字节数。
- 单层连续资源/refresh control 为 **214.701634 µs**，combined FFN control 为 **113.294454 µs**。它们不重复加入五阶段合计。

latency增加来自先完整写入再计算的顺序要求、普通单bank DRAM写入开销，以及后续暂存读取与bank0权重访问的行切换；DQ空闲比例提高不代表推理加速。

![DQ comparison](reference/channel-breakdown.png)

## 两个 SRF slot 连续装载的修正

之前的1102（`e679df8`、job1902573）每读完一个 fragment 就切回权重行，只有一个 slot 的输入用于该次行访问。当前顺序为 `LOAD f0(slot0) → LOAD f1(slot1) → MAC f0 → MAC f1`。V/PV/Gate/Up/Down 每批装满1KiB；Q/K 为8-token布局，每批两个256B fragment，保留原MAC寻址语义。

| 阶段 | 之前每批一个 slot（µs） | 当前每批两个 slot（µs） | 缩短 |
|---|---:|---:|---:|
| Q/K + RoPE + QKT | 61.460574 | 58.708482 | 4.48% |
| V | 24.498684 | 21.992348 | 10.23% |
| PV | 22.823416 | 20.364918 | 10.77% |
| Gate/Up | 50.736420 | 48.579958 | 4.25% |
| Down | 75.231352 | 64.708868 | 13.99% |
| **合计** | **234.750446** | **214.354574** | **8.69%** |

Down 每channel的 `MACAB` 仍为1024条、`LOCALRD` 仍为4096条；channel0 的 `ACTAB` 从264降到132，`PREab` 从275降到141。整批输入写入过程保持不变，减少的是读回输入与权重之间的行往返。所有16个channel均已核对：除ACT/PRE/refresh外，原生指令计数与上一版完全一致。

Down 第一批在 CK22784–22908 连续发出32次列读取，两个slot在CK22933全部就绪，首个MAC在CK22959发出；读取两片之间没有权重行ACT。独立审计对全部批次检查容量、就绪时间、消费顺序和行切换。广播仍逐32B与列读取流水交错，不是新增一次1KiB外部DQ传输。合法PRE/ACT可以覆盖读返回尾部，refresh也可能打断连续读取。

[与上一版逐channel命令对比](reference/staging-batch-comparison.json)、[上一版记录](reference/previous-single-slot.json)、[Down原生审计](reference/down_gpu_input/native-audit.json)。本次仅改输入装载顺序，rank命令仲裁及RoPE实现保持原样；剩余latency增量仍包含这些模型与调度假设。

## 总命令总线与 rank 命令总线

16个 channel 的总 CA 占用范围为 **18.744054 µs**；rank0 CA 占用为 **0.049714 µs**；rank1 CA 占用范围为 **34.230434 µs**。这些是不同资源，不能相加到 latency。

每个256/512B输入 fragment 使用一个总CA描述符（1CK），rank controller展开8/16条32B读取及广播；每批先连续发送两个fragment描述符，再进入对应计算。广播控制命令占rank CA 1CK；实际DRAM命令使用模型规定的命令宽度（ACTAB为2CK）。这些是本版采用的控制接口时序假设。

DQ 空闲进一步分为：总 CA 忙 **9.568538–9.577918 µs**、DQ 与总 CA 同时空闲 **184.245712–184.255092 µs**（各channel范围；逐channel数值见CSV）。后者只是资源可用时间上界；未注入并发 GPU 请求，尚不能视为实际 GPU 获得的服务时间。

[逐channel分解](reference/channel-partitions.csv)、[逐阶段类型分解](reference/channel-phase-kind-breakdown.csv)、[窗口边界](reference/channel-phase-windows.csv)、[命令总线分解](reference/channel-command-bus.csv)、[完整JSON](reference/channel-analysis.json)。

## 计时假设

GPU 输入落地使用普通 WR，包括 ACT/PRE、CWL、burst 和写恢复；全部写入完成后才允许本地读取。新增 DRAM 暂存区是每个 rank 的 bank0 row256–319，最多128KiB，各算子复用。每 bank 仍只有两个512B SRF slot。

地址验证针对当前单层布局；18层×10步为耗时折算，尚未建立18层同时驻留时的完整物理地址分配。

bank读取和SRF广播的数据延迟包含在 `nCL+nBL_min` 中，不再占用外部DQ；广播命令仍占rank CA。暂存行与bank0权重行共用行状态，行切换和refresh开销计入。普通列读取保留模型允许的流水能力。QKT的定向Q交换、V suffix写回、GPU输出传输保持原路径。

**这不是完整AE端到端 latency。** GPU softmax、GeLU/multiply/cast/pack、O projection、norm/residual、dispatch 和背景VLM竞争尚未计时；模型核对地址、时序和依赖，不计算BF16数值。DQ沿用AttAcc WRGB/RDRES抽象，普通RD/WR使用CL/CWL后的burst；每channel区间均检查无重叠。

## 复现与验证

Linux、C++20、CMake≥3.18、Python3.11及开发头文件。编译依赖源码已随附；绘图可选安装matplotlib。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh-h16
python verify_reference.py results/fresh-h16
python compare_staging_batches.py results/fresh-h16
# Optional comparison figure:
python experiments/pi05-unified-gemv/plot_comparison.py results/fresh-h16
# Baseline with direct GPU -> SRF inputs:
python experiments/pi05-unified-gemv/reproduce.py --output results/direct-baseline --direct-input-baseline
python verify_baseline.py results/direct-baseline
```

本次通过 **56项Python测试、27项独立进程原生约束检查、7个原生案例**。同一原生库另外重跑7个旧路径案例，周期、字节数、trace和原始事件日志哈希均与1101一致。原始trace与timeline由复现脚本生成；reference保存结果、日志哈希和源码快照证明。`baseline-xinyao_1101.json`单独保留旧结果用于比较；新结果全部来自重新仿真。

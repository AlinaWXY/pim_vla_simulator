# xinyao_1102：单 bank 输入暂存与 rank controller

从 `xinyao_1101` Full DQ 的提交 `4073452` 复制，保留 GPU GeLU/multiply 与按输出维分片的 Down。GPU 的广播输入先整批写到 rank1 的 BG0/bank0 DRAM，完成后再通过 `LOCALRD → bank0 SRF → 同 rank 其它 SRF` 装载。读取、广播和原有 PIM 计算共用 rank 命令总线；总 controller 下发描述符。[模型、指令和地址布局](experiments/pi05-unified-gemv/MODEL.md)。

## 新版原生结果

H=16，batch=1，8 attention heads，head_dim=256，hidden=1024，FFN=4096；16 个 LPDDR5X-8533 channel，每个 channel 的 rank0 预留给 GPU、rank1 的16个 bank 执行 PIM。以下为 **µs/层/denoising step**；DQ 与窗口列为单 channel，16个 channel 均已分别验证且相同。

| 阶段 | 1101 latency | 1102 latency | DQ 占用 | DQ 空闲 | 完整无外部传输窗口 |
|---|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 35.273490 | 61.460574 | 3.301760 | 58.158814 | 48.880118 |
| V projection/cache | 10.283294 | 24.498684 | 3.121664 | 21.377020 | 17.262014 |
| PV | 9.190524 | 22.823416 | 2.341248 | 20.482168 | 16.709532 |
| Gate/Up + GPU 回传 | 37.620366 | 50.736420 | 3.842048 | 46.894372 | 41.600300 |
| GPU Z + Down + 输出 | 17.395210 | 75.231352 | 7.924224 | 67.307128 | 53.416286 |
| **五阶段合计** | **109.762884** | **234.750446** | **20.530944** | **214.219502** | **177.868250** |

- 相对1101，已建模 latency 变化 **+113.87%**。按18层×10步折算为 **42.255080 ms**，基线为 **19.757319 ms**。
- DQ 空闲比例 **91.25%**；完整工作窗口比例 **75.77%**。
- 时间闭合：`234.750446 = 20.530944 + 177.868250 + 36.351252`，分别是 DQ 占用、完整无传输工作窗口、其它 DQ 空隙。
- 外部流量仍为 **5472 KiB/层/步（全部16个 channel）**，累计 DQ 占用仍为 **20.530944 µs/channel**。集中暂存改变空闲时间的分布，没有压缩原输入字节数。
- 单层连续资源/refresh control 为 **235.247586 µs**，combined FFN control 为 **126.094402 µs**。它们不重复加入五阶段合计。

latency增加来自先完整写入再计算的顺序要求、普通单bank DRAM写入开销，以及后续暂存读取与bank0权重访问的行切换；DQ空闲比例提高不代表推理加速。

![DQ comparison](reference/channel-breakdown.png)

## 总命令总线与 rank 命令总线

16个 channel 的总 CA 占用范围为 **18.742178–18.749682 µs**；rank0 CA 占用为 **0.053466 µs**；rank1 CA 占用范围为 **35.651504–35.659946 µs**。这些是不同资源，不能相加到 latency。

每个256/512B输入 fragment 使用一个总CA描述符（1CK），rank controller展开8/16条32B读取及广播。广播控制命令占rank CA 1CK；实际DRAM命令使用模型规定的命令宽度（ACTAB为2CK）。这些是本版采用的控制接口时序假设。

DQ 空闲进一步分为：总 CA 忙 **9.559158–9.576042 µs**、DQ 与总 CA 同时空闲 **204.643460–204.660344 µs**（各channel范围；逐channel数值见CSV）。后者只是资源可用时间上界；未注入并发 GPU 请求，尚不能视为实际 GPU 获得的服务时间。

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
# Optional comparison figure:
python experiments/pi05-unified-gemv/plot_comparison.py results/fresh-h16
# Baseline with direct GPU -> SRF inputs:
python experiments/pi05-unified-gemv/reproduce.py --output results/direct-baseline --direct-input-baseline
python verify_baseline.py results/direct-baseline
```

本次通过 **51项Python测试、25项独立进程原生约束检查、7个原生案例**。同一原生库另外重跑7个旧路径案例，周期、字节数、trace和原始事件日志哈希均与1101一致。原始trace与timeline由复现脚本生成；reference保存结果、日志哈希和源码快照证明。`baseline-xinyao_1101.json`单独保留旧结果用于比较；新结果全部来自重新仿真。

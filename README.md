# xinyao_1108：布局、RoPE 与空闲窗口 Evaluation

在 dev0 完成 5 种配置 × 5 个 native cases，复用已有 GPU/VLM 测量。三种布局/RoPE 的 DQ 总量分别为每层每步 5904、4968、4776 KiB；完整 kernel 按序放入全部16 channel 的 DQ/共享CA联合空闲窗口，不能直接减去累计 idle。b–g 为已知项混合模型，尚非真实 GPU/PIM 硬件端到端测量。

本次交付：[完整报告](reference/evaluation-1108/REPORT.zh-CN.md)、[a–g 延迟](reference/evaluation-1108/evaluation-a-g.csv)、[DQ 总量](reference/evaluation-1108/DQ-traffic.csv)、[DQ 时间线](reference/evaluation-1108/dq-timeline.png)。来源、复现命令和验证记录见报告。

## xinyao_1107 历史记录：MAC_RUN / ROPE_RUN 粗细命令

新版本在原有 layout、本地 RoPE、QKT score 集中导出和 PIM O projection 上调整命令接口。计算统一为每 rank 每算子一条256-bit描述符，QKᵀ和O各一条；输入、结果和跨rank搬运保留直接DQ读写。字段、CA周期、执行语义和复现命令见[粗细命令说明](experiments/pi05-unified-gemv/COMMANDS.zh-CN.md)，原生前后对比与图见[本次报告](reference/commands-1107/REPORT.zh-CN.md)。

以下保留调整前的历史实验记录。

## 四条历史数据流比较（仅分析已有记录）

最新比较按用户指定的四条路径排列，复用原生 jobs `1903036`、`1924418`，未重跑仿真。完整表格、DQ 时间线、DQ/CA 最长单段空闲及来源见[四条路径报告](reference/dataflow-comparison/REPORT.zh-CN.md)。

| 路径 | 已建模 AE 部分 / ms（18层×10步） | O projection 是否计时 | DQ utilization（16-channel平均） |
|---|---:|---|---:|
| ① 直接广播至 GEMV buffer | 19.757319 | GPU O 未计时 | 18.705% |
| ② 两 bank 暂存，score 流式导出 | 30.831197 | GPU O 未计时 | 9.621% |
| ③ 两 bank 暂存，score 集中导出 | 32.550157 | GPU O 未计时 | 9.113% |
| ④ 同③，PIM O projection | 37.841434 | 已计 PIM O：5.039536 ms | 8.524% |

上表是独立单层阶段和的外推，**不是完整 AE 端到端测量**。①另有 QKV 融合、PV 分片、输出精度和调度差异，因此①→②不是单独改变输入路径的消融。

![四条路径 AE 计时与 DQ breakdown](reference/dataflow-comparison/ae-latency-dq-utilization.png)

主窗口指标不累计碎片：DQ 取最长完整工作窗口；CA 取最长实际单段空闲；joint 取完整 DQ 工作范围内最长的一段 DQ/CA 同时空闲。QKT 的 DQ 最长窗口①②③④分别为 `0.490574 / 0.559048 / 19.383770 / 19.383770 µs`，其中①②各为31个分离窗口，不能相加称为连续时间。②③④虽然都有47.638206 µs的Down DQ完整窗口，但共享CA最长单段仅1.093708 µs。

## 原有两个开关的2×2实验

在 `xinyao_1104` 基础上实现两个独立开关，并在 dev0 完成四组原生模拟。实验 1：每处理 16 个本地 K token，将 score tile 写回 DRAM，全部完成后统一读取到 GPU。实验 2：PV 部分结果写回 DRAM，经 bank0/4 读出并广播到 16 bank，执行 O projection。

四组共同采用 PV/O 的 BF16 输入、权重和结果，FP32 累加；QKT score 保持 FP32。**主比较改为最长连续完整工作窗口，而非累计无 DQ 时间；实验 2 的时延包含 PV 和 PIM O projection。** 此前累计口径的报告保留在 [previous-cumulative-report](reference/o-projection/previous-cumulative-report/)。

下表为 channel 0 在各工作范围内的**最长一个连续、无外部 DQ 的完整工作窗口**，单位 µs。窗口由完整计算/本地操作阶段组成；相邻阶段之间若只有无 DQ 的控制段则连接，遇到外部传输阶段即分开。传输阶段内部的 DQ idle 小间隙不计入，也不跨独立案例拼接。

| 工作范围 | 非1非2 | 1非2 | 2非1 | 1和2 |
|---|---:|---:|---:|---:|
| QKT | 0.559048 | 19.383770 | 0.559048 | 19.383770 |
| PV；实验 2 为 PV+O | 11.784094 | 11.784094 | 25.124330 | 25.124330 |
| Gate / Up | 40.757038 | 40.757038 | 40.757038 | 40.757038 |
| Down | 47.638206 | 47.638206 | 47.638206 | 47.638206 |

实验 1 将 QKT 的 31 个分离窗口变为 1 个连续窗口；实验 2 的 PV+O 有 2 个窗口，最长为 25.124330 µs，不能把两者相加称为连续。完整 QKV/RoPE/QKT/V chain 的最长窗口四组均为 32.435102 µs，会掩盖 QKT 的变化，因此单列 QKT。窗口包含本地等待、refresh 和内部控制，不等于 MAC 活跃时间，也不是已测 GPU 可用时间。完整数据见 [continuous-dq.json](reference/o-projection/continuous-dq/continuous-dq.json)、[范围汇总](reference/o-projection/continuous-dq/continuous-dq-summary.csv)和[逐连续工作窗口](reference/o-projection/continuous-dq/continuous-dq-complete-work-windows.csv)。

以下 latency 与 DQ utilization 保留作总量核算。时间为每层、每 denoising step 的**内存/PIM 模型时间**，由 QKV/RoPE/QKT、PV 或 PV+O、Gate/Up、Down 四个独立阶段相加；它不是一个连续窗口。非实验 2 组未计 GPU O projection，所有组未计 GPU softmax、最终跨 rank 求和等算术，因此不能直接解读为端到端加速比较。

| 组合 | 模型 latency (µs) | 相对非1非2 (µs) | DQ busy/ch0 (µs) | DQ utilization/ch0 | 外部 DQ 总量 (KiB) |
|---|---:|---:|---:|---:|---:|
| 非1非2（GPU O 未计时） | 171.284428 | 0 | 16.478784 | 9.621% | 4392 |
| 1非2（GPU O 未计时） | 180.834206 | +9.549778 | 16.478784 | 9.113% | 4392 |
| 2非1（已计 PIM O） | 200.680410 | +29.395982 | 17.919552 | 8.929% | 4776 |
| 1和2（已计 PIM O） | 210.230188 | +38.945760 | 17.919552 | 8.524% | 4776 |

![Latency 与 DQ breakdown](reference/o-projection/latency-dq-breakdown.png)

详细分析见[实验报告](reference/o-projection/REPORT.zh-CN.md)，机器可读数据包括[四组汇总](reference/o-projection/comparison.json)、[各阶段](reference/o-projection/stage-comparison.csv)、[全部 16 channel](reference/o-projection/channel-comparison.csv)、[逐 phase](reference/o-projection/phase-comparison.csv)。

PV 基线为 14.953596 µs，实验 2 的 **PV+O 为 44.349578 µs**。其中 O 子流程为 27.997424 µs：读取/广播/计算 20.257986、结果导出 7.731934、控制 0.007504；另有 PV 结果落 DRAM 的 3.320520 µs。相对基线净增 29.395982 µs，还扣除了被替换的 PV 导出 1.921962 µs，因此该差值不是纯 O 耗时。

实验 1 不减少外部 score 字节数，增加本地写回和最终普通 DRAM 读取，改变 DQ 发生的时机。实验 2 在各 rank 上先投影、最后交给 GPU 求和，导出的 BF16 部分结果由 128 KiB 变成 512 KiB，外部 DQ 增加 384 KiB。DQ 百分比下降也受到模型总时间增长影响，不能单凭该比例判断流量减少或性能改善。

原 QKT 每计算 16 个 K token 就导出 1 KiB FP32 score tile，计算和导出交替，因此 QKT 整段中的 DQ 不是持续为零。实验 1 将逐 tile 的外部导出变成本地写回，最后统一读出。具体说明与时间线见报告。

当前 PV 每个 rank 保留某个 head 的 token 奇/偶分片部分和，因此线性 O projection 可以在 rank 内先执行，再对 8 heads × 2 parity 的结果求和。每 rank 配置完整 `[2048,1024]` BF16 W_O 常驻副本（4 MiB/rank、共 64 MiB），当前计算只访问本 head `[256,1024]` 子块（共 8 MiB 活跃权重）。其余权重预置且不访问；初始化时间不计入本次稳态实验。完整常驻布局在测量后补充，20 组 trace 与配置的精确一致性证明见[来源绑定](reference/o-projection/source-result-binding.json)。

原生运行 job `1924418` 包含四组各 5 案例，以及新版 backend 的 5 个旧模式回归案例。原实验的 141 项 Python 测试和 73 项 native guards（52 项旧 guards、8 项 score guards、13 项 O guards）通过；其中 O 的一项报错文字断言经过单项重试，原失败与修正记录均保留。验证包含所有 16 channel 的命令/地址覆盖、ACT1/ACT2、CA/DQ 冲突、SRF/累加器容量、scratch 与 DRAM 生命周期。详见[原实验验证](reference/o-projection/validation.json)、[补充验证](reference/o-projection/postchecks/summary.json)；本次连续窗口分析的新增测试另见[连续窗口复核](reference/o-projection/continuous-review-validation.json)。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce_o_projection.py --output results/four-way
python experiments/pi05-unified-gemv/reproduce.py --skip-tests \
  --cases chain pv gate_up_gpu down_gpu_input serial_layer_control --output results/legacy
python experiments/pi05-unified-gemv/validate_o_projection_results.py \
  --experiment results/four-way --legacy results/legacy --reference reference
```

布局、资源成本和建模范围见[O-PROJECTION.md](experiments/pi05-unified-gemv/O-PROJECTION.md)。这是时序与数据映射模型，没有执行神经网络 tensor 数值；BF16 部分结果舍入与求和顺序的精度影响尚未验证。原始事件和源码快照归档位于本地 `results/o-projection-1924418/`，SHA256 见[归档清单](reference/o-projection/archive-manifest.json)。此前 token-PV 版本的 FP32 输出结果保留在[历史 README](README-pv-token.md)及原 `reference/` 中，不作为本次四组的共同 BF16 基线。

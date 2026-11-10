# Streaming 输入双缓冲

Case e 使用 token 布局、bank-local RoPE、PIMHead 和逐请求闭环 VLM。两个版本都运行 18 层 × 10 个去噪步 × 16 通道，沿用同一份 Version3 缩放输入和校准、`W_max=128`、96/32 写水位以及 `mac_freq_mult=4`。并发和 AE solo 的配置对照中，唯一变化是 PIM 程序路径。 [Version3 校准与基线](../ablation-scaled-w128/README.md)

双缓冲把每 bank 的 1 KiB 输入 SRF 划成两个 512 B 半区。当前半区执行 `MACAB` 时，`WRGB` 可以填下一半。开始使用下一半前要等输入全部到齐，复用半区前要等该半区的上一批 MAC 完成。 [生成器](../../programs/gen_pim_program.py#L70)

| 指标 | e 基线 | e 双缓冲 | 变化 |
|---|---:|---:|---:|
| 端到端延迟（ms） | 128.883328 | 129.504899 | +0.621570 |
| AE solo（ms） | 53.312744 | 48.672570 | −4.640174 |
| 并发 AE 完成（ms） | 103.304466 | 85.724947 | −17.579519 |
| AE GPU 等待（ms） | 49.665476 | 36.626022 | −13.039454 |
| PIM 阶段累计（ms） | 34.576720 | 30.036656 | −4.540064 |
| VLM 尾部（ms） | 25.578862 | 43.779951 | +18.201089 |
| PIM 阶段内的 GPU RD/WR | 120,969,226 | 93,645,278 | −22.59% |
| 每 μs PIM 阶段的 GPU RD/WR | 3,498.57 | 3,117.70 | −10.89% |
| PIMHead 就绪候选通过率 | 88.17% | 85.70% | −2.48 个百分点 |

双缓冲版有 **94.10%** 的 `WRGB` 发令和 **84.89%** 的 `WRGB` DQ 周期与 MAC 执行重叠。PIM 阶段中，下一半已填好且 MAC 在执行时发出的 GPU RD/WR 有 **32,053,255 条**，占阶段内全部 GPU RD/WR 的 **34.23%**，占 MAC 执行期间发出的 **65.52%**。PIMHead 候选统计按判定次数计数，同一请求可以多次参与。 [时长和计数 CSV](latency-comparison.csv) · [逐算子仲裁](pimhead-arbitration.csv) · [双缓冲就绪状态](e-db-gpu-placement.csv) · [机器可读汇总](summary.json)

实跑源码为 `a08db04cfeb8630f74ccdce1f109fe6260d193ca`，V4 移入的实现与实跑源码逐文件一致。原版 e 的并发和 AE solo 各有两次已发布的重复。作业 `1989805` 复跑了一次带计数的基线并发，并各跑一次双缓冲并发与 AE solo。基线并发的 `raw.json` 与 Version3 归档逐字节一致。作业 `1991040_1` 独立复跑了双缓冲并发并采集 GPU 插入位置，`raw.json` 与双缓冲并发结果逐字节一致。读写字节、逐单元 VLM 输入和覆盖范围也已核对。

[双缓冲原始证据](native-evidence.tar.gz) 包含原生结果、原始 dev0 配置、命令与计数、作业日志和观测器源码。[归档清单](archive-manifest.json) 记录 SHA-256，基线原始并发与 AE solo 位于 [Version3 证据](../ablation-scaled-w128/dev0-evidence.tar.gz)。GPU 地址和读写到达顺序沿用合成闭环模型，插入计数和端到端差值均为该配置的仿真结果。

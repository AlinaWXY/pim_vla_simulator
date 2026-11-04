# FFN：专用数据 bank 与计算 bank

基线为 xinyao_1103 的 `7d2730d`。本实验仅改变 FFN 的 rank 内 bank 分工、权重布局及输入调度。每个 rank 仍负责 256 个 Gate/Up feature、64 个 Down 输出；每 rank 都处理全部 16 个 token。QKV、V cache、PV 的布局不变。

| 配置 | 数据 bank | 计算 bank | Gate/Up 每 pass 最大输出列/bank | Down 最大输出列/bank |
|---|---|---:|---:|---:|
| 1＋15 | 0 | 15 | 18 | 5 |
| 2＋14 | 0、4（不同 BG） | 14 | 19 | 5 |

每 bank 保持独立的 **1 KiB 输入 SRF＋2 KiB FP32 output buffer**。FFN 的 SRF 分成两个 512 B 缓冲，每个缓冲存放 `[16 token][16 reduction lane] BF16`。一个缓冲被 MAC 消费时，另一个可以读取和广播下一片输入；必须等待消费者完成才能覆盖该缓冲。其余路径仍使用 1103 的整块 SRF 消费规则。

双缓冲模型允许两个半区分别接收输入和供给 PE；它假设相应 SRAM 端口可同时工作，没有额外增大 SRF 容量。输入广播的额外数据延迟仍按原约定包含在列读取周期内，命令总线仲裁单独计时。

## 输出容量与权重布局

原 Gate/Up 全部输出占 32 KiB/rank，15 或 14 个计算 bank 的 output 总容量分别只有 30、28 KiB。因此分为 Gate、Up 两个输出 pass，每个 pass 都遍历全部 X。最大 output 占用分别为 1152、1216 B/bank；Down 为 320 B/bank。

这意味着 **GPU 的 X 只写入一次，但内部从数据 bank 读取两遍**。两个 pass 各自导出结果后才复用 output buffer。GPU 的 GeLU、乘法和 BF16 转换依旧是未计时的边界。

设计算 bank 数为 C、按物理编号排列的计算 bank 列表为 B，rank 内逻辑输出 feature 为 f：

```text
bank = B[f % C]
column = f // C
weight_offset = base + (input_fragment * ceil(rank_features/C) + column) * 32 + lane*2
```

Gate/Up 的 rank_features 为 256，Down 为 64。每列 MAC 的有效 bank mask 只包含 `column*C+bank_ordinal < rank_features` 的 bank；保留的 padding 地址不参与 MAC，也不回传 GPU。Gate、Up、Down 的新权重基址分别为 1048576、1087488、2097152 B，避开原权重、cache 与暂存区域。

结果按 `[valid rank-local feature][token16] FP32` 导出，保留完整的逻辑 feature/token 坐标，不需要跨 rank 求和。

## 数据 bank 的满行访问

DRAM 行大小为 2 KiB。一行恰好容纳四个 512 B 输入片段。设数据 bank 数为 D、片段编号为 f：

```text
logical_row = f // 4
source_bank = [0,4][logical_row % D]
offset = 524288 + (logical_row // D)*2048 + (f % 4)*512
```

GPU 写入时，在两数据 bank 当前打开行的对应列之间交错，直到每行的 64 个 32 B 列全部写完。读取时按逻辑片段顺序，同一源行连续供应四个片段。数据 bank 不参与权重 MAC 或权重行准备，避免每片输入之后切换到权重行。这里的“满行”是保持 DRAM 行并连续消费，**没有额外配置 2 KiB 输入 SRAM**。

整个 X 或 Z 的外部写入及恢复先完成，再开始内部读取与计算。此次 overlap 指数据 bank 内部读取/广播与计算 bank MAC 的重叠，不包含尚未收到的 GPU 输入。

## 原生命令与时序假设

本实验增加 ACTMASK、PREMASK、MACMASK，分别是现有 ACTAB、PREab、MACAB 的 bank-mask 扩展，rank CA 占用分别为 2、1、1 CK。目标 bank 的行状态、前置条件和时序都会实际更新；不能只修改事件日志中的 mask。PREMASK 保留原 nRPab 恢复时间，但只作用于被选中的计算 bank。共享 rank/BG 的命令、激活功耗和刷新约束继续生效。

新 ACTMASK 另外进入既有 ACT1/REFpb 的 nFAW 窗口，保持旧命令之间的基线行为不变。该模型把一次 masked rank 激活记为一个激活事件，沿用 rank 级 PIM 激活抽象；它不是 14/15 个独立 ACT1 的电气电流验证。硬件支持这些 masked PIM 命令是本实验的架构假设。

## 对比与统计口径

两种配置都测量 Gate/Up、Down 和连续 FFN 控制；同时运行相同 trace、相同布局但关闭 overlap 的对照，区分双缓冲收益与布局变化。连续 FFN 不重复加入两个独立阶段的总和；混合 serial 案例用于检查 attention 到新 FFN 路径的切换。

| FFN 流量，全部 16 channel | 基线 | 两种新配置 |
|---|---:|---:|
| GPU X 写入 | 512 KiB | 512 KiB |
| Gate＋Up FP32 导出 | 512 KiB | 512 KiB |
| GPU Z 写入 | 2048 KiB | 2048 KiB |
| Down FP32 导出 | 64 KiB | 64 KiB |
| X 内部读取 | 512 KiB | 1024 KiB |
| Z 内部读取 | 2048 KiB | 2048 KiB |

有效标量 MAC 总量保持 201326592。latency、读写和 MAC 重叠时间、DQ/CA 空闲均从原始事件时间线统计，不从旧 latency 中直接扣除估计开销。保留共享 rank CA、DRAM 时序、刷新、数据依赖和 barrier。DQ 与总 CA 同时空闲是资源空闲窗口，未模拟并发 GPU 请求，不代表已测得的 GPU 服务带宽。

本模型验证时序、地址映射、数据覆盖与容量，不执行 BF16/FP32 数值推理。结果不包含 GPU GeLU/乘法/转换、输入生产、其它 GPU 算术、后台竞争以及初始权重搬迁。

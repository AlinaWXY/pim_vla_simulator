# 命令、延迟与能量对应关系

来源为 [HBM3-PIM.cpp](sources/attacc/pim_ramulator_src/HBM3-PIM.cpp)、[请求枚举补丁](sources/attacc/pim_ramulator_src/patches/src_base_request.patch)、[前端](sources/attacc/pim_ramulator_src/pim_loadstore_trace.cpp)、[控制器](sources/attacc/pim_ramulator_src/hbm3_pim_controller.cpp) 和 [action/preq 补丁](sources/attacc/pim_ramulator_src/patches/)。下述范围是该实现的含义，不是厂商 ISA 编码；状态机层面尚存在 [静态不一致](findings.md)。

## 所有新增命令

| 命令 | 对外请求 ID / trace | 作用及范围 | latency 从哪里来 | energy 从哪里来 |
| --- | --- | --- | --- | --- |
| ACTAB | MACAB 的前置命令 | 同一 channel 所有 pseudochannel/rank/BG/bank 打开指定行 | 复用 nRC/nRCDRD/nRAS；行命令占用 2 cycles | 无逐 ACTAB 计能；Python 按 MAC 数据字节摊销 ACT/PRE |
| ACTSB | MACSB 的前置命令 | 上述各 BG 中同一 bank index | 同上，但若干约束在 bank 层级 | 同上 |
| ACTPB | MACPB 的前置命令 | 指定 rank/BG/bank，跨两个 pseudochannel 广播 | 同上 | 同上 |
| PRESB | MACSB 行冲突前置命令 | 意图为 same-bank 预充电；实际关闭范围与 ACTSB 不一致 | nRP；MACSB→PRESB 用 nRTPL | 无独立系数 |
| PREPB | MACPB 行冲突前置命令 | 指定 rank/BG/bank，跨 pseudochannel 关闭 | nRP；MACPB→PREPB 用 nRTPL | 无独立系数 |
| MACAB | 4 / PIM_MAC_AB | bank 放置方案；默认每 channel 涉及 64 个 bank | channel 层 nCCDAB；ACTAB→MACAB 用 nRCDRD | MAC 请求数×32B×64，再乘 BA mem 系数；ALU 能量另按 FLOPs 算 |
| MACSB | 5 / PIM_MAC_SB | BG 放置方案；默认同编号 bank 跨 16 个 BG/rank/pch 位置 | channel 层 nCCDSB；ACTSB→MACSB 用 nRCDRD | MAC 请求数×32B×16，再乘 BG mem 系数 |
| MACPB | 6 / PIM_MAC_PB | buffer die 放置方案；每 pch 指定一个 bank | channel nBL、rank nCCDS、BG nCCDL 同时约束；没有 nCCDPB 字段 | MAC 请求数×32B×1 或 ×2，fresh/cache 路径不一致 |
| WRGB | 7 / PIM_WR_GB | 输入向量写入 GEMV buffer | 与搬移/普通 RD/WR 共用 nBL/nCCDS/nCCDL | 32B/请求；计 interposer、TSV、GIO/BG 路径流量 |
| MVSB | 8 / PIM_MV_SB | GEMV 输出移到 softmax buffer | 同 WRGB | 32B/请求；计 TSV、GIO/BG 路径流量 |
| MVGB | 9 / PIM_MV_GB | softmax 输出移到 GEMV buffer | 同 WRGB | 同 MVSB |
| SFM | 10 / PIM_SFM | softmax 阶段的调度标记 | 约束表将其列入搬移间隔，但channel scope跳过下层ready检查；外层另算softmax时间 | 外层 SRAM 字节能量 + FLOPs 能量 |
| SETM | 11 / PIM_SET_MODEL | set-model 请求占位 | 没有专属时序约束、结果完成事件或状态动作 | 只计请求；无专属能量项 |
| SETH | 12 / PIM_SET_HEAD | set-head 请求占位 | 同 SETM | 同 SETM |
| BARRIER | 13 / PIM_BARRIER | 调度排序边界；到队首后被删除 | 无固定等待周期；不等待真实 ALU pipeline 完成 | 无能量项 |

`SB` 在此指 **same bank**，不是 LPDDR5 PIM 文献中可能出现的 single bank。`AB=64`、`SB=16` 来自默认 `2 pch × 2 rank × 4 BG × 4 bank`；改变组织后必须重新核查外层流量倍率。命令名和调度设计的证据等级为 CODE；对应物理延迟/能量的等级见 [timing_sources.csv](timing_sources.csv) 和 [energy.md](energy.md)。

PIM 设备保留 `ACT/PRE/PREA/RD/WR/REFab/REFsb`，共 22 个命令名。原版的 `RDA/WRA/RFMab/RFMsb` 和两个 RFM 请求未纳入 PIM 设备。`PREA` 在原版已经存在，ATTACC 改的是层级遍历，不能重复算作新命令。

## MACAB 的 latency 到底是什么

默认配置的有效 `tCK=769ps`，不是时序表末列的 `1300ps`，因为初始化会执行 `int(1e6/(5200/4))` 并覆盖该列。

| 约束 | 5.2 Gbps PC | 时间 | 依据 |
| --- | --- | --- | --- |
| ACTAB→MACAB | nRCDRD=19 | 14.611ns | 借用 HBM2 RCDRD 绝对时间并转换周期，PARTIAL/DERIVED |
| MACAB→MACAB | nCCDAB=6 | 4.614ns | 作者功耗节流表，PARTIAL |
| MACAB→MACAB（NPC） | nCCDAB=4 | 3.076ns | 作者无功耗约束表，CODE/PARTIAL |
| MACAB→PREA | nRTPL=8 | 6.152ns | 复用 DRAM read-to-precharge，PARTIAL/DERIVED |
| PREA→ACTAB | nRP=19 | 14.611ns | 复用预充电时间，PARTIAL/DERIVED |
| ACTAB→ACTAB | nRC=63 | 48.447ns | 复用行周期，PARTIAL/DERIVED |
| ACTAB→PREA | nRAS=45 | 34.605ns | 最短行打开时间，PARTIAL/DERIVED |

假设同一 channel、目标行起初关闭、没有刷新或其他阻塞，且一行内发出 N 个 MACAB，**从 ACTAB 发射到最后一个 MACAB 发射**的理论下界为：

\[
T_{\mathrm{last\ issue}}=[nRCDRD+(N-1)nCCDAB]\,tCK.
\]

例如 N=8、PC 得到 `(19+7×6)×0.769=46.909ns`。这不是最后一个乘加结果的完成时间，也不包括后续 MVSB、softmax、context GEMV、跨行和队列开销。若已经 row hit，则连 nRCDRD 这段也不需要重新支付。

完整 attention 耗时来自生成 score/context 等请求 → 调度器插入 ACT/PRE 等前置命令 → 满足逐层时序约束 → 统计 `memory_system_cycles`。外层乘 `0.769ns/cycle`。不能用“矩阵 FLOPs ÷ 单个 MACAB latency”代替这条执行链。

作者 README 说明 PC 间隔按 activation/read energy 计算；论文方法段说明 GEMV 算术单元的频率和综合方法。但公开实现没有给出每档能量、供电预算到 nCCDAB 的完整公式、脚本或中间数据。**666MHz 是单元工作频率，不足以直接推出 `nCCDAB=6`。**

## 其他命令不应被赋予虚构的固定 latency

WRGB/MVSB/MVGB 与 RD/WR 在 pseudochannel/rank/BG 层分别共享 `nBL=2`、`nCCDS=2`、`nCCDL=4`，具体最早发射时间受访问地址和前驱命令决定。SFM也被列入这些约束声明，但其channel scope会跳过下层ready检查，见F10。代码允许部分数据搬移与 MAC 重叠，因为将二者视作不同路径。

这些命令没有 SRAM/ALU 数值仿真，也没有独立结果就绪事件。控制器仅为普通 `Read` 设置 `depart=clk+nCL+nBL`；PIM 最终命令发出后就移出队列。`is_pending()` 因而不是 PIM 计算单元 busy 信号。

三个 trace 脚本的注释写着 MACAB 8tCK、WRGB/MVSB/MVGB 4tCK、SFM 16tCK（L=256）。生成的 trace 只有请求和地址，不含这些时间；真正生效的是 C++ 约束以及外层计算。特别是注释中的 MACAB 8tCK 与默认 PC=6/NPC=4 不同，不能拿注释作为已执行的模型参数。

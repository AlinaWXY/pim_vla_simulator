# 能量模型与每个系数的来源

数据链是 [PIM_DRAM_system.cpp](sources/attacc/pim_ramulator_src/PIM_DRAM_system.cpp) 的成功接收请求计数 → [ramulator_wrapper.py](sources/attacc/src/ramulator_wrapper.py) 的流量估算 → [config.py](sources/attacc/src/config.py) 能量系数 → [devices.py::PIM](sources/attacc/src/devices.py) 的累加。

**C++ Ramulator 增量没有新增“某条命令执行一次就消耗 X pJ”的能量求和器，也没有读取 DRAMsim3 的 IDD 表。** 下列能量是外层分析模型，不是新增命令的逐条硅后测量。

## 原始系数

| 系数 | 代码值/使用单位 | 原始来源核查 | 结论 |
| --- | --- | --- | --- |
| Cell ACT/PRE | 0.11 pJ/bit | 注释指 S5。其 HBM2 1KiB行 ACT+PRE=909pJ；909/(1024×8)=0.11096 | DERIVED：满行摊销可复核；不是每次ACT只有0.11pJ |
| Cell RD/WR | 0.44 pJ/bit | 注释指 S5；全文/Table3未直接给出对应cell项0.44 | PARTIAL：需细分模型/提取表，不能冒称原文直接列出 |
| Column decoder→BG MUX | 1.01 pJ/bit | 注释指 S5；其公开表格并非按此命名划分 | PARTIAL：具体分配和缩放未闭合 |
| BG MUX→GIO MUX | 1.23 pJ/bit | 同上；1.01+1.23=2.24与S5一处平均片内数据路径能量相符 | PARTIAL：只能核对合计，不能证明两段的拆分 |
| TSV | 0.5 pJ/bit | 注释指 S5；未找到明确单列TSV=0.5的对应表项 | PARTIAL：不能未经活动率/边界核对，直接用总I/O减interposer证明 |
| Silicon interposer | 0.3 pJ/bit | S5 §2直接给出此值，考虑应用数据翻转 | VERIFIED（S5条件下）；不是ATTACC HBM3独立测量 |
| PIM SRAM | 0.0034，代码按pJ/B使用 | S4说明修改FinCACTI的方法；S1没有该数值的buffer尺寸/端口/活动率/输出表 | PARTIAL：方法可追溯，确值不能复算 |
| PIM ALU | 0.32，代码乘FLOPs/2 | S4说明DC/ASAP7方法；缺0.32对应操作、精度、工艺/电压、活动率、报告 | PARTIAL：使用方式相当于每MAC等价操作pJ，不能按文件头误读为pJ/B |
| AttAcc↔xPU通信 | 10.4 pJ/B | S1引用S6；GPU一侧写1.3×8，PIM直接写10.4 | PARTIAL：算术换算明确，原始文献条件未核对 |

S5 Table3 的 HBM2 项是：row activation909pJ、pre-GSA1.51pJ/bit、post-GSA1.17pJ/bit、I/O0.80pJ/bit（后两项标50% activity）。这与代码的全部细分项目并非逐项同名对应。S5给出的模型方法与活动率背景能支持研究方向，不能填补ATTACC缺失的拆分系数。

作者论文称内部读能量按HBM数据路径长度缩放。因此确认1.01/1.23等值需要原始路径长度、参考能量和缩放过程。不要把论文的**面积**工艺缩放系数直接用成能量缩放系数。

## 按 PE 放置位置组合的系数

| 位置 | Python `mem` 的公式 | pJ/B | 判定 |
| --- | --- | --- | --- |
| BA | (0.11+0.44)×8 | 4.4 | DERIVED；原始0.44等仍未闭合 |
| BG | (0.11+0.44+1.01)×8 | 12.48 | DERIVED；原始细分系数PARTIAL |
| BUFFER | (0.11+0.44+1.01+1.23+0.5)×8 | 26.32 | DERIVED；原始细分系数PARTIAL |

三种位置均另设 `io=[0.3,0.5,1.23,1.01]`、`sram=0.0034`、`alu=0.32`、`comm=10.4`。注意 **mem乘了8，io没有乘8**，但两者最终乘的流量都来自字节计数。若注释pJ/bit正确，io路径存在8倍单位差异。这里保留原始实现并标注疑点，不擅自改系数。

## 各命令如何进入能量公式

设一次单HBM trace中计到 M 个MAC、W个WRGB、S个MVSB、G个MVGB：

```text
B_interposer = 32 × W
B_TSV = B_GIO = B_BG = 32 × (W + S + G)
B_memory = 32 × M × parallel_factor
parallel_factor: BA=64, BG=16, BUFFER=1（新仿真）或2（缓存路径）
```

wrapper 再把全部流量乘 `num_hbm` 和 fast_mode 的分组数；`devices.PIM` 使用：

```text
E_memory_paths = B_memory × mem + Σ(B_path × io[path])
E_arithmetic = score_layer_FLOPs / 2 × alu
E_softmax = sum(softmax_layer_sizes_bytes) × sram + softmax_FLOPs / 2 × alu
E_communication = communication_bytes × comm
```

最后按各路径乘 `num_attacc`。这些式子表达代码的记账方式，输出能量项以pJ量级累加，后续系统输出再作单位转换。特别注意：

- 每个MAC请求的广播倍数硬编码，不是运行时根据组织自动统计的实际 bank read 次数。
- ACT/PRE按访问字节摊销，没有按真正发出的ACTAB/ACTSB/ACTPB/PRE统计能量；改变row locality不会自然对应真实ACT能量变化。
- REFab/REFsb没有逐命令能量项，背景/待机/refresh电流也没有按周期积分。
- SETM/SETH/BARRIER没有专属能量项，不代表硬件操作物理上免费。
- C++里SFM请求主要影响调度；外层另算softmax时间和能量。不能把一次SFM请求简单赋值为0.32pJ。
- score分支的trace覆盖score和context，但ALU项只调用score layer的FLOPs；context分支返回零。需作者确认是否少计了context算术能量，不能据此声明attention能量全部闭合。

要得到可迁移到LPDDR5-PIM的**逐命令**energy表，应先明确每个命令的广播bank数、每bank数据量、计算精度、有效操作数、激活次数、搬移路径和工作电压，再取得这些组件的能量。ATTACC可以提供分解结构，现有系数不能直接成为LPDDR5-PIM实测参数。

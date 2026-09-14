# 来源与证据等级

## 判定规则

| 标记 | 含义 |
| --- | --- |
| CODE | 能确认作者公开代码中的实现或参数；不声明其物理正确性 |
| VERIFIED | 已核对原始文献/配置直接给出的数值、单位和适用对象 |
| DERIVED | 本次可根据明确输入进行算术复核；如果作者未公开公式，仍标明为审计者推导 |
| PARTIAL | 作者提供参考文献或方法，但不足以唯一重建该具体值 |
| MISSING | 当前公开材料未找到该具体值或设计选择的可复核依据 |

同一项可以同时“代码来源已确认、物理来源待确认”。例如 `nCCDAB=6` 的代码来源确定，但不能据此证明某 LPDDR5 PIM 硬件需要 6 个周期。

## 原始来源

**S0 — 原版 Ramulator。** [固定源码](https://github.com/CMU-SAFARI/ramulator2/tree/b7c70275f04126c647edb989270cc429776955d1)，本项目 `ramulator2/`；HBM3 原始时序表本身含有寻找更多来源的 TODO。作为比较基线，不将原版已有值自动视为已验证的 JEDEC 数据。时序转换函数为 `src/base/utils.cpp::JEDEC_rounding`。

**S1 — ATTACC 作者公开实现。** [固定提交](https://github.com/scale-snu/attacc_simulator/tree/c60005143a6b492d7ef83231723386478b59a506)，以及本目录 [快照](sources/attacc/README.md)。[安装脚本](sources/attacc/set_pim_ramulator.sh) 确定增量范围；[README 的 power constraint 段](sources/attacc/README.md) 明确说利用 activation/read energy 增大连续 MAC 的间隔。逐个有效文件的固定 GitHub URL 记录在 [file_delta.csv](generated/file_delta.csv)。公开文本文件中没有 GEMV RTL、SDC、DC 功耗/时序报告或修改后的 FinCACTI 工程。

**S2 — 作者指定的 DRAMsim3 HBM2 配置。** [`HBM2_8Gb_x128.ini@29817593b3389f1337235d63cac515024ab8fd6e`](https://github.com/umd-memsys/DRAMsim3/blob/29817593b3389f1337235d63cac515024ab8fd6e/configs/HBM2_8Gb_x128.ini)。已获取并核对完整配置。`tCK=1ns`；CL/RCDRD/RCDWR/RP=14，RAS=34，WR=16，CWL=4，RTP_S/L=4/6，WTR_S/L=6/8，FAW=30，REFI=3900。它是 HBM2 配置，不能当成 HBM3 或 PIM 厂商手册。其 `[power]` 有 VDD/IDD，但 ATTACC 的 Python 能量模型没有读取这些 IDD 值。

**S3 — JEDEC JESD238A / JESD235C。** ATTACC 组织表注释声明参考 JESD238A；部分约束注释继续指向 JESD235C Table 3。本次未取得这两份标准全文，故不能确认具体表号、速度档、密度定义、广播语义与每个数字。引用存在的证据来自 S1，而不是标准全文校验。所有这类数值列为 PARTIAL。

**S4 — AttAcc! ASPLOS 2024。** [作者论文](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf)，[DOI:10.1145/3620665.3640422](https://doi.org/10.1145/3620665.3640422)。本次作者站 PDF 下载失败，但检索索引可核对 §7.1 / p.113 的方法段：算术单元用 Verilog、Synopsys DC、ASAP7；GEMV/accumulator 为 666MHz，softmax 为 1.3GHz；SRAM 用修改的 FinCACTI；内部读能量按数据路径长度缩放。它提供方法依据，未提供本审计所需的逐参数计算日志。此前讨论中的 IDD7/具体峰值预算仍需获得原文对应页及作者计算表复核，不据此在本报告反推一个“已证实”的 nCCDAB。

**S5 — Fine-Grained DRAM，MICRO 2017。** [作者 PDF](https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf)，[NVIDIA 官方页面](https://research.nvidia.com/publication/2017-10_fine-grained-dram-energy-efficient-dram-extreme-bandwidth-systems)，[DOI:10.1145/3123939.3124545](https://doi.org/10.1145/3123939.3124545)。已下载、抽取全文核对 §2、§4、Table 2/3。关键依据和不能验证的细分系数列于 [energy.md](energy.md)。不将该论文标题中的 FGDRAM 与表格中的 HBM2 基线混淆；ATTACC 引用该研究不意味着采用其 FGDRAM 结构。

**S6 — 外部互连能耗引用。** [IEEE 文献 10067395](https://ieeexplore.ieee.org/document/10067395)，来自 `src/config.py` 的注释。未核对该文献正文及其接口版本/测量条件；`1.3pJ/bit → 10.4pJ/B` 仅确认代码换算，不标为原始测量已验证。

用户的 `reference/` 下 Samsung LPDDR5X-PIM 文件不是 ATTACC HBM3 模型数值的直接来源。本分支不将其中 LPDDR5X-PIM 命令与 ATTACC 内部请求名一一等同。

## 获取及复用说明

快照来自作者 MIT 仓库并保留 [LICENSE](sources/attacc/LICENSE)；全文论文只在 `/scratch` 作查阅缓存，未纳入仓库。网络访问日为 2026-09-14。此次来源结论受固定公开提交及以上可获取材料限制；未发现资料不等于作者没有这些内部数据。

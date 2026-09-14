# AttAcc Ramulator Analysis

审计日期：2026-09-14。分支：`attacc-ramulator-analysis`。

本目录回答：**ATTACC 对原版 Ramulator 2.0 增改了什么，以及每项功能、latency、energy 的依据能追溯到哪里。** 分析覆盖固定版本的公开实现；“已定位代码数值”与“已证明数值的物理来源”分别记录。

## 对比基线与结果

| 对象 | 固定版本 |
| --- | --- |
| Ramulator 2.0 原版 | [`b7c70275f04126c647edb989270cc429776955d1`](https://github.com/CMU-SAFARI/ramulator2/tree/b7c70275f04126c647edb989270cc429776955d1) |
| ATTACC 公开仓库 | [`c60005143a6b492d7ef83231723386478b59a506`](https://github.com/scale-snu/attacc_simulator/tree/c60005143a6b492d7ef83231723386478b59a506) |
| 对比依据 | ATTACC 的 `set_pim_ramulator.sh` 明确 reset 到上述 Ramulator 提交，再复制文件、应用补丁 |
| 本项目源码 | `ramulator2/` 保留原版；仅在临时目录重建 ATTACC 补丁结果 |

确认有 **21 个原文件修改、18 个新增有效文件，共 39 个文件变化**。安装时另外复制的 21 个 `.patch` 属于补丁载荷，不重复算作 21 项模型功能。新增命令名 15 个：13 个列在作者的 PIM 命令区，另有 `PRESB/PREPB`；对外新增 PIM 请求 10 类。PIM 模型相对原版 HBM3 还去掉了 `RDA/WRA/RFMab/RFMsb`，不是原命令集的简单超集。

审计材料包括：436 个时序表单元格（原版 26、ATTACC HBM3 130、HBM3-PIM 280）、240 个组织表单元格、151 条时序约束声明。数量包含三个模型各自的条目，**不表示它们全部是新增参数**。真正新增的时序名称只有 `nCCDAB/nCCDSB`，其余是旧参数的新值、层级调整、复用或未使用字段。

## 核心结论

1. **时延由命令 trace、状态机和调度约束共同产生。** MACAB 没有一个独立的“从发射到结果可用”延迟字段。5.2 Gbps 下，`nCCDAB=6`（功耗受限）或 `4`（NPC）表示连续命令间隔；ACT→MACAB 还要满足 `nRCDRD=19`。详见 [命令与 MAC 延迟](commands.md)。
2. **普通 HBM3 时序大部分参考 HBM2 的绝对时间。** 作者注释指向固定提交的 DRAMsim3 配置；13 个直接对应字段和 `nRC` 可以通过换算复核，但 `nRRD`、刷新扩展值、MAC 功耗节流等不能凭该引用全部解释。详见 [时序来源表](timing_sources.csv)。
3. **能量计算位于外层 Python。** C++ Ramulator 返回周期和请求计数，Python 将计数换成字节流量，再乘能量系数；没有一张覆盖所有新增命令的独立实测 energy 表。详见 [能耗来源](energy.md)。
4. **公开来源不足以闭合所有物理数据链。** 尤其缺少 `nCCDAB/nCCDSB` 的逐档功耗计算表、综合报告、FinCACTI 参数和细分数据路径能量的推导。本审计已经逐项定位这些缺口；没有将“代码里写了”标成“厂商已测量”。

## 阅读入口

| 内容 | 文件 |
| --- | --- |
| 39 个有效文件逐项功能、数据依据 | [file_review.csv](file_review.csv)；对应原始文件/补丁、固定链接和前后哈希见 [file_delta.csv](generated/file_delta.csv) |
| 新命令、广播范围、latency/energy 对应关系 | [commands.md](commands.md) |
| 组织、地址映射、控制器、trace、配置和外层接口 | [architecture.md](architecture.md) |
| 28 个时序字段逐项来源与可信程度 | [timing_sources.csv](timing_sources.csv) |
| 全部原始时序值、组织值、约束表达式 | [timing_presets.csv](generated/timing_presets.csv)、[organization_presets.csv](generated/organization_presets.csv)、[timing_constraints.csv](generated/timing_constraints.csv) |
| HBM2数值换算复核、默认实际加载值 | [hbm2_crosscheck.csv](generated/hbm2_crosscheck.csv)、[timing_effective_default.csv](generated/timing_effective_default.csv) |
| 能耗各系数及计算链 | [energy.md](energy.md) |
| 代码问题与尚需索取的具体资料 | [findings.md](findings.md) |
| 原始文献、访问范围与证据等级 | [sources.md](sources.md) |
| 完整文本差异、补丁重建记录 | [effective.diff](generated/effective.diff)、[patch_application.txt](generated/patch_application.txt) |

`sources/attacc/` 保存上述固定提交的 51 个文本文件，附原 MIT LICENSE；没有保存二进制程序、子模块源码或论文全文。逐文件 SHA-256 见 [manifest.json](sources/manifest.json)。这些是审计快照，不参与本项目的仿真器构建。

## 复核方法与边界

在项目根目录运行（临时目录必须预先存在且位于 `/scratch` 下）：

```bash
python docs/attacc-ramulator-analysis/audit.py --scratch-root /scratch/xinyaowang/attacc-audit-20260914-nvvMSy
```

脚本仅使用 Python 标准库和 GNU patch：验证快照、原始目标文件及依赖文件哈希；在临时目录按安装清单复制文件；逐个应用补丁，禁止 fuzz、检查无 offset；重新抽取数据并比较已保存结果；确认每个有效变更文件、每个时序名称均有来源记录，并复核70个HBM2换算值。`--refresh` 可重新生成 `generated/`，不修改 `ramulator2/`。重建后的普通HBM3文件另存为 [attacc_HBM3.cpp](generated/attacc_HBM3.cpp)，方便按表中行号查阅；许可沿用原版Ramulator和ATTACC的MIT条款。

已通过以上重建与覆盖检查。此次没有编译或运行完整 ATTACC 性能仿真，没有做 RTL/硅后校验；[findings.md](findings.md) 的代码疑点标明了静态证据和适用范围。对于能源和时序的物理可信度，校验脚本不能代替数据手册、综合结果或测量。

范围包括 Ramulator 内部全部安装增量，以及外层直接用于 PIM 时间/能量的 wrapper、配置和计算。完整 GPU/CPU roofline 校准、所有语言模型参数、论文性能结果复现不是本次 Ramulator 增量审计的结论范围。

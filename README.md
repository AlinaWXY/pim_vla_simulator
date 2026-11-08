# PIM-VLA：基于 Ramulator2 的新版仿真器

本仓库对应 `xinyao_1109` 重写：GPU rank0 与 PIM rank1 在同一原生仿真时间线上竞争，16 个通道均调用 Ramulator2 的 prerequisite、timing 与 command issue 机制。LPDDR5X-8533 时序只在 spec 中定义。PIM 请求直接由张量布局生成，运行不依赖旧 trace、外部仲裁重放或延迟常数抵消。

上游基线为 CMU-SAFARI/ramulator2 **`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`**（2026-11-03，项目版本 2.1）。原任务保存在 [重写方案](docs/rewrite-plan.txt)，逐项交付状态见 [需求矩阵](docs/requirements-matrix.md)。

## 当前验收状态

2026-11-09，服务器 `dev0` 上的验证作业 **1976509** 已 `COMPLETED / 0:0`：spec、原生命令对矩阵、四个微基准、lookahead、资源生命周期、故障注入和合成前端验证通过；五种实际 PIM 程序均通过**单层**运行及 DQ 字节守恒检查。

**完整 18 层 × 10 步 AE、完整 VLM 校准、b–g 与三个仲裁消融的完整运行及重复一致性验收仍待完成。** 单层 smoke 不是完整实验结果，本 README 不提供未经完整验收的延迟或加速比。

验证日志：`/fact_data/xinyaowang/pim_vla_sim2_20261109/validate-1976509.log`。单层原始输出及 `validation.json` 位于该工作区的 `results/pim-smoke/`。这些运行文件不纳入 Git；导出的 spec 与 48 项命令对矩阵保存在 [reference/spec](reference/spec/)。

## 旧版本备份

旧源码冻结为 `legacy-1108`，原提交 `2b8645e`。服务器完整备份目录：

```text
dev0:/fact_data/xinyaowang/backups/pim-vla-legacy-1108-20261109
```

其中包含三个旧工作目录的完整 tar、Git 历史、未提交修改、构建与结果文件，以及 `legacy-1108.bundle`。每个归档的 SHA-256 和备份范围见 [legacy-backup.json](docs/legacy-backup.json)。新版源树采用干净上游核心，并保留旧 Git 历史用于追溯；旧 AttAcc PIM 控制器和外部时序引擎不进入新版执行路径。

## 构建与运行

在 Linux 上使用支持 C++20 的编译器、CMake ≥ 3.18、Python ≥ 3.10 和 Git。首次 CMake 配置会取得固定版本的 yaml-cpp 与 fmt；Python bindings 在此构建中关闭。服务器验证使用 GCC 14.2.1。

```bash
# 构建并运行 spec / native / resource / fault / synthetic frontend 验证
bash build.sh

# 生成五种无时序常数的单层程序；大文件不提交 Git
python3 programs/gen_pim_program.py --output programs/generated --compact

# 五种实际程序单层验证
python3 tests/run_pim_smoke.py

# 准备完整实验配置，不执行实验
python3 experiments/run.py --output results/rewrite --prepare-only

# 五种 AE solo → VLM 校准 → b–g 与仲裁消融，默认各重复两次
python3 experiments/run.py --output results/rewrite

# 完整结果的科学绘图；额外需要 matplotlib，仿真本身不依赖它
python3 analysis/plot.py results/rewrite/evaluation-b-g.csv
```

完整实验在 VLM 校准误差超过 5% 时停止，不进入并发矩阵。已有输出目录受到保护；`--resume` 核验配置、输入、二进制和运行库来源。Slurm 分拆校准、solo 和 case 作业的方法见 [实验说明](experiments/README.md)。硬件灵敏度可用 `--mac-freq-mult 1`、`2`、`4` 对整个矩阵分别运行；默认值为 4。

## 消融配置

| Case | Layout | RoPE | Dataflow | Mode | Arbitration |
|---|---|---|---|---|---|
| b | feature | gpu | streaming | serial | — |
| c | feature | gpu | streaming | concurrent | pim_head |
| d | token | gpu | streaming | concurrent | pim_head |
| e | token | local | streaming | concurrent | pim_head |
| f | token | gpu | fused | concurrent | — |
| g | token | local | fused | concurrent | — |
| e-frfcfs | token | local | streaming | concurrent | frfcfs |
| e-lookahead | token | local | streaming | concurrent | pim_lookahead |
| e-gap | token | local | streaming | concurrent | gap_batch |

Case a 沿用实测，不由本模型仿真。只有 concurrent + streaming 接受 `arbitration`；其他模式必须省略此字段。feature 只允许 gpu RoPE + streaming。旧 `pim_*`、`shift`、`constant_offset`、`dq_latency` 开关被拒绝。

## 代码与产物

| 内容 | 入口 |
|---|---|
| 唯一时序定义、接口 alias 派生及硬件 metadata | [lpddr5x_pim.py](ramulator2/python/ramulator/dram/lpddr5x_pim.py) |
| 原生双来源调度、刷新、私有命令路径与 epoch | [pim_controller.cpp](ramulator2/src/ramulator/controller/impl/pim_controller.cpp) |
| 有界原生 lookahead / SRF 与输出生命周期 | [pim_lookahead.cpp](ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp)、[pim_resources.h](ramulator2/src/ramulator/controller/impl/pim_resources.h) |
| 旁路 CA/DQ 断言与统计 | [bus_checker.cpp](ramulator2/src/ramulator/plugin/bus_checker.cpp) |
| AE 阶段驱动、VLM 闭环访存、GPU 分时 | [pim_vla_frontend.cpp](ramulator2/src/ramulator/frontend/impl/pim_vla_frontend.cpp) |
| 张量布局、地址空间与请求程序 | [workloads/layout.py](workloads/layout.py)、[gen_pim_program.py](programs/gen_pim_program.py) |
| 原始 Thor 测量及输入转换来源 | [input-data/measured](input-data/measured/)、[provenance.json](input-data/provenance.json) |
| 全局参数校准、实验与原生时间戳报表 | [calibrate_vlm.py](calib/calibrate_vlm.py)、[run.py](experiments/run.py)、[report.py](analysis/report.py) |
| 只消费新版 evaluation CSV 的延迟分解图 | [plot.py](analysis/plot.py) |

每次 native 运行保存配置、`native.log`、`raw.json`、`provenance.json`；来源记录包含 spec/程序/测量输入哈希、源码提交、可执行文件及运行库哈希。完整矩阵输出：

- `evaluation-b-g.csv`：AE solo/并发、PIM 干扰、AE GPU wait、VLM stall/remaining、总延迟。
- `<case>-PIM-and-AE-waits.csv`、`<case>-VLM-placement.csv`、`<case>-bus.csv`：阶段、kernel、通道总线明细。
- `<case>-checks.json`、`summary.json`、`REPORT.zh-CN.md`：守恒检查与汇总；重复运行比较原生输出字节一致性。
- `latency-breakdown.png/.svg/.pdf` 与同名 `.json`：绘图命令生成的可分享图和输入哈希/数值清单。分解为 AE solo + PIM interference + AE GPU wait + VLM remaining；负干扰保留在零线左侧，黑色菱形标出代数和，绘图前再次检查总时间恒等式。

这些完整产物目前仍待实验结束后验收，文件格式的实现不等于已有结果。

`evaluation-b-g.csv` 的前21列保留旧 publication 表的列名与顺序，后面追加新版 canonical 列；兼容性测试已通过。`AE_native_schedule_ms` 等于 `AE_solo_ms`，`raw_model_total_ms` 等于未经修正的 `total_ms`。`completed_units` 统计 AE 完成时已结束的 VLM 单元，`VLM_completed_model_ms` 累加这些单元的 native solo 时长；`VLM_remaining_model_ms` 是实际尾部时间，可能包含尚在运行的单元，因此不要求它与 completed 列加总为 VLM solo。GPU 命令/行命中列来自 rank0 原生计数。完整映射写入运行后的 `summary.json`。

## 采用的模型边界与文档冲突处理

详细依据见 [时序来源与决定](docs/timing-source-and-decisions.md)、[程序与张量契约](docs/program-design.md)、[原生执行模型](docs/native-model.md)。以下会影响结果解释：

- 同 rank 的全 BG RDRES→WRGB 为 **17 CK**，跨 rank 为 **15 CK**；同 rank WRGB→read 保留更严格的 WTR_L。alias 只继承传输接口约束，不把 PREab 前的阵列恢复约束错误加给 SRAM 命令。
- MAC 的 `M × 4/k` 完成时间，以及本地搬运、归约、cast、RoPE 延迟是明确的架构假设。每 rank 一个 RoPE 执行单元，每 bank 1 KiB SRF 和 2 KiB FP32 输出池，无 rank score tile 缓存。RoPE/布局校验不执行 BF16 数值推理。
- 全 bank MAC 通过各 bank 的合法 ACT1/ACT2 准备，遵守 RRD/FAW；不存在旧 ACTAB。Q 交换使用有地址的 DRAM WR + LOCALRD，不能把不同 bank 的 Q 分片伪装成同值 WRGB。描述符、局部 RoPE、score 立即导出等变化意味着任务 §10.4 的“仅 I/O 转向造成差异”不能字面成立，差异在程序文档中逐项披露。
- 每层地址为独立的 1 MiB/bank 区域，18 层互不重叠；10 个去噪步复用相应层。W_O 完整常驻，各布局访问自己的有效分片；token O 的标量 MAC 数是 feature O 的两倍，这是保留的旧分片语义。
- VLM 使用逐 kernel 实测时长和扇区量、均匀释放、有限 outstanding window、合成顺序地址区域。kernel 的写流量由全局 alpha 拟合，memcpy 写流量保持实测值。逐单元实测时长之和为 105.2205782 ms，独立 phase 目标为 104.950496035 ms；两者均保留，不归一化。
- Case b 按两个独立初始化的 solo epoch 定义：同一进程内 AE 后重置 DRAM 行状态与刷新相位，再运行 VLM，保留全局时间与计数；它不代表保留缓存行状态的真实交接。并发 case 不进行这种重置。
- WRGB 搬运的 SM 占用、额外 GPU↔PIM 控制交接时间不建模；AE GPU 与 VLM kernel 按不可抢占 kernel 分时，不执行窗口适配。实测归约/cast 辅助 kernel 是算术开销，仍计时。这些结果是条件仿真估计，不是 Thor 并发实测。

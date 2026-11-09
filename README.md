# PIM-VLA：基于 Ramulator2 的新版仿真器

本仓库对应 `xinyao_1109` 重写：GPU rank0 与 PIM rank1 在同一原生仿真时间线上竞争，16 个通道均调用 Ramulator2 的 prerequisite、timing 与 command issue 机制。LPDDR5X-8533 时序只在 spec 中定义。PIM 请求直接由张量布局生成，运行不依赖旧 trace、外部仲裁重放或延迟常数抵消。

上游基线为 CMU-SAFARI/ramulator2 **`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`**（2026-11-03，项目版本 2.1）。原任务保存在 [重写方案](docs/rewrite-plan.txt)，逐项交付状态见 [需求矩阵](docs/requirements-matrix.md)。

## 当前分支：FIFO32 按序发射

本分支实现用户 2026-11-09 指定的 **fused kernel 窗口模型**：输入上传和粗指令交付后，PIM 私有 FIFO32 计算与按校准时长推进的 VLM kernel 重叠。按最新澄清，**QKV→本地 RoPE、PV→O 连续执行**：前段本地存储完成后立即按原生 CA 时序交付下一条粗指令，当前 VLM kernel 继续运行；真实数据上传/导出、Q 交换及 AE GPU 算子仍保留独占交接。PIM 完整落盘后，在需要导出时等待当前 kernel 结束。VLM 不再逐笔仿真访存；连续粗指令造成的微小共享 CA 干扰不另行延长校准 kernel 时长，PIM 指令的 CA/decode 成本仍完整计时。执行边界、校准来源及指标区别见 [模型说明](docs/fused-kernel-execution.md)。

**连续执行策略的完整 f/g 已各运行一次并通过审计**：18 层 × 10 步、2357 个校准 VLM 工作单元，源码 `924141c`，作业 `1983025`、审计 `1983030`。远端 24 项原生前端回归和 91 项 Python 测试通过；新增 10 项审计单测后，本地 22 项审计单测全部通过。

| 当前连续执行模型 | e2e / ms | AE 完成 / ms | 独占交接前等待 VLM / ms | 实际运行耗时 |
|---|---:|---:|---:|---:|
| f：GPU RoPE | 137.737998608 | 124.515315582 | 59.276970970 | 478.752 s |
| g：本地 RoPE | 131.832380624 | 118.609697598 | 59.036419932 | 477.272 s |

f/g 的 180/360 次连续粗指令交付均实际跨越正在运行的 VLM kernel，全部私有 PIM 计算被 VLM 执行覆盖。与下方旧策略相比，g 的 AE 提前 **15.371028512 ms** 完成，但随后仍有 **13.222683026 ms** VLM 尾部，因此总 latency 减少 **2.148345486 ms（1.60347%）**。前后工作负载、校准输入、非路径配置和运行库均相同，见 [新结果审计](reference/fused-kernel-continuous-e2e/summary.json)、[策略对比与时间分解](reference/fused-kernel-continuous-e2e/policy-comparison.json)。本轮 f/g 均在 einstein 运行；旧 g 使用另一主机，实际耗时差不能作为严格同机加速比。

以下是**此前每个描述符都等待 VLM 的历史结果**，包括 QKV→本地 RoPE、PV→O 的额外等待，不能当作最新连续执行策略的结果。该轮完整 18 层 × 10 步、2357 个校准 VLM 工作单元的 f/g 各运行一次并通过独立审计（源码 `0046e8b`，作业 `1982852`、审计 `1982856`）：

| 历史每描述符等待模型 | e2e / ms | 等待当前 VLM / ms | 实际运行耗时 |
|---|---:|---:|---:|
| f：GPU RoPE | 137.764615296 | 71.018671122 | 484.588 s |
| g：本地 RoPE | 133.980726110 | 74.431107618 | 487.233 s |

该历史运行的 VLM 原生访存请求数为 0；全部 kernel 保持各自的校准时长，上传/导出/AE GPU 区间不与 VLM kernel 重叠。全部 PIM 阶段、逐通道字节、刷新与 FIFO 顺序检查通过。见 [历史结果审计](reference/fused-kernel-e2e/summary.json)、[历史阶段分解](reference/fused-kernel-e2e/phase-durations.csv)。f 在 einstein、g 在 infinite-frontier 运行，实际耗时是该次运行记录，不作为同机严格性能对照。

本分支以 `6bcd57b` 的 gio-bcast 实现为对照，按用户 2026-11-09 的澄清，将 fused 描述符内部恢复为 **32 项缓冲、按程序顺序发射**。只检查队首的依赖、资源及 Ramulator 原生 prerequisite/timing；队首未发射时，后续子指令及其 ACT/PRE 不得提前发射。队首发射后立即推进 FIFO，不等待它全部完成；显式 completion 依赖、资源冲突和 barrier 仍须等待，因此保留合法流水重叠。

此前 FIFO32 的首次验证范围为 **第 0 层 × 第 0 个 denoising step 的 AE solo**，使用相同输入比较修改前后的模拟 latency 和 native 进程实际耗时。下述历史结果绑定各自的源码提交，不代表当前窗口模型的结果。

## Version2 单层历史记录

以下为此前 **第 0 层 × 第 0 个 denoising step 的 AE latency** 验证记录，固定 **GPU rank0 REFpb、PIM rank1 REFab**。该轮未运行 VLM 校准、完整 AE 或端到端并发实验。Down权重与输入放在同一行：12 fragments/行、22行/44KiB每bank，输入按query散布16个bank，每行统一ACTAB，预写跨BG交错。结果及PIM/GPU breakdown见 [Down同排单层报告](docs/down-row-single-layer.md)；该历史版本的 fused 描述符仍使用 32 项窗口内有限乱序，不能将其数值作为当前按序 FIFO32 的结果。

五种单层总延迟依次为 **345.013 / 316.757 / 287.631 / 318.337 / 289.798 μs**（feature/GPU/stream、token/GPU/stream、token/local/stream、token/GPU/fused、token/local/fused）。源提交 `5fcd84b`、作业 `1981031`；52项Python、224项原生时序检查与5组刷新回归通过，五配置均两次逐字节一致。Token/GPU fused Down由61.918降到45.000 μs（−27.32%），整层仍比同布局streaming慢约0.50%；bank RoPE组合约0.75%。[独立暂存布局基线及VLM比较](docs/rank-refresh-single-layer.md)和更早的[双REFpb结果](docs/version2-single-layer.md)均作为历史对照保留。

独立入口为 `python experiments/run_single_layer.py --output results/down-row-single-layer --repeats 2`；先用当前生成器重建五份程序。该入口仅构造五种单层AE配置，不加载VLM、不调用完整实验入口，并核对两个rank的实际刷新类型和计数。

## Version1 历史验收记录

2026-11-09，已通过 38 项 Python 测试、132 项原生检查、四个时序微基准及 lookahead/barrier 回归、资源生命周期、故障注入和前端验证。五种实际 PIM 程序及八种并发配置的单层检查通过；**五种完整 18 层 × 10 步 AE solo 也已通过独立审计**，覆盖每阶段180次、逐通道DQ字节与占用周期守恒、二进制/运行库/输入来源一致性。

正式 AE 运行绑定源码 `1e0b73bc8da695268fbf79f1013653b54120316e`，`mac_freq_mult=4`，Slurm 数组作业为 `1976674`：

| 程序 | 完整 AE solo / ms |
|---|---:|
| feature_gpu_before | 80.825733142 |
| token_gpu_before | 76.648556502 |
| token_local_before | 72.381632022 |
| token_gpu_after | 97.128931984 |
| token_local_after | 92.878612918 |

验收记录见 [完整 AE 审计](reference/validation/full-k4-final-solos.json)，五份原始输出、配置与provenance见 [结果归档](reference/completed-ae-solos.tar.gz)（[归档校验信息](reference/completed-ae-solos.json)）；旧新差异及口径见 [对比说明](docs/legacy-comparison.md)。构建、微基准和单层并发日志保存在 [reference/validation](reference/validation/)。导出的 spec 与48项命令对矩阵位于 [reference/spec](reference/spec/)。

**按用户2026-11-09要求，VLM校准及后续实验已暂停。** 六个校准候选作业与等待汇总作业已停止，已有日志和冻结输入保留；尚无通过验收的校准参数，完整 b–g/仲裁矩阵及两次重复一致性实验未运行。不得将上述 AE solo 结果当作并发消融结果。暂停位置见 [交接记录](docs/checkpoint-20261109.md)。

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

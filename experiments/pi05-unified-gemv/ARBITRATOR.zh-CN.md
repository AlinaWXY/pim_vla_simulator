# 仲裁器机制与复现

当前入口为 `run_arbitrator.py`，核心为 `sequential_transactions.cpp`。模型放置为 GPU 普通 RD/WR 在 rank0，PIM 在 rank1；同 bank 的 GPU/PIM 共享阵列状态不受支持，启用行状态时会拒绝 rank0 PIM 阵列访问。

## 每个请求检查什么

1. **CA 发射**：在 channel 时钟边沿发射，遵守当前 CA 预约与原生命令宽度。CA 为控制器向 DRAM 发送命令/地址的通道，读写方向切换主要约束 DQ。
2. **独立命令间隔**：`sequential_timing.h` 保留近期传输命令历史。AttAcc 所有传输跨 channel 至少 2 CK，同 BG 至少 4 CK；普通跨 rank RD→RD / RD→WR / WR→RD / WR→WR 分别至少 4 / 15 / 2 / 3 CK。WR→RD 的 2 CK 是原生 sibling 1 CK 与 AttAcc channel 2 CK 共同约束后的结果。
3. **未来 DQ 返回**：RD 在发射后 23 CK、WR 在发射后 12 CK 占用 2 CK；AttAcc WRGB/RDRES 不加 CL/CWL。所有已提交返回均不可撤销，并检查前后方向/rank 保护间隔。
4. **行状态与刷新**：GPU 关闭行先 ACT1/ACT2，同 bank 换行先 PREpb；检查 RCD/RCDWR、RAS、RC、RP/RPab、RTP、写恢复、RRD、FAW、PPD、RFC/REFI。固定刷新前预留 PREab。连续 ACT1/ACT2 是当前调度策略，不代表原生协议只能如此发射。

原生表来自仓库内 `LPDDR5XPIM(pim_data_path="attacc")` 与 `LPDDR5X_8533`；双 rank、WCK always_on、x16、32 B 事务。矩阵测试直接解析原生配置，不用另一张手写表充当期望值。

## 仲裁与进度

- PIM 原生请求序列的间隔作为因果下界。请求受阻后，后续 PIM 序列整体顺延；不重建 PIM bank 状态。
- 同时就绪时 PIM 优先。未发射的未来 PIM DQ 不会提前预约总线；已提交 GPU 返回必须完成。已就绪 PIM 等待时，新 GPU 请求不能通过 CA、DQ 或发射间隔进一步延长这次等待。
- VLM kernel 保持顺序，只在 PIM 阶段启动；已启动 kernel 可以跨过 PIM 边界，后续 AE GPU 算子等待它完成。完成检查点推动虚拟 GPU 进度，不等于 warp/SM 模型。
- c–e 允许细粒度交错。f/g 的整个外部输入/输出 epoch 独占端口，所有权交接先排空已提交返回；内部细命令不占外部 CA。
- GPU 独立参考由同一仲裁器生成并重放，要求完成时间相同、附加等待为零，避免把 GPU 自身行操作和访存冲突全部计入 PIM 干扰。

## 结果口径

```text
raw_model_total = AE_solo + PIM_interference + AE_GPU_wait + VLM_remaining_model
total = raw_model_total - (VLM_model_reference - VLM_retained)
```

`total_ms` 是沿用旧报告的归一化诊断值；`raw_model_total_ms` 是原始模型时间。每阶段的 `PIM_interference` 用相同仲裁器的 PIM 独立重放作基准。全局 PIM 阶段结束取 16 个 channel 中最晚完成时间，不能把各通道等待相加当作端到端延迟。

## 输入与源码

| 文件 | 用途 |
|---|---|
| `reference/evaluation-1108/online-arbitration-preparation/native-programs.json.gz` | 原生 PIM CA/DQ 因果程序及 after 所有权区间 |
| `reference/evaluation-1108/uniform-transactions/native-reservations.json.gz` | 原生刷新日历、时钟相位及来源校验 |
| `reference/evaluation-1108/corrected-overlap/*-explicit_native_io-solo-schedule.csv` | AE 阶段拆分、原生 source tick 范围 |
| `reference/evaluation-1108/online-arbitration-preparation/natural-ae-kernel-ledger.json` | AE GPU 自然缓存计时 |
| `reference/evaluation-1108/gpu/uniform-memory-demand.json` | 2353 个 VLM kernel 的读写事务总量 |
| `reference/evaluation-1108/gpu/warm-summary.json` | 保留的 VLM GPU 计时 |
| `uniform_transactions.cpp/.py` | 事务、日历、绑定、到达计划等公共实现；旧固定日历入口不是当前主模型 |
| `online_transactions.cpp/.py` | 不启用 GPU 行状态的因果版本；同样修复了发射间隔 |
| `sequential_transactions.cpp/.py`、`sequential_rows.h`、`sequential_timing.h` | 当前带 GPU 行状态的仲裁器 |
| `arbitrator-audit/` | 只读插桩、e 全通道统计、原始基线空闲统计和绘图 |

数据准备脚本 `prepare_online_program.py`、`prepare_uniform_bus.py`、`prepare_uniform_demand.py`、`extract_natural_ae.py` 和 `collect_uniform_vlm_memory.py` 用于原始采集后的整理。正常复现直接使用随分支发布的完整冻结输入。

## 操作命令

在仓库根目录执行，Python 3.10+、g++ C++20：

```bash
python3 experiments/pi05-unified-gemv/run_arbitrator.py --output outputs/arbitrator --audit-e
python3 experiments/pi05-unified-gemv/verify_arbitrator.py --results outputs/arbitrator/with-rows --audit outputs/arbitrator/e-dq-insertion-audit

# 只跑 e，不改其他参考结果
python3 experiments/pi05-unified-gemv/run_arbitrator.py --output outputs/arbitrator-e --cases e --audit-e

# 可选绘图，需要 Matplotlib；中文图建议安装 Noto Sans CJK SC 或 Microsoft YaHei
python3 experiments/pi05-unified-gemv/arbitrator-audit/analyze_and_plot.py --data outputs/arbitrator/e-dq-insertion-audit

# 重算原始 GPU 广播基线空闲窗口，读取随仓库发布的 16 个原生 channel trace
python3 experiments/pi05-unified-gemv/arbitrator-audit/analyze_broadcast_idle.py --output outputs/arbitrator/broadcast-idle
```

`native-timing-fix.sbatch` 保留原 job 1974571 的运行记录与路径；跨机器使用上面的 Python 入口。历史 `SHA256SUMS.json` 保留原运行来源（其中 `.so` 不随分支分发，编译器不同也可能产生不同二进制）；本次发布文件以 `reference/arbitrator-publication/SHA256SUMS.json` 为准。

## 已知边界

混合 buffer/array 的 1/2 CK 跨 rank DQ 保护间隔是保守扩展，原生 AttAcc 表没有对应的明确规则。固定刷新接纳与连续 ACT 策略也可能严于协议最小间隔。

均匀事务到达、每 kernel/channel 从 sector0 开始的连续地址、L2 write sectors 的同步写代理仍是假设。bank 状态只在连续 PIM 阶段内保留；没有真实物理地址复用。GPU→PIM 搬运 kernel、SM/copy engine 竞争、输入 cache 驻留和补充尚未计费。AE 非线性 GPU 算子与 VLM 已串行，但不能据此推断所有搬运期间 VLM 都能自由计算。

VLM 模型与已有计时差距显著，归一化偏差在并发场景中能否抵消未经硬件验证。此分支用于检查资源时序、统计插入位置和定位模型问题，不作为硬件性能结论。

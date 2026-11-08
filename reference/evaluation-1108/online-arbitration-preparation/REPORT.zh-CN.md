# b–g：因果总线重放、融合后隔离与自然缓存 AE 计时

**这是条件模型估计，尚未实现原生控制器上的 GPU/PIM 联合执行，也不是硬件端到端实测。** 采用用户确认的均匀访存假设；不再使用 ρ、预设在途数或读写比。

作业 `1972166`，分支 `xinyao_1108`；原生 PIM 输入为 `evaluation-1971385`。本次不重跑纯 GPU/VLM benchmark。

| 配置 | 定义 | AE 独立/ms | PIM 被 GPU 推迟/ms | 后续 AE GPU 等待/ms | 净重叠收益/ms | 校准总延迟/ms | 原始模型/ms |
|---|---|---:|---:|---:|---:|---:|---:|
| b | feature-wise / GPU RoPE / 融合前 / 串行 | 55.013 | 0.000 | 0.000 | 0.000 | **159.963** | 167.177 |
| c | feature-wise / GPU RoPE / 融合前 | 55.013 | 7.995 | 62.746 | 19.661 | **140.303** | 147.516 |
| d | 当前 layout / GPU RoPE / 融合前 | 51.921 | 6.996 | 56.882 | 19.937 | **136.935** | 144.148 |
| e | 当前 layout / 本地 RoPE / 融合前 | 47.089 | 6.967 | 48.445 | 20.837 | **131.203** | 138.417 |
| f | 当前 layout / GPU RoPE / 融合后 | 60.963 | 0.109 | 47.303 | 28.940 | **136.974** | 144.187 |
| g | 当前 layout / 本地 RoPE / 融合后 | 56.089 | 0.145 | 32.726 | 29.158 | **131.881** | 139.095 |

e→g：AE 独立时间变化 +8.999511 ms；净重叠收益变化 +8.321181 ms；总延迟变化 +0.678330 ms。f/g 的隔离并不自动消除输入落 DRAM、结果导出和已启动 VLM kernel 对后续 AE GPU 算子的等待。

本轮 g 比 e 慢 0.678330 ms。另一项对账表明，本次每个 PIM 阶段均被已启动 VLM kernel 覆盖（包括访存阻塞），因此总延迟等于保留的 VLM 时间 + AE GPU 算子时间 + VLM 额外访存阻塞。e/g 的 AE GPU 工作分别为 19.171626/18.273066 ms，VLM 额外访存阻塞分别为 7.080654/8.657544 ms。这是当前输出的核对关系，不是另行以空闲时间推算进度，也不能证明真实硬件的 e/g 排序。

## c–e 与 f/g 使用不同的资源规则

c–e 无 rank controller、无输入 DRAM staging：GPU 持续 WRGB 广播，细命令占外部 CA。每次 GPU RD/WR 发射必须同时预留 CL/CWL 后的完整 DQ burst，检查读写切换和目标 rank refresh。已经发出的 GPU 返回不可撤销；尚未发出的 PIM 命令可等待这些返回。已就绪的 PIM 命令优先，后来的 GPU 请求不能继续延长它的等待。

f/g 有 rank controller、粗命令、输入 DRAM staging 和本地读取。本地细命令不占外部 CA，计算阶段不存在 c–e 那种 GPU/PIM 细命令竞争。整个输入/输出阶段使用外部总线独占交接；交接前排空已预约返回，阶段内部不插入 GPU 访存 burst。外部粗命令本身仍使用其原生 CA 时长。所有 f/g 外部 DQ burst 已核对位于这些交接阶段。

两类配置都保留 GPU 自身的内存时序和 refresh。tCK=0.938 ns、CL=23 CK、CWL=12 CK、burst=2 CK；REF 的外部 CA 命令为 1 CK，目标 rank blackout 为 224 CK。1 CK 的 CA 占用不能与 224 CK 的 rank 阻塞混为一谈。

## kernel 顺序和进度

VLM kernel 按原顺序启动，不要求完整 kernel 放入单一窗口；允许访存被阻塞并跨阶段继续执行。PIM 活跃时启动的 VLM kernel 若未完成，后续 AE GPU 算子等待它完成。本模型没有预测未来全部 PIM DQ 占用来选择 GPU 发射槽，只查询已释放命令、已预约返回和已知 refresh 时序。

独立 GPU 请求流先用同一个因果仲裁器执行，记录实际发射和完成时刻，作为后续重放的进度检查点；重新执行独立流必须得到相同完成时刻且额外 stall 为零。竞争导致返回超过检查点时才推迟后续进度，未把任意正长空隙直接计为 GPU 进度。这仍是请求完成检查点抽象，没有重建 warp/SM 内部计算依赖。

PIM 被阻塞后，后续原生命令间隔整体顺延；各 channel 分别执行。该方法保存原生因果间隔，但没有重新生成 bank/row 状态，不能据此声称已通过完整 DRAM 控制器验证。

## 计时和访存证据

VLM 请求量沿用独立单 pass 采集的 2353 个 kernel 计数和 4 次 memcpy 字节数。读侧为 L2 read-miss sectors；写侧为 L2 write sectors 的全写回代理，后者不是直接测得的 DRAM 写量。每个 kernel 各有自己的读写量，没有人为固定读写比。均匀分配到 16 channel、4 bank group，并假设 GPU 请求命中 rank 0 已打开的行；缺少真实地址、ACT/PRE 和内存依赖。

AE 原 GPU 算子改用已有自然缓存 trace：5 个请求 × 10 个 denoising step，每步 1033 个 kernel，名称与原有 ledger 逐项对齐。RoPE 原始 kernel 时长和为 **5.528525 ms/AE**，旧逐 kernel 清缓存值为 11.656213 ms/AE。本次替换所有保留的 AE GPU 算子计时，而非只替换 RoPE，所以不能把新旧主表之差全归因于仲裁策略。

框架是 OpenPI PyTorch eager，未使用 torch.compile、CUDA Graph 或 fused RoPE；原实验扩展了物理 MHA head。自然缓存记录本身带 profiler，部分 kernel 时间戳有重叠；按顺序累加其时长是抽象，并非优化后的推理端到端时间。PIM↔GPU 专用交接算子仍沿用已有 logical-operation cold32MiB 微基准，未假装重新测得自然缓存交接数据。

## 总延迟口径

`净重叠收益 = 已完成 VLM 参考工作量 − PIM 被 GPU 推迟的时间 − 后续 AE GPU 等待`

`校准总延迟 = AE 独立时间 + 保留的 VLM latency − 净重叠收益`

`原始模型总延迟 = AE 独立时间 + PIM 推迟 + AE GPU 等待 + 剩余 VLM 参考工作量`

保留的 VLM latency 为 104.950496035 ms；同一因果仲裁器下的独立 VLM 参考为 112.164321085 ms，偏差 7.213825051 ms。CSV 同时保存原始模型值与校准值；校准假设此参考偏差在配置比较中抵消，尚未经并发硬件实测验证。两列不是置信区间或严格上下界。

未臆造 CPU dispatch、跨设备通知、总线所有权交接或队列容量数值，这些实现成本因此没有被本模型测量。当前结果不应直接用来宣称相对纯 GPU 端到端的硬件加速比。

## 实时实现边界

CA 发射时预留未来 DQ 返回可以由内存控制器实现，不需要预知整个 GPU kernel。c–e 要有可反压的 PIM 命令队列和已提交返回记分牌；f/g 主要需要计算与外部 I/O 阶段的所有权交接。两者都不能仅凭 CUDA stream priority 获得纳秒级外部总线控制。

现有 native `lpddr5_controller.cpp::can_overlap` 仍仅支持 PIM_MAC_HB 与普通 Read 的特定跨 rank/half 情形，没有接入本次通用策略。因此这里交付的是独立因果总线模型和数据，不是已实现的 Thor 并发运行时。

## 验证与复现

31 项测试通过；768 份原生源文件哈希复核；20 组原生配置、843264 个 DQ 事件逐项一致；f/g 所有 DQ burst 的交接阶段覆盖检查通过。全部输出核对 kernel 顺序、每 kernel 字节数、访存 stall、PIM 延迟、AE GPU 等待和总延迟等式。

入口为 `experiments/pi05-unified-gemv/online-transactions.sbatch`；结果在 `evaluation-isolated/`。`independent-audit.json` 保存审计结果，`SHA256SUMS.json` 保存源码、输入和结果清单。`evaluation/`、`pre-reference-normalization/` 为调试中间产物，不是本报告主表。

旧 `uniform-transactions` 表使用已知未来 PIM 日历且保持 PIM 时间固定，并沿用逐 kernel 冷缓存 AE 计时，仅保留历史溯源。当前主表同时更改了因果调度、融合后隔离与 AE 计时，不能作为单变量消融。

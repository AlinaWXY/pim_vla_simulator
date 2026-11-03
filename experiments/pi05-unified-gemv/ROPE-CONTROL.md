# RoPE 为什么在新版变慢：原指令与控制路径

RoPE 没有新增广播，读写数据量和算术参数也没有改变。`4073452` 到 `cbc1ee4` 的变化来自统一启用 rank controller：本地命令开始竞争 rank CA，同时既有单 bank 细指令被逐条施加 global CA 准入。后者与 trace 顺序结合，产生跨 channel 队头阻塞（head-of-line blocking，HOL）。RoPE 的大幅增长不能全部算作输入暂存或广播方案的硬件代价。本文前四节解释该历史问题，最后一节说明当前默认模式的修正；旧行为通过 `--legacy-rank-dispatch` 保留。

## 原来的指令是什么

H16 的每个 channel 持有 8 个本地 query。对每个本地 query，双缓冲模式生成以下七组请求；每组遍历 16 个 channel、每 channel 的 16 个 PIM bank。

```text
PIM_ROPE_COEF  coefficient_address
PIM_ROPE_LOAD  q_address  0
PIM_ROPE_EXEC  q_address  0
PIM_ROPE_LOAD  k_address  1
PIM_ROPE_EXEC  k_address  1
PIM_ROPE_STORE q_address
PIM_ROPE_STORE k_address
```

每条请求指向 **单个 bank 的一个 32 B 向量**，不是对整 rank 的广播。COEF、LOAD 转为 LOCALRD，STORE 转为 LOCALWR；EXEC 使用已装载的输入与系数，处理该向量的 8 个 RoPE pair。序列源码见 [horizon_attention_workload.py 的 rope](source/horizon_attention_workload.py#L133)，遍历顺序见 [active_bank_locations](source/horizon_attention_layout.py#L225)。该序列在旧版和新版相同。

每 channel 共 `8 × 7 × 16 = 896` 条 RoPE 请求，全部 16 channel 共 **14336 条**。每 channel 的操作量为 COEF128、LOAD256、EXEC256、STORE256，即 LOCALRD384、LOCALWR256、EXEC256。Q/K projection 的 spill 在此前独立执行，不属于这七组 RoPE 请求。

## 为什么旧版没有同样的问题

旧基线 `4073452` 的 RoPE 走 bank-local 路径。LOCALRD/LOCALWR 的最终本地操作不占共享 CA，多个 bank 的本地命令能够同 CK 发出；必要的 ACT/PRE 仍走 DRAM 行准备和相应时序。EXEC 也没有共享 rank CA 准入，实测同一个 CK 可以发出 16 个 bank 的 EXEC。输入就绪、系数复用、两个输入 slot、输出 FIFO 和计算流水依赖仍受检查，旧版并非所有操作都零延迟。

前端配置为 **32 个 frontend tick / memory CK**，每个 frontend tick 尝试发送一条请求。旧路径没有新增的逐请求 1 CK global CA 门限，按 channel 排列的连续请求能较快进入队列；原有队列背压仍会等待，但没有新版每条 PIM 请求触发的准入等待。前端仅在发送成功时推进 trace，逻辑见 [loadstore_trace.cpp:283](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/frontend/impl/memory_trace/loadstore_trace.cpp#L283)；32:1 配置见 [config_for](reproduce.py#L45)。

## 新版增加了哪些控制限制

`cbc1ee4` 的 staged 模式启用 `pim_rank_controller=True`，开关作用于整个 PIM controller，不只作用于新输入广播。具体有三项变化：

1. **本地 array 命令竞争 rank CA。** device admission/observer 钩子让 LOCALRD、LOCALWR 与同 rank 其它命令共享 CA，见 [rank_staging_extension.inc:78](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/rank_staging_extension.inc#L78)。
2. **EXEC 也占 rank CA。** `rank_pseudo("PIM_ROPE_EXEC")` 占用 1 CK，见 [rank_rope_extension.inc:132](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/rank_rope_extension.inc#L132) 和 [rank_pseudo](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/rank_staging_extension.inc#L71)。
3. **每条细指令都先占 global CA 1 CK。** `send()` 依据 `req.type_id >= PIM_MAC_AB` 将所有这些 PIM 请求视为 coarse 请求，在 global CA 忙时返回 false，见 [lpddr5_controller.cpp:252](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/lpddr5_controller.cpp#L252)。RoPE 仍以单 bank、32 B 为请求单位，没有封装为由 rank controller 自主展开的 RoPE 循环描述符。

如果硬件假设是 rank 内传数与计算共用一条命令总线，前两项是合理的统一仲裁，不需要操作包含广播。第三项中的粗细粒度选择属于实现政策；它与单前端、channel-major 顺序产生的跨 channel 阻塞，并非共享 rank 内 CA 的必然结果。

生成器先连续写出 ch0 的 16 个 bank 请求，再写 ch1 的 16 个，依次到 ch15。前端对同一 channel 的第二条请求遇到 1 CK 准入门限时，无法跳过它向其它 channel 派发。这使多个独立 channel 大部分时间轮流获得请求，再进入下一组 COEF/LOAD/EXEC/STORE；请求尚未到达时，rank CA 即使空闲也无法执行它。

## 1102 日志中的量化证据

以下是历史运行 `baseline-1902586` 与 `validated-1902586` 的数据，分别对应 `4073452` 直接输入模式和 `cbc1ee4` 单 bank、成对 SRF 模式；**不是 1103 双 bank 的新测量值**。原始文件位于相应运行的 `chain/simulation/`，可核对 `barriers.csv`、`bgk_events.csv.ch0`、`rank_ca_events.csv.ch0` 和各 channel 同名文件。

| 指标 | 旧直接输入基线 | 1102 单 bank + rank controller |
|---|---:|---:|
| RoPE 阶段 latency | 0.529032 µs（564 CK） | 12.652682 µs（13489 CK） |
| projection spill latency | 0.206360 µs | 3.812970 µs |
| 单次 LOCALRD 数据就绪延迟 | 25 CK | 25 CK |
| 单次 RoPE EXEC 完成延迟 | 11 CK | 11 CK |
| 单次 LOCALWR 服务及恢复 | 41 CK | 41 CK |
| 每 channel RoPE 请求数 | 896 | 896 |

1102 的 ch0 global CA 派发日志给出以下前三组：

| 请求组 | ch0 派发时刻（memory CK） |
|---|---|
| 第一个 query 的 COEF，16 banks | 29509–29524 |
| 同 query 的 LOAD Q，16 banks | 29749–29764 |
| 同 query 的 EXEC Q，16 banks | 29989–30004 |

相邻组的开始相隔约 **240 CK**，主要是前端遍历其余 channel，不是一次读必须等待 240 CK：ch0/bank0 的 LOAD Q 在 29750 CK 发出，29775 CK 数据就绪，EXEC Q 却直到 29990 CK 才发出。

整个 RoPE 的 **14336 条跨 channel 请求**从 29509 CK 派发到 42956 CK，包含最后 1 CK 派发槽的跨度为 **13448 CK = 12.614224 µs**，已接近整个阶段的 12.652682 µs。与此同时，ch0/rank1 CA 仅实际占用 **1030 CK = 0.966140 µs，即阶段时间的 7.64%**；它没有达到命令总线饱和。该占用包含本地命令、EXEC、行准备与该 rank 的 refresh 命令。

旧版同 channel 的 256 条 EXEC 分布在 16 个不同 CK，最多 16 banks 同 CK 发出；新版分布在 256 个不同 CK，最多每 CK 1 条。这说明 rank CA 仲裁改变了可并行发出的命令数量，但不足以把整个 24 倍增长解释为满载 CA 排队。阶段变长后跨越 refresh、重新激活工作行，又带来少量额外开销；读写字节数、计算量和上述单次服务时间均未增加。

## 当前默认模式怎样修正

`786842a` 的原始 1103 只改变双 bank 暂存，仍有上述派发问题。当前修订保留双 bank 布局、所有数据依赖和共享 rank CA，通过两个可独立对照的控制变化修正它：

1. **独立 channel 入口。** 前端仅暴露当前合法阶段，为各 channel 保持独立 FIFO 指令次序。每个 channel 有容量为 32 的 memory-system 入口请求队列；一个 channel 遇到 controller/global CA 背压时，其它 channel 可以继续前进。每个 memory CK，每个 channel 最多尝试派发一条队首请求。全局 BARRIER/WAIT、普通访问与 PIM 的完成边界以及 Q_REMOTE 的 peer Q_EXPORT 完成依赖不被绕过。
2. **rank 描述符。** 一条 rank 请求在 controller 内展开该 rank 的 16 个原 bank 请求，原子操作不再分别向总 controller 申请 coarse 派发；实际 LOCALRD/LOCALWR、RoPE EXEC 和数据传输仍按原时序发出，并竞争 rank CA。接收父请求需要原有 PIM 队列中至少 16 个空位，父完成等待全部 16 个子操作完成。

新 trace 语法如下，`ADDRESS` 为目标 channel 的 rank1/BG0/bank0 地址，保留对应 row/column；LOAD、EXEC 和 REMOTE 的 `SLOT` 为 0/1（REMOTE 固定 0）：

```text
PIM_RANK_SPILL      ADDRESS
PIM_RANK_ROPE_COEF  ADDRESS
PIM_RANK_ROPE_LOAD  ADDRESS SLOT
PIM_RANK_ROPE_EXEC  ADDRESS SLOT
PIM_RANK_ROPE_STORE ADDRESS
PIM_RANK_Q_EXPORT   ADDRESS
PIM_RANK_Q_REMOTE   ADDRESS SLOT
```

一条描述符覆盖所有 16 个 bank 的同一逻辑向量位置，不是把不同 bank 的数据广播成同一份。Q_EXPORT 和 Q_REMOTE 仍产生全部真实 RD/WRGB 外部 DQ；projection spill 仍写入全部本地列。Q_EXPORT 借普通读队列执行的 RD 及其行准备使用请求级 rank-fine 标记，只占 rank CA，不因共用普通 scheduler 而再次收取 global CA；原 ACT2/CAS/WCK、DQ 与读完成路径保留，见 [内部普通读的 CA 归属](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/lpddr5_controller.cpp#L242)。每个本地 query 仍按前述七组 RoPE 操作执行，只是每组现在每 rank 一个父请求。因此 RoPE 的全 channel **14336 个细操作变成 896 个父描述符**，细操作数与读写字节数不变。`PIM_MAC_AB`、`PIM_MAC_QK` 等 GEMV 计算指令维持原样。

默认 CLI 开启两项修正；`--no-rank-descriptors` 只开启独立 channel 派发，可量化前端变化；`--legacy-rank-dispatch` 同时关闭两者，复现修改前 1103。需要旧 1102 单 bank 对照时再加 `--staging-banks 1`。直接输入 baseline 始终关闭这两个新开关。

生成器在 [Writer.banks](source/horizon_attention_workload.py#L61) 压缩请求，native 在 [send_rank_descriptor](../pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/rank_descriptor_extension.inc#L42) 校验并展开。`rank_descriptors.csv.chN` 记录 DISPATCH、16 个 CHILD_COMPLETE 和最终 COMPLETE，可核对完成时间；`frontend_dispatch.csv`、`channel_ingress.csv` 可核对跨 channel 进度、有限队列深度与 FIFO 次序。

这里的 0.529032/12.652682 µs 等数字仍专指上述历史结果，不能作为本次修正后的测量值。修正后的 latency 必须来自当前代码的新仿真，并分别对照旧模式及仅修正前端的 ablation，不能由直接扣除“估计 HOL 时间”得到。

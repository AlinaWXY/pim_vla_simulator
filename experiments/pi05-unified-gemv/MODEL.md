# xinyao_1102 单 bank 输入暂存与 rank controller

本实验从 `xinyao_1101` 的 `4073452`（Full DQ、GPU GeLU/multiply、Down 按输出维分片）派生。权重、KV cache、计算指令及输出归属保持一致。新增的是 GPU 广播输入的 DRAM 暂存和 rank 内命令仲裁。

## 输入路径与存储布局

每个 channel 的物理 rank0 保留给 GPU，rank1 包含 16 个 PIM bank。所有需要在 rank 内广播的 GPU 输入先完整写入该 rank 的 **BG0/bank0 DRAM array**。全部写入和恢复完成后，rank controller 才读取暂存数据到该 bank 的 GEMV buffer（SRF），并广播到同 rank 其余 15 个 bank 的同一 SRF slot。

```
GPU → channel DQ → rank1 bank0 DRAM（整批输入）
                         ↓ 普通行准备 + LOCALRD
                    bank0 SRF
                         ↓ rank 内广播
                bank1…bank15 的 SRF
                         ↓
                 原有 PIM MAC 指令
```

| 输入阶段 | 每 rank 必需输入 | 暂存字节/rank | 暂存 DRAM rows | 每个 SRF fragment |
|---|---|---:|---|---|
| Q/K projection | 本 rank 的 8 token × 1024 feature X；Q/K 共享一次输入 | 16 KiB | 256–263 | 8 × 16 BF16 = 256 B |
| V projection | 16 × 1024 X | 32 KiB | 256–271 | 16 × 16 BF16 = 512 B |
| PV | 16 query × 992 physical KV position P | 31 KiB | 256–271，最后一行只使用 1 KiB | 16 × 16 BF16 = 512 B |
| Gate/Up | 16 × 1024 X；Gate/Up 共享一次输入 | 32 KiB | 256–271 | 16 × 16 BF16 = 512 B |
| Down | 16 × 4096 GPU Z | 128 KiB | 256–319 | 16 × 16 BF16 = 512 B |

PV 的逻辑 KV 长度为 968 + 16 = 984，物理长度按原布局补齐到 992；暂存包含原本就传输的 padding。各算子复用 row256–319 的最多 128 KiB DRAM 保留区，不与已有权重/cache 区域重叠。每 bank 原有 **两个 512 B SRF slot，共 1 KiB**，保持不变；新增 128 KiB 是一个 bank 内的 DRAM 地址区，不是额外 SRAM。

暂存采用 `[input_fragment][token][lane16] BF16`：

`bank_offset = 524288 + (fragment × rows_per_fragment + token) × 32`。

Q/K 使用 64 个 fragment、每个 8 token；V/Gate/Up 使用 64 个 fragment、每个 16 token；PV 使用 62 个；Down 使用 256 个。fragment 轮流使用 slot0/slot1。GPU 源 tensor 的打包/可用时间与基线一样未计入。

QKT 的 Q 是按 feature 定向送到对应 bank，**不是同一份数据向所有 bank multicast**。因此原有 `PIM_Q_EXPORT`、`PIM_Q_REMOTE` 及其 GPU 往返路径保留；其命令仍受新 rank 命令仲裁。V suffix cache 的普通 DRAM 回写也保留。

## 时序与命令

- `PIM_STAGE_BEGIN ADDRESS BYTES ROWS` 声明一个 rank 的完整输入，`ROWS` 为 8 或 16。
- 随后的普通 `ST` 使用真实 DRAM `WR`：计入 ACT/PRE、CWL、DQ burst、列间隔和写恢复，不直接填 SRF。
- `PIM_STAGE_SEAL ADDRESS` 必须等待全部声明列写入完成；它使整批输入可供消费。
- `PIM_STAGE_FRAGMENT ADDRESS SLOT` 是一个完整 fragment 的粗粒度描述符：Q/K 为 256 B，其它输入为 512 B。rank controller 将其展开成 8/16 次真实 bank0 `LOCALRD` 和对应的 SRF 广播；这些细粒度操作不再向总 controller 单独请求 CA。`PIM_STAGE_LOAD ADDRESS SLOT` 的单 32 B 入口只保留给局部测试。
- bank→SRF 与 SRF→同 rank SRF 的数据延迟均包含在 DRAM 的 `nCL + nBL_min` 内，不额外加数据传输周期；广播命令本身仍与读取、计算争用 rank 命令总线。
- 源 bank 的暂存行和该 bank 的权重行共享真实行状态。因此加载输入与全 bank MAC 之间的 PRE/ACT 切换会计入，不假设独立暂存 SRAM 或额外读端口。
- 输入覆盖、前次消费完成、双缓冲 slot 复用和 release 检查保持生效。所有计算沿用原有 PIM 指令：Q/K projection 为 8 CK/MAC，其余 GEMV 为 16 CK/MAC，QKT 及 RoPE 参数保持基线。

总 controller 负责向 rank controller 下发请求描述符，每个描述符占用 1 CK 总 CA。输入装载的粗粒度单位是一个 256/512 B fragment：后者由 rank controller 展开成 8/16 次读取所需的 PRE/ACT、LOCALRD 和 BCAST，而不是总 controller 逐条下发这些细粒度命令。MAC 仍按原有 PIM 指令逐条下发。该模型没有假设一个描述符启动整个算子的自主循环。新模式中只移除旧的自动生成 channel CA 占用约束，由 native controller 显式记录总线仲裁；显式 DRAM 行/列/refresh 和 DQ 约束全部保留。日志分别保存 staging 生命周期、rank command bus、DRAM command timeline 和 SRF 生命周期。staged load 按队列顺序执行，不越过前面的 MAC 请求；它可以与已发出但尚未完成的计算重叠，但没有为每个 32 B 输入额外优化跨 fragment 的预取顺序。

## 流量与统计解释

每 rank 仍只接收一份所需 GPU 输入，rank 内广播不重复占用外部 DQ。整批暂存改变 DQ 使用的时间分布，**不会降低原输入字节数**。五阶段合计的外部流量仍为 **5472 KiB/层/步（全部 16 channel）**；其中本次改为单 bank 暂存的输入为 **3824 KiB**。读取这些输入的同量 DRAM 内部流量及向 16 个 SRF 写入的合计流量单独记录。

报告区分累计 DQ 占用、完整无外部传输工作窗口、其它 DQ 空隙。完整窗口可用于评估 GPU channel 带宽机会，但未向 rank0 注入真实并发 GPU 请求，故不能直接称为已测量的 GPU 可用时间。新增粗粒度/细粒度命令仲裁为时序模型，不是已验证的硬件接口或电路实现。

沿用基线的未建模范围：GPU softmax、GeLU/multiply/cast/pack、O projection、norm/residual、GPU dispatch 和后台 VLM 竞争均未计时；实验不计算 BF16 tensor 数值，完整 AE 端到端 latency 保持未定义。

## 复现模式

默认 `reproduce.py` 使用 `pim_rank_staging=True` 与 `pim_rank_controller=True`。`--direct-input-baseline` 恢复旧 WRGB 直接广播路径及旧命令调度，用于同一份源码的对照。底层 generator 的 `rank_staging=False` 入口保留，原有故障注入 guards 仍覆盖该兼容路径。

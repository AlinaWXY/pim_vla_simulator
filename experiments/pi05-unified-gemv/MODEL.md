# xinyao_1103 双 bank 输入暂存与 rank controller

本实验从 `xinyao_1102` 的 `cbc1ee4` 派生；其直接输入基线是 `xinyao_1101` 的 `4073452`。1103 将 GPU 广播输入分布到不同 bank group 的两个 DRAM bank，并交错写入。权重、KV cache、计算指令、输出归属及每 bank 的两个 SRF slot 保持不变。当前默认模式在双 bank 数据路径上进一步启用 channel 独立派发和 rank 粗粒度描述符，修正旧模式的跨 channel 队头阻塞。[RoPE 命令说明](ROPE-CONTROL.md) 保留问题的历史证据，并解释本次控制路径修改；`--legacy-rank-dispatch` 可复现修改前的 1103。

## 输入路径与双 bank 布局

每个 channel 的物理 rank0 保留给 GPU，rank1 有 16 个 PIM bank。默认使用 rank1 **BG0/bank0（flat bank 0）** 和 **BG1/bank0（flat bank 4）** 暂存需要广播的 GPU 输入。整批输入的普通 DRAM 写入及写恢复完成后，rank controller 才从各 fragment 的源 bank 读取到该 bank 的 GEMV buffer（SRF），再广播到同 rank 其余 bank 的对应 SRF slot。

```text
GPU → channel DQ → rank1 bank0 / bank4 DRAM（整批输入，交错 ST）
                         ↓ 各 fragment 从自身源 bank 连续 LOCALRD
                    源 bank 的 SRF
                         ↓ 同 rank 广播
                 全部 16 banks 的同一 SRF slot
                         ↓
                    原有 PIM MAC 指令
```

两个源 bank 各保留 **64 KiB、row256–287**，总计仍为 **128 KiB/rank**。该区域属于 DRAM array，不是新增 SRAM；各算子依次复用。每个物理 bank 仍只有 **两个 512 B SRF slot，共 1 KiB**。

| 输入阶段 | 每 rank 的输入 | 总字节/rank | 每个源 bank 实际字节 | 每个源 bank 实际使用的 DRAM rows | 每个 fragment |
|---|---|---:|---:|---|---|
| Q/K projection | 本 rank 的 8 token × 1024 feature X；Q/K 共享 | 16 KiB | 8 KiB | 256–259，共 4 行 | 8 × 16 BF16 = 256 B |
| V projection | 16 × 1024 X | 32 KiB | 16 KiB | 256–263，共 8 行 | 16 × 16 BF16 = 512 B |
| PV | 16 query × 992 physical KV position P | 31 KiB | 15.5 KiB | 256–263，共 8 行；末行使用 1.5 KiB | 16 × 16 BF16 = 512 B |
| Gate/Up | 16 × 1024 X；Gate/Up 共享 | 32 KiB | 16 KiB | 256–263，共 8 行 | 16 × 16 BF16 = 512 B |
| Down | 16 × 4096 GPU Z | 128 KiB | 64 KiB | 256–287，共 32 行 | 16 × 16 BF16 = 512 B |

PV 的逻辑 KV 长度为 968 + 16 = 984，物理长度按原布局补齐到 992；上述字节数包含原本就传输的 padding。Q/K、V、Gate/Up 各有 64 个 fragment，PV 有 62 个，Down 有 256 个。

令 `f` 为逻辑 fragment，`t` 为 fragment 内 token，`R` 为每 fragment 的 token 数（Q/K 为 8，其余为 16）。默认双 bank 映射为：

```text
physical_rank = 1
bank_group    = f % 2
bank          = 0
flat_bank     = 4 * (f % 2)
bank_offset   = 524288 + (floor(f / 2) * R + t) * 32
SRF slot      = f % 2
```

每个 fragment 的 8/16 个 32 B 列在自身源 bank 内连续排列。公式与保留区见 [地址 helper](source/rank_staging.py#L17) 和 [reservation](source/rank_staging.py#L94)。manifest 的 `staging_inputs` 保存 `fragment_mapping`、`store_sequence`、`store_ranges` 和两个源 bank 的 reservation；`store_sequence` 同时记录 trace 行号范围，串联控制实验同步平移这些范围。

## GPU 写入顺序与成对读取

GPU 暂存写入按 **fragment pair → token → source bank → channel** 排列。对于一对 fragment `(2p, 2p+1)`，每个 channel 看到：

```text
ST bank0(fragment 2p, token 0)
ST bank4(fragment 2p+1, token 0)
ST bank0(fragment 2p, token 1)
ST bank4(fragment 2p+1, token 1)
...
```

相邻写请求交替 bank group，为模型中的跨 BG 2 CK 写列间隔提供机会；实际发出时间仍由 DRAM 列时序、行准备、DQ 和 refresh 决定，不在 trace 中插入固定等待，也不保证每条 WR 都恰好间隔 2 CK。生成循环见 [stage_input](source/rank_staging.py#L49)。

整批输入经 `PIM_STAGE_SEAL` 完成后，沿用两个 SRF slot 的成对消费顺序：

```text
FILL fragment 2p   ：bank0 DRAM → 源 SRF slot0 → 全 rank slot0
FILL fragment 2p+1 ：bank4 DRAM → 源 SRF slot1 → 全 rank slot1
MAC fragment 2p
MAC fragment 2p+1
```

每个 descriptor 内仍从自身源 bank 连续读取；本次没有把两个 descriptor 的 LOCALRD 改成跨 bank 交错流水，也没有令下一对 staged load 越过前面的 MAC。先填好两个 slot，再发对应 MAC，消费完后再装载下一对。V/PV/Gate/Up/Down 每批向每个 bank 的两个 SRF 共填入 1 KiB；Q/K 每批共 512 B，仅使用每个 slot 的前 256 B。成对策略见 [fragment_batches](source/rank_staging.py#L37)。

## 时序与命令

- `PIM_STAGE_BEGIN ADDRESS BYTES ROWS` 仍使用 canonical rank1/BG0/bank0 的 base 地址，声明两个源 bank 合计的完整输入字节数；`ROWS` 为 8 或 16。bank 数量由配置 `pim_staging_banks` 指定，trace 指令格式不变。
- 普通 `ST` 使用真实 DRAM `WR`，计入 ACT/PRE、CWL、DQ burst、列间隔和写恢复，不直接填 SRF。
- `PIM_STAGE_SEAL ADDRESS` 仍使用 canonical bank0/base，等待所有声明列在各源 bank 的写入及恢复完成。
- `PIM_STAGE_FRAGMENT ADDRESS SLOT` 指向 fragment 的真实源 bank。rank controller 展开 8/16 次 LOCALRD 与对应 SRF 广播；`PIM_STAGE_LOAD` 的单 32 B 入口保留给局部测试。
- bank→SRF 和 SRF→同 rank SRF 的数据延迟包含在 `nCL + nBL_min` 内，不额外增加数据传输周期。广播命令仍与读取、计算争用 rank CA。
- 两个源 bank 都保留权重计算职责。暂存行和权重行共享各 bank 的真实行状态，加载与全 bank MAC 之间的 PRE/ACT 成本继续计入，不假设独立 array 读端口。
- 输入覆盖、写完成、SRF 消费完成、slot 复用和 release 检查继续生效。Q/K projection 为 8 CK/MAC，其余 GEMV 为 16 CK/MAC，QKT 和 RoPE 参数保持基线。

内部 staging 的 ACT1/ACT2 与 ordinary RD/WR 使用相同的 `can_start_act1` lookahead：只有下一命令周期的 ACT2 已满足 DRAM 与 rank CA 约束，才发出 ACT1。fragment 请求保留在 PIM FIFO 队首，并登记普通控制器的 bank owner 与 `ACT1_clock+nAAD` deadline。已开始的激活优先完成 ACT2，刷新和其它命令随后恢复调度；ACT2 仍占 rank CA，并保留 nRP、nRFC、nRCD 等时序。超时或 owner 丢失会报错。SRF、数据布局、预取批量及所有时序参数均不改变。

本项修正仅覆盖 `PIM_STAGE_LOAD/FRAGMENT` 的内部激活。完整命令日志另行审计其它内部路径，并显式报告尚未修正的非 staging 协议问题；staging 专项通过不能等同于所有内部路径均满足 ACT 协议。

总 controller 对输入装载继续采用 256/512 B fragment 描述符；内部 PRE/ACT、LOCALRD、BCAST 由 rank controller 发出。新默认模式还把 projection spill、RoPE COEF/LOAD/EXEC/STORE、Q_EXPORT/Q_REMOTE 合为 rank 级描述符，每个描述符占用 1 CK global CA，再在 rank controller 内展开 16 个原有单 bank 操作。所有实际 bank 操作继续竞争 rank CA；原有 `PIM_MAC_AB`、`PIM_MAC_QK` 等 GEMV 计算指令不改变。

描述符地址使用目标 channel 的 canonical rank1/BG0/bank0；row/column 和可选 slot 对所有 16 个 bank 相同。只有全部子操作完成，父描述符才回调完成。native 仅在现有 PIM 请求队列有 16 个空位时接收描述符，保留有限队列背压；这些控制队列不是额外的数据 SRAM。语法与展开规则见 [ROPE-CONTROL.md](ROPE-CONTROL.md)。

前端在当前合法阶段内为各 channel 保留独立指令游标，按 channel FIFO 顺序提交到各自容量为 32 的入口请求队列；一个 channel 的总 CA 或 controller 背压不再阻止其它 channel 前进。memory system 每个 memory CK 对每个 channel 最多尝试派发一条队首请求，controller 继续检查 global/rank CA 与所有数据依赖。全局 BARRIER、WAIT、原普通访问/PIM 阶段完成边界和 Q_REMOTE 对 peer Q_EXPORT 的完成依赖仍保留。

QKT 的 Q 按 feature 定向送到对应 bank，并非同一数据对所有 bank multicast，因此保留原子 `PIM_Q_EXPORT`、`PIM_Q_REMOTE` 的真实 GPU 往返路径；新描述符仅合并控制请求，仍执行全部单 bank RD/WRGB 数据传输。V suffix cache 的普通 DRAM 回写也保留。

## 流量与统计解释

每 rank 仍只接收一份所需 GPU 输入，bank 数量变化不复制输入字节。五阶段外部流量仍为 **5472 KiB/层/步（全部 16 channel）**，其中经暂存的数据为 **3824 KiB**。rank 内读取和广播的内部流量单独记录，不重复计入 DQ。

报告区分累计 DQ 占用、完整无外部传输工作窗口和其它 DQ 空隙。未向 rank0 注入真实并发 GPU 请求，故空闲窗口不能直接称为已测量的 GPU 可用时间。GPU softmax、GeLU/multiply/cast/pack、O projection、norm/residual、GPU dispatch 和后台 VLM 竞争未计时；不计算 BF16 tensor 数值，完整 AE 端到端 latency 保持未定义。

## 复现模式与兼容性

默认 `reproduce.py` 使用双 bank 暂存，并同时启用 `pim_rank_descriptors=True`、`channel_independent_dispatch=True`。两个新开关在 Python API/MemoryConfig 中默认为 false，CLI 显式开启，避免改变既有 guards 的调用含义。

| CLI 模式 | rank 粗描述符 | channel 独立派发 | 用途 |
|---|---|---|---|
| 默认 | 开 | 开 | 修正后的双 bank 1103 |
| `--no-rank-descriptors` | 关 | 开 | 单独测 channel HOL 修正，trace 仍为原细请求 |
| `--legacy-rank-dispatch` | 关 | 关 | 复现 `786842a` 的双 bank 1103 |
| `--staging-banks 1 --legacy-rank-dispatch` | 关 | 关 | 复现 1102 单 bank 控制与数据路径 |
| `--direct-input-baseline` | 关 | 关 | 关闭暂存与 rank controller，恢复 1101 直接 WRGB 路径 |

`--staging-banks 1` 本身仅选择单 bank 数据布局，仍默认使用新的派发模式。其布局为 rank1/BG0/bank0、128 KiB、row256–319，offset 为 `524288 + (f*R+t)*32`。需要旧时序对照时必须同时指定 `--legacy-rank-dispatch`。两个 bank 模式均保留两 slot 成对装载/计算顺序。

legacy 模式的 7 cases × 单/双 bank × staged/direct 共 28 个生成 trace 已与 `786842a` 逐字节核对一致。新描述符展开成原 16 个 bank 操作后，同样逐字节恢复原指令、地址、slot 与次序，DQ 总量一致；native 时序与日志是否一致由实际重新运行的验证结果判定，不由 trace 一致直接推断。

ACT1/ACT2 修正对所有启用 staging 的模式生效，包括 `--legacy-rank-dispatch`；该开关只恢复旧派发方式，不能撤销 deadline 修正。需要逐字节重现修正前时序或日志时，应检出 `14fb3bf`（或对应历史 commit），并使用该版本的独立 backend。历史 verify_legacy/单 bank 脚本的旧事件哈希不作为本版修正后的验收标准。

`--direct-input-baseline` 无论附带哪种 bank/派发选项都使用旧控制路径。有效 `staging_banks=0` 表示没有暂存，backend 的兼容参数为 1。summary、manifest、provenance 记录有效 bank 数及两个派发开关。

新模式附加 `frontend_dispatch.csv`、`channel_ingress.csv` 和 `rank_descriptors.csv.chN`，分别记录前端请求接受/完成、有限队列入队/派发和父子描述符生命周期。manifest 的 `counts` 统计实际 trace 父请求，`expanded_counts` 统计细操作；backend 的 `requests.total` 是父请求数，旧 `pim_q_export` 等流量计数按展开后的操作数计，避免把描述符压缩误当成数据量下降。

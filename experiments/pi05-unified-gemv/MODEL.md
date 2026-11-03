# xinyao_1103 双 bank 输入暂存与 rank controller

本实验从 `xinyao_1102` 的 `cbc1ee4` 派生；其直接输入基线是 `xinyao_1101` 的 `4073452`。1103 将 GPU 广播输入分布到不同 bank group 的两个 DRAM bank，并交错写入。权重、KV cache、计算指令、输出归属及每 bank 的两个 SRF slot 保持不变。rank controller 的派发政策沿用 1102；[RoPE 命令说明](ROPE-CONTROL.md) 记录其现有跨 channel 队头阻塞，本次不修复该问题。

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

总 controller 对输入装载采用 256/512 B fragment 描述符，每个占用 1 CK global CA；内部 PRE/ACT、LOCALRD、BCAST 由 rank controller 发出。其余既有 PIM 请求继续沿用 1102 的派发路径：包括 RoPE 的单 bank、32 B 请求也分别经过 global CA 准入。现有 channel-major trace 与单前端重试造成跨 channel 队头阻塞，不能全部解释成硬件 rank 内命令竞争。[ROPE-CONTROL.md](ROPE-CONTROL.md) 区分这两种影响；1103 的双 bank 数据路径修改不修复该控制问题。

QKT 的 Q 按 feature 定向送到对应 bank，并非同一数据对所有 bank multicast，因此保留 `PIM_Q_EXPORT`、`PIM_Q_REMOTE` 和 GPU 往返路径。V suffix cache 的普通 DRAM 回写也保留。

## 流量与统计解释

每 rank 仍只接收一份所需 GPU 输入，bank 数量变化不复制输入字节。五阶段外部流量仍为 **5472 KiB/层/步（全部 16 channel）**，其中经暂存的数据为 **3824 KiB**。rank 内读取和广播的内部流量单独记录，不重复计入 DQ。

报告区分累计 DQ 占用、完整无外部传输工作窗口和其它 DQ 空隙。未向 rank0 注入真实并发 GPU 请求，故空闲窗口不能直接称为已测量的 GPU 可用时间。GPU softmax、GeLU/multiply/cast/pack、O projection、norm/residual、GPU dispatch 和后台 VLM 竞争未计时；不计算 BF16 tensor 数值，完整 AE 端到端 latency 保持未定义。

## 复现模式与兼容性

`reproduce.py` 默认 `--staging-banks 2`，启用 `pim_rank_staging=True`、`pim_rank_controller=True` 和 `pim_staging_banks=2`，见 [CLI 参数](reproduce.py#L117)。

`--staging-banks 1` 使用 1102 的单 bank 布局：rank1/BG0/bank0、128 KiB、row256–319，offset 为 `524288 + (f*R+t)*32`。保留 1102 的 GPU 写入顺序与成对 SRF 装载顺序；7 个 staged case 的生成 trace 已与 `cbc1ee4` 逐字节核对一致。Python generator 和 backend 参数默认仍为 1，兼容原 API 与 guards；CLI 显式传入默认值 2。

`--direct-input-baseline` 关闭暂存与 rank controller，恢复 1101 的直接 WRGB 输入路径。此时 `--staging-banks` 不改变 trace，backend 使用兼容值 1；summary/provenance 的有效 `staging_banks=0` 表示没有暂存。7 个 direct case 的生成 trace 已与 `cbc1ee4` 的 direct 模式逐字节核对一致。summary、manifest 和 provenance 均记录有效 bank 数量，便于区分运行结果。

# xinyao_1107 粗细命令

当前 layout、本地 RoPE、两 bank 输入暂存、QKV 融合、token 分片 PV、QKT score 集中导出、PIM O projection 都保留。`pim_coarse_commands=true` 启用新接口；关闭时保留旧模式用于同源码回归。主入口是 `reproduce_commands.py`。

## 粗命令与细命令

计算粗命令只有 `MAC_RUN` 和 `ROPE_RUN`。QKV、QKᵀ、PV、Gate/Up、Down、O 每个 rank 各发一条 `MAC_RUN`；QKᵀ 的 31 个输出 tile、O 的 2 个输出 tile 由控制器内部遍历。每个 rank 一条 `ROPE_RUN` 覆盖本地 8 个 token 的 Q/K。

输入、结果和跨 rank 搬运直接发实际 `WR / RD / RDRES / WRGB`。Q exchange 在 trace 中保留 `PIM_Q_EXPORT / PIM_Q_REMOTE` 路由标记，分别产生实际 RD 和 WRGB；每个请求只对应一个 32B DQ burst，不再包装成 rank 批处理粗命令。

rank CA 上保留数组操作 `ACT1 / ACT2 / PRE / PREab / LOCALRD / LOCALWR / MAC`、外部读写及必要维护。SRF multicast、gather、cast、reduce、buffer release、RoPE execute 都是本地资源操作，不占用虚构的 1 CK CA 指令；其实际 port、PE、转换、恢复等待仍由原生模型计时。native timeline 中保留 `MACAB / MACQK` 两种时序适配类，rank CA 日志统一记为 `MAC`；这不是更改乘加算法或取消 QK 的归约时间。

`BARRIER` 和 `STAGE_BEGIN / STAGE_SEAL` 是仿真的生产者、覆盖率和依赖元数据，不传输 CA。外部数据写入本身仍逐 burst 计入 DQ 和真实 CA。此版本只实现本文固定 layout；没有声称任意 tensor 布局、数值精度或硬件电气协议已经验证。

## 256-bit MAC_RUN

字段内所有地址/stride/jump 单位均为 32B burst，除非另行注明。

| Word | 位 | 字段 |
|---|---|---|
| W0 | 3:0 / 7:4 | opcode=1 / version=2 |
| W0 | 9:8 / 10 / 11 | A_MODE（1 两bank条带，3 resident Q）/ REDUCE / A_KEEP |
| W0 | 27:12 / 28 / 31:29 | BANK_MASK / B_TILE_MAJOR / reserved |
| W1 | 3:0 / 13:4 / 25:14 | M−1 / N_total−1 / F−1 |
| W1 | 27:26 / 31:28 | log₂(PACK) / last_valid_lanes−1 |
| W2 | 23:0 / 27:24 / 31:28 | A_base / A_bank0 / A_bank1 |
| W3 | 15:0 / 31:16 | A_mstride / A_fstride |
| W4 | 23:0 / 31:24 | B_base / B_break[7:0] |
| W5 | 15:0 / 31:16 | B_nstride / B_fstride |
| W6 | 23:0 / 29:24 / 31:30 | signed B_jump / N_tile−1 / B_break[9:8] |
| W7 | 4:0 / 10:5 / 18:11 / 26:19 / 31:27 | input buffer base / result buffer base / WAIT / DONE / reserved |

当前约束：BANK_MASK=ffff；A_KEEP=0；两 buffer base=0；last_valid_lanes=16；保留位为0；DONE非0。输入 SRF 每 bank 1KiB，输出 FP32 池每 bank 2KiB，rank score tile 1KiB。WAIT=0 不等待事件；非0等待此前 DONE。DQ producer/consumer 就绪仍以真实读写完成为准。

普通 B 地址为 `B_base + f*B_fstride + n*B_nstride`；tile-major 为 `B_base + (floor(n/N_tile)*F+f)*B_fstride + (n%N_tile)*B_nstride`。`n>=B_break` 时加 B_jump（B_break=0 表示禁用）。条带 A 为 bank `[0,4][f%2]`、`A_base+floor(f/2)*A_fstride+m*A_mstride`。

| 算子 | M | N_total | N_tile | F | PACK | A_base（byte） | A_fstride（byte） | B_base（byte） | B_fstride（byte） | 特殊字段 |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| QKV | 8 | 48 | 48 | 64 | 4 | 0x80000 | 256 | 0x60000 | 1536 | — |
| QKᵀ | 16 | 496 | 16 | 1 | 1 | resident Q, 0 | 512 | 0x10000 | 0 | REDUCE=1，break=484，jump=1408B |
| PV | 16 | 16 | 16 | 31 | 2 | 0x80000 | 512 | 0x28000 | 512 | 最后批次只有1个 fragment |
| Gate/Up | 16 | 32 | 32 | 64 | 2 | 0x80000 | 512 | 0x40000 | 1024 | — |
| Down | 16 | 4 | 4 | 256 | 2 | 0x80000 | 512 | 0x50000 | 128 | — |
| O projection | 16 | 64 | 32 | 16 | 2 | 0x80000 | 512 | 0xc0000 | 1024 | B_TILE_MAJOR=1 |

以上 A_mstride 和 B_nstride 均为32B。QKᵀ 的 split 将第484个槽映射到本地 K suffix 的0x14200。O 第二输出 tile 权重从0xc4000开始；第一块512个 RDRES burst 必须实际读完后才能复用输出池。输出去向沿用当前模型固定布局策略：QKV写本地Q/K/V，PV写O输入暂存，QKᵀ写score DRAM，O/FFN直接导出。这里没有把这些去向伪装成额外的通用可编程字段。

## 256-bit ROPE_RUN

W0 为 opcode=2、version=2，bit8/9启用Q/K，bits27:12为bank mask。W1低16位为token_count−1，bits23:16为WAIT，bits31:24为DONE。W2/W3/W4/W5/W6依次描述Q输入、K输入、系数、Q输出、K输出；每个word低24位为base，高8位为token stride。W7保留为0。

当前一条：token_count=8；Q输入/输出0x14100；K输入/输出0x14200；系数0x14000；stride均32B。每 rank 产生384次 LOCALRD、256次 LOCALWR，完成2048次pair rotation。Q/K与系数使用BF16，运算/转换时序沿用已有本地RoPE模型；没有数值tensor执行。

## CA 封包及执行

每条命令 `14-bit header + 256-bit descriptor + 16-bit CRC + 8-bit zero padding = 294 bit`。7根CA线DDR每CK传14bit，因此**21 CA CK**；解码再用1CK，此1CK不占CA。header固定为0x04ed；CRC16/CCITT-FALSE，初值ffff，对8个word按小端byte序计算。没有省略CRC、免费的缓存描述符启动或每tile追加粗指令。

每 channel 一层共有6条 MAC_RUN +1条 ROPE_RUN，计算描述符合计**147 CA CK**。16个channel并行，不把2352个累计channel-CK当成串行延迟。普通输入、输出、跨rank传输和刷新CA另行计入。

控制器每个活动rank只保存一个描述符和循环游标，内部请求进入已有32项有限FIFO。输入加载和MAC允许原生有限前瞻，保留取输入时预开权重行的能力。SRF覆盖、输出复用、写恢复、ACT2期限、实际DQ结束和本地port互斥仍由原生时序与独立审计检查。

## 复现与统计

```bash
export PYTHON=/path/to/python3.11
bash experiments/pi05-unified-gemv/build.sh
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2/python:$PWD/experiments/pi05-bgk-gpuq/backend"
export LD_LIBRARY_PATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2"
$PYTHON experiments/pi05-unified-gemv/validation/test_descriptors.py
$PYTHON experiments/pi05-unified-gemv/reproduce_commands.py --output results/commands
$PYTHON experiments/pi05-unified-gemv/compare_commands.py --before results/commands/before --after results/commands/after --output results/commands/comparison
```

DQ使用真实burst区间，普通RD/WR包含CL/CWL，AttAcc RDRES/WRGB按原模型服务区间。每个channel的总空闲与最长单段分别统计；所有channel同时空闲取busy区间并集的补集。只合并相邻/重叠区间，不跨busy脉冲、独立案例或GPU未知边界把碎片伪装成长窗口。连续层图来自单次 `serial_layer_control` 原生运行，不拼接独立阶段。

报告同时给出独立阶段时延与连续层时延。它们都是已建模部分，未计GPU softmax、归约/非线性/归一化/残差等未知算术与真实后台争用；不等同于完整AE端到端延迟。

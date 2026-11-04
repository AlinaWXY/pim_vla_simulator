# xinyao_1104：FFN 专用数据 bank 与双缓冲

已实现两种配置：**1 个数据 bank＋15 个计算 bank**、**2 个数据 bank＋14 个计算 bank**。基线为 `7d2730d`，每 bank 保持 **1 KiB SRF＋2 KiB output**；SRF 分成两个 512 B 缓冲，在共享 CA 和 DRAM 约束下重叠下一片读入/广播与当前片 MAC。

每 rank 的 feature 分片不变；权重重排到计算 bank。Gate、Up 分两个 output pass，X 经 DQ 只写一次、内部读取两遍。每个数据 bank 的行包含四个连续输入片段，每行 64 个 32 B 列全部使用；计算和权重行准备排除数据 bank。满行指数据覆盖，刷新仍可能迫使同一行重新激活，实际开销均已计时。完整映射、容量和 masked 指令假设见 [FFN-BANK-SPLIT.md](experiments/pi05-unified-gemv/FFN-BANK-SPLIT.md)。

## Latency

单位 µs/层/step。before 为 1103 的两个暂存 bank 也参与计算；db1/db2 为本轮专用数据 bank 配置。

| 阶段 | before | 1＋15 | 2＋14 |
|---|---:|---:|---:|
| Gate/Up | 47.127934 | 47.645710 | 46.856852 |
| Down | 58.691598 | 45.266942 | 34.191976 |
| 连续 FFN 控制 | 105.910518 | 92.971746 | 81.314282 |
| **独立 Gate/Up＋Down 合计** | **105.819532** | **92.912652** | **81.048828** |

连续 FFN 单独列示，不重复计入合计。包含 attention 的单层 serial 控制为 171.960726 → 158.779950 / 147.458290 µs；GPU 算术仍未计时。

在这组 workload 下，1＋15、2＋14 的独立 FFN 总和分别下降 12.20%、23.41%。收益主要来自 Down：专用数据 bank 保持输入行，补数可与 MAC 重叠；两数据 bank 的写入交错进一步减少输入阶段等待。Down 在 15/14 个计算 bank 下均需 5 个输出列，计算命令数相同。Gate/Up 则从每 pass 18 列增加到 19 列，且两种配置均需内部读取 X 两遍，收益较小。每个 512 B 片段仍从一个数据 bank 读取，两个数据 bank 按整行轮换。

## 双缓冲对照

每一对使用相同 trace、布局、有效 MAC、DRAM 时序和编译库，仅切换 FFN overlap 开关。

| 阶段 | 配置 | overlap 关闭 | overlap 开启 | 开启后减少 |
|---|---|---:|---:|---:|
| Gate/Up | 1 数据 bank | 58.025618 | 47.645710 | 10.379908 |
| Gate/Up | 2 数据 bank | 57.780800 | 46.856852 | 10.923948 |
| Down | 1 数据 bank | 64.791412 | 45.266942 | 19.524470 |
| Down | 2 数据 bank | 53.999722 | 34.191976 | 19.807746 |
| 连续 FFN 控制 | 1 数据 bank | 123.069352 | 92.971746 | 30.097606 |
| 连续 FFN 控制 | 2 数据 bank | 112.141652 | 81.314282 | 30.827370 |

读入、广播与 PE 计算的实际重叠时间和逐 bank 生命周期证据保存在各案例的 `native-audit.json` 的 `ffn_split` 字段。上述差额为完整原生运行实测，包含刷新及调度相位变化，不是直接从总时间中扣除重叠窗口。

下表来自连续 FFN 控制的 channel0；重叠时间按两个资源区间的交集取并集，避免把流水中的多个读请求重复相加。关闭 overlap 的对应统计均为零。

| 配置 | pass | 内部读与MAC重叠 µs | MAC期间发出的LOCALRD | MAC期间发出的广播 |
|---|---|---:|---:|---:|
| db1 | gate | 4.819444 | 965 | 966 |
| db1 | up | 4.824134 | 965 | 966 |
| db1 | down | 16.514428 | 3441 | 3477 |
| db2 | gate | 4.828824 | 968 | 968 |
| db2 | up | 4.834452 | 968 | 969 |
| db2 | down | 16.823030 | 3493 | 3538 |

原生命令数如下（全部 16 channel 合计）。ACT1 还包括数据 bank 的读写行准备；MACMASK 仅作用于有效计算 bank，命令数不能直接当成标量 MAC 数。

| 阶段 | 配置 | ACTAB | ACTMASK | ACT1 | LOCALRD | MACAB＋MACMASK |
|---|---|---:|---:|---:|---:|---:|
| Gate/Up | before | 672 | 0 | 1280 | 16384 | 32768 |
| Down | before | 2096 | 0 | 5216 | 65536 | 16384 |
| Gate/Up | db1 | 0 | 768 | 944 | 32768 | 36864 |
| Down | db1 | 0 | 512 | 2192 | 65536 | 20480 |
| Gate/Up | db2 | 0 | 784 | 928 | 32768 | 38912 |
| Down | db2 | 0 | 496 | 2160 | 65536 | 20480 |

## DQ 空闲 breakdown

以下为 channel0，单位 µs；[全部16channel](reference/ffn-bank-split/channels.csv) 保留完整结果。外部 DQ 流量在所有 FFN 配置中均为 **3136 KiB**；两阶段 DQ 占用总量为 11.766272 µs/channel。

| 阶段 | 配置 | DQ占用 | DQ空闲 | 完整无外部传输工作窗口 | 其它空闲 | DQ与总CA同时空闲 |
|---|---|---:|---:|---:|---:|---:|
| Gate/Up | before | 3.842048 | 43.285886 | 40.757038 | 2.528848 | 41.160378 |
| Gate/Up | db1 | 3.842048 | 43.803662 | 38.503962 | 5.299700 | 41.348916 |
| Gate/Up | db2 | 3.842048 | 43.014804 | 40.482204 | 2.532600 | 40.447498 |
| Down | before | 7.924224 | 50.767374 | 47.638206 | 3.129168 | 49.139006 |
| Down | db1 | 7.924224 | 37.342718 | 23.450938 | 13.891780 | 35.471408 |
| Down | db2 | 7.924224 | 26.267752 | 23.159220 | 3.108532 | 24.415202 |
| 连续 FFN 控制 | before | 11.766272 | 94.144246 | 88.414004 | 5.730242 | 90.380052 |
| 连续 FFN 控制 | db1 | 11.766272 | 81.205474 | 61.941768 | 19.263706 | 76.877542 |
| 连续 FFN 控制 | db2 | 11.766272 | 69.548010 | 63.592648 | 5.955362 | 65.113146 |

“完整工作窗口”排除纯控制、release 及传输阶段碎片，且扣除进入阶段的传输尾部。CA 与 DQ 时间是不同资源维度，不能再次加到阶段 latency。没有注入并发 Thor 请求，空闲窗口不等于实测 GPU 可用带宽。

## 各子阶段

等待、行准备和刷新延迟均计入实际所在阶段；阶段边界余量见 [完整 phase CSV](reference/ffn-bank-split/phases.csv)。

| 配置 | 路径 | phase | latency µs |
|---|---|---|---:|
| before | Gate/Up | gate_up_gpu_input_stage | 2.502613 |
| before | Gate/Up | gate_up_local_input_and_compute | 40.757038 |
| before | Gate/Up | gate_up_local_input_and_compute_gemv_release | 0.002814 |
| before | Gate/Up | gate_up_result_export | 3.864560 |
| before | Down | down_gpu_input_stage | 10.547839 |
| before | Down | down_local_input_and_compute | 47.638206 |
| before | Down | down_local_input_and_compute_gemv_release | 0.002814 |
| before | Down | down_result_export | 0.501830 |
| db1 | Gate/Up | gate_pass_begin | 0.000967 |
| db1 | Gate/Up | gate_up_gpu_input_stage | 5.270622 |
| db1 | Gate/Up | gate_local_input_and_compute | 19.254326 |
| db1 | Gate/Up | gate_local_input_and_compute_gemv_release | 0.002814 |
| db1 | Gate/Up | gate_result_export | 1.920086 |
| db1 | Gate/Up | up_pass_begin | 0.001876 |
| db1 | Gate/Up | up_local_input_and_compute | 19.249636 |
| db1 | Gate/Up | up_local_input_and_compute_gemv_release | 0.002814 |
| db1 | Gate/Up | up_result_export | 1.941660 |
| db1 | Down | down_pass_begin | 0.000967 |
| db1 | Down | down_gpu_input_stage | 21.331996 |
| db1 | Down | down_local_input_and_compute | 23.450938 |
| db1 | Down | down_local_input_and_compute_gemv_release | 0.002814 |
| db1 | Down | down_result_export | 0.479318 |
| db2 | Gate/Up | gate_pass_begin | 0.000967 |
| db2 | Gate/Up | gate_up_gpu_input_stage | 2.503522 |
| db2 | Gate/Up | gate_local_input_and_compute | 20.252358 |
| db2 | Gate/Up | gate_local_input_and_compute_gemv_release | 0.002814 |
| db2 | Gate/Up | gate_result_export | 1.941660 |
| db2 | Gate/Up | up_pass_begin | 0.001876 |
| db2 | Gate/Up | up_local_input_and_compute | 20.229846 |
| db2 | Gate/Up | up_local_input_and_compute_gemv_release | 0.002814 |
| db2 | Gate/Up | up_result_export | 1.920086 |
| db2 | Down | down_pass_begin | 0.000967 |
| db2 | Down | down_gpu_input_stage | 10.548748 |
| db2 | Down | down_local_input_and_compute | 23.159220 |
| db2 | Down | down_local_input_and_compute_gemv_release | 0.002814 |
| db2 | Down | down_result_export | 0.479318 |

## 验证与复现

原生 job 1908477：130 项 Python 测试、58 项原生 guards 全部通过。24 个案例覆盖修改前、新 backend 旧模式、两种 bank 配置、两种关闭 overlap 的对照及混合 serial。旧模式逐周期、trace、流量及被审计原始 CSV 精确复现 1103；所有新模式核对有效 tensor 坐标、mask、完整行、SRF/output 容量、覆盖前消费、CA 和 ACT 时序。[验证](reference/ffn-bank-split/validation.json)、[源码—结果绑定](reference/ffn-bank-split/source-result-binding.json)、[完整比较](reference/ffn-bank-split/comparison.json)。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --ffn-only --ffn-data-banks 1 --output results/db1
python verify_ffn_banks.py --mode db1 results/db1
python experiments/pi05-unified-gemv/reproduce.py --ffn-only --ffn-data-banks 2 --output results/db2
python verify_ffn_banks.py --mode db2 results/db2
# 相同布局关闭 overlap；默认 ffn-data-banks=0 保留 1103 路径
python experiments/pi05-unified-gemv/reproduce.py --ffn-only --ffn-data-banks 2 --no-ffn-overlap --output results/no-overlap2
```

修改前备份位于本地 `../backups/xinyao_1103_before_ffn_bank_split_7d2730d/`。原始结果在 `results/*-1908477/`，根 `reference/` 保留 1103 基线，新的参考结果位于 `reference/ffn-bank-split/`。

这是时序、地址映射与容量模型，不执行完整数值推理。Masked PIM 指令及广播成本是架构假设；GPU GeLU/乘法/cast、输入生产及打包、其它 GPU 算术、后台竞争和初始权重搬迁未计时。所报结果不是完整 AE 端到端 latency。

双缓冲假设两个 512 B SRAM 半区能同时分别接收数据和供给 PE。模型保留原有 rank/BG/DRAM 约束及 bank 内 LOCALRD/MAC 间隔；不同 bank 的内部读取与 MAC 之间未另外增加共享内部数据网络带宽限制。硬件实现需要支持这些端口及内部并行数据路，当前结果不构成电气规格验证。

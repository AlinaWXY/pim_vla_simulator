# Pi0.5 AE：H=16 action horizon，8-head MHA

这是 `xinyao_1025` 的独立 **H=16** 版本。H 表示 action/query token 数，attention head 数仍为 8，每 head 256 维。H=20 原版在 [`../pi05-bgk-gpuq/`](../pi05-bgk-gpuq/README.md)。

本版重新生成 H=16 的 trace 并执行原生仿真。QKV 仍共享 X，Q/K 经 GPU RoPE 边界后写回目标 DRAM 布局；QK 从 DRAM 读 Q 再广播到 GEMV buffer。沿用原版每 channel 32 B 暂存的串行 read→relay 调度。

## 复现

使用原版同一个原生后端和固定 Python/C++ 依赖，不复制或修改 backend：

```bash
git clone --branch xinyao_1025 https://github.com/AlinaWXY/pim_vla_simulator.git
cd pim_vla_simulator/experiments/pi05-bgk-gpuq-h16
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../pi05-bgk-gpuq/requirements-lock.txt
PYTHON="$VIRTUAL_ENV/bin/python" JOBS=4 bash ../pi05-bgk-gpuq/build.sh
python reproduce.py --skip-build
```

默认引用 `../pi05-bgk-gpuq/backend`，也可用 `--backend` 指定已构建的同一版本后端。执行环境要求见原版 README。每次运行创建新目录，也可指定 `--output results/H16-my-run`。

## H=16 mapping

16 channel、2 rank/channel、4 BG/rank、4 bank/BG，共 512 bank。每 head 两个 channel；prefix 仍为 968 tokens。有效 KV 为 **984**，物理长度仍为 **992**，padding 为 **8** tokens。

| 项目 | H=16 |
|---|---|
| 单个 Q/K/V projection | `[16,1024] × [1024,2048]` |
| 联合 QKV | `[16,1024] × [1024,6144]` |
| QK，每 head | `[16,256] × [256,992]` |
| PV，每 head | `[16,992] × [992,256]` |
| FFN Gate/Up，各自 | `[16,1024] × [1024,4096]` |
| FFN Down | `[16,4096] × [4096,1024]` |

Q/K/V projection 的输出 feature ownership 不变：`d=64*slot+32*channel_in_head+16*rank+4*BG+bank`。每 bank 为每个 projection 计算 4 个 feature，处理全部 16 个 action tokens；WQ/WK/WV 按 `[input_fragment64][QKV3][slot4][lane16]` 放置。每次 X fragment 输入完成后依次扫三个 projection，不重复搬 X。

K 的 token `t` 位于 `channel=2*h+t%2`、`rank=(t//2)%2`、`BG=(t//4)%4`、`token_slot=t//16`。256 维在该 BG 四 bank 内切分：`bank=(d%128)//32`、`local_feature=(d//128)*32+d%32`。**16 个 suffix tokens 正好每 head 的 16 个 BG 各一个**。K 仍使用每 bank `[62 token slots,64 local features]` 布局，padding 在 GPU softmax 时屏蔽。

Q 每个 head 的两个 channel 各保存完整副本，复制的写回流量已计入。每 channel 仅 rank0/BG0 四个 source bank 存 Q，bank 内顺序 `[fragment4][query16][lane16]`；每 source bank **2048 B，恰好一行**。一条 32 B Q 读回后，广播至同 channel 所有 BG/rank 的相同 bank index。V/PV 仍按输出 feature 分片。

| 每 bank 地址范围 / B | 用途 |
|---|---|
| `[0,24576)` | WQ/WK/WV |
| `[24576,26624)` | Q cache；仅 source banks 有 payload |
| `[26624,34816)` | K cache |
| `[34816,43008)` | V cache |
| `[43008,75776)` | FFN Gate+Up |
| `[75776,92160)` | FFN Down |

Attention 预留 **42 KiB/bank**，加 FFN 后每层共 **90 KiB/bank**。权重体积不随 H 改变。

## Buffer 与流量

| 项目 | H=16 大小 |
|---|---:|
| QKV/PV 输入双缓冲 | 1024 B/rank |
| 联合 QKV 累积器 | 768 B/bank |
| QK 输入双缓冲 | 1024 B/bank |
| QK 部分和 | 3968 B/bank，预留 4096 B |
| QK 输出双缓冲 | 2048 B/BG |
| Q relay 暂存 | 32 B/channel |
| FFN Gate+Up / tiled Down 累积器 | 1024 B/bank |
| FFN rank-local Z | 4096 B/rank |

联合 QKV 的 X 输入为 **512 KiB**；Q/K/V 各导出 **128 KiB FP32**。Q 双副本写回 **128 KiB BF16**，K suffix 写回 **64 KiB BF16**。每次 QK 的 Q DRAM 读取与 relay 各 **128 KiB**（各 4096 次 burst），扇出后 buffer 交付 1 MiB。

V 的尾部仍涉及两个 16-token 存储块：8 prefix +16 suffix +8 padding。因此保持 **64 KiB 尾部读取、128 KiB 写回**；此项不会随 H 简单乘 16/20。P 的物理 BF16 payload 为 248 KiB，在每 head 两个 channel 各送一份，外部传入为 **496 KiB**；QK score 导出为 **496 KiB FP32**；PV 输出 **128 KiB FP32**。FFN Down 的 32 组 rank partials 导出共 **2 MiB FP32**。

本版将 `nCCDAB` 和 `nCCDQK` 从 20 CK 改为 **16 CK**，对应同一 PE 吞吐假设下处理 16 条 query vector；除 MAC 批次 service 外的 DRAM 时序、频率与 refresh 配置不变。ACT/PRE、padding、固定时序和重叠仍由模拟器执行，因此 latency 不是 H=20 结果乘 0.8。

## Latency 与验证

单位为 μs/层/step，包含已建模的 DRAM/PIM 传输。

| 阶段 | H=20 | H=16 |
|---|---:|---:|
| 联合 QKV + 导出/写回 | 20.423074 | 16.751742 |
| QK | 17.728200 | 14.305438 |
| PV | 5.682404 | 4.408600 |
| FFN Gate + Up | 22.290632 | 17.945816 |
| FFN Down | 20.176380 | 16.017288 |
| **已测合计** | **86.300690** | **69.428884** |

H=16 合计 ×18 层×10 步为 **12.49719912 ms**，相对 H=20 小计减少 **19.55%**。这反映 workload 从20个 action tokens 改成16个，以及其实际调度结果，不是同一 workload 的硬件加速比。

H=16 QK 的互斥分解：Q DRAM 读 **6.003200 μs**、relay **0.480256 μs**、阵列 MAC **3.721984 μs**、GPU 边界导出 **1.909768 μs**、其他/边界 **2.190230 μs**。

完整 [breakdown 与图](reference/H16/all-linear-breakdown/) 单独标为 H=16。

完整输出包括互斥墙钟 breakdown、逐 phase 时间、系统流量、驻留地址、独立事件审计和结果图。独立 Q/K/V controls 不与联合 QKV 重复相加。整体乘 18 层×10 步仅表示已测阶段累计。

**边界**：GPU RoPE/转换/packing、softmax、O、最终 FFN 归并、GELU 等算术未测量。Q relay 的 32 B 单条串行暂存限制仍保留，尚未充分隐藏 DRAM 返回延迟；channel 内 fanout/BG reduction 仍采用原版命令吞吐假设。因此报告不代表完整 AE 或优化后的 Q 预取极限。

2026-10-25 验证通过：12 项 mapping/workload 测试、10 项 analysis/FFN 测试、4 项参考结果保护测试、9 个 attention/control 独立审计、8 类故障注入和 80 个主计时 channel 分解守恒检查。完整复跑的 13,787 个 JSON 数值字段和 100 个 CSV 数值字段一致；[验证记录](verification/2026-10-25/) 随版本保存。共享原生后端没有修改。

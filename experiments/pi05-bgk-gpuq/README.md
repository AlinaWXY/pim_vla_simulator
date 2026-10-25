# Pi0.5 AE MHA8：共享 X 的 QKV projection 与 DRAM Q 回读

`xinyao_1025` 从 `xinyao_1024` 的原生时序后端继续实现：Q/K/V projection 全部使用 V 的输出 feature 切分，三者共用一次 X 输入；Q/K 经 GPU RoPE 后以 QK 目标布局写回 DRAM；QK 通过真实 DRAM 读和 channel 内广播填充 GEMV buffer。V/PV 和 rank-local FFN 保留原有计算方案。

本目录包含完整后端、构建依赖源码、trace 生成器、独立审计器、breakdown 和参考结果。GPU 算术未执行；下述测量是控制器/DRAM/PIM 阶段的时间，不是完整 AE 延迟。

## 从干净 checkout 运行

需要 Linux、Python 3.11 及开发头文件、CMake 3.18+、C++20 编译器。验证环境为 Python 3.11.16、GCC 11.5.0、CMake 3.26.5，建议 8 GiB 内存。Windows 可用 WSL2/Linux。

```bash
git clone --branch xinyao_1025 https://github.com/AlinaWXY/pim_vla_simulator.git
cd pim_vla_simulator/experiments/pi05-bgk-gpuq
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
PYTHON="$VIRTUAL_ENV/bin/python" JOBS=4 bash build.sh
python reproduce.py --skip-build
```

编译和运行使用同一 Python 环境。只构建本目录的 `backend/ramulator2`，使用随代码提交的 C++ 依赖；仓库根目录 `ramulator2/` 是历史基线，不参与本实验。每次运行创建新的结果目录，不覆盖已有结果，也可指定 `--output results/my-run`。

运行入口执行 11 项 workload 测试、7 项 analysis 测试、9 个 attention/control 仿真、6 个原生定向检查、4 个 FFN/control 仿真、独立事件审计、8 类故障注入、breakdown、绘图、诊断和参考结果比较。后端 pytest 可另外运行：

```bash
export PYTHONPATH="$PWD/backend/ramulator2/python:$PWD/backend"
export LD_LIBRARY_PATH="$PWD/backend/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd backend && python -m pytest tests -q)
```

## 矩阵与物理映射

固定形状：8 heads × 256 维，20 action tokens，hidden=1024，FFN=4096；prefix=968，KV 有效长度 988，物理长度 992。每 head 两个 channel；每 channel 两个 rank；每 rank 四 BG；每 BG 四 bank，共 512 bank。LPDDR5X-8533，tCK=0.938 ns，refresh 开启。

| 算子 | 矩阵乘法（省略 batch=1） | bank 内持有/归约 |
|---|---|---|
| Q/K/V projection | 各自 `[20,1024] × [1024,2048]` | 每 bank 每个 projection 4 个输出 feature，完整输入维度归约 |
| 联合 QKV | `[20,1024] × [1024,6144]` | 一次 X fragment 扫 Q4/K4/V4，共 12 个权重 column |
| QK | 每 head `[20,256] × [256,992]` | K 按 token 分到 BG，feature 在 BG 的四 bank 内切分；BG 局部归约 |
| PV | 每 head `[20,992] × [992,256]` | V 输出 feature 切分，bank 内归约 token 维 |
| FFN Gate+Up | 两个 `[20,1024] × [1024,4096]` | 共享 X，输出 feature 切分；rank-local gather 后非线性 |
| FFN Down | `[20,4096] × [4096,1024]` | 输入 feature 分片，导出 rank partials 供 GPU 归并 |

**Projection ownership**：head 内 feature `d = 64*slot + 32*channel_in_head + 16*rank + 4*BG + bank`，`slot=0..3`。每 bank 的融合权重按 `[input_fragment64][Q/K/V3][slot4][lane16]` 排列，BF16，共 24 KiB。每个 projection 全系统 4 MiB，三者合计 12 MiB；没有以前 K 的 16 倍复制。

每次载入 20 条 X 的 16 维局部片段，同一片段完成 Q/K/V 的 MAC 后才换下一片。X 外部传输总计 640 KiB；独立 Q/K/V controls 各传一遍 X，仅作对照，不加入主计时合计。

**GPU 回程与 cache**：Q/K/V 各导出 160 KiB FP32。GPU 对 Q/K 做 RoPE、转换和打包后，Q 写回 160 KiB BF16，K suffix 写回 80 KiB BF16。Q 明确在两个 channel 各存完整副本，复制包含在这 160 KiB 写回中。V 沿用 64 KiB prefix 尾部读取 + 128 KiB suffix RMW 写回。

**QK ownership**：token `t` 位于 `channel=2*h+t%2`、`rank=(t//2)%2`、`BG=(t//4)%4`、`token_slot=t//16`。feature `d` 位于 `bank=(d%128)//32`、`local_feature=(d//128)*32+d%32`。K 每 bank 顺序为 `[token_slot62][local_feature64]`。同 token 的四 bank 部分和在 BG 内归约；不需要跨 channel 归约。

Q 在每 channel 的 rank0/BG0 四个 source bank 保存，沿用 K 的 feature-bank 对应关系，bank 内顺序为 `[fragment4][query20][lane16]`，每 source bank 2560 B。Q 回读一条 32 B 后，广播至该 channel 所有 BG/rank 的同一 bank index，共 8 个消费者。一次 QK 共 5120 次读 + 5120 次 relay：DRAM 读 160 KiB、广播入口 160 KiB、扇出后目标 buffer 写入 1.25 MiB。这里不是再从 GPU 输入 Q。

| 每 bank 地址范围 / B | 用途 |
|---|---|
| `[0,24576)` | WQ/WK/WV |
| `[24576,28672)` | Q 区域；只有 rank0/BG0 实际存储 |
| `[28672,36864)` | K cache |
| `[36864,45056)` | V cache |
| `[45056,77824)` | FFN Gate+Up 权重 |
| `[77824,94208)` | FFN Down 权重 |

每层物理预留 92 KiB/bank。完整公式与 trace 契约见 `source/ATTENTION_CONTRACT.txt`。

## Buffer 与时序边界

- QKV 输入：rank-shared 双缓冲 `2 × 20 × 16 × 2 = 1280 B/rank`；三个 projection 累积器合计 `20 × 12 × 4 = 960 B/bank`。
- QK 输入：bank-local 双缓冲 1280 B/bank；部分和预留 5 KiB/bank，BG 输出双缓冲 2.5 KiB/BG。
- Q 回读控制器 staging：**32 B/channel，单条 read→return→relay 串行完成**。真实 RD 经过 ACT/PRE、CL 和 burst，relay 再消耗 WRGB 时序，消费者等待最后一个 burst 完成。
- 当前 RD_Q 不越过排队中的 MAC，普通读等待期间也限制 PIM 发射。因此双缓冲容量与覆盖保护已实现，但下一 Q fragment 的预取尚未与当前扫描充分重叠；这是本版保守调度限制，不是 DRAM 必然只能达到的带宽。
- 本版同时修正 legacy WRGB 的就绪时间：从 issue 即就绪改为等待 `nBL_min=2 CK`，避免消费尚未传完的最后一个 burst。

channel 内 fanout 和 BG reduction 仍采用 AttAcc 命令级吞吐抽象；计入共享通路和依赖，但不另设内部网络传播/加法器流水延迟。Q 回读的 32 B staging 是原生有界状态；其余 SRAM 容量通过 workload/审计约束，没有统一 SRAM 溢出硬件模型。

## 测量结果与 breakdown

单位：每层、每个 denoising step 的微秒；下表包含已建模的 DRAM/PIM 传输。

| 主计时阶段 | latency / μs |
|---|---:|
| 联合 QKV + 导出/写回 | 20.423074 |
| QK（DRAM Q 回读 + 广播） | 17.728200 |
| PV | 5.682404 |
| FFN Gate + Up | 22.290632 |
| FFN Down | 20.176380 |
| 已测 PIM 阶段合计 | **86.300690** |

合计 ×18 层 ×10 步为 **15.5341242 ms**，是独立计时阶段之和，不是包含未知 GPU 算术的 AE 端到端时间。独立 QKV→QK chain 另外用于验证跨算子存取依赖；行状态/refresh 相位会使 chain 时间与独立阶段相加不同。

QK 的互斥 breakdown：Q DRAM 读 **7.504 μs**、Q relay **0.600320 μs**、阵列 MAC **4.652480 μs**、结果导出 **2.375016 μs**、其他/边界 **2.596384 μs**。串行回读明显暴露每个 burst 的 CL；这组结果不能用于声称更深预取队列没有收益。

完整结果见 [`reference/xinyao_1025/all-linear-breakdown/`](reference/xinyao_1025/all-linear-breakdown/)。历史 `xinyao_1024` 不含 PIM Q projection，且 K 映射不同，不能把两版小计直接解释为完整 AE 的加速比。


主计时是 **联合 QKV + QK + PV + Gate/Up + Down**。联合 QKV 包含三个 projection 及上述 GPU 边界传输；不人为把共同输入/重叠时间拆成三个可相加的独立 latency。

`q/k/v`、`qkv_serial/qk_serial` 是 controls。`chain_qkv_qk` 验证写回可见性与后续 Q 读取，GPU 计算是显式未知依赖边界；它不代表 GPU/PIM 完整端到端运行。

主要输出：

- `all-linear-breakdown/all-linear-breakdown.csv`：互斥墙钟 breakdown。
- `phase-breakdown.csv`（同目录）：逐阶段 projection、Q/K 导出与写回时间。
- `breakdown.json`、`channel-partitions.csv`：各 channel 事件区间与守恒核对。
- `system-traffic.csv`、`resident-layout.json`：字节数与地址布局。
- `linear-latency-breakdown.png/.svg`：breakdown 图。
- `<case>/simulation/timeline.csv.ch*`、`bgk_events.csv.ch*`：原生命令和读回/relay 完成事件。
- `reference-comparison.json`、`validation-summary.json`、`audit-fault-detection.json`、`artifact-sha256.json`、`runner-logs/`：复现与审计记录。

`q_dram_read` 覆盖 RD issue 到返回（CL+burst）；行准备/刷新等等待在 `other` 或所属完整传输 phase 中。`q_broadcast` 是 relay 的 burst 服务。共享资源时间取事件区间并集，overlap 只计一次；input-only 列不是全部输入服务量。`other` 不能直接叫作 row miss。

**未测量**：GPU Q/K RoPE、转换/packing、softmax、O projection、Down 最终归并、额外 GPU 源端 DMA/到达处理、V 转换/packing、FFN GELU/逐元素乘法/转换、action I/O、norm/residual/conditioning，以及权重/prefix 初始安装。未知计算时间没有补作零；也未模拟这些未知 GPU 间隔引入的后续 refresh 相位变化。

## 来源与复现证据

`SOURCE_PROVENANCE.json` 记录分支基线和固定 C++ 依赖提交。原有许可证保留。`reference/run-1849371/`、`reference/run-1833560/` 及 `verification/2026-10-24/` 是历史资料；新同版本参考放在 `reference/xinyao_1025/`。同 schema 复现比较全部数值字段：整数必须一致，浮点容差仅 `1e-12`。

2026-10-25 验证：从本分支源码编译的后端通过 200 项 pytest；同版本完整复跑通过，13,780 个 JSON 数值字段和 100 个 CSV 数值字段一致，9 个 attention/control 独立事件审计、8 类故障注入和 80 个主计时 channel 分解守恒检查通过。证据见 [`verification/2026-10-25/`](verification/2026-10-25/)。

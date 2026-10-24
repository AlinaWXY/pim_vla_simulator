# Pi0.5 AE MHA8：线性算子时序与 breakdown 复现

本目录复现 2026-10-23 的 `run-1849371`：GPU Q、BG 内 K 特征分片与归约、原 V/PV 映射、rank-local FFN。它包含实际使用的 Ramulator 原生后端完整源码、构建依赖源码、trace 生成器、审计器、绘图和参考结果。运行时不需要原服务器目录、旧实验代码、GPU 或 Slurm。

这里仍是**每 head 的 16 个 BG 按 token 切分 K projection、每 BG 保留一份完整 head 的 W_K**的已测版本。讨论过的“改按输出维度切分 K projection”尚未实现，也没有混入这组结果。

## 从干净 checkout 运行

需要 Linux、Python 3.11 及开发头文件、CMake 3.18+、支持 C++20 的编译器。参考运行使用 Python 3.11.16、GCC 11.5.0、CMake 3.26.5。内存建议 8 GiB；使用 2～4 个编译线程即可。Windows 请使用 WSL2/Linux。

```bash
git clone --branch xinyao_1024 https://github.com/AlinaWXY/pim_vla_simulator.git
cd pim_vla_simulator/experiments/pi05-bgk-gpuq
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
PYTHON="$VIRTUAL_ENV/bin/python" JOBS=4 bash build.sh
python reproduce.py --skip-build
```

编译和运行必须使用同一个 Python 环境。`build.sh` 只构建本目录的 `backend/ramulator2`，使用随代码提交的 fmt、yaml-cpp、nanobind 和 robin-map；不需要从网络再次获取 C++ 依赖。仓库根目录的 `ramulator2/` 是历史上游基线，不是这组实验的执行后端。

每次运行生成新的 `results/reproduce-<timestamp>/`，已有结果目录不会被覆盖。也可指定一个尚不存在的路径：

```bash
python reproduce.py --skip-build --output results/my-run
```

运行入口执行：8 项 workload 测试、6 项 breakdown 测试、8 个 attention/control 原生仿真、6 个原生定向检查、4 个 FFN/control 原生仿真、独立事件审计、5 个故障注入检查、breakdown 分解、绘图、诊断、参考结果比较和产物 SHA256 清单。参考文件只用作比较；所有测量都重新运行模拟器。

若使用 Slurm，在激活上述 Python 环境、完成编译后，从本目录提交：

```bash
sbatch --partition=YOUR_PARTITION source/run.sbatch
```

## 预期结果

单位为每层、每个 denoising step 的微秒。Gate 和 Up 共用输入且联合测量，不能各自重复计入整段时间。

| 算子 | latency / us |
|---|---:|
| K projection | 19.466314 |
| V projection | 6.723584 |
| QK | 8.074304 |
| PV | 5.682404 |
| FFN Gate + Up | 22.290632 |
| FFN Down | 20.176380 |
| PIM 阶段合计 | **82.413618** |

合计乘以 18 层、10 步为 **14.83445124 ms**。这是上述已测 PIM 阶段的累计时间，**不是完整 AE 的端到端 latency**。

入口会核对六个主算子的时间、`breakdown.json` 中全部 9,909 个数值字段和 CSV 中 108 个数值字段；整数时钟/计数要求相等，浮点字段只允许 `1e-12` 的显示误差。比较失败会以非零状态退出。

2026-10-24 已在独立 Linux 目录和新的 Python 环境从源码重新编译并复跑，六项时间及上述全部字段一致，190 项后端回归测试通过。记录见 [`verification/2026-10-24/`](verification/2026-10-24/)，Python 依赖版本固定在 `requirements-lock.txt`。

输出文件：

- `reference-comparison.json`：是否与历史测量一致。
- `all-linear-breakdown/all-linear-breakdown.csv`：各算子的互斥 latency 分解。
- `all-linear-breakdown/breakdown.json`、`channel-partitions.csv`：逐 channel 时序及资源证据。
- `all-linear-breakdown/system-traffic.csv`：GPU、DRAM 和本地数据路径流量。
- `all-linear-breakdown/linear-latency-breakdown.png`、`.svg`：breakdown 图。
- `all-linear-breakdown/resident-layout.json`：互不重叠的权重/KV 地址区间。
- `<case>/simulation/timeline.csv.ch*`、`bgk_events.csv.ch*`：原生命令和本地事件。
- `validation-summary.json`、`audit-fault-detection.json`、`artifact-sha256.json`：审计及哈希。
- `runner-logs/`：各阶段命令、输出和退出状态。

breakdown 列是互斥的墙钟时间区间，overlap 只计一次。输入工作有一部分位于 overlap 中，不能仅用 input-only 列代表全部输入搬运开销。阵列列间隔限制与 MAC 服务分别列出；`other` 含剩余调度、行准备和刷新等等待，不能全部称为 row miss。

## 映射与边界

- 8 heads × 256 维；20 action tokens；968 prefix tokens；988 有效 KV tokens，物理 padding 到 992。
- 每 head 一颗 LPDDR、两个 channel；每 channel 两个 rank；每 rank 四 BG；每 BG 四 bank，共 512 bank。
- LPDDR5X-8533，tCK=0.938 ns，refresh 开启。`nCCDAB=20`、`nCCDK=2`、`nCCDQK=20`、本地写端口 `nLOCALWR=4` CK；原生同 bank 列约束仍生效。
- K cache 按 token 划分到 BG，256 维在同 BG 四 bank 内切分，并让 RoPE 的维度对保留在同 bank。W_K 每 BG 一份完整 head 权重，全层 64 MiB。K projection 输出在本 bank 累积，最后本地写回。
- Q projection 在 GPU；Q 以分片传入 PIM，没有 Q DRAM cache。QK 输入双缓冲为 1,280 B/bank；K 输入最多使用其中 128 B/bank。QK 部分和另预留 5 KiB/bank，输出另预留 2.5 KiB/BG。
- V/PV 保留既有输出 feature 切分，V 的 GPU 边界导出、prefix 尾部 RMW 和写回按原 trace 计时。O projection 在 GPU。
- FFN Gate+Up 联合运行；rank-local gather 计入。Down 导出 rank partials，最终合并计算在 GPU。

已测量原生 DRAM 行/列时序、ACT/PRE/refresh、输入/输出传输、双缓冲就绪与覆盖保护、本地写回、FFN 本地 gather/feed。channel 内 multicast 与 BG reduction 使用 AttAcc 命令级抽象：统计流量、共享通路和依赖，但假设内部扇出/加法器能跟上命令吞吐，不另设独立网络/加法器流水延迟。buffer 字节数由 workload 和审计约定，模拟器没有独立的 SRAM 容量溢出硬件模型。

**未测量** GPU Q/O projection、softmax、GPU 最终归并、GPU 端额外 DMA/到达处理、K 的本地 RoPE/转换/打包、V 的 GPU 转换/打包、FFN GELU/逐元素乘法/转换，以及 action I/O、norm/residual 和 conditioning。未测项保留为未知，不解释成零开销。权重与 prefix KV 的初始安装也不计入上述在线阶段。

## 源码与参考数据

`SOURCE_PROVENANCE.json` 记录来源和 C++ 依赖的固定提交；许可证随源码保留。`reference/run-1849371/` 是历史输出，`reference/run-1833560/` 仅包含 FFN 回归所需的两个结果文件，不会被当成新测量。

原生后端的额外回归测试可单独运行：

```bash
export PYTHONPATH="$PWD/backend/ramulator2/python:$PWD/backend"
export LD_LIBRARY_PATH="$PWD/backend/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd backend && python -m pytest tests -q)
```

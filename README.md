# PIM-VLA · 2026-11-15

OpenPI π0.5，BF16，action horizon 16，10 个去噪步。这里汇总 Thor 全 GPU 的 ViT/VLM 与 AE 并行实测，以及 Ramulator2 的 PIM/GPU 分工仿真。

## Thor 全 GPU 实测

每组 pair 含当前帧的 ViT/VLM 与前一帧的 AE。AE 使用前一帧的 KV cache。表内是 pair 墙钟 p50，单位 ms。

| 模型 | PyTorch | 串行 | 双 CUDA stream 并行 |
|---|---|---:|---:|
| MQA | eager | 250.66 | 228.92 |
| MQA | compile | 120.70 | 117.50 |
| MHA8 | eager | 247.25 | 236.71 |
| MHA8 | compile | 141.29 | 137.07 |

连续运行另测。MHA8 使用对齐 VLM 校准采集的 compile 选项时，双流实测为 AE 5.07 chunk/s、VLM 10.14 次/s。详见 [MHA8 profile](reference/2026-11-15/thor/mha8/)、[MQA profile](reference/2026-11-15/thor/mqa/)和[基准脚本](reference/2026-11-15/benchmarks/)。MHA8 将 K/V 权重实质扩成 8 个头，MQA 保留 1 个 K/V 头。

## Ramulator2 仿真

顶层 MC、MAC 2×、PIM rank ACTAB 四批及分批选列。c–g 在 AE 需要 GPU 时，以 2 µs VLM 调度量子请求让出 GPU，并等待当前待处理访存排空。各 case 保留 Thor 冷缓存 GPU 非线性实测时长及 GPU→PIM 写，仅从模拟命令中扣除重复的 PIM 输出 DRAM→GPU 读回。e/g 保留本地 RoPE 路径的 Q 交换读。

| Case | 布局 | RoPE | 数据流 | AE (ms) | AE 结束前 VLM 含 ViT (ms) | 端到端 (ms) |
|---|---|---|---|---:|---:|---:|
| b | feature | GPU | streaming，串行 | 56.947 | 0 | 122.883* |
| c | feature | GPU | streaming | 60.327 | 47.478 | 86.766 |
| d | token | GPU | streaming | 57.898 | 47.418 | 84.036 |
| e | token | local | streaming | 56.829 | 47.164 | 83.571 |
| f | token | GPU | fused | 69.271 | 60.147 | 78.165 |
| g | token | local | fused | 68.645 | 60.295 | 77.678 |

\* b 实跑 AE solo，端到端为两个独立 solo 相加。b 的 AE 是 PIM 46.012 ms + GPU 10.935 ms，后者含 0.127 ms 的 action 投影。c–g 的 VLM 并发列扣除 AE 抢占暂停时间，保留 VLM 自身访存等待。

VLM 请求模型以 Thor 上 compile 的 MHA8 ViT+VLM solo **66.803 ms** 为目标，校准后模拟 **65.936 ms**（−1.30%），含 1,416 个 kernel，`alpha=0.5`、窗口 1024、写高/低水位 768/256。[b 原生汇总](reference/2026-11-15/simulation/case-b-single-read-summary.json) · [c–g 完整 CSV](reference/2026-11-15/simulation/cg-single-read-compiled-vlm.csv) · [校准](reference/2026-11-15/simulation/compiled-vlm-calibration.json)

## 构建与重放

Linux、GCC 14（C++20）、CMake ≥ 3.18、Python ≥ 3.10。`bash build.sh` 构建并执行测试。首次配置拉取固定版本的 fmt 和 yaml-cpp。实现见[调度入口](ramulator2/src/ramulator/frontend/impl/pim_vla_frontend.cpp)及 [MC 控制器](ramulator2/src/ramulator/controller/impl/pim_controller.cpp)。

完整证据在 `dev0:/fact_data/xinyaowang/PIM_for_VLA/Important data/xinyao_1115-2026-11-15/`。`ramulator-runs.tar.gz` 含可直接重放的配置、生成后的 PIM 程序、AE 时间表、VLM workload 及逐 case 的 `raw.json`。解包后把配置中 `spec_path`、`program_path`、`ae_schedule_path` 和 `vlm_path` 的原实验目录前缀改成解包目录，再用 `build/pim_vla_sim --config <配置> --output <新目录>`。b 的配置为 `case-b-single-read.bauEXp/10/config.json`，c–g 为 `compiled-single-read-matrix.683RuH/compiled/<case>/config.json`。

Thor 的原始 Nsight、CUDA 时间样本、NCU 读写指标和锁定版 OpenPI 在同一备份。GPU benchmark 使用 `lerobot/pi05_base` snapshot `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`、OpenPI `215abfb` 和 PyTorch `2.11.0+cu130`。基准脚本的 `--support` 指向含 token fixture 的目录，OpenPI `src` 加入 `PYTHONPATH`。

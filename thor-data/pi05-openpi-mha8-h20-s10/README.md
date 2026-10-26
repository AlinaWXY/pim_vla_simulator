# Thor / OpenPI pi0.5 / MHA8 / H20 / S10

本目录仅收录 **2026-10-22 已完成** 的这一组 Thor GPU 测量，2026-10-26 整理上传。
不是新跑的数据，也不是 PIM 仿真或 GPU+PIM 端到端结果。

## 配置

| 项目 | 配置 |
|---|---|
| 模型与实现 | pi0.5，OpenPI PyTorch |
| Horizon / flow steps / batch | **20 / 10 / 1** |
| Attention | **8 Q heads + 8 个物理 KV heads，head_dim=256** |
| AE | 18 层，residual width 1024，attention width 2048，FFN 4096 |
| 输入 | 3 个 224x224 图像，200 个 padded text/state positions |
| Prefix / 总 KV | 968 / 988 个物理位置；原有 padding mask 保留 |
| 条件缓存 | 时间编码、time MLP、37 个 AdaRMS conditioning linears 预计算 |
| 精度 | **选择性 BF16/FP32**，不是整个模型统一 BF16 |
| Runtime | Torch 2.11.0+cu130，Transformers 4.53.2，单 CPU 线程 |
| 编译/缓存 | 未开启 torch.compile/CUDA Graph；没有主动清缓存 |
| OpenPI source | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| 测量代码 commit | `5bbcdd47685402998621db9a99bed8cc2f9f1070` |

MHA 是将原单 KV head 权重物理扩展到 8 个 heads 的**结构实验**，VLM 与 AE 均扩展；
不是重新训练的 MHA 模型。Q/O、FFN、ViT 等不变。固定样例的 cache on/off 和转换检查
记录在 validation 数据里，不代表机器人任务成功率验证。

## 关键数据

| 范围 | 结果 |
|---|---:|
| 完整模型 clean p50 / p95 | 221.256 / 230.602 ms |
| ViT 同 pass phase mean | 25.638 ms |
| VLM 同 pass phase mean，已排除 ViT | 77.706 ms |
| AE 10 steps 同 pass phase mean | **120.817 ms** |
| 第 5 个 denoising step clean p50 | **11.845 ms** |
| 该 step profiled GPU-active union | 10.724 ms |
| Profiled matrix / nonlinear+norm / auxiliary | 4.669 / 1.026 / 5.020 ms |

完整模型使用 clean-before / phase / clean-after，每个 block **20 warmups + 100 samples**。
细分使用第 5 个真实 denoising step，包含 18 AE 层和 action input/output，**不包含外层 Euler**。
CPU+CUDA 语义标注和 CUDA-only 计时分开，CUDA-only profile 20 次；每段另有 clean 计时。

**不能混用的口径：**

- `step_categories.csv` 是 profile 中互斥的 GPU-active 分类，不是 clean wall-time 的占比。
- `step_operators.csv` 是 inclusive operator 统计，不应与 exclusive 分类重复相加。
- profiler 区间扰动约 **+54.72%**，profile 后漂移约 **+26.79%**，未用比例强行修正。
- `rms_norm` 已含其 casts 和 affine modulation；不能再加一遍这些子操作。
- KV concat 约 3.973 ms/step 是带 profiler 的移动时间，不是非线性计算；不能直接从 clean 基线扣除。
- clean p50、phase mean 与 profiled kernel mean 是不同统计范围。

## 文件

- `CONFIGURATION.json`：机器可读配置、版本、计时边界和摘要。
- `figures/summary.json`、`step_categories.csv`、`step_operators.csv`、`step_geometry.csv`：收益建模最常用的数据。
- `figures/clean_samples.csv`、`phase_latency.csv`：样本与 phase 统计。
- `figures/*.png/*.svg/*.pdf`：总延迟、细分和 profiler 扰动图。
- `runs/`：validation、coarse、labels、fine 的原始 JSON、单样本 kernel 列表及 Chrome trace。
- `parameters.json`、`architecture/`：该配置的参数、维度和结构图。
- `capture-source/`：采集与分析源码快照及直接辅助模块。启动脚本引用原验证环境，
  不是自带权重/第三方 runtime 的一键部署包；精确 runtime 源码哈希见 environment 记录。
- `provenance/`：原有来源与校验清单；`SHA256SUMS.json` 校验本上传包。

## HTML 快照

按用户要求，`html/latest-h16-report.html` **逐字节保留上传时最新 HTML**。
它讲的是 **xinyao_1025、H16 的 layout 与条件性收益估算**，不是本目录的 H20 GPU 实测，
也不是已完成 H16 GPU 测量的证明。HTML 自带数据，可下载后独立打开；其中原本的相对
源码链接按要求未改写，在本目录中未必可用。HTML 的原始 SHA256 记录于配置文件。

本包不包含权重、其他模型的测量、H16 新测量、账号凭据或服务器使用笔记。

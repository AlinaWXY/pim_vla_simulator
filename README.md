# PIM-VLA · 2026-11-18

## WCK rank switching

[时序、配置与 DQ 统计方法](docs/wck-rank-switch.md)

## GR00T N1.6 H=15

本目录从 `PIM_for_VLA/experiments/2026-11-17-g-power-rerun/source` 复制。
`--model gr00t_n16` 选择 GR00T，省略时使用 π0.5。π0.5 继续使用原 JSON 格式，缺省的 `model` 按 π0.5 解析。

GR00T 的程序依据 Thor 捕获的 VLM 总序列长度生成。单帧 cross K/V 在 4 个去噪步间复用。
DiT 交替执行 16 个 cross 层和 16 个 self 层，输入为 15 个 action token 和 1 个 state token。
cross 的图像和文本 mask 保留完整的 VLM 序列长度。旧的 fused kernel-window 执行路径仍用于 π0.5，
GR00T 的 n16-f 使用同一原生请求执行路径中的本地 PV→O 交接。

Thor 脚本和捕获在 `thor/` 下。捕获使用已校验的 GR00T N1.6 3B 权重、BF16、
PyTorch 2.11.0+cu130 和 batch 1。早期对照使用三张固定种子的 256×256 图像和固定英文提示；640×480 结果见下节。
Eagle 使用保留图像边界的 PyTorch SDPA。基准包括完整编译 AE，以及按仿真边界编译和计时的 AE 算子。

```bash
python3 experiments/prepare_inputs.py --model gr00t_n16 --source thor/capture
python3 programs/gen_pim_program.py --model gr00t_n16 \
    --vl-tokens "$(python3 -c 'import json; print(json.load(open("thor/capture/fixture.json"))["vl_tokens"])')" \
    --output programs/generated --compact
python3 experiments/run.py --model gr00t_n16 --mac-freq-mult 2 --spec reference/spec/four-batch-k2.json \
    --vlm-gpu-quantum-us 2 --output results/gr00t-n16-h15/full
```

引用同一份 LPDDR5X/MAC 配置时，可用 `--spec` 显式指定 π0.5 发布回放的 K2/ACTAB 四批 spec。
VLM 单跑以 Thor 的编译时延为校准目标。校准使用 CUPTI 时序与 Nsight Compute 的 L2 扇区计数，
保留逐 kernel 时长，通过窗口和写回比例搜索在 5% 内的配置。四个结果 case 为
`n16-b`、`n16-c`、`n16-d`、`n16-f`。π0.5 验证记录在 `results/pi05-regression/`。

[640×480 冷缓存单读版汇总](results/gr00t-n16-h15-640x480-single-read/matrix/REPORT.zh-CN.md) ·
[256×256 冷缓存对照汇总](results/gr00t-n16-h15-single-read/matrix/REPORT.zh-CN.md) ·
[自然缓存对照](results/gr00t-n16-h15/matrix/REPORT.zh-CN.md) ·
[Thor AE 算子时延](results/gr00t-n16-h15/thor-ae-operators.csv) ·
[π0.5 回归校验](results/pi05-regression/summary.json)

### H=15 冷缓存单读版（256×256 对照）

按 π0.5 发布口径，在 Thor 上用 192 MiB 的读写扫掠逐次驱逐 L2，再采集编译版 GPU 消费算子。
输入的 shape、dtype 和布局与 PIM 导出匹配，数值采用固定种子的样本。
准备脚本在每个去噪步和层核对 GPU 消费的字节数，将匹配的输出 RD 改为保留依赖的 barrier。
GPU→PIM 写入、每帧 cross K/V 写入和原 VLM trace 随配置保留。

| Case | AE 单跑 ms | AE 完成 ms | 端到端 ms |
|---|---:|---:|---:|
| n16-b | 45.607596 | 45.607596 | 84.374216 |
| n16-c | 45.607596 | 46.763032 | 59.772845 |
| n16-d | 43.869200 | 45.079250 | 59.335780 |
| n16-f | 59.163636 | 60.132686 | 60.132686 |

每个完整 case 重复两次，raw.json 逐字节一致。读取扣除与 GPU 消费输入按步、层逐字节配对，
四个 case 均通过 DQ 流量守恒。PIM 写入字节数与自然缓存对照一致。

```bash
out=results/gr00t-n16-h15-single-read
python3 experiments/prepare_gr00t_single_read.py \
    --cold-profile thor/cold-consumers/measurements.json --output "$out/input-set"
python3 experiments/run.py --model gr00t_n16 --mac-freq-mult 2 --spec reference/spec/four-batch-k2.json \
    --input-dir "$out/input-set/input-data" --program-dir "$out/input-set/programs" \
    --calibration results/gr00t-n16-h15/calibration/calibration.json --vlm-gpu-quantum-us 2 \
    --output "$out/replay"
```

### 640×480 冷缓存单读版（2026-11-18）

三路固定随机原图为 640×480，按 checkpoint 默认评估预处理和 Eagle 对齐，
实际进入 ViT 的张量均为 `[1,3,252,336]`。324 个图像 token 加 47 个文本 token，
总长 371、KV 对齐至 384。H=15，每步 16 个 query，4 个去噪步。

Thor 编译版 VLM p50 为 58.679392 ms，完整 AE p50 为 44.007904 ms。
VLM 校准使用同分辨率采集均值 58.547801 ms 为目标，得到 58.629582 ms（+0.140%）。
采用 `alpha=0.5, window=256, write_high=192, write_low=64`，硬件继续使用 K2/ACTAB 四批、MAC 2× 和 2 µs GPU 让出量子。

| Case | AE 单跑 ms | AE 完成 ms | 端到端 ms |
|---|---:|---:|---:|
| n16-b | 45.950081 | 45.950081 | 104.579663 |
| n16-c | 45.950081 | 47.083475 | 78.358497 |
| n16-d | 44.045465 | 45.167811 | 77.823912 |
| n16-f | 59.458015 | 60.442819 | 74.409592 |

14 组 GPU 消费算子重采冷 L2 编译时延：逐次进行 192 MiB 驱逐，驱逐不计入时延。
只扣除与 GPU 消费输入逐步、逐层配对的 PIM 输出 RD；GPU→PIM 写入和每帧 cross K/V 写入保留。
四个 case 各重复两次，raw.json 逐字节一致，PIM/GPU DQ 守恒、写入保留和来源哈希校验全部通过。
本分辨率下 f 比 d 快 3.414320 ms；f 的 VLM 尾部为 13.966773 ms，d 为 32.656100 ms。

VLM trace 含 757 个 kernel 和 1 个内存复制单元。12 处 fused SigLIP LayerNorm 计数使用每 CTA 两行、时序使用每 CTA 一行，均覆盖相同的 1296×1152 张量；代理及编译代码单独留档，CUPTI 时间保留。
采集使用 packed SDPA；VLM 编译输出与 eager 的相对 RMS 差异为 10.71%，结果用于当前实现的性能分析。

[完整报告](results/gr00t-n16-h15-640x480-single-read/matrix/REPORT.zh-CN.md) · [结果 CSV](results/gr00t-n16-h15-640x480-single-read/matrix/evaluation-b-g.csv) · [输入与复跑校验](results/gr00t-n16-h15-640x480-single-read/matrix/640x480-checks.json) · [本地证据校验](results/gr00t-n16-h15-640x480-single-read/local-evidence-checks.json)

复跑使用这一组独立输入和校准文件：

```bash
base=results/gr00t-n16-h15-640x480-single-read
python3 experiments/run.py --model gr00t_n16 --mac-freq-mult 2 \
    --spec reference/spec/four-batch-k2.json --vlm-gpu-quantum-us 2 \
    --program-dir "$base/input-set/programs" --input-dir "$base/input-set/input-data" \
    --calibration "$base/calibration/calibration.json" --output "$base/replay"
```

## π0.5 发布基线

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

备份保留原始运行二进制。发布源码补上了 `gap_batch` 在无地址 barrier 上的检查。b–g 使用 `frfcfs` 或 `pim_head`，保留原始结果。

Thor 的原始 Nsight、CUDA 时间样本、NCU 读写指标和锁定版 OpenPI 在同一备份。GPU benchmark 使用 `lerobot/pi05_base` snapshot `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`、OpenPI `215abfb` 和 PyTorch `2.11.0+cu130`。基准脚本的 `--support` 指向含 token fixture 的目录，OpenPI `src` 加入 `PYTHONPATH`。

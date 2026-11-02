# H16：统一 bank GEMV Buffer 原生时序复现

基于 `xinyao_1028` 的 bank-local RoPE / DRAM-Z 布局，实现所有 PIM GEMM 共用每 bank 两个 512 B 输入 slot，并重新进行原生 Ramulator 仿真。此分支只包含当前实验、必要后端、离线 C++ 构建依赖及紧凑参考结果。

## 结果

H=16、batch=1、8 heads、head_dim=256、hidden=1024、FFN=4096。16 个 LPDDR5X-8533 channel，各 channel 的 rank0 预留给 GPU，rank1 的16个bank执行PIM。下表单位为 **µs/层/denoising step**；DQ 列为每 channel 时间，已独立核对全部16个channel，当前工作负载的各channel结果相同。

| 阶段 | Latency | 模型 DQ 占用 | 模型 DQ 空闲 | 完整无外部传输工作窗口 |
|---|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 35.273490 | 3.301760 | 31.971730 | 9.368744 |
| V projection/cache | 10.283294 | 3.121664 | 7.161630 | 0.000000 |
| PV | 9.190524 | 2.341248 | 6.849276 | 0.000000 |
| Gate/Up + Z 写入 | 36.198358 | 1.921024 | 34.277334 | 2.440676 |
| Down | 33.931212 | 3.842048 | 30.089164 | 26.172105 |
| **五阶段合计** | **124.876878** | **14.527744** | **110.349134** | **37.981525** |

- 模型 DQ 空闲比例：**88.37%**；完整工作窗口比例：**30.42%**。
- 时间严格闭合：`124.876878 = 14.527744 + 37.981525 + 72.367609`，依次为 DQ 占用、完整窗口、其它 DQ 空隙。
- 全16channel外部传输：**3872 KiB/层/步**。五阶段合计乘18层、10步：**22.477838 ms**。
- 单次 Simulation 的连续 memory/resource control：**125.829886 µs**；combined FFN control：**70.889350 µs**。两项均单列，不重复加入五阶段求和。

![Channel time decomposition](reference/channel-breakdown.png)

这里的 DQ 区间来自 AttAcc movement 模型：WRGB/RDRES 从 issue 起占用2CK；普通 RD/WR 保留 CL/CWL 后的数据 burst。分析拒绝区间重叠，逐channel核对计数和阶段边界。完整窗口只统计整个阶段都没有外部 RD/RDA/WR/WRA/WRGB/RDRES 的有效工作，排除输入/输出阶段中的零碎空隙及 release-only 控制段，并扣除前序数据尾。本次扣除后的窗口长度与扣除前相同。

**完整 AE 端到端 latency 与实际 GPU 可使用时间仍为 null。** GPU softmax、V cast/pack、O projection、GELU/multiply/cast、GPU partial reduction、norm/residual、dispatch及背景VLM没有计时；也未注入 rank0 并发GPU工作。连续control只在这些边界假定输入已准备好，保持PIM资源与refresh状态连续，不能视为完整模型推理时间。

## 从源码复现

Linux；C++20编译器（验证环境GCC14.2.1）、CMake>=3.18、Python3.11及其开发头文件。fmt、yaml-cpp、nanobind/robin_map的必要源码与许可证已随附，C++构建不访问网络。构建和执行必须使用同一Python。

```bash
git clone --depth 1 --single-branch --branch xinyao_1101 https://github.com/AlinaWXY/pim_vla_simulator.git
cd pim_vla_simulator
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh-h16
python verify_reference.py results/fresh-h16
```

输出目录必须不存在。原生日志较多，建议在本地SSD目录运行输出，完成后再复制到共享存储；输出位置不影响仿真cycle。默认执行25项单元/故障注入检查、12项独立进程原生guards、7个原生case、独立地址/生命周期/转置审计及channel时间分解。

输出包括 `summary.json`、`channel-analysis.json`、`channel-partitions.csv`、`channel-phase-windows.csv`、`CHANNEL-ANALYSIS.md`；每个case还包含trace、per-channel原生命令、GEMV事件及审计。`verify_reference.py`比较全部case的cycle、DQ、trace和原生CSV哈希，以及逐channel的时间分解。

## 共享 GEMV 的硬件与调度假设

- 每个PIM bank仅一套 `B[slot2][token16][lane16] BF16`，共1KiB；所有算子使用同一slot状态。没有独立Q SRAM或完整rank MAC输入SRAM；Z仍在bank DRAM row176。
- 每rank只有一个32B external ingress暂存：WRGB完成2CK模型DQ传输后，再占2CK内部multicast端口，完成前不接收下一burst。这是明确的保守背压调度。
- rank端口默认32B/2CK，可将一行输入广播到16个bank；Down将每个source bank的32B列直接scatter到全部目标bank的对应feature lane。GU gather/return、Q填充和Z输入共用该端口；scatter写入能力及吞吐是架构假设，不是实测硬件参数。
- Q/K projection为8CK/MAC，其它GEMM为16CK/MAC；两个slot共用bank MAC资源。只有全部目标bank的有效token/lane填完，MAC才可启动；消费完成前禁止覆写。QKT持有Q slot直到496个本地K全部扫描完成。
- `PIM_GEMV_RELEASE`显式结束slot生命周期，不发DRAM命令。RoPE II=1CK/pair、pipeline=4CK，cast=1CK、rank reduction=4CK；DRAM行开关、refresh、本地读写与完成依赖保留原生模型。
- 地址/ownership保持既有布局：Q/K按token分rank并复制权重；V/PV按feature分片；FFN按256 intermediate features/rank分片；Down最终由GPU归约16份partial。模拟器检查时序和路由，不计算BF16 tensor数值。

## 代码与验证证据

- `experiments/pi05-unified-gemv/`：一个runner、H16地址/trace生成器、channel分析、独立审计和guards。
- `experiments/pi05-bgk-gpuq/backend/`：仅作共享后端目录；精简CMake编译LPDDR5XPIM及23个必需原生单元和Python bindings。小型元数据/DSL导入链保留，用于codegen；未编译其它DRAM模型。
- `reference/expected.json`：本次新仿真的可比较数值与哈希；`reference/validation.json`：25+12检查及连续refresh证据；`reference/independent-repeat.json`：两次独立构建的5个primary逐cycle、DQ、trace及160份原生CSV完全一致。
- `reference/provenance.json`：运行时源码与原生库哈希；构建codegen可能重写生成文件。`PACKAGE-MANIFEST.json`记录发布输入文件的字节哈希与裁剪规则。

最小源码约3.2MiB（参考结果另计），不包含历史结果包、预编译库或其它实验。所有原始许可证保留。参考结果来自新原生仿真；runner没有旧latency常量断言或缓存结果回填。

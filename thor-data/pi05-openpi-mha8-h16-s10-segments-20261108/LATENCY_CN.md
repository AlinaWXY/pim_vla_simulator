# 2026-11-08 连续 GPU 段数据使用说明

H16、MHA8、head_dim256、step10、B1、AdaRMS 预计算缓存。**每段入口只驱逐一次，内部所有小 kernel 连续执行。**
以下不是逐 kernel 冷缓存之和，也不是完整 GPU/PIM E2E 实测。

## 一、先选计时列

- `graph_entry_cold`：Graph 内部 events 测 GPU 段内服务时间，**不含 CPU graph launch/输入提交**；入口驱逐是缓存近似。
- `eager_entry_cold`：原生函数的 event 区间，包含 CPU 提交造成的 GPU 间隙。
- `*_reused`：相同固定地址的复用对照，不代表真实整模型的缓存状态。

两种提交方式分开使用，不能拼接各自最低值。Graph 没有融合 RoPE/Norm，也不是完整模型的 Graph 部署结果。

## 二、连续段的主要数值

下列为 **Graph + 入口驱逐**，合计已汇总 18 层；不能再乘 18。

| 原生输入段 | ms/step 的调用均值合计 |
| --- | ---: |
| Q/K RoPE 应用 | 0.615944 |
| RoPE 位置表（只生成 1 次） | 0.022523 |
| scale + mask + softmax + cast | 0.353058 |
| GELU + gate×up | 0.153910 |
| Attention 残差 + FFN 前 norm | 0.545320 |
| FFN 残差 + 下一层/最终 norm | 0.548933 |
| SV 排布 + O 投影 + 残差 + norm（O 在 GPU） | 0.964560 |

**连续组合行替换其组成部分，不得重复相加。** FFN 后组合已经包括下一层入口 norm，最后一层包括 final norm；
第 0 层的初始 norm 仍单独取一次。位置表不能每层都再计一次。

## 三、FP32 逻辑 PIM 接口候选

以下输入是已解包的逻辑稠密缓冲区，不含物理 bank 数据搬运。FP32 值从原生模型结果构造，不验证 PIM 数值精度。

| 完整 GPU 段 | Graph 单次均值 µs | 18 次合计 ms |
| --- | ---: | ---: |
| FP32 Gate/Up → GELU×Up → BF16 | 11.155 | 0.200784 |
| FP32 Q/K → RoPE → BF16 | 37.678 | 0.678208 |
| FP32 score → scale/mask/softmax → BF16 | 17.135 | 0.308426 |
| 上述 score 流程 + 补齐/奇偶 KV 分片 | 25.442 | 0.457952 |
| FP32 O → BF16 → 残差 + norm（O 在 PIM） | 31.714 | 0.570851 |
| FP32 Down → BF16 → 残差 + 下一次 norm | 31.943 | 0.574979 |
| 两份 FP32 PV 部分和 → sum/cast/排布/O/残差/norm（O 在 GPU） | 55.452 | 0.998128 |

softmax 基础行与含分片行二选一；O 在 GPU/PIM 的两个出口二选一；PIM 已做 RoPE 时不再计 GPU RoPE。
两份 PV 部分缓冲区为各自独立的 FP32 数组，形状 `[1,8,16,256]`；P 分片输出每 head/rank 为 `[16,496]` BF16。
全部明细及 Eager 列见 [基础块](analysis_verified/native_blocks.csv)、[连续段](analysis_verified/handoff_composites.csv)、
[FP32 接口](analysis_verified/pim_interface_candidates.csv) 与 [cases.csv](analysis_verified/cases.csv)。

## 四、I/O 是否包含，以及 E2E 怎么加

**包含：**段内实际输入/权重读取、内部中间结果读写/cast、输出 store 的执行成本。不会在小 kernel 之间人为清缓存。

**未测或未保证：**输出全部回写 LPDDR/PIM 可见、物理 bank 解包/地址写入、额外 DMA、跨 GPU/PIM 同步/dispatch，
以及后台 VLM 争用。Graph 列还不含 CPU launch、Graph 构建和输入/参数更新。

必须先对应仿真器 RDRES/ST/DQ 与 GPU 访存：同一笔传输不能重复计；确实额外发生的转存、可见性和布局步骤也不能漏。
不能把总时间减 `bytes/峰值带宽` 当成纯计算时间。无重叠时按实际段顺序加，异步时按依赖/资源争用算关键路径。

本批只对真实第 5 步的各段重复计时；乘 10 是“各步成本相同”的估算，不是全 AE 实测。
正常完整模型的独立 API wall p50 为 **224.134475 ms**；其 phase 均值为 ViT **25.775816**、VLM **78.444153**、
AE **118.689149**、Other **0.848260 ms**。这些仍是原生 eager，不能和 Graph 分段混用来宣称纯 PIM 收益。

原始数值及完整方法见 [README](README.md) 和 [summary.json](analysis_verified/summary.json)。

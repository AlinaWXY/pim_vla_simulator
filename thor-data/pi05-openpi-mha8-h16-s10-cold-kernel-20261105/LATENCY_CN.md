# π0.5 H16 冷缓存算子耗时与 E2E 使用说明

配置：Thor、OpenPI π0.5、MHA 8 heads、head_dim=256、H=16、step=10、B=1，保留 AdaRMS 预计算缓存；沿用原生未融合 attention 和 selective BF16/FP32 精度。

**下表是一个真实 denoising step 内的 GPU kernel 耗时之和，已经汇总全部 18 层，不是单层耗时，也不是完整 step 的端到端 latency。** 取每个请求的第 5 步，重复 3 次求平均；每步 1,033 个 kernel，每个 kernel 仅采集 1 pass。权重提前加载、完整 warmup 20 次，但 NCU 在每个被测 kernel 前清硬件缓存；不是重新从硬盘加载模型。

## 1. 已有算子和耗时

单位均为 **ms/step**；同一行多个数值与算子名称一一对应。

| 算子 | 耗时 |
| --- | ---: |
| Q / K / V projection | 0.3948 / 0.4169 / 0.3912 |
| QKᵀ / SV 矩阵乘法 | 0.3888 / 0.4026 |
| O projection | 0.4236 |
| FFN gate / up / down projection | 0.7899 / 0.7631 / 0.8665 |
| Action input / output projection | 0.0074 / 0.0088 |
| Softmax | 0.1766 |
| GELU | 0.0878 |
| RMS 平方 / 均值归约 / rsqrt | 0.1763 / 0.3744 / 0.1206 |
| Norm、AdaRMS 调制的加法 / 乘法 | 0.5639 / 0.5087 |
| Residual gate 乘法 / residual 加法 | 0.2232 / 0.1950 |
| FFN gate × up 逐元素乘法 | 0.0982 |
| Attention scaling / mask 加法 | 0.0923 / 0.2274 |
| RoPE 位置生成 / 应用 | 0.0238 / 1.1656 |
| K/V 拼接 | 4.1007 |
| GPU kernel 执行的 copy、dtype cast | 0.8888 |
| 其他小操作：掩码、cumsum、sum、where、比较、逻辑与、arange、ones、cat 及少量加减乘 | 0.0597 |
| **全部合计** | **13.9368** |

分类汇总：**线性矩阵运算 4.8538；非线性/RMS 核心 0.9357；逐元素调制/门控 1.9086；RoPE 1.1894；拼接/拷贝/转换 4.9895；其他 0.0597 ms**。AdaRMS conditioning dense 已预计算，在线为 **0 个 kernel**，但实际归一化、调制乘加和 residual 仍需执行。RMS/Norm 含 36 次层内归一化及 1 次最终归一化。

精确数值见 [operators.csv](analysis_verified/operators.csv) 和 [families.csv](analysis_verified/families.csv)；表内数值有四舍五入。最终分类以 `analysis_verified/` 为准。

## 2. 包不包含 memory input/output latency

**包括 kernel 执行期间实际发生的访存工作，不是“纯算术计算时间”；但不等于一次完整的 GPU↔PIM 数据交换时间。**

| 项目 | 是否计入上述时间 |
| --- | --- |
| kernel 内读取输入、权重、中间张量，以及相应的访存等待 | **包含**；由实际缓存/内存访问决定，没有扣除 DRAM 时间 |
| kernel 内输出 store，以及执行期间相关的等待 | **包含**；输出写入的是 GPU 内存层级，不保证计时覆盖所有后续 LPDDR 回写 |
| 等待所有输出最终落到 LPDDR、并可被 PIM 正确读取 | **未单独测量或保证**；若实际接口需要排空、flush/fence 或一致性操作，应另计尚未覆盖的成本 |
| GPU kernel 实现的 K/V concat、copy、cast | **包含**，已在表中单列；这些不是额外的 GPU↔PIM 传输 |
| 独立 DMA memcpy、H2D/D2H、GPU↔PIM 额外搬运及 PIM 输入布局转换 | **不包含额外路径**；仅包含当前 GPU 实现中已列出的 kernel |
| kernel 外的 CPU launch、调度/同步等待、执行间隙，以及 NCU 清缓存/采集开销 | **不包含**在这个 kernel duration 合计中 |
| 权重加载、AdaRMS 预计算、ViT/VLM、step 外的 Euler 更新 | **不包含**在这张单步表中 |

硬件缓存清空不删除 AdaRMS 预计算张量。单个 kernel 运行时仍可复用刚读入缓存的数据；本次没有把每个算子的时间拆成“DRAM 读 + 计算 + DRAM 写”三项，也不能用总时间减去 `bytes/峰值带宽` 来得到纯计算时间。**尤其不要把 kernel 结束直接当作 PIM 已能读到全部输出。**

## 3. 后续统计 E2E 时怎么用

1. 表中层内算子已累计 18 层，**不要再乘 18**。只测了第 5 步；若乘 10，必须标注“假设各步成本相同的估算”，不能称为完整 AE 实测。
2. 先确定新方案实际保留哪些 GPU 算子。线性移到 PIM 后，并非只剩 0.9357 ms；逐元素、RoPE、布局处理等应按最终分工计入。若新布局消除了旧 GPU 路径的 KV concat，就不要机械地再加 4.1007 ms。
3. **不要重复加入同一份输入/输出的访存时间。** GPU kernel 时间已含其内部访存；另加的应是未覆盖的跨 GPU/PIM 交接、数据可见性、额外复制/重排、调度和同步。共享 LPDDR 不一定需要一次额外物理复制，但是否需要一致性或布局操作仍须确认；PIM 仿真已包含的传输也不能再加一遍。
4. 无重叠的串行估算可用 `PIM 实际执行部分 + GPU 实际保留部分 + 尚未覆盖的交接开销`；异步方案按依赖关系计算关键路径，而不是把两边所有时间直接相加。这批冷缓存数据是估算输入，不是已验证的异构 E2E latency。

已有正常缓存 GPU 基线另存于 [上一批数据](../pi05-openpi-mha8-h16-s10-adarms-20261105/README.md)：同轮 phase 均值 ViT **26.2905**、VLM **78.6600**、完整 AE **118.0245**、Other **1.0125 ms**，合计 **223.9875 ms**；独立完整 API wall 的 p50 为 **219.2633 ms**。这些是另一种测量口径，不要与本表重复累加，也不要把均值与 p50 混加。

口径参考：[Nsight Compute 缓存与计时说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[PTX load/store 缓存说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

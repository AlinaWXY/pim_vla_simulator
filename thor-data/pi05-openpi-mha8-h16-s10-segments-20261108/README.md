# 2026-11-08 π0.5 H16 连续 GPU 段延迟

本批修正 11 月 5 日“每个小 kernel 前清缓存”的计时边界，旧数据不覆盖。
配置仍为 Thor、OpenPI π0.5、H16、S10、MHA8、head_dim256、B1、AdaRMS 条件预计算缓存，
VLM 和 AE 均采用此前的物理 K/V 扩展；原生 selective BF16/FP32、未融合 attention、CPU/interop 各 1 线程。

入口先执行一次 256 MiB `.ca` 读流作为缓存驱逐近似，再连续执行整个逻辑段。
**段内不清缓存；没有将旧结果按 kernel 个数相除。** 不用 NCU/Nsight 采集模型计时。
这不是实物 PIM 已接入后的交接/E2E 测量，也不声称每个输入的所有字节必然从 DRAM 读取。

## 核心结果

下表为对应调用次数的实测均值合计，单位 ms。RoPE 应用和各层算子按 18 个真实层分别测量。

| 完整计算段 | 次数/step | Graph 段内时间，入口驱逐 | Eager 段时间，入口驱逐 |
| --- | ---: | ---: | ---: |
| 原生 Q/K RoPE 应用 | 18 | 0.615944 | 1.789467 |
| RoPE 位置表生成 | 1 | 0.022523 | 0.171099 |
| 上述 RoPE 合计 | 18+1 | **0.638467** | **1.960566** |
| scale + mask + softmax + cast | 18 | 0.353058 | 0.984486 |
| GELU + gate×up | 18 | 0.153910 | 0.475453 |
| Attention residual + FFN 前 norm | 18 | 0.545320 | 2.646555 |
| FFN residual + 下一层/最终 norm | 18 | 0.548933 | 2.592205 |

**两列不是同一计时口径。** Eager 的外部 CUDA events 包围原生函数，包含 CPU 发射造成的 GPU 间隙。
Graph 的计时 events 放在图内部，记录 GPU 段内服务时间，不含 CPU graph launch/输入提交。
Graph 捕获同一数学函数，不做 `torch.compile` 或 RoPE/Norm 融合；不称为最优融合 GPU baseline。
不能把新 Graph 与旧 NCU 数据的差额全归因于缓存，也不能将新 Graph 分段和旧 Eager 整体的差异全归因于 PIM。

正常完整模型另外测了 60 次 clean 和 60 次 phase：API wall p50 **224.134475 ms**；
phase 均值为 ViT **25.775816**、VLM **78.444153**、AE **118.689149**、Other **0.848260 ms**。
完整模型仍是原生 eager、自然缓存，未启用模型级 Graph。独立块的合计不能替代这个完整执行时间。

## 三张表怎样用

- [native_blocks.csv](analysis_verified/native_blocks.csv)：278 个原生逻辑块，矩阵与非线性都保留。
  RMSNorm 连同调制/cast 完整测；QKV 三个投影成组，Gate/Up 两个投影成组；K/V concat 成对测。
- [handoff_composites.csv](analysis_verified/handoff_composites.csv)：54 个连续组合段。
  残差之后直接接归一化；O 留 GPU 时，提供 SV layout→O→残差→norm 整段。
  **这些行替换重叠的基础行，不能把两表一起相加。**
- [pim_interface_candidates.csv](analysis_verified/pim_interface_candidates.csv)：126 个 FP32 逻辑接口候选段。
  包含 FP32 Gate/Up→GELU/乘法→BF16；FP32 Q/K→RoPE→BF16；FP32 score→softmax→BF16；
  O/Down FP32→BF16→残差/norm；两份 FP32 PV 部分和→sum/cast/layout/O/residual/norm；
  以及 P 补齐至 992 列后按奇偶 KV 分成每 head/rank `[16,496]` BF16。
  FP32 输入由真实模型输出构造；PV 的两份缓冲区由 native SV 各取一半构造。只验证 GPU 实现，
  不声称模拟了真实 PIM 数值误差或物理 bank 布局。传输前的物理解包/地址写入没有混入这些数值。

当前可读的同学分支 `xinyao_1104@8764996de08741f248746b7094830360f666fbe6` 将 RoPE 放在 PIM，
并要求 GPU 合并两个 token 分片的 PV 部分结果；GPU-RoPE 行仅供另一种分工选择，不能在 PIM-RoPE 方案里重复收费。
P 分片和 PV 求和的逻辑形状参考该提交的 `experiments/pi05-unified-gemv/TOKEN-PV.md`。

## 计时与缓存验证

- 有效主采集：`pi05_h16_segments_20261108_02`；接口补充：`pi05_h16_interfaces_20261108_01`。
  共 **458 个段 × 4 种口径 × 20 次 = 36,640 次计时**，覆盖真实第 5 步的所有 18 层。
  每个运行捕获同一确定性输入的第 5 步，固定缓冲区重复不等于不同模型请求。
- 四种口径为 Eager/Graph × 入口驱逐/地址复用控制；交错并反转执行顺序降低顺序漂移。
  每段先预热，Graph 在 side stream 捕获；`enable_timing=True, external=True` 的 events 位于 graph 内部。
  每次入口驱逐都在计时前完成。未修改时钟；这类分段微基准不等同于正常整模型运行。
- Graph、Eager 与各自参考输出精确相等（`torch.equal`）；原生完整模型输出也与 11 月 5 日固定输入基线相同。
  FP32 接口候选只与其声明的接口函数比较，不是机器人任务精度测试。
- 1 MiB 输入、两 kernel 链的 UCF non-CPU DRAM 读量中位数：地址复用 **224 B**，
  入口一次驱逐 **1,053,824 B**，每 kernel 驱逐 **2,104,832 B**。256 MiB 驱逐读流也通过字节量校准。
  这验证探针的段内复用；不等于每个模型算子的 DRAM 字节量已被测量。
- 地址复用列只是固定地址对照，不是完整模型自然缓存状态。入口驱逐会同时影响 GPU 常量/权重缓存，
  没有选择性地只使 PIM 输入冷却。未模拟与后台 ViT/VLM 的并发争用和抢占。
- 空 graph 计时均值 **1.424 µs**，作为参考保留、未做减法修正；小段需考虑计时节点和调度粒度。
- 初始化失败的 `pi05_h16_segments_20261108_01` 保留在 raw 中，**不用于统计**。
  原因是参数被创建在 inference_mode 中而缺少版本计数；`runner_v2.py` 将构造移出该上下文，未更改模型算术。

## 内存 I/O 和 E2E

本数据包含段内实际输入/权重读取、内部中间结果读写、cast 和输出 store 的执行成本。
**不保证计时终点已把全部输出排空到 LPDDR 或使 PIM 可见**；物理数据交接、DMA、coherency/fence、
host/PIM dispatch、Graph 输入/参数更新、图构建和并发争用不在段内服务时间里。
静态 suffix mask 常量预分配，CPU/H2D 创建开销不在分块计时内。

估算前必须逐项对应仿真器的 RDRES/ST/DQ 传输与 GPU 输入输出路径：若两者描述同一笔传输，不能重复计；
若确实先落一个缓冲区再读到另一端，应计实际额外步骤。不能直接用 `bytes/峰值带宽` 从本结果扣出纯计算时间。
共享 LPDDR 并不自动消除可见性、布局和调度成本。

若最终部署使用 Graph GPU 段，应选择优化程度可比的完整 GPU baseline，再归因 PIM 收益。
本轮没有给出未经实物验证的异构 E2E 或加速比。更简洁的取数规则见 [LATENCY_CN.md](LATENCY_CN.md)。

## 文件与复现

- `figures/index.html`、SVG/PNG/PDF：可视化报告；`analysis_verified/`：正式汇总，`cases.csv` 保留每层数值。
- `raw/`：缓存探针、作废初始化、有效主采集和接口补充；`logs/`：完整控制台日志。
- `code/run_v2.sh` 是有效主采集入口，`code/run_interfaces.sh` 是接口补充入口。
  原 `runner.py/run.sh` 仅保留为探针/作废尝试的历史证据，不作为新主采集入口。
- `COMMANDS.json`：本次准确命令；`MANIFEST.json`：文件哈希；`RESOURCE_STATUS_FINAL.txt`：最终资源状态。

仅 CPU 分析/绘图；在服务器上按 SKILL 使用 CPU Slurm 和现有分析环境：

```bash
python -B -m unittest discover -s code -p 'test_*.py' -v
python -B code/analyze.py --root . --output analysis_reproduced
python -B code/render.py --root . --output figures_reproduced
```

GPU 重跑必须先确认 Thor 空闲、磁盘余量足够，并改用新的 run tag；不要覆盖已有输出。
没有重新下载权重，没有修改共享环境或同学的仿真源码。

方法参考：[PyTorch 2.11 CUDA Graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)、
[CUDA Event](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html)。

上传仓库 `AlinaWXY/pim_vla_simulator`，独立分支
`thor-data/pi05-openpi-mha8-h16-s10-segments-20261108`。

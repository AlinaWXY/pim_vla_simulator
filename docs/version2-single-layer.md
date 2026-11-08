# Version2：单层、单个 denoising step

用户于 2026-11-09 要求先观察单层 latency，暂停校准和端到端仿真。本轮只运行 AE 的 step 0 / layer 0；不加载 VLM、不恢复旧校准作业、不运行 18 层 × 10 步或并发消融矩阵。Version2 文档中的完整实验要求不改变本轮范围。

任务文档 SHA-256：`a8273123c0f306776eb09a8cc98269071b11d5e379e52398c1ae18f8a4c1fbfa`。原件保存在服务器 `results/task-documents/version2-rewrite.docx`；分支保存实现说明及来源哈希。

## 统计边界

单层包含自然 kernel ledger 的 ordinal 30–84，以及对应的 PIM 阶段、GPU RoPE / helper、注意力和 FFN 残差与归一化。mask、RoPE 系数已在层外准备；全局 action 输入投影、下一层、最终 norm / 输出投影不计入。单层总时延 = 原生 PIM 阶段时延 + 实测 GPU 算子时延（向上取整到 CK）。GPU 部分采用已有实测，不在本轮重新测量或校准。

streaming 在 PV 与 O 之间恢复实测转置：feature 4.352 μs，token 4.992 μs；删除原始序列中对应的自然 contiguous 记账，避免重复计算。fused 无此转置。表中的 layout / RoPE / dataflow 是五种 AE 结构；这里没有 VLM 并发，因此不等同于完整 b–g 消融。

Layout 继续遵循用户要求保留 1108 的实际 feature/token 映射，包括 feature PV 的分片语义；没有把文档中“PV 按 token 分片”的概述用于悄然改变 feature 路径。详细映射见 [程序设计](program-design.md)。

## 本轮实现

- ACTAB 由原生 spec 展开为四批、每批四个 bank 的激活，批间 16 CK，共用滚动 FAW 历史；读/计算就绪分别为 20/36/52/68 CK。外部 CA 2 CK，私有 ACTAB 1 CK。
- 每个 bank 独立 RoPE，两输入槽、双输出 FIFO；八对元素向量完成 11 CK、启动间隔 8 CK，Q 执行可与 K 装载重叠。streaming 固定功能操作不占 CA，fused 私有命令占 1 CK。
- QKT 的归约结果存入现有每 bank 2 KiB 输出池，496 列批满后集中 LOCALWR；没有额外 rank score cache。streaming 仍逐列 RDRES。
- 描述符内部 FIFO32 允许已满足依赖的后续操作及安全 ACT/PRE 提前发射。完成依赖保护数据，issue 依赖保护流水顺序；barrier、I/O epoch 和未发请求所需行限制重排。
- Q 导入先完成整个 WR epoch，再 LOCALRD 装载 SRF。生成器为阵列、SRF、输出池、gather/CAST 和 RoPE 资源生成 RAW/WAR/WAW 依赖。

## 结果与验证

实现与验证进行中；结果完成后补入本节。历史完整 AE 原始结果保持原样，不用于声称 Version2 已完成完整验证。

复现入口：

```bash
python programs/gen_pim_program.py --output programs/generated --compact
python experiments/run_single_layer.py --binary build/pim_vla_sim \
  --output results/version2-single-layer --repeats 2
```

后续恢复并发实验前仍需验证 GPU 仲裁路径；本轮所有 rank0/VLM 请求计数应为零，不能据此证明并发性能或校准误差。

已知后续检查项：`next_gpu` 在外部 ACT1/ACT2 lane 被占用时，可能先选中不能发射的 ACT1 候选而没有回选其他 bank 的就绪请求。本轮 PIM 窗口中的同类问题已修复并回归；GPU 路径须在恢复校准/并发前处理。它不影响这里没有 rank0 请求的单层 AE 结果。

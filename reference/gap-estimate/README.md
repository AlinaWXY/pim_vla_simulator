# c–g 同切片实验复现

[对照表和分阶段报告](gap-estimate.md) · [完整结果与校验和](gap-estimate.json)

[MAC 4× → 2× 消融结果与复现命令](mac-frequency-2x.md)

在仓库根目录运行以下命令。构建环境见[仓库 README](../../README.md)，需要 Linux、支持 C++20 的编译器、CMake ≥ 3.18 和 Python ≥ 3.10。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pim_vla_sim --parallel
c++ -std=c++20 -O3 tools/gap_replay.cpp -o build/gap_replay

python3 tools/capture_gap_traces.py \
  --binary build/pim_vla_sim \
  --output build/gap-traces

python3 tools/estimate_dq_gaps.py \
  --traces build/gap-traces \
  --engine build/gap_replay \
  --output build/gap-report
```

采集脚本生成 c–g 的 PIM 程序并运行首层与末层共十组 AE solo 切片，每组对比 trace 开关两次运行的 `raw.json`。重放脚本将两个候选切片分别重复 180 次，使用仓库内的 AE 调度、VLM profile 和命令对矩阵。主表的 c–e 使用 α=1.0 和 `same_first`，f/g 使用融合阶段的重叠估算及旧版共享张量 I/O 去重。Q 交换完整计入 f/g 的 AE 时间。

生成的 `build/gap-report/gap-estimate.md` 中，c–g 同切片对照应选中 step 9 / layer 17。五行端到端时长四舍五入到 0.001 ms 后依次为 145.946、141.545、134.735、131.398 和 124.900。归档的整 kernel 融合窗口结果用于报告中的历史对照，复现本表只需上述十组 AE solo trace。

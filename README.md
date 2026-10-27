# PIM VLA Simulator

Pi0.5 AE MHA8 线性算子时序复现（H 表示 action horizon，head 数均为 8）：

- [H=16 最终版：bank 内 RoPE、DRAM Z 与全套已建模 AE 阶段](experiments/pi05-bank-rope-h16/README.md)，[冻结结果与完整复现包](experiments/pi05-bank-rope-h16/results/published-h16/README.md)。
- [H=20/30/40：扩大 buffer、整批执行的原生对比实验](experiments/pi05-bank-rope-horizon/README.md)，[延迟、布局及完整无外部传输命令阶段统计](experiments/pi05-bank-rope-horizon/results/published-sweep/REPORT.md)。
- [H=20：共享 X 的 QKV、GPU RoPE 边界、DRAM Q 回读](experiments/pi05-bgk-gpuq/README.md)。
- [H=16：独立 mapping、原生仿真与 breakdown](experiments/pi05-bgk-gpuq-h16/README.md)。

两个版本分别保存结果与审计，共用 `experiments/pi05-bgk-gpuq/backend/` 的原生后端和构建依赖；
不依赖旧服务器实验路径，也不执行根目录的历史上游 `ramulator2/`。

独立的 Thor + Ramulator（LPDDR PIM）仿真器项目，目前处于源码基线准备阶段。

## 仓库结构

- 项目根目录管理自己的 Git 历史，初始分支为 `main`。
- `ramulator2/` 是纳入本项目版本管理的普通源码目录，可以直接修改；它不是子模块或嵌套仓库。
- 项目远端：[AlinaWXY/pim_vla_simulator](https://github.com/AlinaWXY/pim_vla_simulator)（私有仓库），本地名称为 `origin`。
- `git commit` 只创建本地提交；`git push` 才会上传到指定远端。

## Ramulator 来源

- 上游：[CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2)
- 源码基线：`b7c70275f04126c647edb989270cc429776955d1`（Ramulator 2.0）
- 选择依据：[ATTACC 的安装脚本](https://github.com/scale-snu/attacc_simulator/blob/main/set_pim_ramulator.sh)使用此提交，再应用自己的 HBM3 PIM 扩展。
- 原始说明和许可证保留在 `ramulator2/README.md` 和 `ramulator2/LICENSE`。

当前已有上游 LPDDR5 模型，尚未编译，也尚未实现 LPDDR PIM 扩展或 Thor 接口。

本机原始 Ramulator Git 元数据（包括提交历史和原分支）保存在
`.git/upstream-backups/ramulator2.git`。此备份不参与本项目提交，也不会随本项目的 clone/push 传输。
可在项目根目录用以下命令查看原始历史：

```bash
git --git-dir=.git/upstream-backups/ramulator2.git log -5 --oneline
```

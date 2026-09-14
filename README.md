# PIM VLA Simulator

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

## ATTACC 增量分析

[`AttAcc Ramulator Analysis`](docs/attacc-ramulator-analysis/README.md) 对照 ATTACC 使用的固定 Ramulator 提交，记录全部有效文件变化、新命令、时序与能量来源，并区分已确认代码、可复核推导和待补资料。分析分支为 `attacc-ramulator-analysis`；报告附固定源码快照及可重复的补丁重建检查。

本机原始 Ramulator Git 元数据（包括提交历史和原分支）保存在
`.git/upstream-backups/ramulator2.git`。此备份不参与本项目提交，也不会随本项目的 clone/push 传输。
可在项目根目录用以下命令查看原始历史：

```bash
git --git-dir=.git/upstream-backups/ramulator2.git log -5 --oneline
```

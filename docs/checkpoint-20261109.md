# 2026-11-09 上传与暂停记录

最新范围：用户随后提供 Version2，并要求仅查看 **一层、一个 denoising step** 的 latency。当前实现冻结源码为 `f7c7af6eab0655b526773b9b15ed63e8e6f73add`，新版单层验证与结果见 [Version2 单层报告](version2-single-layer.md)。校准及端到端实验仍暂停。

以下是 Version1 的历史交接记录。`results/full-k4-final` 的冻结文件保持原样，不能把它们的校准/恢复配置与 Version2 的二进制或程序混用。将来恢复完整实验须先处理 Version2 报告中的待办，并建立新的冻结目录。

用户要求：先上传 `xinyao_1109`，校准及之后的工作先暂停。

已完成：旧代码完整备份及 `legacy-1108` 标签；Ramulator2原生重写；时序、资源、故障注入和单层并发验证；五种完整18层×10步AE solo及独立守恒/来源审计。具体时长和文件哈希见 [完整AE验收记录](../reference/validation/full-k4-final-solos.json)。

正式实验使用的冻结源码为 `1e0b73bc8da695268fbf79f1013653b54120316e`。之后的提交补充验收记录、审计工具和交接文档，不改变该次运行的二进制、程序或输入。

服务器位置：

```text
dev0:/fact_data/xinyaowang/pim_vla_sim2_20261109
results/full-k4-final/
  frozen/                 源码、五程序、可执行文件、libramulator.so及SHA清单
  configs/                固定实验配置
  solos/ae-solo/           五份已完成的完整AE原始结果和provenance
  calibration/            已中断的候选配置及日志；不是有效校准结果
  jobs/、logs/            作业记录
  audit-frozen.json        冻结文件和Git对象独立审计
results/acceptance/full-k4-final-solos.json
```

AE数组作业 `1976674_0..4` 全部成功。校准数组 `1976675_0..5` 和等待汇总作业 `1976676` 按用户要求取消；完整case作业没有提交。取消前产生的配置和日志继续保留，不能据此宣称校准已通过。

恢复工作时应先核验冻结清单，重新运行被中断的完整VLM候选，再选择首个通过总时长±5%门槛的参数并冻结；然后才运行九个完整case、每case两次，完成独立审计和图表。原生进程没有中途checkpoint，`--resume`只复用具备完整provenance且校验通过的结果，不会从被取消的kernel位置继续。不得覆盖或把现存中断日志误当作完成输出。

源代码已提供 `tools/audit_campaign.py`。完整矩阵恢复并结束后，可在服务器仓库根目录运行：

```bash
python3 tools/audit_campaign.py results/full-k4-final \
  --expected-source 1e0b73b --git-repo . \
  --output results/full-k4-final/audit-campaign.json
```

该完整审计要求有效校准、五AE、2357个VLM工作单元和全部九case的两次原始输出；当前暂停状态不能通过完整验收。

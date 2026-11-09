# Version3 缩放流量消融

运行源码为 `a766f13d2316cf8d6d4e2b32091af5765dbeefce`。dev0 作业 `1984774` 使用每通道 `W_max=128`、`α=1.0` 和写水位 `96/32`。各组共享同一份缩放后的 VLM 输入与校准文件。发布时已完成五组，e-lookahead（`1984774_2`）仍在运行。

| Case | 端到端延迟（ms） |
|---|---:|
| b | 169.576248 |
| e | 128.883328 |
| e-lookahead | 运行中 |
| e-frfcfs | 129.343722 |
| e-gap | 147.622221 |
| g | 127.487527 |

g 比 e 提前 6.610079 ms 完成 AE，VLM 尾部增加 5.214278 ms，端到端缩短 1.395801 ms。完整延迟分项、逐单元校准误差和机器可读汇总见 [报告](ablation.md)、[CSV](ablation.csv) 和 [JSON](ablation.json)。

VLM solo 总时长 105.354238 ms，相对 profile 合计误差 +0.1270%。137 个工作单元偏差超过 1%，最大绝对偏差为 9.129%。最初的启动门槛校验与后续按哈希绑定的豁免均保留在证据中。 [初始门槛检查](validation.json) · [校准账本](calibration-scaled.json) · [缩放输入](final-vlm.json)

每组已完成实验的 `cases/` 子目录包含原生汇总、CSV 和守恒检查。[原始证据归档](dev0-evidence.tar.gz) 另含两次并发、两次 AE solo 的 `raw.json`、配置、日志和来源记录，以及校准的初始与最终 solo。配置中的 dev0 绝对路径是原始来源记录。

发布前核对了原始重复结果、校准和输入哈希、运行库与可执行文件哈希、工作单元覆盖、DQ 字节守恒和端到端时间恒等式。解包前可按 [归档清单](archive-manifest.json) 核对 SHA-256。

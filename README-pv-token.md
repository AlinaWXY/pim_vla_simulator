# xinyao_1103：QKV融合、V本地写回、token分片PV

测得主路径latency **183.857380 → 171.282552 µs/层/step**。基线为810e01e；修改前、新版、新backend旧模式各重新运行7案例。GPU对PV部分结果的求和尚未计时，所报时间包含两边结果的DQ导出。

GPU X一次性交错写入每rank两个bank，QKV共同使用每个SRF输入片段；V经本地cast、打包及读改写进入token分片cache。GPU已完成softmax的P按KV列分片并整批存入两bank，seal后再逐批读入SRF、广播、计算PV。两rank的部分结果回GPU相加。每bank独立1KiB SRF与2KiB共享FP32 output；整块SRF消费完成后才覆盖，QKV输出占1.5KiB、PV1KiB。完整布局与约束见[TOKEN-PV.md](experiments/pi05-unified-gemv/TOKEN-PV.md)。

## Latency对照

单位µs；attention块的“修改前”为原Q/K链与独立V之和，修改后为融合chain。新独立V仅为诊断，未重复计入总和。

| 阶段 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| QKV + RoPE + QKT + V 本地写回 | 59.043348 | 50.511300 | -8.532048 |
| PV | 18.994500 | 14.951720 | -4.042780 |
| Gate/Up | 47.127934 | 47.127934 | +0.000000 |
| Down | 58.691598 | 58.691598 | +0.000000 |
| **主路径合计** | **183.857380** | **171.282552** | **-12.574828** |

单层串联控制：184.809450 → 171.960726µs；单独列示，不重复计入上述总和。逐phase值与边界余量见[phase对照](reference/pv-token/phase-latency.csv)，新stage详细值见[summary](reference/summary.json)。

attention 与 PV 的子阶段墙钟时间如下，等待和行切换计入所在阶段；它们不是可与 DQ/CA 忙时再次相加的独立资源开销。

| 路径 | 子阶段类别 | 修改前 | 修改后 | 变化 |
|---|---|---:|---:|---:|
| attention | dram_read | 0.300160 | 0.000000 | -0.300160 |
| attention | dram_write | 0.532784 | 0.000000 | -0.532784 |
| attention | 外部读出 | 8.687756 | 7.746004 | -0.941752 |
| attention | 外部写入 | 0.481194 | 0.481194 | +0.000000 |
| attention | gemv_release | 0.007504 | 0.006566 | -0.000938 |
| attention | GPU 输入整批暂存 | 2.491387 | 1.245693 | -1.245693 |
| attention | 内部补数与 MAC（含行切换等待） | 36.626086 | 29.840594 | -6.785492 |
| attention | 本地写回 | 0.347998 | 1.704346 | +1.356348 |
| attention | QKT 计算与归约 | 8.681190 | 8.600522 | -0.080668 |
| attention | RoPE | 0.885472 | 0.885472 | +0.000000 |
| attention | 阶段边界余量 | 0.001817 | 0.000909 | -0.000909 |
| PV | 外部读出 | 0.959574 | 1.920086 | +0.960512 |
| PV | gemv_release | 0.002814 | 0.002814 | +0.000000 |
| PV | GPU 输入整批暂存 | 2.442581 | 1.243817 | -1.198764 |
| PV | 内部补数与 MAC（含行切换等待） | 15.588622 | 11.784094 | -3.804528 |
| PV | 阶段边界余量 | 0.000909 | 0.000909 | +0.000000 |

## DQ空闲breakdown

下表为channel0，全部16channel均独立审计。完整无传输窗口扣除了边界传输尾部。

| 阶段 | 原DQ空闲 | 新DQ空闲 | 新DQ占用 | 新完整无传输工作窗口 | 新其它空闲 |
|---|---:|---:|---:|---:|---:|
| QKV + RoPE + QKT + V 本地写回 | 53.580436 | 47.209540 | 3.301760 | 41.030934 | 6.178606 |
| PV | 16.653252 | 13.060712 | 1.891008 | 11.784094 | 1.276618 |
| Gate/Up | 43.285886 | 43.285886 | 3.842048 | 40.757038 | 2.528848 |
| Down | 50.767374 | 50.767374 | 7.924224 | 47.638206 | 3.129168 |
| **合计** | **164.286948** | **154.323512** | **16.959040** | **141.210272** | **13.113240** |

外部DQ总量5216→4520KiB；输入X减少256KiB，V GPU导出及cache读写减少320KiB，P输入减少248KiB，PV部分结果导出增加128KiB，净减少696KiB。DQ与总CA同时空闲：155.684550 → 145.953738µs/channel。这些是多个窗口累计，未注入并发Thor请求，不等于已测得的GPU可用带宽。

单层连续控制的DQ空闲：165.239018 → 155.001686µs/channel；DQ与总CA同时空闲：156.575650 → 146.599082µs/channel。

[全部channel对照](reference/pv-token/channel-dq-comparison.csv)、[phase类别](reference/pv-token/phase-kind-dq-comparison.csv)、[原生命令数](reference/pv-token/native-command-comparison.csv)、[完整比较](reference/pv-token/pv-token-comparison.json)。

## 证据与复现

原生job 1904762：121项Python测试、52项native guards通过；21案例全部16channel的ACT1/ACT2 owner/deadline/干扰检查通过。独立审计核对QKV权重与输入覆盖、SRF整批生命周期、output容量、V本地RMW的prefix/suffix/padding、P分片和PV部分结果映射。旧模式逐周期、trace、流量与native CSV精确复现归档1904020；[源码—结果绑定](reference/pv-token/source-result-binding.json)、[验证](reference/validation.json)、[旧结果](reference/previous-pv-token/summary.json)。

修改前完整备份位于本地`../backups/xinyao_1103_before_pv_token_810e01e/`，新raw结果位于`results/token-1904762/`（路径相对于本仓库）。全部文件绑定SHA256。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh
python verify_reference.py results/fresh
python experiments/pi05-unified-gemv/reproduce.py --legacy-pv-feature-sharding --output results/legacy
python compare_pv_token.py --before results/legacy --after results/fresh
```

这是时序、容量和数据映射模型，未执行完整BF16神经网络数值推理。GPU softmax、PV部分和求和、其它GPU算术及后台竞争未计时；prefix按所选布局预置，初始cache迁移不计入每step时间。当前结果不是完整AE端到端latency。历史模型见[MODEL.md](experiments/pi05-unified-gemv/MODEL.md)。

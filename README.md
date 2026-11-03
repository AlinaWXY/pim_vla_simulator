# xinyao_1103：两个 bank 交错写入

从已备份并上传的 `xinyao_1102`（`cbc1ee4`）派生。GPU 输入按 fragment 奇偶分配到 rank1 的 BG0/bank0 与 BG1/bank0，逐 token 交错发出普通 WR，以利用跨 bank group 的短列间隔。全部写入完成后，仍先填两个 SRF slot，再计算对应的两个 fragment。

总 DRAM 暂存容量仍为128KiB，分成两个64KiB区域；每 bank 的 SRF 仍为2×512B。MAC、输出、输入字节数及rank controller派发政策保持原样。详见[模型与地址映射](experiments/pi05-unified-gemv/MODEL.md)。

## 原生结果：每层、每denoising step，单位µs

H16、batch1、16个LPDDR5X-8533 channel，每channel的rank0预留给GPU、rank1包含16个PIM bank。DQ列为每channel时间；16个channel分别审计。

| 阶段 | 1102 单bank | 1103 双bank | DQ占用 | DQ空闲 | 完整无外部传输窗口 |
|---|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 58.708482 | 58.708482 | 3.301760 | 55.406722 | 47.611004 |
| V | 21.992348 | 20.592852 | 3.121664 | 17.471188 | 16.064188 |
| PV | 20.364918 | 18.994500 | 2.341248 | 16.653252 | 15.587684 |
| Gate/Up | 48.579958 | 47.127934 | 3.842048 | 43.285886 | 40.756100 |
| Down | 64.708868 | 58.691598 | 7.924224 | 50.767374 | 47.637268 |
| **合计** | **214.354574** | **204.115366** | **20.530944** | **183.584422** | **167.656244** |

总latency降低 **4.78%**，按18层×10步折算为 **36.740766ms**（1102为38.583823ms）。这仍是已建模阶段之和，不是完整AE端到端时间。

## 整批输入写入

| 输入阶段 | 单bank写入µs | 双bank写入µs | 缩短 | 写入阶段DQ利用率 |
|---|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 2.486667 | 1.246631 | 49.87% | 38.63% → 77.05% |
| V | 5.270651 | 2.503551 | 52.50% | 36.45% → 76.73% |
| PV | 5.150587 | 2.443519 | 52.56% | 36.13% → 76.16% |
| Gate/Up | 5.270651 | 2.503551 | 52.50% | 36.45% → 76.73% |
| Down | 21.332025 | 10.548777 | 50.55% | 36.02% → 72.84% |

原生日志确认跨BG写入达到2CK，同BG写入间隔仍至少4CK；没有放宽DRAM时序或增加DQ带宽。写入latency的改善不会全部变成总latency改善：读取需要准备两个源bank，且refresh相位也会变化。计算指令及非ACT/PRE/refresh的原生命令计数在全部16个channel均与1102一致。

[写入阶段数据](reference/input-write-comparison.json)、[逐channel命令对比](reference/staging-bank-comparison.json)、[原生审计](reference/down_gpu_input/native-audit.json)。

## DQ及命令总线分解

时间闭合：`204.115366 = 20.530944 + 167.656244 + 15.928178`，分别为DQ占用、完整无外部传输工作窗口、其它DQ空隙。DQ空闲比例为89.94%。外部流量仍为5472KiB/层/步（全部16channel），其中暂存输入3824KiB。

总CA占用范围18.706534µs；rank0 CA为0.047838µs；rank1 CA为34.907670µs。DQ空闲时，总CA忙9.517886–9.527266µs，总CA也空闲174.057156–174.066536µs。不同资源的占用不能相加到latency；未注入并发GPU请求，这些空闲窗口只是资源可用上界。

![单bank和双bank DQ对比](reference/channel-breakdown.png)

[逐channel时间分区](reference/channel-partitions.csv)、[阶段窗口](reference/channel-phase-windows.csv)、[CA分解](reference/channel-command-bus.csv)、[完整数据](reference/channel-analysis.json)。单层连续控制为204.338610µs，combined FFN控制为105.910518µs，不重复加入主结果。

## 保留的已知问题与RoPE旧指令

1102备注的“引入rank controller导致命令堆积”问题本次仍保留。当前单bank细粒度RoPE指令逐条经过总CA准入，单前端按channel分组发令，受阻时挡住其它channel，产生跨channel队头阻塞。双bank写入实验不修复这条控制路径，当前latency仍包含其额外代价。

旧版的RoPE是每个本地query依次执行 `COEF → LOAD Q(slot0) → EXEC Q → LOAD K(slot1) → EXEC K → STORE Q → STORE K`，每组遍历各bank；每条请求处理一个32B向量。旧版本地LOCALRD/LOCALWR不占共享CA，EXEC也可跨bank同CK发出；前端没有新增的逐请求1CK总CA门限。因此旧0.529032µs是更理想化的bank内执行模型。新版RoPE工作量不变，但新增仲裁与前端HOL叠加。详细旧指令、代码位置和日志证据见[RoPE控制路径说明](experiments/pi05-unified-gemv/ROPE-CONTROL.md)。

GPU softmax、GeLU/multiply、packing、O projection、norm/residual、dispatch与后台VLM竞争未计时。单层地址布局已核对；18层×10步只是耗时折算，未构建18层同时驻留的地址布局。

## 复现与验证

Linux、C++20、CMake≥3.18、Python3.11及开发头文件。依赖源码随附，绘图另需matplotlib。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh-h16 --staging-banks 2
python verify_reference.py results/fresh-h16
python compare_staging_banks.py results/fresh-h16
python experiments/pi05-unified-gemv/plot_comparison.py results/fresh-h16
python experiments/pi05-unified-gemv/reproduce.py --output results/single --staging-banks 1
python verify_single_bank.py results/single
python experiments/pi05-unified-gemv/reproduce.py --output results/direct --direct-input-baseline
python verify_baseline.py results/direct
```

原生job1902629通过 **65项Python测试、33项原生guards、7个双bank案例**。同一编译库另跑7个1102单bank及7个1101直接输入案例，周期、字节、trace、事件CSV哈希与逐channel时间窗口均与历史结果一致。实验/backend/vendor的502个文件与实际原生运行源码逐字节一致。reference保存紧凑证据；原始trace、timeline由复现脚本生成。

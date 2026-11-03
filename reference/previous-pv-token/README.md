# xinyao_1103：V projection 按 token 分 rank

五阶段总 latency **187.293274 → 183.857380 µs/层/step**。修改前后和新 backend 的旧布局控制均重新运行，共21案例；旧布局与归档23e6614逐周期、trace、native CSV和DQ分区完全一致。

## V 与 PV 的数据流

```text
GPU X → 奇偶token分别送每head的两个rank → 各算8token×256feature的V
    → 原有FP32输出回GPU → BF16转换、按feature重排 → 原有V cache写回
GPU P → 同head两rank各接收完整P → 与各自128feature的V cache相乘
    → 各自产生完整KV归约结果 → 按feature拼接，无跨rank partial-sum reduction
```

以一个 head 为例，A、B 是不同 channel 中的两个 PIM rank（均为各自 channel 的物理 rank1）：

| 步骤 | rank A | rank B |
|---|---|---|
| GPU 供应 X | 偶数 token，X[8,1024] | 奇数 token，X[8,1024] |
| V projection | X[8,1024] × Wv[1024,256] → V[8,256] | 同样计算另 8 个 token 的全部 256 个 feature |
| GPU 重排后 V cache | 全部 KV token 的 feature 集合 A，Vcache[992,128] | 全部 KV token 的 feature 集合 B，Vcache[992,128] |
| GPU 供应 P | 完整 P[16,992] | 相同的完整 P[16,992] |
| PV | P × Vcache → O[16,128] | P × Vcache → O[16,128] |

cache 的 feature 集合沿用旧交错布局：A 为 0–15、32–47、…、224–239，B 为 16–31、48–63、…、240–255，并非连续的前后 128 维。GPU 按原 feature 索引组合两个 PV 结果，得到 O[16,256]。992 个物理 KV 位置含 968 个已有 prefix、16 个新 token 和 8 个 padding；新 V 写入位置为 968–983。GPU 重排同时调整 rank、bank 和列地址，保留 prefix 并处理 padding。

X 和 P 都先经外部 DQ 整批交错写入各 rank 的 bank0、bank4，写完成后再由 rank controller 分片读取到 SRF、广播到本 rank 的全部 16 个 bank。V 每 rank 的 X 为 16 KiB，PV 每 rank 的 P 为 31 KiB；这些输入分别装载，V 计算时的 token 归属不会限制 PV 的 query 行数。

V每rank输入32→16KiB；Wv由每head两rank分片改为各自完整副本，总4→8MiB，新权重区为每bank[131072,163840)。V cache仍为[100352,116736)，PV trace完全不变。全部算子继续使用16个计算bank，SRF仍2×512B/bank；新V每slot只用8×16 BF16=256B，16个输出列全部消费后才释放。accumulator仍8×16×4=512B/bank。

V MACAB条数翻倍、每条服务16→8CK，scalar MAC总数保持33554432；standalone V的nCCDAB同步16→8，其余普通DRAM时序、CA约束及ACT owner/deadline检查保留。V外部输入少256KiB，FP32导出仍128KiB，cache写回仍走原prefix-tail合并与padding路径。GPU cast/pack/reorder算术尚未计时，本模型核验映射/时序和流量，不执行BF16神经网络数值运算。

## 各阶段 latency

单位µs/层/denoising step，H16、batch1。

| 阶段 | 修改前 | token分片后 | 变化 |
|---|---:|---:|---:|
| Q/K + RoPE + QKT | 41.886390 | 41.886390 | +0.000000 |
| V | 20.592852 | 17.156958 | -3.435894 |
| PV | 18.994500 | 18.994500 | +0.000000 |
| Gate/Up | 47.127934 | 47.127934 | +0.000000 |
| Down | 58.691598 | 58.691598 | +0.000000 |
| **五阶段合计** | **187.293274** | **183.857380** | **-3.435894** |

单层连续控制：187.887966 → 184.809450µs；该控制不重复计入五阶段总和。下游连续运行的phase变化可能受行状态与refresh相位影响，不能解释成PV布局改变。

| V 子阶段 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| v_gpu_input_stage | 2.502613 | 1.245693 | -1.256920 |
| v_local_input_compute | 16.065126 | 14.092512 | -1.972614 |
| v_gemv_release | 0.002814 | 0.002814 | +0.000000 |
| v_result_export | 0.983024 | 0.982086 | -0.000938 |
| v_prefix_tail_read | 0.505582 | 0.300160 | -0.205422 |
| v_suffix_write | 0.532784 | 0.532784 | +0.000000 |
| phase 外边界计时余量 | 0.000909 | 0.000909 | +0.000000 |

## DQ 空闲 breakdown

下表为channel0，全部16channel独立核验；完整无传输窗口已扣除边界数据尾部。

| 阶段 | 原DQ占用 | 新DQ占用 | 原DQ空闲 | 新DQ空闲 | 新完整无传输窗口 | 新其它空闲 |
|---|---:|---:|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 3.301760 | 3.301760 | 38.584630 | 38.584630 | 32.448234 | 6.136396 |
| V | 3.121664 | 2.161152 | 17.471188 | 14.995806 | 14.092512 | 0.903294 |
| PV | 2.341248 | 2.341248 | 16.653252 | 16.653252 | 15.588622 | 1.064630 |
| Gate/Up | 3.842048 | 3.842048 | 43.285886 | 43.285886 | 40.757038 | 2.528848 |
| Down | 7.924224 | 7.924224 | 50.767374 | 50.767374 | 47.638206 | 3.129168 |
| **合计** | **20.530944** | **19.570432** | **166.762330** | **164.286948** | **150.524612** | **13.762336** |

五阶段外部流量5472 → 5216KiB（16channel总和）。DQ与总CA同时空闲：158.591412 → 155.684550µs/channel。单层连续控制DQ空闲：167.357022 → 165.239018µs，DQ/总CA同时空闲：159.121382 → 156.575650µs。空闲是多个窗口累计，未注入并发Thor请求，不能等同已测得的可用带宽。

[逐phase对照](reference/v-token/phase-latency.csv)、[所有channel DQ/CA对照](reference/v-token/channel-dq-comparison.csv)、[phase类别](reference/v-token/phase-kind-dq-comparison.csv)、[完整比较及ACT审计](reference/v-token/v-token-comparison.json)。

## 验证、备份与复现

原生job 1904020通过112项Python测试、43项native guards。独立映射审计覆盖32768个V输出、实际RDRES/cache包和旧cache地址；serial还检查V→PV共享SRF的阶段切换。before、token、legacy各7案例的全部16channel测量日志均通过ACT1/ACT2 owner、deadline和干扰检查；这不等同所有DRAM物理规格验证。[验证记录](reference/validation.json)、[源码和原始结果绑定](reference/v-token/source-result-binding.json)、[修改前结果](reference/previous-v-token/summary.json)。

修改前Git bundle、source.zip和job1903036原始日志备份在本地`backups/xinyao_1103_before_v_token_23e6614/`，附SHA256；新原始结果保存为`results/token-1904020/`。

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce.py --output results/fresh
python verify_reference.py results/fresh
python experiments/pi05-unified-gemv/reproduce.py --legacy-v-feature-sharding --output results/legacy
python compare_v_token.py --before results/legacy --after results/fresh
```

GPU softmax、cast/packing/reorder、GeLU/multiply、O projection、norm/residual和背景VLM未计时，所报总和不是完整AE端到端latency。详细模型见[MODEL.md](experiments/pi05-unified-gemv/MODEL.md)。

# 只考虑 DQ 空闲的补充分析

沿用 native job 1971246，不重跑 GPU 或 PIM；仅将 VLM 可放置窗口的条件从“全部16 channel DQ + 共享CA空闲”改成“全部16 channel DQ空闲”。保持完整 kernel、原顺序、不可拆分。原主表的联合窗口结果保留。

| 配置 | 原最长联合窗口 / µs | 最长 DQ-only 可放置窗口 / µs | 可重叠 VLM / µs | 并行 VLM+AE / ms |
|---|---:|---:|---:|---:|
| feature + GPU RoPE / before | 0.504644 | 6.999356 | 1.0368 | 187.940550 |
| token + GPU RoPE / before | 0.494326 | 13.814864 | 1.0368 | 185.046536 |
| token + local RoPE / before | 0.494326 | 13.814864 | 1.0368 | 174.280447 |
| token + GPU RoPE / after | 3.885196 | 47.595967 | 1.0368 | 191.111248 |
| token + local RoPE / after | 3.883320 | 47.595967 | 1.0368 | 180.312573 |

a（纯GPU）和b（串行）不变。这里只放进第一个1.0368µs kernel，后续144.5696µs convolution仍长于最长窗口，不能跳过它继续放后续kernel。融合后的原始Down阶段最长DQ idle为47.634454µs；表中47.595967µs是按原AE合成规则剔除重复计时的共享tensor输入ST阶段后，可用于放置kernel的片段。

所有latency仍是已知项混合模型，不是实际GPU/PIM并发测量。DQ-only放宽了资源条件，不能保证GPU目标rank在整个窗口中都可访问，也没有额外估计GPU内存请求因refresh或CA仲裁而增加的延迟。

## Refresh：指令占用与内部执行时长

当前模型tCK=0.938ns，实际运行使用REFab。原生日志的`scope=global`事件显示，一条REFab占用共享CA **1 CK = 0.938ns**。例如channel0的`[4164,4165)`向rank0发refresh，`[4238,4239)`向rank1发refresh；两条分别占1CK。

核对了`token_local_after/down_gpu_input`全部16 channel，480条全局REFab事件的`ready_at-clock`均为1CK。原始日志位于服务器 `results/evaluation-1971246/token_local_after/down_gpu_input/simulation/rank_ca_events.csv.ch0` 至 `ch15`。

| 参数 | CK | 时间 | 含义 |
|---|---:|---:|---|
| 一条 REFab 的 CA 占用 | 1 | 0.938ns | 命令发送占用 |
| nRFC | 224 | 210.112ns | 被刷新rank的refresh执行/时序约束 |
| nREFI | 4164 | 3.905832µs | nominal refresh间隔 |
| nRFCpb | 128 | 120.064ns | per-bank refresh参数；本次日志未使用REFpb |

`nRFC`不是CA持续占用224CK。先前约3.88µs的联合窗口来自“任何CA脉冲都切断连续窗口”的严格筛选，不表示GPU kernel必须每3.88µs停止或重启。进一步的联合执行模型应允许GPU计算继续、将refresh/CA争用作用于实际内存请求，而不是把1CK的CA脉冲当作kernel不能跨越的边界。

来源：[native时序参数](../native-provenance/token_local_after/down_gpu_input/result.json)、[CA预约实现](../../../experiments/pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl/rank_staging_extension.inc)。复现：

```bash
python experiments/pi05-unified-gemv/analyze_dq_only.py --reference reference/evaluation-1108
```

[完整精度比较](comparison.csv)、[分阶段原始DQ窗口](per-case-raw-windows.csv)、[数据来源及口径](summary.json)。

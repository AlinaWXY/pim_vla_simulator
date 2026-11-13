# MAC 2× 逐请求 CA 冲突延发回放

使用 step 9 / layer 17，重复 180 次。两个已采样切片中，该切片的 e−g 绝对 E2E 时差最大。关闭 CA 时为 7.990581 ms，开启 CA 并允许补发时为 8.036519 ms。step 0 / layer 0 的对应时差为 7.659334 ms 和 7.701896 ms。

| Case | DQ-only E2E（ms） | CA 延发 E2E（ms） | E2E 差值（µs） | CA 延发时 PIM 期间的 VLM 请求 |
|---|---:|---:|---:|---:|
| b | 180.250566 | 180.250566 | 0.000 | 0 |
| c | 145.038693 | 144.944832 | −93.861 | 173,005,944 |
| d | 141.414056 | 141.478249 | +64.193 | 175,594,595 |
| e | 133.536861 | 133.591890 | +55.030 | 171,243,932 |
| f | 134.733394 | 134.740919 | +7.525 | 247,619,916 |
| g | 125.546280 | 125.555372 | +9.092 | 266,857,991 |

CA 开关共用同一份 DQ 输入、整 kernel GPU 交接和 f/g 共享 I/O 计账。回放使用 CA trace 中各通道已占用的时隙，按 RD 提前 23 CK、WR 提前 12 CK 查找列命令的发令周期。发生冲突时，该请求推迟到下一个合法周期。后续请求保留原定节奏，同时受前一笔 burst 的实际结束时刻约束。VLM 的 ACT/PRE 和 bank 时序处于这项对照的抽象范围之外。

c 的负差值来自重新安排后的 VLM kernel 完成时刻和 AE GPU 交接边界。两种回放的 AE 单跑时长、VLM 单跑时长及总请求数一致。

## 复现

从仓库根目录执行。需要支持 C++20 的编译器和 Python 3.10 或更高版本。

```sh
mkdir -p build results/mc-dq-replay/traces
tar -xzf reference/mc-dq-replay/traces-s9-l17.tar.gz -C results/mc-dq-replay/traces
c++ -std=c++20 -O3 tools/dq_request_replay.cpp -o build/dq_request_replay
python3 tools/run_dq_request_replay.py \
  --engine build/dq_request_replay \
  --traces results/mc-dq-replay/traces \
  --prior-summary reference/mc-dq-replay/prior-summary.csv \
  --slice 9 17 --compare-ca --preserve-pace \
  --output results/mc-dq-replay/ca-catch-up
cmp reference/mc-dq-replay/ca-catch-up/ca-comparison.csv results/mc-dq-replay/ca-catch-up/ca-comparison.csv
cmp reference/mc-dq-replay/ca-catch-up/stage-ca-comparison.csv results/mc-dq-replay/ca-catch-up/stage-ca-comparison.csv
```

结果 JSON 记录整数 CK、请求数、CA 推迟计数及输入哈希。两种回放共用同一个原始 trace 压缩包。

[E2E 与请求数对照](ca-comparison.csv) · [逐阶段对照](stage-ca-comparison.csv) · [DQ-only 结果](../dq-only/README.md)

# MAC 2× 逐请求 DQ-only 回放

发布切片为 step 9 / layer 17。按同一切片内 e−g 的绝对 E2E 时差比较，step 0 / layer 0 为 7.659334 ms，step 9 / layer 17 为 7.990581 ms。每个切片重复 180 次。

| Case | E2E（ms） | PIM 期间插入的 VLM 请求 |
|---|---:|---:|
| b | 180.250566 | 0 |
| c | 145.038693 | 172,360,190 |
| d | 141.414056 | 175,927,352 |
| e | 133.536861 | 171,647,110 |
| f | 134.733394 | 247,673,515 |
| g | 125.546280 | 266,929,527 |

每个 case 都回放同一份 VLM profile 的 619,790,815 笔 32 B 请求。PIM 期间按通道放入 DQ 空隙，考虑跨 Rank DQ 切换。AE 需要 GPU 时，当前 VLM kernel 完成后再交接。b 先执行 AE 再执行 VLM。f/g 的共享 I/O 归入现有 AE GPU profile 一次，Q 交换保留原来的 DQ 独占窗口。

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
  --slice 9 17 --output results/mc-dq-replay/dq-only
cmp reference/mc-dq-replay/dq-only/summary.csv results/mc-dq-replay/dq-only/summary.csv
cmp reference/mc-dq-replay/dq-only/stage-summary.csv results/mc-dq-replay/dq-only/stage-summary.csv
```

压缩包包含已捕获的原始 CA/DQ trace、MAC 2× spec、AE 切片和捕获校验记录。逐请求回放直接读取这些输入。每个 case 的 JSON 记录整数 CK、请求计数及回放输入 SHA-256。

[E2E 汇总](summary.csv) · [逐阶段统计](stage-summary.csv) · [原始 trace](../traces-s9-l17.tar.gz)

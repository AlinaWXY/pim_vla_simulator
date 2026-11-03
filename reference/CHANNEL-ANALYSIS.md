# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 58.708482 | 3.301760–3.301760 | 55.406722–55.406722 | 47.611004 | 47.611004 |
| v | 20.592852 | 3.121664–3.121664 | 17.471188–17.471188 | 16.064188 | 16.064188 |
| pv | 18.994500 | 2.341248–2.341248 | 16.653252–16.653252 | 15.587684 | 15.587684 |
| gate_up_gpu | 47.127934 | 3.842048–3.842048 | 43.285886–43.285886 | 40.756100 | 40.756100 |
| down_gpu_input | 58.691598 | 7.924224–7.924224 | 50.767374–50.767374 | 47.637268 | 47.637268 |

Five-stage sum: 204.115366 us; x18x10: 36.740766 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 18.706534 | 0.047838 | 34.907670 | 9.527266 | 174.057156 |
| 1 | 18.706534 | 0.047838 | 34.907670 | 9.517886 | 174.066536 |
| 2 | 18.706534 | 0.047838 | 34.907670 | 9.517886 | 174.066536 |
| 3 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 4 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 5 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 6 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 7 | 18.706534 | 0.047838 | 34.907670 | 9.518824 | 174.065598 |
| 8 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 9 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 10 | 18.706534 | 0.047838 | 34.907670 | 9.518824 | 174.065598 |
| 11 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 12 | 18.706534 | 0.047838 | 34.907670 | 9.517886 | 174.066536 |
| 13 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 14 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |
| 15 | 18.706534 | 0.047838 | 34.907670 | 9.519762 | 174.064660 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

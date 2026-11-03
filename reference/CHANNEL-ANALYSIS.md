# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 41.886390 | 3.301760–3.301760 | 38.584630–38.584630 | 32.448234 | 32.448234 |
| v | 17.156958 | 2.161152–2.161152 | 14.995806–14.995806 | 14.092512 | 14.092512 |
| pv | 18.994500 | 2.341248–2.341248 | 16.653252–16.653252 | 15.588622 | 15.588622 |
| gate_up_gpu | 47.127934 | 3.842048–3.842048 | 43.285886–43.285886 | 40.757038 | 40.757038 |
| down_gpu_input | 58.691598 | 7.924224–7.924224 | 50.767374–50.767374 | 47.638206 | 47.638206 |

Five-stage sum: 183.857380 us; x18x10: 33.094328 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 1 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 2 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 3 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 4 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 5 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 6 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 7 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 8 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 9 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 10 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 11 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 12 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 13 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 14 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |
| 15 | 17.277960 | 0.042210 | 33.787698 | 8.602398 | 155.684550 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

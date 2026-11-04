# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| gate_up_gpu | 58.025618 | 3.842048–3.842048 | 54.183570–54.183570 | 48.882932 | 48.882932 |
| down_gpu_input | 64.791412 | 7.924224–7.924224 | 56.867188–56.867188 | 42.975408 | 42.975408 |

Two-stage FFN sum: 122.817030 us; x18x10: 22.107065 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 1 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 2 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 3 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 4 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 5 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 6 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 7 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 8 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 9 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 10 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 11 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 12 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 13 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 14 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |
| 15 | 9.933420 | 0.028140 | 21.578690 | 4.345754 | 106.705004 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

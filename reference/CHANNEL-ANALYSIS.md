# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 58.708482 | 3.301760–3.301760 | 55.406722–55.406722 | 46.370968 | 46.370968 |
| v | 21.992348 | 3.121664–3.121664 | 18.870684–18.870684 | 14.905758 | 14.905758 |
| pv | 20.364918 | 2.341248–2.341248 | 18.023670–18.023670 | 14.228522 | 14.228522 |
| gate_up_gpu | 48.579958 | 3.842048–3.842048 | 44.737910–44.737910 | 39.443838 | 39.443838 |
| down_gpu_input | 64.708868 | 7.924224–7.924224 | 56.784644–56.784644 | 42.893802 | 42.893802 |

Five-stage sum: 214.354574 us; x18x10: 38.583823 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 18.744054 | 0.049714 | 34.230434 | 9.577918 | 184.245712 |
| 1 | 18.744054 | 0.049714 | 34.230434 | 9.568538 | 184.255092 |
| 2 | 18.744054 | 0.049714 | 34.230434 | 9.568538 | 184.255092 |
| 3 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 4 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 5 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 6 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 7 | 18.744054 | 0.049714 | 34.230434 | 9.569476 | 184.254154 |
| 8 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 9 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 10 | 18.744054 | 0.049714 | 34.230434 | 9.569476 | 184.254154 |
| 11 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 12 | 18.744054 | 0.049714 | 34.230434 | 9.568538 | 184.255092 |
| 13 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 14 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |
| 15 | 18.744054 | 0.049714 | 34.230434 | 9.570414 | 184.253216 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

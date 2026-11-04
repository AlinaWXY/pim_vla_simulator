# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 50.511300 | 3.301760–3.301760 | 47.209540–47.209540 | 41.030934 | 41.030934 |
| pv | 14.951720 | 1.891008–1.891008 | 13.060712–13.060712 | 11.784094 | 11.784094 |
| gate_up_gpu | 47.127934 | 3.842048–3.842048 | 43.285886–43.285886 | 40.757038 | 40.757038 |
| down_gpu_input | 58.691598 | 7.924224–7.924224 | 50.767374–50.767374 | 47.638206 | 47.638206 |

Four-stage sum (fused QKV): 171.282552 us; x18x10: 30.830859 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 1 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 2 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 3 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 4 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 5 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 6 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 7 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 8 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 9 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 10 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 11 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 12 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 13 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 14 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |
| 15 | 15.816556 | 0.039396 | 31.103142 | 8.369774 | 145.953738 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

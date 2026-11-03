# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 61.460574 | 3.301760–3.301760 | 58.158814–58.158814 | 48.880118 | 48.880118 |
| v | 24.498684 | 3.121664–3.121664 | 21.377020–21.377020 | 17.262014 | 17.262014 |
| pv | 22.823416 | 2.341248–2.341248 | 20.482168–20.482168 | 16.709532 | 16.709532 |
| gate_up_gpu | 50.736420 | 3.842048–3.842048 | 46.894372–46.894372 | 41.600300 | 41.600300 |
| down_gpu_input | 75.231352 | 7.924224–7.924224 | 67.307128–67.307128 | 53.416286 | 53.416286 |

Five-stage sum: 234.750446 us; x18x10: 42.255080 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 18.742178 | 0.053466 | 35.651504 | 9.560096 | 204.659406 |
| 1 | 18.742178 | 0.053466 | 35.651504 | 9.567600 | 204.651902 |
| 2 | 18.742178 | 0.053466 | 35.651504 | 9.568538 | 204.650964 |
| 3 | 18.742178 | 0.053466 | 35.651504 | 9.559158 | 204.660344 |
| 4 | 18.742178 | 0.053466 | 35.651504 | 9.567600 | 204.651902 |
| 5 | 18.742178 | 0.053466 | 35.651504 | 9.559158 | 204.660344 |
| 6 | 18.742178 | 0.053466 | 35.651504 | 9.566662 | 204.652840 |
| 7 | 18.742178 | 0.053466 | 35.651504 | 9.559158 | 204.660344 |
| 8 | 18.744054 | 0.053466 | 35.654318 | 9.570414 | 204.649088 |
| 9 | 18.744054 | 0.053466 | 35.654318 | 9.570414 | 204.649088 |
| 10 | 18.745930 | 0.053466 | 35.656194 | 9.562910 | 204.656592 |
| 11 | 18.745930 | 0.053466 | 35.656194 | 9.572290 | 204.647212 |
| 12 | 18.747806 | 0.053466 | 35.658070 | 9.563848 | 204.655654 |
| 13 | 18.747806 | 0.053466 | 35.658070 | 9.574166 | 204.645336 |
| 14 | 18.749682 | 0.053466 | 35.659946 | 9.566662 | 204.652840 |
| 15 | 18.749682 | 0.053466 | 35.659946 | 9.576042 | 204.643460 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

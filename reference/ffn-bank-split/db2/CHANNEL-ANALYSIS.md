# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| gate_up_gpu | 46.856852 | 3.842048–3.842048 | 43.014804–43.014804 | 40.482204 | 40.482204 |
| down_gpu_input | 34.191976 | 7.924224–7.924224 | 26.267752–26.267752 | 23.159220 | 23.159220 |

Two-stage FFN sum: 81.048828 us; x18x10: 14.588789 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 1 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 2 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 3 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 4 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 5 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 6 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 7 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 8 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 9 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 10 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 11 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 12 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 13 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 14 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |
| 15 | 10.014088 | 0.017822 | 21.637784 | 4.419856 | 64.862700 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

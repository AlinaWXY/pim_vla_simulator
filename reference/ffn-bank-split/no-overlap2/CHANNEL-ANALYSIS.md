# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| gate_up_gpu | 57.780800 | 3.842048–3.842048 | 53.938752–53.938752 | 51.427726 | 51.427726 |
| down_gpu_input | 53.999722 | 7.924224–7.924224 | 46.075498–46.075498 | 42.966966 | 42.966966 |

Two-stage FFN sum: 111.780522 us; x18x10: 20.120494 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 1 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 2 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 3 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 4 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 5 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 6 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 7 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 8 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 9 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 10 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 11 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 12 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 13 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 14 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |
| 15 | 10.036600 | 0.025326 | 21.686560 | 4.442368 | 95.571882 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

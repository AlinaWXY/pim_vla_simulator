# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| gate_up_gpu | 47.645710 | 3.842048–3.842048 | 43.803662–43.803662 | 38.503962 | 38.503962 |
| down_gpu_input | 45.266942 | 7.924224–7.924224 | 37.342718–37.342718 | 23.450938 | 23.450938 |

Two-stage FFN sum: 92.912652 us; x18x10: 16.724277 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. Actual concurrent GPU request admission and service are not simulated.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

| Channel | Global CA busy | Rank0 CA busy | Rank1 CA busy | DQ idle, global CA busy | DQ and global CA both idle |
|---|---:|---:|---:|---:|---:|
| 0 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 1 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 2 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 3 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 4 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 5 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 6 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 7 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 8 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 9 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 10 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 11 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 12 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 13 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 14 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |
| 15 | 9.913722 | 0.021574 | 21.547736 | 4.326056 | 76.820324 |

Global CA and each rank CA are separate resources. Their durations must not be added to DQ time. The last two columns partition DQ idle exactly; simultaneous DQ/global-CA idle is a resource upper bound, not measured GPU usable time.

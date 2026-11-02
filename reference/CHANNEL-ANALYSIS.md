# Native H16 channel accounting

All times below are microseconds per layer and denoising step. DQ intervals describe the AttAcc movement model, not a validated electrical waveform. GPU usable time is unmeasured.

| Stage | Wall | Model DQ occupied/channel (min–max) | Model DQ idle/channel (min–max) | Whole zero-external phases, all channels | After incoming data tails |
|---|---:|---:|---:|---:|---:|
| chain | 35.273490 | 3.301760–3.301760 | 31.971730–31.971730 | 9.368744 | 9.368744 |
| v | 10.283294 | 3.121664–3.121664 | 7.161630–7.161630 | 0.000000 | 0.000000 |
| pv | 9.190524 | 2.341248–2.341248 | 6.849276–6.849276 | 0.000000 | 0.000000 |
| gate_up_z_store | 36.198358 | 1.921024–1.921024 | 34.277334–34.277334 | 2.440676 | 2.440676 |
| down_tiled_dram | 33.931212 | 3.842048–3.842048 | 30.089164–30.089164 | 26.172105 | 26.172105 |

Five-stage sum: 124.876878 us; x18x10: 22.477838 ms. This is a sum of measured stages, not complete AE end-to-end latency.

Each channel closes exactly as wall = model DQ occupied + model DQ idle. Whole-phase candidate windows exclude transfer-phase fragments and release-only control phases. Incoming delayed data is removed before counting each remaining suffix. These windows still require GPU command/address, rank-turnaround and row-admission modelling before they can be called usable GPU time.

The JSON and CSV files retain every channel and semantic phase, including boundary overhead and any data tail beyond stage completion.

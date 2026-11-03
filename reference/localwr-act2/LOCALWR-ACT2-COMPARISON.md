# LOCALWR ACT1/ACT2 correction (baseline a987650)

Times are microseconds per layer/denoising step.

| Case | Scope | Before | After | After minus before |
|---|---|---:|---:|---:|
| chain | primary_stage | 41.969872 | 41.886390 | -0.083482 |
| v | primary_stage | 20.592852 | 20.592852 | +0.000000 |
| pv | primary_stage | 18.994500 | 18.994500 | +0.000000 |
| gate_up_gpu | primary_stage | 47.127934 | 47.127934 | +0.000000 |
| down_gpu_input | primary_stage | 58.691598 | 58.691598 | +0.000000 |
| combined_timing_chain | control_only | 105.910518 | 105.910518 | +0.000000 |
| serial_layer_control | control_only | 188.328826 | 187.887966 | -0.440860 |
| primary_five_stage_sum | primary_sum | 187.376756 | 187.293274 | -0.083482 |

Only the five primary stages are summed. The combined FFN and serial-layer cases are separate controls.

All seven corrected cases pass the complete ACT protocol audit across all 16 channels. The historical a987650 spill failures are retained in the before audit and are not treated as passing.

[Full ACT protocol audit](localwr-act2-protocol-audit.json), [comparison JSON](localwr-act2-comparison.json), [case latency](case-latency.csv), [all semantic phases](phase-latency.csv), [all-channel DQ/CA](channel-dq-comparison.csv), [phase-kind DQ](phase-kind-dq-comparison.csv), [native commands](native-command-comparison.csv).

DQ/CA idle is modeled resource availability, not measured concurrent GPU service. Phase differences include row-state and refresh effects; they are not pure ACT-stall measurements.

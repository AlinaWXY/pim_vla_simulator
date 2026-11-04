# QKT DRAM spill / PIM O projection: 2×2 native experiment

All values are per layer and denoising step. PV and O operands/weights/results are BF16; accumulators remain FP32. Scores remain FP32.

The latency column sums modeled memory/PIM stages. GPU O projection is unmeasured in non-O variants, so these differences are not end-to-end speedups.

| Variant | Modeled latency (µs) | Timed O subflow (µs) | Longest complete QKT DQ-free work window (µs/ch0) | QKT work-window count | DQ utilization/ch0 |
|---|---:|---:|---:|---:|---:|
| neither | 171.284428 | GPU O unmeasured | 0.559048 | 31 | 9.621% |
| score_only | 180.834206 | GPU O unmeasured | 19.383770 | 1 | 9.113% |
| oproj_only | 200.680410 | 27.997424 (included) | 0.559048 | 31 | 8.929% |
| both | 210.230188 | 27.997424 (included) | 19.383770 | 1 | 8.524% |

Timed O includes its input read/broadcast/compute, release/rewind and BF16 export. It is not the net latency increase, because the baseline PV export is replaced by local PV spill.

| Variant | Longest physical QKT DQ-idle window (µs/ch0) | Longest complete PV/PV+O DQ-free work window (µs/ch0) | Serial control (µs) |
|---|---:|---:|---:|
| neither | 0.559986 | 11.784094 | 171.960726 |
| score_only | 19.443773 | 11.784094 | 181.253492 |
| oproj_only | 0.559986 | 25.124330 | 201.204752 |
| both | 19.443773 | 25.124330 | 210.646660 |

Continuous work windows stop at a score/result transfer phase. DQ-free control bridges between work phases are allowed; idle fragments inside transfer phases are excluded. Physical DQ-idle windows are reported separately and can include such gaps. Neither metric is MAC-active time or measured GPU-usable time.

Historical cumulative diagnostic only: sum of separate complete semantic work phases after incoming burst tails. Superseded for uninterrupted-window comparison; never interpret this sum as one window.

Independent-stage interaction: +0.000000 µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: +0.149142 µs.

Sum of independently timed QKV/attention, PV or PV+O, Gate/Up and Down stages. GPU O projection is unmeasured in non-O variants; GPU reductions, softmax, nonlinearities and contention are unmeasured in all variants.

These are timing and data-mapping simulations, not numerical network execution or full end-to-end latency. BF16 partial rounding and the changed reduction order require separate numerical validation.

Detailed evidence: [continuous windows](continuous-dq/continuous-dq.json), [all complete windows](continuous-dq/continuous-dq-complete-work-windows.csv), [stages](stage-comparison.csv), [all channels](channel-comparison.csv), [all phases](phase-comparison.csv), [source provenance](provenance.json).

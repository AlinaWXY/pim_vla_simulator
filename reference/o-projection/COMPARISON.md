# QKT DRAM spill / PIM O projection: 2×2 native experiment

All values are per layer and denoising step. PV and O operands/weights/results are BF16; accumulators remain FP32. Scores remain FP32.

The latency column sums modeled memory/PIM stages. GPU O projection is unmeasured in non-O variants, so these differences are not end-to-end speedups.

| Variant | Modeled latency (µs) | Δ (µs) | DQ busy/ch0 (µs) | DQ utilization/ch0 | All-channel external DQ (KiB) |
|---|---:|---:|---:|---:|---:|
| neither | 171.284428 | +0.000000 | 16.478784 | 9.621% | 4392 |
| score_only | 180.834206 | +9.549778 | 16.478784 | 9.113% | 4392 |
| oproj_only | 200.680410 | +29.395982 | 17.919552 | 8.929% | 4776 |
| both | 210.230188 | +38.945760 | 17.919552 | 8.524% | 4776 |

| Variant | Whole work without DQ (µs/ch0) | Other DQ idle (µs/ch0) | DQ and global CA idle (µs/ch0) | Serial control (µs) |
|---|---:|---:|---:|---:|
| neither | 141.210272 | 13.595372 | 146.429304 | 171.960726 |
| score_only | 151.993520 | 12.361902 | 156.272676 | 181.253492 |
| oproj_only | 164.788778 | 17.972080 | 173.098520 | 201.204752 |
| both | 175.572026 | 16.738610 | 182.941892 | 210.646660 |

The DQ breakdown closes as latency = DQ busy + whole work without DQ + other DQ idle. A lower utilization can result from longer computation or local-memory work; it does not by itself imply less data transfer or greater performance.

Sum of complete semantic work phases with no external DQ commands after removing incoming burst tails. Includes local-memory waits and refresh; it is neither MAC active time nor one contiguous GPU-available window.

Independent-stage interaction: +0.000000 µs (additive by construction when the switches affect separate stages). Continuous serial-control interaction: +0.149142 µs.

Sum of independently timed QKV/attention, PV or PV+O, Gate/Up and Down stages. GPU O projection is unmeasured in non-O variants; GPU reductions, softmax, nonlinearities and contention are unmeasured in all variants.

These are timing and data-mapping simulations, not numerical network execution or full end-to-end latency. BF16 partial rounding and the changed reduction order require separate numerical validation.

Detailed evidence: [stages](stage-comparison.csv), [all channels](channel-comparison.csv), [all phases](phase-comparison.csv), [source provenance](provenance.json).

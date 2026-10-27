# Whole-horizon AE sweep: H=16,20,30,40

All entries are fresh native Ramulator runs with the same compiled backend. H is action-token count; the number of attention heads remains8. Token buffers and accumulators expand with H, and all H tokens execute as one batch.

H16 reproduces the frozen xinyao_1027_h16 baseline exactly. Timings preserve the earlier sum of separately measured stages; nonlinear/GPU arithmetic and VLM concurrency remain unmeasured.

| H | Padded KV | Stage sum/us | x18x10/ms | Zero-external-command phases/us | Fraction | External DQ/KiB |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 992 | 124.474476 | 22.405406 | 37.981525 | 30.51% | 3872 |
| 20 | 992 | 154.011158 | 27.722008 | 47.224577 | 30.66% | 4792 |
| 30 | 1024 | 223.439104 | 40.219039 | 64.421869 | 28.83% | 7280 |
| 40 | 1024 | 295.939000 | 53.269020 | 85.366471 | 28.85% | 9600 |

## Timing by stage

| H | Q/K chain | V | PV | Gate/Up + Z write | Down |
|---:|---:|---:|---:|---:|---:|
| 16 | 34.886096 | 10.282356 | 9.189586 | 36.195544 | 33.920894 |
| 20 | 42.556122 | 12.683636 | 11.267256 | 45.179708 | 42.324436 |
| 30 | 64.378692 | 18.505802 | 17.188850 | 66.739638 | 56.626122 |
| 40 | 85.006250 | 24.649702 | 23.010078 | 88.422446 | 74.850524 |

All stage times are microseconds per layer and denoising step.

## Complete phases without external transfer commands

| H | Q/K local spill, RoPE and QKT compute | Z local store | Down local input/compute | Total/us |
|---:|---:|---:|---:|---:|
| 16 | 9.378124 | 2.440676 | 26.162725 | 37.981525 |
| 20 | 11.151882 | 3.401188 | 32.671507 | 47.224577 |
| 30 | 17.448676 | 4.841956 | 42.131237 | 64.421869 |
| 40 | 23.376836 | 6.442184 | 55.547451 | 85.366471 |

The phase segmentation is unchanged. All short and long gaps inside input/output phases are excluded. This is a candidate-window statistic before data tails, read/write turnaround, rank-switch, CA and GPU row constraints; it is not measured usable GPU bandwidth. No rank0 background workload is supplied, and the AttAcc PIM transfer model retains its prior limitations.

## Layout and buffer changes

| H | Q buffer/B per bank | V/FFN input double buffer/B per rank | GU accumulator/B per bank | Down accumulator/B per bank | Work rows/bank | Z DRAM/B per bank |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 512 | 1024 | 2048 | 1024 | 1 | 512 |
| 20 | 640 | 1280 | 2560 | 1280 | 1 | 1024 |
| 30 | 960 | 1920 | 3840 | 1920 | 1 | 1024 |
| 40 | 1280 | 2560 | 5120 | 2560 | 2 | 1536 |

- Weight ownership and copies do not change: Q/K retain two token-owner copies per head; V/FFN weights remain single-copy and feature-sharded.
- Q/K production assigns H/2 tokens to each peer rank, keeps RoPE pairs in their original banks, and exchanges only the missing half of Q. Each enlarged bank query buffer holds all H queries, so QKT still scans each local K column once.
- The work area stores H/2 coefficient, H/2 Q, H/2 suffix-K columns, followed by K padding. H40 needs a second reserved row only for the final four padding-K columns; coefficient/Q/K RoPE data still fit in the first row. V regions shift accordingly; FFN weight bases stay fixed.
- Z uses [feature_slot16][ceil(H/16) token blocks][token_lane16] BF16. The last column has padded lanes; padded lanes are discarded when filling the H-token Down buffer. It remains within the existing2KiB DRAM reservation per bank.
- The32B bank staging and streamed128B+32B nonlinear staging remain bounded; no full-Z rank SRAM is introduced. Nonlinear arithmetic is still excluded.
- Projection/QK MAC service scales to H/2 and H cycles respectively; V/PV/FFN MAC service is H cycles. RoPE pair II/latency, local-port throughput, bank topology, frequency and refresh policy stay fixed.

## Reproduction and evidence

Each H directory contains workload traces, per-channel native timelines, completion events, independent layout/dataflow audits, source provenance and the channel-window analysis. Combined FFN cases check write-before-read dependencies and are not added to the five-stage sum. See README.md for build/run commands.

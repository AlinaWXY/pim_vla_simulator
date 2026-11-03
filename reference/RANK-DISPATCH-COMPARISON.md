# Independent channel dispatch and rank descriptors

Fresh native runs; latency in µs per layer/step. Baseline: xinyao_1103 commit 786842a.

| Stage | Before | Independent dispatch only | Both fixes | Saved |
|---|---:|---:|---:|---:|
| Q/K + RoPE + QKT | 58.708482 | 41.969872 | 41.969872 | 16.738610 |
| V | 20.592852 | 20.592852 | 20.592852 | 0.000000 |
| PV | 18.994500 | 18.994500 | 18.994500 | 0.000000 |
| Gate/Up | 47.127934 | 47.127934 | 47.127934 | 0.000000 |
| Down | 58.691598 | 58.691598 | 58.691598 | 0.000000 |
| Total | 204.115366 | 187.376756 | 187.376756 | 16.738610 |

DQ table shows channel 0; the CSV audits all 16 channels. Full no-transfer windows exclude boundary data tails; other idle includes gaps within transfer phases.

| Mode | DQ busy | DQ idle | Full no-transfer windows | Other DQ idle |
|---|---:|---:|---:|---:|
| before | 20.530944 | 183.584422 | 167.656244 | 15.928178 |
| independent_only | 20.530944 | 166.845812 | 152.563824 | 14.281988 |
| after | 20.530944 | 166.845812 | 152.563824 | 14.281988 |

Per-phase latency: [CSV](rank-dispatch-phase-comparison.csv). Per-channel DQ and CA comparison: [CSV](rank-dispatch-channel-comparison.csv).

Retained constraints: finite 32-entry per-channel ingress FIFO, one dispatch opportunity per channel/CK, actual external/rank CA arbitration, DRAM timing, SRF ownership, peer-export completion, and global barriers.

These are modeled memory/PIM stages, not complete AE end-to-end latency. DQ availability is an upper bound; no competing GPU requests are injected.

# H=16 bank-local RoPE and token-sharded PIM ranks

This experiment implements the agreed Q/K projection → local RoPE → GPU-boundary
Q exchange → rank-local QKT dataflow. `reproduce_ae.py` also runs every previously
modeled AE stage: V projection/cache update, PV, FFN Gate/Up, and tiled Down.
`reproduce_ae_dram.py` replaces FFN full-batch Z SRAM with bank-resident DRAM,
while retaining the SRAM version as a regression control. All PIM work uses
physical rank1, leaving rank0 reserved for GPU traffic. It is
separate from the published H16 baseline. The backend extension is opt-in;
old traces keep their original mode.

## Organization and layout

- 16 LPDDR5X-8533 channels, two physical ranks/channel. Rank 0 is reserved for
  GPU/Thor traffic; rank 1 executes PIM. No background VLM trace is assumed.
- Eight MHA heads, head dimension 256, projection input dimension 1024, 16 new
  tokens. Each head owns channels `2*h` and `2*h+1`; token parity chooses the
  producing PIM rank, so each rank produces eight new tokens.
- Each PIM rank has 16 banks. Bank `b` owns eight split-half RoPE pairs
  `(8*b+u, 8*b+u+128)` for `u=0..7`. Both ranks of a head store the same WQ/WK
  columns. Each column's entire input reduction stays within its owning bank.
- Each rank holds 484 prefix K tokens, eight suffix tokens, and four padding
  tokens. Its 496 K slots cover all 256 features across its 16 banks.
- WQ/WK layout is `[input_fragment64][Q/K2][pair8][component2][input_lane16]`,
  BF16. Capacity is 64 KiB/bank and 16 MiB/layer across the system.
- The per-bank FP32 projection accumulator holds `[Q/K2][local_token8][pair8]
  [component2]`, totaling 1024 B. Projection finishes before its BF16 local spill.

A bank has a 2 KiB row and a 32 B column transaction. Its reusable working row is:

| Columns | Contents | Bytes |
|---|---|---:|
| 0–7 | BF16 cos/sin for eight local positions | 256 |
| 8–15 | Eight locally produced Q slices | 256 |
| 16–23 | Final suffix K slices | 256 |
| 24–27 | Four zero-padding K slices | 128 |
| 28–63 | Reserved | 1152 |

Prefix K is a separate address range. Address holes are not extra attention
tokens. Remote Q is delivered directly to bank SRAM, not stored in the work row.

## RoPE and QKT

The chosen RoPE interface follows the official PyTorch BF16 path: coefficients
are generated in FP32 and stored as BF16; Q/K operands and outputs are BF16.
Ramulator models timing, dependencies and movement, not tensor numerical values
or bitwise equivalence. Rounding rules need a separate numerical implementation.

Each token's coefficient column is reused for its Q and K columns. Two 32 B
input buffers allow K prefetch while Q is being rotated. A separate 32 B coefficient
buffer and a bounded two-column (64 B) output FIFO give 160 B total per bank.
All columns are in the same working row;
native column timing, ACT/PRE, refresh and local write recovery remain active.

One RoPE command processes eight pairs. The unit's initiation interval defaults
to one CK/pair; pipeline latency defaults to four CK and is swept separately.
There are 128 pairs/bank/layer/step. This does not imply 128 CK end-to-end latency.
Projection conversion and rank reduction delays are explicit assumptions, not
hardware measurements. Scaling by 1/16 belongs on the score path, not in RoPE.

QKT preserves the existing H16 query batching: all 16 queries' current 16-feature
slice occupy 512 B/bank (1024 B with two buffer slots). Source banks retain their
local eight queries. The GPU-boundary transfer fills the other eight queries in
the corresponding bank of the peer channel, and has an export-before-delivery
dependency. One K column is reused for all 16 queries. Each rank scans its 496 K
slots once, reduces bank → BG → rank, then exports complete FP32 scores. The GPU
concatenates token partitions; there is no cross-rank partial-score sum.

## Steady-state external DQ budget

Per layer and denoising step, batch one:

| Operation | KiB |
|---|---:|
| X input, shared by Q/K | 256 |
| Q export | 64 |
| Remote Q buffer delivery | 64 |
| Complete FP32 score export | 496 |
| Total | 880 |

This assumes X is already available at the GPU boundary. A single DRAM source
read of X, reused across all heads, would add 32 KiB. Weight installation
(16 MiB/layer), current-window coefficient installation (64 KiB), prefix K
installation, and GPU staging spills are outside the steady-state budget.
Local DRAM reads/writes must not be counted as external DQ.

## Run

Use a Linux C++ build environment and the same Python interpreter for build and
execution. The vendored dependencies are used without fetching remote sources.

```bash
PYTHON=/path/to/venv/bin/python JOBS=4 bash build.sh
/path/to/venv/bin/python reproduce.py --output results/fresh-run \
  --rope-latency-ck 4 --sensitivity 1 8 --check-baseline
```

Each run writes fresh traces, native command timelines, local operation events,
phase timing, a provenance manifest, independent native audits and injected-fault
audit checks. `--check-baseline` reruns the original QKV/QK cases with the new
feature disabled and checks their published timings. Those timings are regression
checks, not directly comparable end-to-end speedup baselines.

The `reproduce.py` report covers only Q/K/RoPE/QKT. For the full set of previously
modeled AE stages, run:

```bash
/path/to/venv/bin/python reproduce_ae.py --output results/fresh-ae-run
```

The AE runner generates and simulates all five cases afresh, independently audits
the native commands, and reports their per-layer/per-step sum and its 18-layer,
10-step scaling. It preserves the previous separately timed stage methodology;
it does not claim continuous refresh state or overlap across stage boundaries.

## Remaining AE stages on one PIM rank/channel

V and PV retain feature partitioning. The 256 features of each head are divided
over two channels' 32 active banks, giving eight output features/bank. V weights
have one physical copy. V projection processes all16 query tokens and uses its
own X pass; that additional input traffic is charged. Its output still crosses
the original GPU BF16 conversion/packing boundary, including the prefix-tail
read-modify-write. PV scans all992 physical KV slots, with padding masked by the
unmeasured softmax boundary.

The following describes the retained SRAM baseline; the revised primary DRAM
storage variant is described below. FFN retains full4096 intermediate features
and single-copy weights. Each of the
16 PIM ranks owns256 intermediate features. Gate/Up now needs2KiB FP32 accumulators
per bank. The old rank-local gather assumption is retained at32B/2CK, therefore
2048CK/rank, and Z storage grows to8KiB/rank. Down still uses four256-output tiles
and1KiB accumulators/bank, but each tile scans16 input fragments. It exports16
rank partials (1MiB FP32) for the original GPU reduction boundary.

| Region | Per-bank byte range |
|---|---|
| Q/K weights, prefix K, RoPE working row | `[0,83968)` |
| V weights | `[83968,100352)` |
| V cache | `[100352,116736)` |
| Reserved gap | `[116736,262144)` |
| FFN Gate/Up weights | `[262144,327680)` |
| FFN Down weights | `[327680,360448)` |

| All modeled AE stages | External DQ / KiB per layer/step |
|---|---:|
| Q/K + RoPE + Q exchange + QKT | 880 |
| V projection + cache update | 832 |
| PV | 624 |
| FFN Gate/Up | 512 |
| FFN Down partial export | 1024 |
| Total | 3872 |

V/PV/FFN use the original rank-scoped MACAB path with16CK service and explicit
completion waits. Only rank1 addresses are emitted; channel rank broadcasting
is disabled. Independent audits check matrix coverage, native rank selection,
row activation, input-buffer dependencies, result timing, and actual DQ commands.

As requested, the prior GPU/nonlinear partition is preserved. O projection,
softmax/scaling, FFN GELU/multiply/cast, final partial reduction, norm/residual,
action I/O, GPU forwarding kernel execution and background VLM contention remain
unmeasured. The reported sum covers every formerly modeled AE stage; complete AE
end-to-end latency remains `null` rather than assigning zero to these operations.

## FFN Z in DRAM

The revised primary FFN storage choice is bank DRAM. It preserves the feature
partition and single-copy weights above; it does not use the proposed token-only
FFN partition or replicate FFN weights. The previous 8KiB/rank Z SRAM model and
its saved results remain available for comparison.

```bash
PYTHON=/path/to/venv/bin/python JOBS=4 bash build.sh
/path/to/venv/bin/python reproduce_ae_dram.py --output results/fresh-ae-dram
```

Each bank reserves row176, byte range `[360448,362496)`, for512B of BF16 Z.
Its sixteen used columns are `[local_feature_slot16][token16]`: one32B column
holds one feature across all16 tokens. For a Down input fragment, the rank reads
the same column index from all16 banks and assembles/transposes those values into
the512B `[token16][input_lane16]` MAC input slice. The existing1KiB input double
buffer remains; no full-batch8KiB Z SRAM is present.
These input and staging sizes exclude the existing accumulators:2KiB/bank for
Gate/Up and1KiB/bank for tiled Down.

Z production is streamed: each `PIM_Z_STORE` gathers128B of FP32 G/U for one
feature from its bank accumulator, crosses the explicitly unmeasured nonlinear
boundary, returns32B BF16 Z to that bank's staging buffer and writes DRAM using
native `LOCALWR`. This avoids assuming a separate32KiB rank G/U queue. The rank
has128B nonlinear input and32B output staging; each bank has32B column staging.
Rank-local transport remains a configurable32B/2CK assumption. The previous
bulk2048CK gather WAIT is replaced, not added again.

`PIM_Z_LOAD` performs native `LOCALRD` with array prerequisites and completion,
then uses the bounded rank-local port to fill the selected MAC input buffer.
All16 source banks must be present and ready before consuming a fragment.
Z and weight rows compete for the same banks, so ACT/PRE, refresh, write recovery
and repeated row switches are included. Four Down output tiles reread Z four
times:128KiB local Z writes and512KiB local Z reads across the system per layer
and step. These paths do not use external DQ; the rank-local links are an explicit
hardware assumption, not a free GPU round trip.
Array preparation follows trace order to prevent conflicting Z/weight row
activations from livelocking. Once a LOCALRD is issued, its completion and local
port delivery may overlap a MAC using the other input buffer. The measured
schedule does not claim the lowest latency attainable with reordered prefetch.

The runner also executes a combined FFN transfer chain with Z initially absent,
so loads depend on prior native stores. Independent Down timing explicitly
declares resident Z. The AE headline retains the previous separately timed stage
sum for comparison, and does not add the combined control a second time. Neither
mode assigns a measured latency to nonlinear arithmetic or GPU partial reduction.

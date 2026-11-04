# QKT score DRAM buffering and local PIM output projection

This branch adds independent `score_dram_spill` and `o_projection` switches to
the H16 token-sharded PV path. The four-way runner uses BF16 PV results in
every variant. All PV and O operands/weights are BF16, accumulation is FP32,
and results are rounded to BF16 when written/exported. QKT scores remain FP32.
The old FP32-output mode remains available for exact regression checks.

## Experiment 1: score buffering

Each rank computes a 16-query × 16-K tile, then transfers its 1 KiB FP32
scores through the rank-local 32 B port into DRAM. The 32 packets alternate
between bank0 and bank4. The physical offset is
`655360 + tile*512 + (packet//2)*32`, with `flat_bank=4*(packet%2)`.
All 31 tiles fit in 15.5 KiB per source bank. Storage is disjoint from Q/K/V
caches, weights and the reusable input staging region.

`PIM_SCORE_SPILL` is a rank descriptor. Every physical LOCALWR goes through
row preparation, rank CA arbitration, ACT1/ACT2 ownership and write recovery.
Each packet transfer uses 2 CK of the shared local port. The single 32 B
packet scratch can be reused after the local write data service; bank write
recovery remains enforced. The next tile waits for all writes of the previous
tile to complete. After all 31 tiles are stored, ordinary LD reads every
packet and sends the scores to the GPU. These reads include CL and real DQ
bursts. The score export byte count remains 496 KiB across all channels.

## Experiment 2: PV to local Wo

Each head occupies two PIM ranks in different channels. Each rank retains
the KV parity partial `[16 query,256 head feature]`. Each source bank owns
`[16 query,16 feature]`. A `PIM_PV_SPILL` packet gathers 16 FP32 elements from
one source bank, casts them to BF16, transfers the 32 B result to bank0 or
bank4, and performs a real LOCALWR.

The 8 KiB per-rank O input uses the existing `[fragment][query][lane16]`
staging layout at 524288: source-bank number is the logical feature fragment,
even fragments go to bank0 and odd fragments to bank4. A full producer
coverage and write-completion check precedes SEAL. Both banks subsequently
feed LOCALRD plus same-rank broadcast into all 16 bank SRFs.

Every rank holds one complete `[2048,1024]` BF16 `Wo` copy: 4 MiB per
rank and 64 MiB across the 16 PIM ranks. Each rank computes with its own
head's `[256,1024]` block, so the active blocks are 512 KiB per rank and
8 MiB in total; the two parity ranks read identical active head blocks.

Each bank reserves `[786432,1048576)`, or 256 KiB, for the full copy:
`[resident head slot8][output tile2][input fragment16][output column32][input lane16]`.
For rank/channel `c`, head block `h` occupies slot `(h-c//2)%8`. The active
head therefore remains in slot0 at `[786432,819200)`, preserving every
previously measured MAC address. Slots1–7 store the remaining head blocks
and are not read during this rank's projection. Weight input rows within
each head follow the PV/RoPE pair-major feature order.

Full residency is a capacity/layout extension made after the native runs.
The additional 56 MiB is preinstalled storage, with initialization traffic
and time excluded from the per-step results. The native runs validate the
8 MiB of active addresses; separate mapping tests prove the full-copy
bijection, bank bounds and unchanged active addresses. Frozen run artifacts
retain their original active-block metadata, and regenerated traces must
match them exactly. This extension adds no native commands or timing claims
for the inactive weight blocks.

Each bank computes 64 output features in two tiles of 32. Each tile uses
`16*32*4=2048 B` of the existing FP32 output pool. The 1 KiB SRF holds two
512 B input fragments, both of which must be fully consumed before overwrite.
After the first tile is exported and all SRF views released, explicit
`PIM_O_REWIND` permits a second scan of the retained 8 KiB DRAM input.

Each rank exports `[16,1024]` BF16 partials, 32 KiB; all ranks export 512 KiB.
The GPU must sum over 8 heads and 2 parity partials. Bias, if present, belongs
after this sum and must be added once. No cross-rank sum is implied by the
bank-local MACs.

## Resource and precision assumptions

The new conversion path models one reusable 64 B FP32 packet scratch,
separate from the unchanged 2 KiB per-bank accumulator and 1 KiB input SRF.
For each 16-element packet: gather 64 B costs 4 CK on the 32 B/2 CK local
port, cast costs 2 CK at 8 elements/CK, and local staging additionally costs
2 CK to transfer the resulting 32 B. Each operation also occupies rank CA.
The scratch cannot be overwritten while its local write or DQ burst is
consuming it. Port and cast costs are architecture assumptions, not measured
circuit costs.

Four-way comparisons use the same BF16 PV boundary. This changes the old
FP32-export baseline: its PV export shrinks from 256 KiB to 128 KiB and gains
explicit conversion service. The experiment-2 partial output is 512 KiB.
The distributive projection mapping doubles Wo arithmetic relative to
projecting an already-summed O once. BF16 partial rounding and changed
reduction order need separate numerical validation; this simulator executes
timing and data-mapping contracts, not neural-network tensor values.

## Reproduction and interpretation

```bash
PYTHON="$(command -v python)" JOBS=4 bash experiments/pi05-unified-gemv/build.sh
python experiments/pi05-unified-gemv/reproduce_o_projection.py --output results/o-projection
```

The runner executes all four combinations, unit and native negative tests,
per-command address coverage, SRF lifetimes, staging producer dependencies,
ACT1/ACT2 timing, CA arbitration and DQ overlap checks. Results include an
independent sum of QKV/attention, PV or PV+O, Gate/Up and Down, plus a separate
single-layer continuous-memory control. The latter is not added twice.

O-enabled variants include the PIM O input-read/broadcast/compute sequence,
control and BF16 result export in the PV+O stage and in the total latency.
Non-O variants exclude GPU output-projection time. All variants exclude GPU
softmax, final partial sums, norm/residual, FFN nonlinearities, GPU dispatch
and background contention. Reported modeled latency is therefore not an
end-to-end speedup comparison. DQ utilization is busy burst time divided by
native wall time per channel. Continuous-window comparisons use the longest
single complete DQ-free work window. Adjacent work phases may connect through
DQ-free control phases; score/result transfer phases split windows. Idle
holes inside transfer phases are excluded, and independent runs are never
concatenated. The exact maximal physical DQ-idle intervals are provided as
a separate diagnostic. Earlier cumulative whole-work totals are retained
for accounting only; they cannot represent uninterrupted available time.
A lower utilization alone does not establish better performance or measured
GPU bandwidth.

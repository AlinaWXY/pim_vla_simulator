# Whole-horizon AE experiment

This experiment compares H=16,20,30,40 action tokens with eight attention heads.
As requested, token buffers and accumulators grow with H and every operator
executes the entire H-token batch. No token microbatching or H16 latency scaling
is used. The frozen H16 implementation and complete result bundle are available
under tag `xinyao_1027_h16` (commit `41199cee64c40c55cafb330ed838d7e5c13fc0b8`).

## Preserved architecture

There are16 LPDDR5X channels, each with GPU rank0 and PIM rank1. Q/K assign half
the suffix tokens to each of two peer PIM ranks per head and retain two weight
copies per head. Banks own the same RoPE feature pairs. V/PV and FFN keep their
feature partitioning and single-copy weights. FFN Z stays in bank DRAM.

The prefix remains968 tokens. KV storage is rounded to a multiple of32 tokens,
which keeps each rank's QKT score tile at16 K tokens. All H suffix K/V values are
materialized before attention uses them. Padding tokens are masked by the same
unmeasured GPU softmax boundary as the H16 baseline.

| H | Local Q/K tokens/rank | Padded KV | Work columns/bank | Reserved work rows | Z physical bytes/bank |
|---:|---:|---:|---:|---:|---:|
|16|8|992|28|1|512|
|20|10|992|32|1|1024|
|30|15|1024|58|1|1024|
|40|20|1024|68|2|1536|

The work columns are coefficients, local Q, suffix K, then padding K. H40's
last four padding columns occupy a second row; its coefficient/Q/K RoPE data
still fit in the first row. V weights/cache move after this expanded reservation;
FFN weight addresses remain fixed. Z uses feature-major token blocks of16,
with physically transferred padding in the final32B column. It fits within the
existing2KiB row reservation at byte360448 per bank for every tested H.

## Hardware capacities and timing assumptions

- Per-bank Q buffer:32H bytes, enough to scan each local K once for all H queries.
- Q/K rank input double buffer:32H bytes; projection accumulator:64H bytes/bank.
- V/PV/FFN rank input double buffer:64H bytes; V/PV accumulator:32H bytes/bank.
- Gate/Up accumulator:128H bytes/bank; Down accumulator:64H bytes/bank.
- Bank column staging remains32B. Nonlinear staging remains streamed128B input
  plus32B output per rank; it processes up to16 tokens of one feature at a time.
- Q/K projection MAC service is H/2 cycles; QKT, V/PV and FFN MAC service is H.
  RoPE II1/latency4, cast1, rank reduction4 and local-port32B/2CK are unchanged.

DRAM row activation, precharge, refresh, local writes/reads and their completion
dependencies run through the native backend. Nonlinear arithmetic, O projection,
GPU V packing, softmax, final partial reductions, norm/residual, GPU forwarding
kernel execution and background VLM contention remain unmeasured.

## Run

From the repository root, using the same Python interpreter for build and run:

```bash
PYTHON=/path/to/venv/bin/python JOBS=4 bash experiments/pi05-bank-rope-h16/build.sh
/path/to/venv/bin/python experiments/pi05-bank-rope-horizon/reproduce.py \
  --horizon 20 --output experiments/pi05-bank-rope-horizon/results/fresh/H20
```

Repeat with16,30,40 and fresh corresponding H directories. Then aggregate:

```bash
/path/to/venv/bin/python experiments/pi05-bank-rope-horizon/aggregate.py \
  experiments/pi05-bank-rope-horizon/results/fresh
```

`run_remote.sbatch` runs the four independent H cases as a Slurm array after a
completed build. Set `PIM_CHECKOUT` and `PYTHON`. Do not rebuild or edit sources
while these jobs run; each result records source and library hashes.

## Channel-window definition

The requested statistic includes only complete semantic phases with no external
RD/RDA/WR/WRA/WRGB/RDRES commands on any channel. All fragments of idle time
inside input/output phases are excluded. Phase boundaries retain the H16 meaning.

This statistic is a set of candidate windows before boundary data tails,
turnaround, rank-switch and CA/row constraints. It is not actual concurrent GPU
usable time: the AttAcc PIM movement model and full-bank/ordinary admission still
have the limitations documented in the frozen H16 channel analysis. Results keep
`actual_GPU_usable_time_us` and complete AE end-to-end latency as null.

Independent audits check native events against address/coverage oracles, row state,
buffer lifetime, partial-column completion and external traffic. H16 is required
to reproduce all six frozen native timing cases exactly. The combined FFN case
checks actual earlier Z stores and is not counted again in the five-stage sum.

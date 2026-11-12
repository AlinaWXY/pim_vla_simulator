# Memory-controller command control at MAC 2×

The top-level memory controller issues the row and column commands for both
ranks. Fused descriptor children remain a software representation of the
existing operation order. Their decode takes zero cycles and issues no CA
command. Fixed-function GATHER, CAST, and adder-tree work remains on die.

## Rank and column state

- A split ACT to the last row of bank 0 followed by PREpb enters PIM mode.
  The penultimate row of bank 1 followed by PREpb restores DRAM mode.
- In PIM mode, a single-bank ACT to a configured input row selects broadcast.
  The subsequent RD sends the column to all SRFs. PRE closes the input row
  and returns to MB idle. Down uses this path on its existing co-resident
  input and weight row, then reopens that row with ACTAB for MAC.
- ACTAB opens all 16 banks at once. It occupies two external CA cycles and
  uses the same array timings as an ordinary split ACT, with one activation
  event in the rolling activation history.
- ACTAB to the third-last row selects register writes. Each Host WR sends
  32 B over DQ to one SRF slot in all banks. Other PIM-mode RD and WR have no
  DQ transfer. For a Host result, the controller issues an internal store
  into the existing score or fused-result address space and then reads the
  column in DRAM mode.

After an ACTAB to a data row, the column command targets the active rank row,
and the four combined BG/BA bits select the function. Single-bank array
operations use ordinary bank addressing. This includes the existing sparse
spills, read-modify-writes, and the source RD in Down broadcast.

| BG/BA | RD | WR |
| --- | --- | --- |
| `0000` | MAC into the current accumulator | WB_ACC |
| `0001` | MAC_INIT overwrites the accumulator | WB_MAC, reserved for the bypass path |
| `0010` | MAC_RED, reserved for a multi-MAC reduction | WB_ROPE |
| `0011` | MAC_INIT_RED starts and reduces a one-MAC dot product | WB_OBUF |

This QK program issues one MAC per key and reduces its 16 bank contributions.
MAC_INIT_RED schedules the original 4 CK reduction at MAC completion. It
fills the original per-bank output slots without another CA command. Fused
QK uses one WB_OBUF per score column and an implicit bank-enable mask for
banks 0 and 4, matching the two existing output packets. Fused O, Gate/Up,
and Down use one WB_OBUF per 16-bank result stripe. The controller drains
these stripes to DRAM while still in PIM mode, then reads the stage's
results in DRAM mode.

The local-RoPE cases execute WB_ROPE when the Q/K projection first writes
its BF16 result. For local query index `i = (column - 8) % 8`, the token index
in the 16-token window is `2*i + (channel & 1)`. The timing model assumes
the eight existing offline coefficient vectors for that channel are
preloaded and shared by its bank PEs. It models the existing cast order,
8 CK RoPE vector initiation interval, and 11 CK rotation latency before
the local store. There is no separate local RoPE stage on CA. GPU-RoPE
cases retain their GPU steps. Tensor values are outside this timing model.

## Reproduction

Run the original and new configurations with the same MAC frequency.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pim_vla_sim --parallel
c++ -std=c++20 -O3 tools/gap_replay.cpp -o build/gap_replay

python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/baseline-ca-traces --mac-freq-mult 2
python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/mc-final-traces --mac-freq-mult 2 --mc-control
python3 tools/estimate_dq_gaps.py --traces results/mc-final-traces \
  --spec results/mc-final-traces/spec-k2.json --engine build/gap_replay \
  --output results/mc-final/replay
python3 tools/plot_pim_buses.py --traces results/mc-final-traces \
  --output results/mc-final/buses --title 'Top-level memory controller | MAC 2x'
python3 tools/audit_mc_protocol.py --traces results/mc-final-traces \
  --baseline results/baseline-ca-traces --spec results/mc-final-traces/spec-k2.json \
  --output results/mc-final/protocol-audit.json
```

The matched-slice e2e estimator repeats each sampled slice 180 times. Cases
c–e use the prior α=1.0 and same-first DQ replay. Cases f/g use the prior
shared-I/O accounting and ideal overlap outside I/O epochs. The estimator
omits shared CA contention with VLM. The CA and DQ graphs use native AE-solo
timestamps and show 0.5 µs bins, including AE GPU intervals on the time
axis. The full projected AE plots tile those samples 180 times and
average them into 1,600 bins per case. The PIM-stage utilization denominator
is the sum of PIM-stage CK
over the 16 channels. CA counts both ranks' refresh and separately reports
rank 1 CA. DQ counts the physical Host transfers during those stages.

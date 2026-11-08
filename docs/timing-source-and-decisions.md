# LPDDR5X-8533 timing source and modelling decisions

The clean base is CMU-SAFARI/ramulator2 commit
`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da` (2026-11-03).
The only new timing authority is
`ramulator2/python/ramulator/dram/lpddr5x_pim.py`. Native generated C++ uses
Ramulator's DRAMDevice prerequisite, timing and state mechanisms.

The source is **JEDEC JESD209-5C, June 2023**, checked against the original
document's tables. A [readable copy of the standard](https://xjbcode.fun/Notes/006-ai/assets/JESD209-5C.pdf)
was inspected; printed pages differ from PDF page numbers by 38.
The document itself is not redistributed with this code.

| Parameter | Value (CK unless stated) | Source |
|---|---:|---|
| CK | 938 ps | 8533 bin, clock rounded upward |
| CL, CWL | 23, 12 | Tables 225 / 229; RL set 0 / WL set A |
| Write recovery | 37 | Table 230, x16, ECC/DVFSC disabled |
| Read-to-precharge | 8 | Table 225 nRBTP=6 plus BL/n_min=2 |
| Read / write RCD | 20 / 9 | Table 396, ceil(18 ns / CK), ceil(8 ns / CK) |
| Single / all bank RP | 20 / 23 | Table 396, 18 / 21 ns |
| RAS / RC | 45 / 65 | Table 396, 42 ns and RAS + single-bank RP |
| RRD / FAW | 4 / 16 | Table 396, 3.75 / 15 ns |
| WTR_S / WTR_L | 7 / 13 | Table 396, 6.25 / 12 ns |
| BL minimum / column cycle | 2 / 4 | Tables 339-342, x16 BL16 BG |
| RFCab / RFCpb (8 Gb die) | 224 / 128 | Table 240, ceil(210 / 120 ns / CK) |
| REFI / REFIpb | 4164 / 520 | Table 240; maximum intervals round downward |
| Per-bank refresh spacing / to other ACT | 96 / 8 | Table 240, 90 / 7.5 ns |

WCK remains on. DQ/NT ODT and ECC are disabled. Read preamble is 4 WCK
and read postamble 0.5 WCK; Table 377 then gives a 4 CK inter-rank
RD-to-RD issue gap (2 CK burst plus 2 CK rank guard). The 2 CK R-to-W
DQ guard represents upward rounding of the Table 377 WCK-to-DQ output
offset. These choices are fixed operating conditions, not calibration knobs.

## Resolving inconsistent examples in the task document

1. Alias only **transfer-to-transfer interface** constraints. Blindly aliasing
   every Rank rule would incorrectly add array tWR/tRTP before PREab to
   WRGB/RDRES, contradicting their SRAM destinations. Bank rules are never
   inherited by these aliases. Refresh blocking is separately explicit.
2. A buffer command addresses every BG. Consequently same-rank RDRES-to-WRGB
   is **17 CK**, from the same-BG rule `CL + BL_max + 2 - CWL`.
   Inter-rank RDRES-to-WRGB and GPU RD-to-PIM WRGB are **15 CK**.
   The document's unqualified "same-channel 15 CK" is the cross-rank case;
   15 CK cannot replace the stricter same-BG rule.
3. All-BG inheritance also retains WTR_L for same-rank WRGB-to-read (29 CK
   command gap), in addition to the Rank WTR_S rule. This follows the stated
   all-BG interface alias algorithm; removing it would contradict that rule.
4. REFpb is the LPDDR5 paired-bank command: bank n and n+8 are refreshed
   together. The upstream pair-target mechanism is retained.

## PIM resource assumptions

`MACAB_M{M}` / `MACQK_M{M}` cover M=1,2,4,8,16,32. Their work-batch completion
is `M*4/k` CK, and the next column issue separation is
`max(tCCD_L, M*4/k)`. The completion timestamp includes the batch's compute
pipeline; no extra unspecified tail is added. This is a model assumption
from section 5.1, not a measured PE RTL latency. Sensitivity uses k=1,2,4.

Local 32-byte array transfers use 4 CK service. SRF multicast/gather use
2 CK, cast 1 CK, rank reduction 4 CK, and RoPE latency 4 CK with initiation
interval 1 CK. The latter values preserve the old rank-RoPE architectural
assumptions; they are distinct from JEDEC memory protocol timings.
Descriptors occupy external CA for 21 CK and decode for 1 CK.

Native timing rules retain one logical receive tick for local commands,
so suppressing external CA does not introduce erroneous +/-1 CK timing
offsets or shift shared tFAW timestamps. Native `command_cycles` is therefore
one, while `metadata.external_ca` is zero; explicit spec bus constraints
exclude local commands from the external CA. Metadata also carries private
CA occupancy separately. Private ACT/PRE and
MAC aliases update the **same** native bank state and timing histories.
All local operations (`LOCALRD`, `LOCALWR`, broadcast, gather, cast,
reduction, and RoPE) share this one-command-per-CK rank-internal issue path.
This fixed architectural assumption applies both to the autonomous local
state machine in streaming and to descriptor expansion in fused operation;
it does not require a programmable rank controller for streaming. Each local
issue occupies private CA for 1 CK and occupies no external CA or DQ.
Instruction issue is separate from execution: LOCALRD/LOCALWR timing remains
bank-scoped, cast is bank-scoped, and broadcast/gather/reduction/RoPE retain
their own rank-unit initiation intervals. No shared rank-wide data-port lock
serializes these independent units until completion. For example, two local
reads to different banks may issue 1 CK apart, while the same bank retains
its 4 CK read-to-read constraint.
Refresh blocks SRAM IO conservatively as required by the task, and waits
for issued SRAM IO data bursts before entering refresh.

`build_spec(k)` exports native config and command completion/DQ metadata.
The bus checker consumes this metadata, not a replayed trace or a second
timing table. Its physical DQ guards are independent of the alias-derived
command constraint list, allowing fault injection to detect a deleted rule.

## Native lookahead scope

`NativeLookahead` allocates one reusable private DRAMDevice per channel using
the same full spec export. Before forecasting it copies the live node states,
open rows, readiness clocks, per-command histories, and shared rolling-window
histories. It expands at most 32 FIFO operations within the requested time
horizon using native prerequisite resolution, native timing checks, and native
issue into this snapshot. Explicit dependencies and barrier completion tails
use actual or predicted completion timestamps from spec metadata. The live
device is never changed by prediction.

Forecasts omit future GPU commands, future refresh, and PIM resource contention.
They therefore provide conservative earliest issue estimates for arbitration,
not promises of actual service times. A local head can reveal a later WRGB or
RDRES within the window; checking only the current head cannot do this. The
controller still checks every real issue against the live native device. The
bounded predictor is intended for streaming FIFO programs; it stops at a
descriptor instead of inventing a nested program schedule.

`tests/native_lookahead_tests.cpp` covers future DQ behind a local operation,
explicit dependencies, barriers, a 32-operation limit, horizon truncation,
all-bank split-activation prerequisites, deterministic repeated forecasts,
and byte-for-byte-equivalent live state before and after prediction.

## Validation

`python tests/test_spec.py` tests numeric provenance, all 48 transfer pairs,
all-BG broadcast, absent SRAM array prerequisites/recovery, completion
metadata, MAC scaling, refresh coverage, and fixed topology validation.
`python tests/test_spec.py --export reference/spec` writes the full native
configuration and regression command matrix. `tests/native_spec_tests.cpp`
repeats every matrix boundary against real DRAMDevice, checks prerequisite
address resolution and paired-bank refresh. It must be built and run on the
server before claiming native validation passed.

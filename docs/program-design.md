# PIM program interface and retained tensor contracts

`programs/gen_pim_program.py` generates JSON directly from tensor definitions.
It does not read an old native trace, a CA/DQ event file, or a replay schedule.
All five valid layout/RoPE/dataflow variants are generated with source hashes.
The stage driver repeats one layer for 18 layers and 10 denoising steps.
The JSON is a layer-relative address template: the controller adds
`layer * 512` to its array row index, equivalent to 1MiB per bank per layer.
Descriptor base fields are also relative to the current layer; their
expanded children receive the same relocation. Thus all eighteen layers
have distinct weights, KV and temporary regions; denoising steps reuse
the same layer's allocation. The 18MiB per-bank reservation fits within
the 64MiB bank capacity of the 8Gb/rank organization. The largest relative
reservation ends at 1MiB, so adjacent layers cannot overlap.

Each stage contains sixteen ordered channel programs. Stage completion is a
global barrier. `deps` refers to **completion**, with indices local to a
stage/channel program; a descriptor's children use a separate index space.
Operations without dependencies may issue after their predecessor has issued,
subject to the controller's resource and DRAM checks. `BARRIER` has no command
or latency. It completes only after its listed producers complete.

Addresses are `[channel,rank,BG,bank,row,column]`; column is a 32-byte column
within a 2KiB row. Whole-rank MAC, WRGB and RDRES use BG/bank `-1`.
`slot`, `dest_slot`, and `result_slot` use **32-byte units**. Input SRF slots
are 0–31; output-pool slots are 0–63. `source_regions` identifies exact
FP32 source bytes by bank and byte offset, including a packet that collects
data from two banks. Gather/cast operates on this existing output pool; the
program introduces no rank result cache or unlimited packet scratch.

`pim_resources.h` enforces those identities independently of DRAM timing.
It tracks validity, ready timestamps and in-flight readers for 32 SRF slots
and 512 FP32 output words per bank, plus the bounded RoPE input/output state.
The reduction array in this class is a validity scoreboard for sixteen
scores, not allocated result SRAM. First-fragment MAC initializes a word;
subsequent fragments must accumulate the immediately preceding partial sum.
RDRES locks exactly its source words through the spec-supplied completion,
and a subsequent MAC cannot overwrite them before that timestamp. Invalid
dataflow raises an error rather than converting missing producers into time.

MAC output geometry is `rows * output_columns * 4` bytes per bank. QKV uses
1536B; token PV 1024B; feature PV 512B; Gate/Up and each O tile 2048B; Down
256B. Each input batch occupies at most 1024B, and the first refill of the
next batch depends on the last MAC's completion. O's second tile cannot fill
its input or overwrite output until all 512 first-tile exports complete.

`MAC_RUN`/`ROPE_RUN` carries eight retained version-2 wire words and CRC16,
plus a timing-free expansion. The controller must feed these children through
its finite 32-entry internal FIFO and complete the descriptor after all
children finish. Array commands in children use private rank CA. External
RDRES operations between O tiles still use external CA/DQ and their named
I/O epoch. A local-RoPE fused layer has six MAC descriptors plus one RoPE
descriptor (147 CA CK under the spec); GPU-RoPE has six (126 CA CK). Decode
latency and CA occupancy are defined exclusively by the DRAM spec.

The Q exchange has two global stages, `q_exchange_export` and
`q_exchange_import`. The old bank-addressed Q_REMOTE behavior cannot be
represented as WRGB's same-data-to-all-banks broadcast. The new program uses
ordinary addressed WR to a dedicated per-bank `[0x15000,0x15800)` Q-import
region, followed by LOCALRD to the proper query SRF slot. Q export and import
retain their original external byte counts. All peer exports finish before
imports start. This adds real row/write/local-read costs relative to the old
targeted-buffer-write interpretation and is an intentional physicalization.

Feature layout owns all 16 queries and eight feature lanes per bank; token
layout owns eight local queries and sixteen pair-major feature lanes per
bank. Token parity, Q/K addresses, V-cache tiling, FFN weight placement,
two-bank staging, O weight residency, and BF16/FP32 boundaries retain the
1108 definitions. The generator's computed per-layer external totals across
16 channels are 6096/5352/5160/4968/4776 KiB for the five variants. Historical
totals serve only as an independent byte-conservation assertion.

The feature rank's feature sets are `[0:64,128:192]` and
`[64:128,192:256]`; each bank owns eight consecutive members of its set.
Token layout uses the pair-major mapping `bank=(feature%128)//8`,
`lane=2*(feature%8)+feature//128`, with all 256 features present in both
parity ranks. Feature Q and K packets contain two token vectors per 32B:
Q load metadata selects the appropriate half into the query slot, while
each MACQK identifies its K `weight_lane_offset` of zero or eight. It still
issues one dot per K token, without assuming double-dot packed throughput.

| Operator | Token layout: matrices per rank | Feature layout: matrices per rank | Scalar MACs, all channels |
|---|---|---|---:|
| QKV | `[8,1024] × [1024,768]` | `[16,1024] × [1024,384]` | 100,663,296 |
| QKᵀ | `[16,256] × [496,256]ᵀ` | `[16,128] × [992,128]ᵀ` | 32,505,856 |
| PV | `[16,496] × [496,256]` | `[16,992] × [992,128]` | 32,505,856 |
| O | `[16,256] × [256,1024]` | `[16,128] × [128,1024]` | 67,108,864 / 33,554,432 |
| Gate/Up | `[16,1024] × [1024,512]` | same | 134,217,728 |
| Down | `[16,4096] × [4096,64]` | same | 67,108,864 |

Token O performs twice the arithmetic of feature O: each KV parity rank
projects its partial PV output before the GPU adds the sixteen rank
outputs. This existing distributed projection choice must not be described
as equal arithmetic across the layout ablation. Both layouts retain a full
`[2048,1024]` BF16 W_O copy per rank, but access only 512KiB/token rank or
256KiB/feature rank. QKV active weights are 1.5MiB/token rank and
0.75MiB/feature rank. Initialization is outside the steady-state program.

The V tail RMW preserves different lanes in the two layouts. Token layout's
last local block contains four prefix tokens, eight new tokens, and four
padding tokens. Feature layout has two final blocks: eight prefix/eight new,
then eight new/eight padding. Gather source regions explicitly select the
right QKV accumulator rows and lanes. Gate/Up exports FP32 complete feature
shards; Down exports FP32 complete output shards, with no cross-rank sum.

The new task deliberately changes two other old execution assumptions:
one shared RoPE unit per rank replaces independently active bank units; and
QK reductions emit each 64B query-score column immediately, as two RDRES
bursts (streaming) or two LOCALWR packets (fused), with no 1KiB rank score
tile cache. The fused score array is bulk-read only after all QK work ends.
MAC completion tails, correct WRGB/RDRES CL/CWL, row constraints and the
addressed Q-import path also change timing. Therefore the task's request that
every standalone difference be attributable only to buffer I/O turnaround
cannot literally hold alongside these structural requirements. Reports must
identify each modeled change rather than normalize away the difference.
In particular, a whole-bank MAC is not an all-bank ACT primitive: every
target bank must undergo legal ACT1/ACT2 preparation, including RRD/FAW
limits. The old AttAcc-style ACTAB operation is absent from the new command
set. Its removal can materially change activation overhead.

RoPE coefficient values are precomputed BF16 sin/cos for the actual model
position IDs. The generated operations cover every pair, input and output,
but this is a timing/layout model and does not execute tensor numerics.
The per-bank RoPE validity records describe vector views in the existing
SRF/output pools, not additional bank PEs or independently allocated SRAM.
RoPE follows the completed QKV spills and precedes Q residency, so these
views do not overlap live GEMV or QK inputs. The sole rank RoPE execution
resource arbitrates all bank vectors.

Feature W_O full residency is arranged as sixteen rotated feature-shard
slots, each 16KiB per bank, placing the active shard in slot zero. Token W_O
uses eight rotated head slots, each 32KiB per bank. Both occupy the same
256KiB per-bank reservation and preserve their active MAC addresses. This
is a layout/capacity contract for preinstalled weights; the simulator does
not execute initialization or derive a numerical accuracy claim from it.

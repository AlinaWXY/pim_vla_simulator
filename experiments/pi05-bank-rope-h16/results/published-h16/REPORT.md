# H16 AE with FFN Z in bank DRAM

All timings are fresh native simulations. This is a sum of separately timed modeled stages, not complete end-to-end AE latency.

| Stage | us/layer/step | External DQ/KiB |
|---|---:|---:|
| chain | 34.886096 | 880 |
| v | 10.282356 | 832 |
| pv | 9.189586 | 624 |
| gate_up_z_store | 36.195544 | 512 |
| down_tiled_dram | 33.920894 | 1024 |
| **Measured sum** | **124.474476** | **3872** |

Measured stage sum x18 layers x10 steps: **22.405406 ms**.

FFN separate-stage subtotal: **70.116438 us**, previously **60.316214 us** with SRAM. Delta: **+9.800224 us**.
Combined FFN transfer chain: **70.797426 us**; checked as a dependency control and excluded from the above sum.

## Storage and dataflow

- Weight ownership is unchanged: all16 tokens/rank,256 intermediate features/rank,16 Down partials returned to GPU. No weight replication is added.
- Z occupies512B/bank in row176 (byte base360448), columns0..15. Each column holds one feature across16 tokens. One2KiB DRAM row/bank is reserved.
- Each store streams128B G/U from the bank accumulator through a rank-local nonlinear boundary, returns32B Z to that bank, then performs a real local DRAM write. Unknown nonlinear compute is not assigned a hardware latency.
- Down reads one column from each of16 banks for each input fragment. A bounded rank-local path transposes these into the512B MAC input slice. All16 source banks must be ready before MAC.
- The1KiB input double buffer remains. There is no8KiB full-Z SRAM. Temporary storage is32B/bank plus128B nonlinear input and32B output staging/rank.
- Native ACT/PRE, LOCALRD/LOCALWR latency and write recovery, row conflicts, shared local ports and buffer ownership are modeled. Local-path throughput remains a configurable32B/2CK assumption.
- Z adds128KiB local array writes and512KiB local array reads per layer/step; these do not traverse external DQ. Rank-internal links are an architectural assumption.

## Evidence

Each case includes the source trace, manifest, per-channel native timelines, local-transfer events and independent audit. Old SRAM cases and unchanged attention stages pass exact timing regression. Native invalid-sequence tests are in native-rejections/.

## Scope and scheduling notes

- The input and temporary buffer sizes exclude the existing Gate/Up2KiB/bank and tiled Down1KiB/bank accumulators.
- Z stays in its producing rank. Its local array reads/writes and rank-local transfers do not traverse external DQ. Local paths still consume array resources and ACT/PRE command timing.
- Array preparation follows trace order to prevent competing Z and weight activations from livelocking. Read completion and local-port delivery can overlap a MAC using the other buffer. These measurements describe this schedule, not the lowest possible latency with reordered prefetch.
- The same unmeasured operations as the earlier AE experiment remain excluded:
  - GPU_V_cast_pack
  - GPU_softmax_and_score_scaling
  - GPU_O_projection
  - GELU_multiply_cast
  - GPU_FFN_partial_reduction
  - normalization_residual
  - action_input_output_and_step_update
  - GPU_Q_forward_kernel_and_dispatch
  - background_VLM_contention

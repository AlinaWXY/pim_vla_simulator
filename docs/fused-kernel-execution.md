# Fused execution with calibrated VLM kernels

This execution model follows the user's 2026-11-09 scheduling clarification.
`execution_model=fused_kernel_windows` applies only to concurrent fused f/g.
The user selected **one descriptor per window**: PV→O and QKV→local RoPE
also wait for the current VLM kernel before handing off the next descriptor.
The older `native_dram_requests` path remains an explicit, separate model.

Each descriptor stage follows these boundaries across all sixteen channels:

1. Upload inputs and deliver the coarse command using native PIM-side DRAM
   requests, prerequisites and timing. Wait for every channel's input DQ and
   descriptor delivery/decode to complete.
2. Run the descriptor's private FIFO32 program while the GPU executes VLM
   kernels in dependency order. The GPU starts another kernel whenever the
   current one finishes and the private computation is still active; there is
   no kernel-size fitting or preemption.
3. Finish the entire private program, including output writes to DRAM. Close
   admission of new VLM kernels. If a VLM kernel is active, wait for its exact
   calibrated completion event.
4. Export all outputs using native ordinary DRAM reads. Execute the next AE GPU
   helper/nonlinear block, or deliver the next descriptor, while VLM admission
   remains closed. Pure transfer/setup stages have no VLM window.

Native zero-time barriers and descriptor completion are settled before admitting
a kernel at the same cycle. This prevents a kernel from starting just as PIM
results become ready. The native resource scoreboard is preserved across stages,
including Q residency. Native refresh, FIFO dependencies, CA/DQ constraints and
completion tails continue to apply to PIM; only the host's phase ownership gates
issue. This path does not use FIFO-head I/O arbitration against GPU requests.

O, Gate/Up and Down now actually spill results to a bounded DRAM region before
export; O finishes both tiles before returning data. See
[program layout](fused-kernel-programs.md). These additional writes and ordinary
read prerequisites are included in the simulation. Earlier RDRES-based stage
durations must not be reused for this model.

## Calibrated GPU abstraction

`tools/export_calibrated_vlm.py` reads the existing accepted VLM-solo calibration
and checks its hashes, identities, clocks, byte totals and complete coverage. It
exports the observed integer-cycle duration of every work unit, including four
memory units: 2357 units totaling 113958594 cycles / 106.893161172 ms at 938 ps.
The old calibration's binary and source identity are retained as provenance;
this is reuse of its durations, not a newly calibrated binary.

The frontend advances VLM by these completion events. It does not generate the
619790815 GPU read/write transactions and does not claim to model their current
rank0 row state or command-level interference. Their bytes are retained as
`vlm_accounted_*` metadata. `gpu_*_bytes` and native GPU command counts are zero.
This is a deliberate model change under the requested exclusive handoff rules,
not a claim that the detailed concurrent DRAM trajectory is unchanged.

`phases` records upload/compute/join/download boundaries and per-channel ready
times; `ae_gpu` records exclusive GPU helper intervals. `AE_VLM_join_us` measures
waiting **before export**. `AE_GPU_wait_us` retains its older meaning and is zero
here. Stage duration includes its join wait; subtract the explicit join when
reporting native PIM phase time, and do not label that value an AE-solo result.

## Execution and audit

Generate f/g with `programs/gen_pim_program.py --execution-model
fused_kernel_windows --compact --output programs/generated/fused-kernel`.
`experiments/run_fused_kernels.py --prepare-only` freezes the binary, runtime,
programs, schedules, VLM input and calibrated-duration ledger. Run each case with
the same tool's `--case f` / `--case g`, then audit with
`tools/audit_fused_kernels.py --run <campaign>`.

The audit verifies native PIM byte conservation and refresh, all 18 layers × 10
steps, exact calibrated kernel durations, zero native GPU transactions, no
kernel overlap with handoff/nonlinear intervals, and full phase/timestamp
identities. Small native regressions include long/short kernels, same-cycle
completion with a trailing barrier, channel imbalance and pure transfer stages.

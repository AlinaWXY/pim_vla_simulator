# Native execution model

The executable links the clean Ramulator 2.1 core pinned at
`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`. Sixteen channel controllers each own
one native `DRAMDevice` containing both ranks. GPU and PIM candidates consult
the same device's prerequisites, timing state and command issue path. No legacy
PIM trace or external arbitration timeline is read.

The spec exports command completion and bus metadata along with its native
timing table. A passive checker records actual CA intervals and delayed DQ
intervals, splitting occupancy at global stage boundaries. It is never queried
to make an issue decision. The fault test removes a native transfer constraint
and verifies that this observer detects the resulting DQ overlap.

The PIM JSON contains a layer-relative physical program and explicit completion
dependencies. Each layer has a separate 1 MiB region per bank; denoising steps
reuse the corresponding layer. Descriptor children are the statically expanded
instruction program. One descriptor is resident at a time; instructions issue
in program order through the private command path. The stored program is not a
hardware FIFO. The lookahead policy can inspect at most the next 32 instructions.
SRF and output pools are checked by a separate bounded data-lifetime scoreboard.

VLM requests are uniformly released within each measured kernel duration and
bounded by a per-channel outstanding window. Each channel receives its own
balanced read/write sequence: channel striping must not accidentally segregate
reads and writes into different channels. Read and write tensor regions are
distinct and the logical address sequence wraps within the physical rank0
capacity. Requests are queued by bank and direction; the controller compares
ready heads by row hit and arrival. This is equivalent to scanning the generated
monotonic bank/direction streams, without scanning the entire outstanding window.

Both refresh modes have native command semantics. Main experiments use paired
per-bank refresh, with due banks draining before a refresh command. Split ACT
commands have owned ACT2 requests: ACT2 issues as early as legal, with nAAD as
an upper bound rather than an artificial mandatory wait.

Case b implements the document's sum of independently initialized solo epochs.
After AE completes, it resets the DRAM state and refresh phase for the VLM epoch
inside the same simulator, retaining the global elapsed timestamp and all
statistics. Thus its measured total equals AE_solo + VLM_solo without a latency
correction. It is an independent-epoch serial baseline, not a measurement of
retained row-buffer or refresh-phase state across an AE-to-VLM handoff. Concurrent
cases never reset memory state during execution.

The model does not execute tensor values. Actual GPU memory addresses and request
burst traces are unavailable; its uniform arrivals and synthetic sequential
regions are the workload assumptions required by the supplied plan. Measured
arithmetic helpers are counted separately from the zero-cost control handoff.
SM contention from WRGB transport, additional control handoff latency, and
concurrent execution of AE nonlinear kernels with VLM kernels are not modeled.

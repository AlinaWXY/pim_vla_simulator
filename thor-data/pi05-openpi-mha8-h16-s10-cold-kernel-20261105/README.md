# Thor pi0.5 H16 MHA8: Cold-Kernel Timing

Fresh, separate cold-cache measurements on 2026-11-05. Existing normal-cache
results are not replaced. This dataset measures **isolated kernel durations**,
not model cold-start time, normal denoising-step elapsed time or PIM offload time.

## Result

Three full requests were executed. In each, Nsight Compute captured the real
fifth denoising step, including all 18 expert layers and action input/output
projections. The model still uses ten denoising steps in every request.

| Family | Mean cold-kernel sum / ms per selected step |
| --- | ---: |
| Linear matrices, including QK and SV | 4.853845 |
| Nonlinear / RMS core | 0.935733 |
| Elementwise affine / gates | 1.908608 |
| RoPE generation / application | 1.189440 |
| KV concat / copies / casts | 4.989493 |
| Masks and small setup | 0.059669 |
| **Sum of selected kernel durations** | **13.936789** |

The three sums are 13.783328, 13.998400 and 14.028640 ms. Standard deviation:
0.133759 ms. Each captured step contains 1,033 kernels; all 3,099 kernels have
exactly one NCU pass. Repetitions have identical kernel sequences, grid/block
geometry and attribution. The sequence/geometry also matches the earlier
normal-cache Nsight Systems label pass exactly.

**Do not call 13.936789 ms the native one-step latency.** It excludes CPU
submission gaps, profiling synchronization, cache-flush work, DMA memcpy and
outer Euler work. Do not multiply it by ten and label the result a measured
full AE latency. NCU serializes the selected kernel measurements.

The existing no-profiler normal baseline remains separate: mean native
denoise_step interval 11.573248 ms and complete model API-wall p50 219.263322 ms.
Different cache state, tools, scheduling and clock behavior preclude treating
their ratio as a cache-only speedup/slowdown.

## Configuration

- OpenPI pi0.5, B1, H16, S10, Q8/K8/V8, head_dim256.
- Same physical synthetic MHA8 in both VLM and AE as the accepted baseline.
  Original Q/O, residual widths, FFNs and ViT are unchanged. This is not a
  trained MHA-policy accuracy claim.
- VLM: 18 layers, D2048/F16384. AE: 18 layers, D1024/F4096.
- Three 224x224 synthetic RGB cameras, seed17, prefix968, AE KV length984.
- Existing selective BF16/FP32, PyTorch2.11.0+cu130, Transformers4.53.2,
  native eager/unfused VLM/AE attention. One Torch CPU and one inter-op thread.
- AdaRMS time-conditioning and dense modulation outputs remain precomputed:
  370 dense cache hits per request, zero online conditioning-dense kernels.
- No model/environment download, compile, inference CUDA Graph, shared runtime
  change, manual GPU clock setting or new weight-loading method.

## Method

Load the existing local checkpoint once with sequential readinto. Precompute
AdaRMS and run 20 complete model requests before measurement. The cold-cache
condition is then imposed **per selected kernel**, not once per request/step:

```text
--metrics gpu__time_duration.sum
--cache-control all
--replay-mode kernel
--clock-control none
--import-sass off
--profile-from-start off
--nvtx --nvtx-include cold_step/
```

Nsight Compute version: 2026.1.1.0. The default Tensor Core pipeline boost
state is `stable`; system clocks are not manually locked. The NCU warning
about unmodified clocks is retained, so clock-related variability is not
claimed to be eliminated. Only duration is collected for model kernels, and
the report verifies one pass per kernel. Three requests are statistical
repetitions, NOT three metric-replay passes.

Each selected kernel starts with caches cleared by NCU. It may still reuse
data loaded into caches during that kernel. Clearing hardware caches does
not delete the AdaRMS precomputed tensors from device memory or rerun their
linear layers. Memory access and stalls actually incurred inside a kernel
are part of the measured duration; no DRAM-time subtraction is attempted.

A logical operation may launch several kernels. Clearing before every kernel
breaks cross-kernel reuse inside that operation. This is intentionally a
**cold-per-kernel diagnostic**, not a simulation of one logical operation
whose input starts in DRAM while its internal kernels reuse caches normally.
Actual GPU/PIM offload additionally needs transfer, coherency, dispatch and
synchronization costs; this result does not measure those costs.

## Cache-Control Probe

Same 1 MiB FP32 input, 1 MiB output, fixed elementwise add; 50 warmups, four
captured launches for each policy. GPU L2 capacity reported by CUDA:32 MiB.

| NCU policy | L2 read-hit sectors | L2 read-miss sectors | Miss requested bytes |
| --- | ---: | ---: | ---: |
| `none`, each of 4 launches | 0 | 0 | 0 |
| `all`, each of 4 launches | 0 | 32,768 | 1,048,576 |

With no L2 read requests, the `none` result's L2 hit rate is undefined, not
100%. It is consistent with requests being served nearer to the SM. The
contrast demonstrates that the requested cache policy changes this probe's
memory behavior. It is not a measurement of every model kernel's DRAM bytes.

The probe collects hit/miss counters and needs three NCU passes. Its timing
values are retained as raw evidence but are NOT part of the model timing
summary. The model collection is duration-only and single-pass.

## Attribution And Validation

- Every ATen dispatch is given a unique NVTX marker with shapes, strides,
  dtypes and module roles. Kernels are mapped to these markers in the NCU
  report. CUB's nested NVTX names are resolved through the complete outer
  range stack; its scan kernels are not omitted.
- Final classification handles composite 4D `aten::matmul` as QK/SV and
  `aten::square` under RMSNorm as RMS core. The capture JSON's provisional
  category fields are preserved for provenance; **use `analysis_verified/`
  for final categories and numbers**. `kernel_ledger.json` retains both.
- Per repeat: 18 Q, K, V, O, FFN gate/up/down operations, 18 QK and18 SV
  operations, 37 norm calls and37 cached dense calls. No cached dense launch.
- Cold vs cache-enabled reference output and the previous independent normal
  run's final actions are bitwise equal for the deterministic fixture.
- All 1,033 kernel names/grid/block values match the normal-cache kernel
  sequence. Three repeats have identical sequences and attribution.
- 52 recorded GPU-process checks in the model run found no other compute PID.
  These are discrete checks, not proof of continuous whole-SoC isolation.
- After completion: GPU-process list empty, GPU utilization0%, temperature40C.
  The saved resource check is2026-11-05T15:06:51+08:00. No background worker
  remains and no other user's work was stopped.
- Six focused CPU analysis tests cover metric parsing, nested NVTX ownership,
  composite attention geometry and RMS-square attribution.

## Files And Reproduction

- `figures/index.html`, SVG/PNG/PDF figures: standalone local report.
- `analysis_verified/summary.json`, `operators.csv`, `families.csv`: final data.
- `analysis_verified/kernel_ledger.json`: all3,099 mapped kernel records.
- `raw/*/capture.json`, `kernels.csv`: unchanged capture and NCU export.
- `raw/*.ncu-rep.gz`: losslessly compressed original NCU reports; decompress
  into a new analysis directory to open them with NCU.
- `logs/`: exact launch commands and complete console logs.
- `code/`: pinned adapters, capture, launch, extraction, analysis and plotting.
- `MANIFEST.json`: distributed-file digests and uncompressed report digests.

The previous warm dataset is retained in the parent commit
`de6974fe0095098500191211c7bdaa231c895ab7`, sibling directory
`thor-data/pi05-openpi-mha8-h16-s10-adarms-20261105/` on this branch. It also
contains the fixed token fixture and normal-reference output/provenance.

```bash
python -B -m unittest discover -s code -p 'test_*.py' -v
python -B code/analyze.py --root . \
  --warm ../pi05-openpi-mha8-h16-s10-adarms-20261105 \
  --output analysis_reproduced
python -B code/render.py --root . --output figures_reproduced
```

Renderer reads the verified original `analysis_verified/summary.json`.
Use the existing main-cluster analysis environment under CPU Slurm for server
analysis/plotting. A new GPU capture requires checking Thor occupancy, reading
the server SKILL and assigning fresh run/output tags; do not replay commands
blindly over existing outputs. No profiling is started by analysis or plotting.

NVIDIA methodology references:
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay

Destination: `AlinaWXY/pim_vla_simulator`, branch
`thor-data/pi05-openpi-mha8-h16-s10-cold-kernel-20261105`.
Only the new cold-kernel data directory is added; existing warm data and
simulator source are unchanged. Intermediate local analysis drafts are not
part of the published dataset.

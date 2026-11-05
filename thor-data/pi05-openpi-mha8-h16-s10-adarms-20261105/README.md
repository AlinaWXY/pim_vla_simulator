# Thor OpenPI pi0.5 H16 MHA8 Latency

Fresh measurements on 2026-11-05. These are H16 measurements, not H20
scaling estimates. Older data and simulator code are unchanged.

## Result

| Normal inference phase | Mean / ms | Same-pass mean share |
| --- | ---: | ---: |
| ViT, three cameras | 26.290488 | 11.737% |
| VLM prefix | 78.660008 | 35.118% |
| AE, ten steps and outer sampler work | 118.024545 | 52.692% |
| Other | 1.012474 | 0.452% |
| Total GPU event span | 223.987515 | 100% |

The independent clean API-wall pass has 100 observations: mean 223.888307 ms,
p50 219.263322 ms, p95 236.121424 ms. All samples, including the 379.121716 ms
maximum, are retained. Do not add phase medians to reconstruct API-wall p50.

Normal one-step elapsed time, averaged over all ten denoising steps within
each of 50 requests: mean **11.573248 ms**, median of per-request means
11.349067 ms. The fifth step alone: mean 11.532703 ms, p50 11.309616 ms.
The denoise_step boundary includes action input/output projections and all
18 expert layers, but excludes the sampler's outer Euler update.

### Fine GPU Activity, A Separate Instrumented Measurement

| Family | Mean / ms per step |
| --- | ---: |
| Linear matrix operations | 4.203207 |
| Nonlinear / RMS core | 0.361426 |
| Elementwise affine and gates | 0.772395 |
| RoPE generation and application | 0.564111 |
| KV concat, copies and casts | 4.286795 |
| Masks and other small setup | 0.026671 |
| Cross-family overlap | 0.022723 |
| GPU activity union | **10.237328** |

This is the exclusive GPU activity union from five profiled full requests.
Each request contributes the average of its ten actual steps; the reported
statistical unit is a request, n=5, not 50 independent requests. Every selected
step has 1,033 kernels matching the separate label pass in name, order and
launch geometry. There are 18 calls of each Q/K/V/O and FFN projection,
37 AdaRMS/norm calls per step, and no online cached-conditioning dense kernel.

The traced event interval is 17.749819 ms/step, including 7.512492 ms without
recorded GPU activity. **That gap is not accepted as a normal-inference gap.**
Nsight software tracing perturbs execution. Full AE p50 is 36.34% above its
same-process pre-trace control and 16.76% above its post-trace control.
Traced AE mean is 52.49% above the independent no-profiler AE mean. These
differences also include process/state drift; do not claim they isolate pure
profiler overhead, or scale the kernel fractions onto the normal total.

## Configuration

- OpenPI native PyTorch model, declared upstream commit
  `215abfb217dbac7d5f1273282331b9b1866c0479`; exact deployed source SHA256s
  are checked before running and saved in both captures.
- pi0.5, B=1, H=16, S=10, Q8/K8/V8, head_dim=256.
- Same synthetic physical MHA conversion as previous studies: expand K/V
  weights in BOTH VLM and AE into eight independently stored heads. Q/O,
  ViT, residual widths and FFNs are unchanged. No trained MHA policy claim.
- VLM: 18 layers, D=2048, F=16384. AE: 18 layers, D=1024, F=4096.
- Three synthetic valid 224x224 RGB cameras; prefix length 968, AE KV length
  984. One deterministic fixture, seed 17, GPU-resident inputs.
- Selective BF16/FP32; Torch 2.11.0+cu130, Transformers 4.53.2.
- One Torch intra-op and one inter-op CPU thread. No compile, inference CUDA
  Graph, changed attention backend, clock change or shared-environment edit.
- Active VLM/AE attention is the accepted OpenPI **eager, unfused** path.
  OpenPI sets it inside `sample_actions` / `denoise_step`; QK, softmax and SV
  therefore appear separately. The raw `dimensions.attention_implementation`
  value `sdpa` was captured at construction time, BEFORE the first inference,
  and is not the executed backend. See `ATTENTION_BACKEND_AUDIT.md`.
- AdaRMS time conditioning and dense modulation outputs precomputed once.
  370 dense cache hits/request. Actual normalization, affine modulation,
  gates, residuals and RoPE still execute online.

## Cache And Timing Methodology

**Warm steady-state, natural cache. Not cold-start, not forced hot-cache.**

Weights are loaded before measurement using the existing sequential readinto
loader. Each process performs 20 full inference warmups. No cache flush,
isolated kernel repetition, NCU replay, cold-DRAM assumption or memory-time
subtraction is used. Cache hits may avoid DRAM, cache misses still read it.
The measured kernel duration includes the memory accesses and stalls actually
incurred. It cannot be split into pure arithmetic and DRAM time by subtracting
bytes divided by peak bandwidth. The result is also not a measured GPU/PIM
offload latency: migration, coherency, dispatch and synchronization would need
to be measured for that system separately.

### Normal Process, No Profiler Injection

20 warmups, then 50 clean requests, 50 phase requests, 50 step-control
requests, 50 phase requests, 50 clean requests. CUDA events are recorded in
the normal stream; synchronization is at request boundaries only, not at
individual operator boundaries. API wall includes completion synchronization.
Outputs and cache coverage are checked after timing. Full model API timing
excludes model load, cache construction, host input preparation, sensor/network
and robot execution. No Nsight/NCU/PMU runs in this process.

Phase marker mean wall difference: +0.066%; clean before/after p50 drift:
+0.179%. Step events versus phase-marker mean wall difference: -0.474%,
consistent with the run-to-run variation, not a claimed speed improvement.

### Trace Process

Separate checkpoint load, 20 warmups, 20 pre-control requests, two traced
burn-ins, five traced requests, one separate attribution request, ten recovery
requests, 20 post-controls. Nsight Systems 2025.6.3 records CUDA and NVTX only;
CPU sampling, context-switch sampling, GPU metrics and PMU sampling are off.
The baseline process is separate because profiler injection can perturb even
same-process controls outside the active capture range.

The fifth denoising step gets NVTX module and ATen/shape labels. Kernels are
mapped via correlated CUDA launch IDs, not CPU annotation elapsed times.
Labels transfer only after exact kernel-name/order/grid/block/shared-memory
signature matching across all 50 measured steps. GPU start/end timestamps
determine durations. Interval unions prevent double counting overlaps.
RMS square/mean/rsqrt are separated from norm casts/affine operations;
projection casts are not counted as GEMM arithmetic.

Nsight reports unsupported hardware CUDA tracing, so software tracing is used.
Its generic diagnostics include `Not all NVTX/CUDA events might have been
collected`; those messages are retained in the JSON/SQLite. All selected step
kernel sequences and module coverage pass our audits, but these checks do not
prove every possible background event was collected.

## Validation And Resource Use

- Cache on/off and trace/baseline final actions are bitwise equal for the
  deterministic fixture. This is not robot task-success validation.
- Baseline recorded 278 GPU-PID checks; trace recorded 77. No other GPU compute
  process was observed. Checks are discrete, not continuous SoC isolation.
- At completion GPU process list is empty, utilization 0%, temperature 41 C.
- Thor had only about 1.1 GiB local free space. No downloads or cleanup were
  performed; bounded traces reused the authorized private runtime directory.
- Eight focused CPU analysis tests pass. Raw captures, trace, SQLite, per-step
  data, labels, source hashes and unfiltered timing samples are included.

## Files

- `figures/index.html`: local report, four breakdown figures and methodology audit.
- `figures/*.svg`, `*.png`, `*.pdf`: publication/edit-friendly figures.
- `analysis/summary.json`: complete statistics and provenance.
- `analysis/coarse.csv`, `operators.csv`, `steps.csv`: numerical tables.
- `analysis/operator_labels.json`, `first_step_kernels.json`: attribution evidence.
- `analysis/step_samples.json`: all 50 profiled step aggregates.
- `raw/*/capture.json`: unfiltered samples, occupancy, environment, output checks.
- `raw/*.nsys-rep`: original Nsight report.
- `raw/pi_tokens.json`: the fixed token IDs/mask needed to reproduce the input.
- `raw/*.sqlite.gz`: lossless compressed SQLite (the uncompressed SHA256 is in
  `analysis/summary.json`). Local/server working copies may retain `.sqlite`.
- `code/`: captured runtime adapters, loader, capture, analysis, rendering, tests.
- `MANIFEST.json`: sizes and SHA256s for the distributed files.

## Reproduction

Follow the server SKILL and verify Thor is idle before any GPU run. The shared
environment and weights are read-only. The launch script encodes original run
IDs; use new tags/output paths when repeating to preserve existing evidence.

```bash
bash code/run.sh baseline
bash code/run.sh trace
```

For archived data, decompress the SQLite into a separate analysis workspace
and place it alongside the two raw capture directories. Use the existing
main-cluster Python analysis environment inside a CPU Slurm allocation:

```bash
python -B -m unittest discover -s code -p 'test_*.py' -v
python -B code/analyze.py --raw raw --output analysis_reproduced
python -B code/render.py --root . --output figures_reproduced
```

The renderer reads `analysis/summary.json`; use the verified original summary
or point a separate reproduction root at the regenerated analysis. Analysis
and renderer deliberately reject an already-existing output directory.

Destination: `AlinaWXY/pim_vla_simulator`, branch
`thor-data/pi05-openpi-mha8-h16-s10-adarms-latency-20261105`.
Only the new `thor-data/pi05-openpi-mha8-h16-s10-adarms-20261105/` directory
is added. No simulator source or prior result is modified.

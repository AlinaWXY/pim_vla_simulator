#!/usr/bin/env bash
set -euo pipefail
ROOT=/var/tmp/shiyuhuang-vla-baseline
CODE="$ROOT/code/pi05_h16_latency_20261105"
cd "$ROOT"
hostname
pwd
export PYTHONPATH="$CODE:$ROOT/runtime/overlay_extra:$ROOT/runtime/overlay:$ROOT/runtime/openpi/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export HF_HOME="$ROOT/cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache/xdg"
export CUDA_CACHE_PATH="$ROOT/cache/cuda" TORCH_HOME="$ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TRITON_CACHE_DIR="$ROOT/cache/triton"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=1
source /var/tmp/vla-venv/bin/activate
which python
python -B -c 'import sys; print(sys.executable)'
case "${1:?baseline or trace}" in
  baseline)
    python -B "$CODE/capture.py" --mode baseline --tag pi05_h16_latency_baseline_20261105_01
    ;;
  trace)
    /usr/local/bin/nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
      --export=sqlite --output="$ROOT/runs/pi05_h16_latency_trace_20261105_01" \
      python -B "$CODE/capture.py" --mode trace --tag pi05_h16_latency_trace_20261105_01
    ;;
  *) exit 2 ;;
esac

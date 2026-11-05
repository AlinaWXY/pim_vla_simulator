#!/usr/bin/env bash
set -euo pipefail
ROOT=/var/tmp/shiyuhuang-vla-baseline
CODE="$ROOT/code/pi05_h16_cold_kernel_20261105"
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
exec python -B "$CODE/capture.py" "$@"

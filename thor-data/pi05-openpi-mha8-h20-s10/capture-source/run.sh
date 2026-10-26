#!/usr/bin/env bash
set -euo pipefail
ROOT=/var/tmp/shiyuhuang-vla-baseline
cd "$ROOT"
export PYTHONPATH="$ROOT/code/openpi_mha8_20261022:$ROOT/code/baselines_20261021:$ROOT/code/adarms_cache_20261020:$ROOT/code:$ROOT/runtime/overlay_extra:$ROOT/runtime/overlay:$ROOT/runtime/openpi/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export HF_HOME="$ROOT/cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache/xdg"
export CUDA_CACHE_PATH="$ROOT/cache/cuda" TORCH_HOME="$ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TRITON_CACHE_DIR="$ROOT/cache/triton"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENTRY=runner.py
if [[ "${1:-}" == "--mode" && "${2:-}" == "labels" ]]; then
    ENTRY=labels_driver.py
fi
exec /var/tmp/vla-venv/bin/python -B "$ROOT/code/openpi_mha8_20261022/$ENTRY" "$@"

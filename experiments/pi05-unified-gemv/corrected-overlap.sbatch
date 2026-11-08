#!/usr/bin/env bash
#SBATCH --job-name=1108-overlap
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=corrected-overlap-%j.log
set -euo pipefail
if [[ "${ALLOW_RETRACTED_IDLE_SERVICE:-0}" != 1 ]]; then
  echo 'Retired performance model: instantaneous idle service omits RD-to-future-DQ reservations.' >&2
  echo 'Only historical algorithm reproduction is available via ALLOW_RETRACTED_IDLE_SERVICE=1.' >&2
  exit 2
fi
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend:$PWD/experiments/pi05-unified-gemv/source:$PWD/experiments/pi05-unified-gemv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
out=reference/evaluation-1108/corrected-overlap
mkdir -p "$out"
"$PYTHON" -m unittest discover -s experiments/pi05-unified-gemv/validation -p test_idle_service.py -v > "$out/unit-tests.log" 2>&1
"$PYTHON" experiments/pi05-unified-gemv/analyze_corrected_overlap.py \
  --runs results/evaluation-1971385 --gpu reference/evaluation-1108/gpu --output "$out"
"$PYTHON" experiments/pi05-unified-gemv/verify_corrected_overlap.py --output "$out"
"$PYTHON" experiments/pi05-unified-gemv/publish_corrected_overlap.py --output "$out" --job "$SLURM_JOB_ID"
tar -czf corrected-overlap-results.tar.gz "$out"

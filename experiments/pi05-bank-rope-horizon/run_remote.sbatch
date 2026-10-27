#!/usr/bin/env bash
#SBATCH --job-name=pi05-horizon
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-3
set -euo pipefail
: "${PIM_CHECKOUT:?Set PIM_CHECKOUT}"
: "${PYTHON:?Set PYTHON to the native build interpreter}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PIM_CHECKOUT"
horizons=(16 20 30 40)
horizon="${horizons[${SLURM_ARRAY_TASK_ID}]}"
"$PYTHON" experiments/pi05-bank-rope-horizon/reproduce.py --horizon "$horizon" \
  --output "experiments/pi05-bank-rope-horizon/results/sweep-${SLURM_ARRAY_JOB_ID}/H${horizon}"

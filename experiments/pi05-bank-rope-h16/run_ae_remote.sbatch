#!/usr/bin/env bash
#SBATCH --job-name=pi05-ae-rank1
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${PIM_CHECKOUT:?Set PIM_CHECKOUT to the isolated candidate checkout}"
: "${PYTHON:?Set PYTHON to the interpreter used to build the native backend}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PIM_CHECKOUT"
"$PYTHON" experiments/pi05-bank-rope-h16/reproduce_ae.py \
  --output "experiments/pi05-bank-rope-h16/results/ae-${SLURM_JOB_ID:-local}"

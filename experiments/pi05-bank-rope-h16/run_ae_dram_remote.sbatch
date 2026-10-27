#!/usr/bin/env bash
#SBATCH --job-name=pi05-ae-zdram
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${PIM_CHECKOUT:?Set PIM_CHECKOUT}"
: "${PYTHON:?Set PYTHON to the native build interpreter}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JOBS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PIM_CHECKOUT"
bash experiments/pi05-bank-rope-h16/build.sh
"$PYTHON" experiments/pi05-bank-rope-h16/reproduce_ae_dram.py \
  --output "experiments/pi05-bank-rope-h16/results/ae-dram-${SLURM_JOB_ID:-local}"

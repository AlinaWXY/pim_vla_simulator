#!/bin/bash
#SBATCH --job-name=ffn-fused-hybrid
#SBATCH --partition=cpu-scavenger
#SBATCH --array=0-229%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --chdir=/fact_data/xinyaowang/lpddr-pim-simulator
#SBATCH --output=results/ffn-fused-hybrid-v1/logs/case_%A_%a_%j.log

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-pim/bin/python -m simulator.fused_hybrid run \
  --root results/ffn-fused-hybrid-v1 --index "$SLURM_ARRAY_TASK_ID"

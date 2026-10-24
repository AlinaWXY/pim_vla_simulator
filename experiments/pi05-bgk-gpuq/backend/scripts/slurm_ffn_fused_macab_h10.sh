#!/bin/bash
#SBATCH --job-name=ffn-fused-h10
#SBATCH --partition=cpu-scavenger
#SBATCH --array=0-24
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=results/ffn-fused-macab-h10-r6-r10/logs/case_%A_%a.log

set -euo pipefail

reuse_values=(6 7 8 9 10)
nccdab_values=(8 9 10 11 12)
reuse_index=$((SLURM_ARRAY_TASK_ID / 5))
nccdab_index=$((SLURM_ARRAY_TASK_ID % 5))
reuse=${reuse_values[$reuse_index]}
nccdab=${nccdab_values[$nccdab_index]}
output="results/ffn-fused-macab-h10-r6-r10/cases/r${reuse}-ck${nccdab}"

export OMP_NUM_THREADS=1
.venv-pim/bin/python scripts/run_ffn_fused_macab_case.py \
  --reuse "$reuse" --nccdab "$nccdab" --output "$output"

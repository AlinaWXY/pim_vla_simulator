#!/usr/bin/env bash
#SBATCH --job-name=oproj-guard-retry
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/fact_data/xinyaowang/pim_vla_o_projection_20261104
#SBATCH --output=guard-retry-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd /fact_data/xinyaowang/pim_vla_o_projection_20261104
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_frozen="$PWD/r3"
export LD_LIBRARY_PATH="$task_frozen/experiments/pi05-bgk-gpuq/backend/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python op-guard-retry.py \
  --frozen-root "$task_frozen" --original-suite "$task_frozen/runs/four-way-1924380" \
  --output "$PWD/guard-retry-${SLURM_JOB_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=pi05-1108-matrix
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=evaluation-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd "$SLURM_SUBMIT_DIR"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JOBS=4
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2/python:$PWD/experiments/pi05-bgk-gpuq/backend"
export LD_LIBRARY_PATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2"
if [[ "${EVAL_INCREMENTAL:-0}" == 1 ]]; then
  cmake --build experiments/pi05-bgk-gpuq/backend/ramulator2/build-unified-gemv --parallel 4
else
  bash experiments/pi05-unified-gemv/build.sh
fi
"$PYTHON" experiments/pi05-unified-gemv/reproduce_evaluation.py --output "results/evaluation-$SLURM_JOB_ID" "$@"

#!/usr/bin/env bash
#SBATCH --job-name=1108-checks
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=checks-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2/python:$PWD/experiments/pi05-bgk-gpuq/backend:$PWD/experiments/pi05-unified-gemv/source:$PWD/experiments/pi05-unified-gemv"
export LD_LIBRARY_PATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p reference/evaluation-1108/validation
"$PYTHON" -m unittest discover -s experiments/pi05-unified-gemv/validation -p 'test_*.py' -v > reference/evaluation-1108/validation/unit-tests.log 2>&1
"$PYTHON" experiments/pi05-unified-gemv/validation/evaluation_native_guards.py --output "results/native-guards-$SLURM_JOB_ID"
cp "results/native-guards-$SLURM_JOB_ID/validation.json" reference/evaluation-1108/validation/native-guards.json

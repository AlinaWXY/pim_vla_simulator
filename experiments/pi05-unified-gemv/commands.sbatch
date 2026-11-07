#!/bin/bash
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JOBS=4
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2/python:$PWD/experiments/pi05-bgk-gpuq/backend"
export LD_LIBRARY_PATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2"
bash experiments/pi05-unified-gemv/build.sh
"$PYTHON" experiments/pi05-unified-gemv/reproduce_commands.py --output "${OUTPUT:-results}" ${RUN_ARGS:-}

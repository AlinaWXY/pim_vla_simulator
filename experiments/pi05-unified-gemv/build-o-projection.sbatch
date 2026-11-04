#!/usr/bin/env bash
#SBATCH --job-name=oproj-build
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=build-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd "$SLURM_SUBMIT_DIR"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JOBS=4
bash experiments/pi05-unified-gemv/build.sh
"$PYTHON" experiments/pi05-unified-gemv/reproduce_o_projection.py --output "runs/smoke-${SLURM_JOB_ID}" --cases chain pv --skip-tests

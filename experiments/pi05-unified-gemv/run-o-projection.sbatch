#!/usr/bin/env bash
#SBATCH --job-name=oproj-four-way
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=run-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd "$SLURM_SUBMIT_DIR"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JOBS=4
bash experiments/pi05-unified-gemv/build.sh
"$PYTHON" experiments/pi05-unified-gemv/reproduce_o_projection.py --output "runs/four-way-${SLURM_JOB_ID}"
"$PYTHON" experiments/pi05-unified-gemv/reproduce.py --output "runs/legacy-${SLURM_JOB_ID}" --cases chain pv gate_up_gpu down_gpu_input serial_layer_control --skip-tests
"$PYTHON" experiments/pi05-unified-gemv/validate_o_projection_results.py \
  --experiment "runs/four-way-${SLURM_JOB_ID}" --legacy "runs/legacy-${SLURM_JOB_ID}" \
  --reference reference

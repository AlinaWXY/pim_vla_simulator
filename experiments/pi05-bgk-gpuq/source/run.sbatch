#!/bin/bash
#SBATCH --job-name=pi05-bgk-gpuq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=pi05-bgk-gpuq-%j.log
set -euo pipefail
# Submit from the experiment directory after activating the Python environment:
#   sbatch source/run.sbatch [reproduce.py arguments]
# Set PI05_EXPERIMENT_ROOT when submitting from another directory. Slurm copies
# this script into its spool, so its own path cannot locate the checkout.
experiment_root="${PI05_EXPERIMENT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
experiment_python="${PI05_PYTHON:-python}"
exec "$experiment_python" "$experiment_root/reproduce.py" --skip-build "$@"

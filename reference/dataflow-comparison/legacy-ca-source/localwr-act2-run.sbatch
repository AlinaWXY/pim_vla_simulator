#!/usr/bin/env bash
#SBATCH --job-name=localwr-act2-h16
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=run-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd "$SLURM_SUBMIT_DIR"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JOBS=4
bash experiments/pi05-unified-gemv/build.sh
"$PYTHON" check_rank_staging_sources.py
task_scratch=$(mktemp -d /tmp/pim-localwr-act2-${SLURM_JOB_ID}.XXXXXX)
trap 'mkdir -p runs; for mode in before corrected direct; do if test -d "$task_scratch/$mode"; then cp -a "$task_scratch/$mode" "runs/${mode}-${SLURM_JOB_ID}"; fi; done' EXIT
before_root=/fact_data/xinyaowang/pim_vla_act2_deadline_20261103
"$PYTHON" "$before_root/experiments/pi05-unified-gemv/reproduce.py" --output "$task_scratch/before" --staging-banks 2 --skip-tests
"$PYTHON" experiments/pi05-unified-gemv/reproduce.py --output "$task_scratch/corrected" --staging-banks 2
"$PYTHON" experiments/pi05-unified-gemv/reproduce.py --output "$task_scratch/direct" --direct-input-baseline --skip-tests
"$PYTHON" compare_localwr_act2.py --before "$task_scratch/before" --after "$task_scratch/corrected"

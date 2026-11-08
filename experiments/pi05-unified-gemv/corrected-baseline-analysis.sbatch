#!/usr/bin/env bash
#SBATCH --job-name=1108-flat-audit
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=corrected-baseline-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2/python:$PWD/experiments/pi05-bgk-gpuq/backend:$PWD/experiments/pi05-unified-gemv/source:$PWD/experiments/pi05-unified-gemv"
export LD_LIBRARY_PATH="$PWD/experiments/pi05-bgk-gpuq/backend/ramulator2"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
native="$1"
out=reference/evaluation-1108/direct-before-corrected
mkdir -p "$out"
if [[ "${2:-}" != checks-only ]]; then
  "$PYTHON" experiments/pi05-unified-gemv/collect_corrected_baseline.py --input "$native" --original results/evaluation-1971246 --output "$out"
  "$PYTHON" experiments/pi05-unified-gemv/audit_linear_ca.py --input "$native" --output "$out/linear-ca" --no-plot
fi
"$PYTHON" -m unittest discover -s experiments/pi05-unified-gemv/validation -p 'test_*.py' -v > "$out/unit-tests.log" 2>&1
"$PYTHON" experiments/pi05-unified-gemv/validation/evaluation_native_guards.py --output "results/corrected-native-guards-$SLURM_JOB_ID"
cp "results/corrected-native-guards-$SLURM_JOB_ID/validation.json" "$out/native-guards.json"
tar -czf direct-before-corrected-results.tar.gz "$out"

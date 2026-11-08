#!/usr/bin/env bash
#SBATCH --job-name=1108-native-timing-fix
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=reference/evaluation-1108/native-timing-fix/job-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_PY=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
TASK_SRC=experiments/pi05-unified-gemv
TASK_OUT=reference/evaluation-1108/native-timing-fix
mkdir -p "$TASK_OUT"
g++ -std=c++20 -O2 -shared -fPIC "$TASK_SRC/sequential_transactions.cpp" -o "$TASK_OUT/libsequential.so"
UNIFORM_LIBRARY="$PWD/$TASK_OUT/libsequential.so" NATIVE_TIMING_AUDIT="$PWD/$TASK_OUT/native-transfer-matrix.json" \
  "$TASK_PY" -m unittest discover -s "$TASK_SRC/validation" -p 'test_*transactions.py' -v > "$TASK_OUT/unit-tests.log" 2>&1
"$TASK_PY" "$TASK_SRC/run_sequential_transactions.py" --library "$TASK_OUT/libsequential.so" --output "$TASK_OUT/with-rows"

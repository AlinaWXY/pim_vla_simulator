#!/usr/bin/env bash
#SBATCH --job-name=1108-sequential-rows
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=sequential-rows-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
src=experiments/pi05-unified-gemv
out=reference/evaluation-1108/sequential-rows
mkdir -p "$out"
g++ -std=c++20 -O2 -shared -fPIC "$src/sequential_transactions.cpp" -o "$out/libsequential.so"
UNIFORM_LIBRARY="$PWD/$out/libsequential.so" "$PYTHON" -m unittest discover -s "$src/validation" -p 'test_*transactions.py' -v > "$out/unit-tests.log" 2>&1
"$PYTHON" "$src/run_sequential_transactions.py" --row-hit-control --output "$out/row-hit-control"
"$PYTHON" "$src/run_sequential_transactions.py" --output "$out/with-rows"

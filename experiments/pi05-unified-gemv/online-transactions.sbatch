#!/usr/bin/env bash
#SBATCH --job-name=1108-online-bus
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=online-bus-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
src=experiments/pi05-unified-gemv
out=reference/evaluation-1108/online-arbitration-preparation
g++ -std=c++20 -O2 -shared -fPIC "$src/online_transactions.cpp" -o "$out/libonline.so"
UNIFORM_LIBRARY="$PWD/$out/libonline.so" "$PYTHON" -m unittest discover -s "$src/validation" -p 'test_*transactions.py' -v > "$out/unit-tests.log" 2>&1
"$PYTHON" "$src/prepare_online_program.py" --runs results/evaluation-1971385 --output "$out/native-programs.json.gz"
"$PYTHON" "$src/run_online_transactions.py" --output "$out/evaluation-isolated"

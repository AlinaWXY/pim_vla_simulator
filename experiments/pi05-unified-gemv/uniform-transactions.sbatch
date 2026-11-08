#!/usr/bin/env bash
#SBATCH --job-name=1108-uniform
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=uniform-stream-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
src=experiments/pi05-unified-gemv
root=reference/evaluation-1108
out=$root/uniform-transactions
mkdir -p "$out"
g++ -std=c++20 -O2 -shared -fPIC "$src/uniform_transactions.cpp" -o "$out/libuniform-stream.so"
UNIFORM_LIBRARY="$PWD/$out/libuniform-stream.so" "$PYTHON" -m unittest discover -s "$src/validation" -p test_uniform_transactions.py -v > "$out/unit-tests.log" 2>&1
"$PYTHON" "$src/run_uniform_transactions.py" --reference "$root" --demand "$root/gpu/uniform-memory-demand.json" --bus "$out/native-reservations.json.gz" --library "$out/libuniform-stream.so" --output "$out/stream-reference" --stream-reference
"$PYTHON" "$src/publish_uniform_transactions.py" --root "$root" --job "$SLURM_JOB_ID"

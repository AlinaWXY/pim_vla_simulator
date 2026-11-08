#!/usr/bin/env bash
#SBATCH --job-name=1108-request-model
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=request-model-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p results/request-model reference/evaluation-1108/request-model
g++ -std=c++20 -O3 -fPIC -shared experiments/pi05-unified-gemv/request_replay.cpp -o results/request-model/librequest_replay.so
export REQUEST_REPLAY_LIBRARY="$PWD/results/request-model/librequest_replay.so"
"$PYTHON" -m unittest discover -s experiments/pi05-unified-gemv/validation -p test_request_model.py -v > reference/evaluation-1108/request-model/tests.log 2>&1
if [[ ! -f reference/evaluation-1108/request-model/refresh-events.json ]]; then
  "$PYTHON" experiments/pi05-unified-gemv/prepare_request_refresh.py --runs results/evaluation-1971246 --output reference/evaluation-1108/request-model/refresh-events.json
fi
"$PYTHON" experiments/pi05-unified-gemv/run_request_model.py --reference reference/evaluation-1108 --library "$REQUEST_REPLAY_LIBRARY" "$@"

#!/usr/bin/env bash
# Build only the backend shipped with this experiment; no remote source fetches.
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend_root="$experiment_root/backend"
experiment_python="${PYTHON:-python3}"
experiment_python="$("$experiment_python" -c 'import sys; print(sys.executable)')"
build_jobs="${JOBS:-2}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
cmake -S "$backend_root/ramulator2" -B "$backend_root/ramulator2/build-reproduce" \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$experiment_python" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$backend_root/ramulator2/ext/fmt" \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$backend_root/ramulator2/ext/yaml-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$backend_root/ramulator2/ext/nanobind"
cmake --build "$backend_root/ramulator2/build-reproduce" --parallel "$build_jobs"
PYTHONPATH="$backend_root/ramulator2/python:$backend_root" \
LD_LIBRARY_PATH="$backend_root/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$experiment_python" -c 'import ramulator, simulator.backend; print("Native backend imports successfully")'

#!/usr/bin/env bash
# Build the opt-in bank-local RoPE backend in its own build directory.
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend_root="$(cd -- "$experiment_root/../pi05-bgk-gpuq/backend" && pwd)"
experiment_python="${PYTHON:-python3}"
experiment_python="$("$experiment_python" -c 'import sys; print(sys.executable)')"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cmake -S "$backend_root/ramulator2" -B "$backend_root/ramulator2/build-bank-rope" \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$experiment_python" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$backend_root/ramulator2/ext/fmt" \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$backend_root/ramulator2/ext/yaml-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$backend_root/ramulator2/ext/nanobind"
cmake --build "$backend_root/ramulator2/build-bank-rope" --parallel "${JOBS:-4}"
PYTHONPATH="$backend_root/ramulator2/python:$backend_root" \
LD_LIBRARY_PATH="$backend_root/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$experiment_python" -c 'import ramulator, simulator.backend; print("Bank-RoPE backend imports successfully")'

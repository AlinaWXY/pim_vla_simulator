#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend_root="${BACKEND:-$experiment_root/../pi05-bgk-gpuq/backend}"
experiment_python="${PYTHON:-python3}"
experiment_python="$("$experiment_python" -c 'import sys; print(sys.executable)')"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cmake -S "$backend_root/ramulator2" -B "$backend_root/ramulator2/build-unified-gemv" \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$experiment_python" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$backend_root/ramulator2/ext/fmt" \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$backend_root/ramulator2/ext/yaml-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND="$backend_root/ramulator2/ext/nanobind"
cmake --build "$backend_root/ramulator2/build-unified-gemv" --parallel "${JOBS:-4}"
PYTHONPATH="$backend_root/ramulator2/python:$backend_root" \
LD_LIBRARY_PATH="$backend_root/ramulator2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$experiment_python" -c 'import ramulator; from simulator.config import MemoryConfig; assert MemoryConfig(pim=True,pim_unified_gemv=True,channels=16,ranks=2,pim_input_broadcast=False,pim_mac_scope="rank",pim_data_path="attacc").pim_unified_gemv; print("Unified GEMV backend imports successfully")'

#!/usr/bin/env bash
# Reproduce small capability probes on Thor. This does not collect Pi0.5.
set -euo pipefail
probe_dir=${TRACE_PROBE_DIR:-/var/tmp/xinyao-1108-trace-feasibility}
nvbit_lib=${NVBIT_MEMTRACE_LIB:-/var/tmp/xinyao-1108-transaction-evidence/build-memtrace/mem_trace.so}
torch_python=${TORCH_PYTHON:-/var/tmp/vla-venv/bin/python}
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
test -f "$nvbit_lib"
python3 - <<'PY'
import subprocess
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not [x for x in active.splitlines() if x.strip().isdigit()], active
PY
mkdir -p "$probe_dir/pm_sampling"
cp "$source_dir/address_probe.cu" "$source_dir/torch_address_probe.py" "$probe_dir/"
cd "$probe_dir"
/usr/local/cuda/bin/nvcc -arch=sm_110 -cudart shared -O2 -lineinfo address_probe.cu -o address_probe_shared
env PATH=/usr/local/cuda/bin:"$PATH" LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  CUDA_CACHE_DISABLE=1 CUDA_INJECTION64_PATH="$nvbit_lib" TOOL_VERBOSE=1 \
  timeout 20s ./address_probe_shared > injected-shared.log 2>&1
env PATH=/usr/local/cuda/bin:"$PATH" PYTHONDONTWRITEBYTECODE=1 \
  CUDA_CACHE_DISABLE=1 CUDA_INJECTION64_PATH="$nvbit_lib" \
  timeout 40s "$torch_python" torch_address_probe.py > torch-address.log 2>&1
cp /usr/local/cuda/extras/CUPTI/samples/pm_sampling/{pm_sampling.cu,pm_sampling.h,Makefile} pm_sampling/
make -C pm_sampling CUDA_INSTALL_PATH=/usr/local/cuda SMS=110 > pm-build.log 2>&1
env LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 20s ./pm_sampling/pm_sampling \
  --device 0 --queryBaseMetrics > pm-metrics.log 2>&1
env LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 20s ./pm_sampling/pm_sampling \
  --device 0 --metrics gpc__cycles_elapsed.avg --maxsamples 1024 > pm-cycles.log 2>&1
env LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 20s ./pm_sampling/pm_sampling \
  --device 0 --metrics lts__t_sectors.sum --maxsamples 1024 > pm-l2-sectors.log 2>&1
python3 "$source_dir/validate_probes.py" "$probe_dir"

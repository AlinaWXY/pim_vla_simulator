#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=$(dirname -- "$repo")
branch=$(git -C "$repo" branch --show-current)
case "$branch" in
  lpddr5-6400) envdir="$project/.venv-6400"; standard=LPDDR5 ;;
  lpddr5x-8533) envdir="$project/.venv-8533"; standard=LPDDR5X ;;
  *) echo "Unsupported verification branch: $branch" >&2; exit 1 ;;
esac
if [[ "$repo" == /fact_data/* ]]; then
  [[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Use a Slurm CPU allocation.' >&2; exit 1; }
  source "$envdir/bin/activate"
  export TMPDIR=/scratch/shiyuhuang/lpddr5x-simulator/tmp
  export XDG_CACHE_HOME=/scratch/shiyuhuang/lpddr5x-simulator/cache
  export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
  export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
  compiler=/opt/rh/gcc-toolset-14/root/usr/bin/g++
  run="$project/runs/$branch/$(date +%Y%m%d-%H%M%S)-$$"
else
  [[ -n ${VIRTUAL_ENV:-} ]] || { echo 'Activate a Python >=3.11 environment.' >&2; exit 1; }
  compiler=${CXX:-$(command -v g++)}
  run="$repo/results/verify-$(date +%Y%m%d-%H%M%S)-$$"
fi
mkdir -p "$run"
cd "$repo"
hostname
pwd
python -m pip install -e ./ramulator2 -e . > "$run/install.log" 2>&1
python -c 'import pathlib,ramulator,simulator,sys; print(sys.executable); root=pathlib.Path.cwd(); assert pathlib.Path(ramulator.__file__).resolve().is_relative_to(root/"ramulator2"); assert pathlib.Path(simulator.__file__).resolve().is_relative_to(root)'
cmake -S ramulator2 -B ramulator2/build -DCMAKE_CXX_COMPILER="$compiler" \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCLANG_FORMAT="$VIRTUAL_ENV/bin/clang-format" > "$run/configure.log" 2>&1
cmake --build ramulator2/build -j "${SLURM_CPUS_PER_TASK:-2}" > "$run/build.log" 2>&1
python -m pytest tests -q -s --basetemp="${TMPDIR:-$run}/interface-$(basename "$run")" \
  --junitxml="$run/interface-tests.xml" > "$run/interface-tests.log" 2>&1
(
  cd ramulator2
  python -m pytest -q -s tests/calibration tests/smoke tests/device_timings \
    tests/controller_scheduling tests/unit_tests tests/test_LPDDR5_params.py tests/test_REFpb.py \
    --basetemp="${TMPDIR:-$run}/backend-$(basename "$run")" \
    --junitxml="$run/backend-tests.xml"
) > "$run/backend-tests.log" 2>&1
python -m simulator run --config configs/default.yaml --trace traces/smoke.trace \
  --output "$run/smoke" > "$run/smoke.log" 2>&1
python scripts/check_memory.py --standard "$standard" --output "$run/bandwidth.json" \
  > "$run/bandwidth.log" 2>&1
git rev-parse HEAD > "$run/commit.txt"
python -m pip freeze > "$run/environment.txt"
tail -2 "$run/interface-tests.log"
tail -2 "$run/backend-tests.log"
cat "$run/smoke.log"
echo "Results: $run"

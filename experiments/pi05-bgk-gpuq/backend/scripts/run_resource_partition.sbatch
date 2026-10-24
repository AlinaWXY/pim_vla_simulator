#!/usr/bin/env bash
#SBATCH --job-name=lpddr-partition
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/resource-partition-%j.log
set -euo pipefail

# Submit from the repository root, after creating results/.
repo="$SLURM_SUBMIT_DIR"
cd "$repo"
source "$repo/.venv-pim/bin/activate"
export TMPDIR="/scratch/$USER/lpddr-partition-$SLURM_JOB_ID/tmp"
export XDG_CACHE_HOME="/scratch/$USER/lpddr-partition-$SLURM_JOB_ID/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
run_dir="$repo/results/resource-partition-$SLURM_JOB_ID"
mkdir "$run_dir"
cmake -S ramulator2 -B ramulator2/build \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-14/root/usr/bin/g++ \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCLANG_FORMAT="$VIRTUAL_ENV/bin/clang-format" > "$run_dir/configure.log" 2>&1
cmake --build ramulator2/build -j "$SLURM_CPUS_PER_TASK" > "$run_dir/build.log" 2>&1
python -m pytest tests -q --basetemp="$TMPDIR/interface" \
  --junitxml="$run_dir/interface-tests.xml" > "$run_dir/interface-tests.log" 2>&1
(
  cd ramulator2
  python -m pytest -q tests/calibration tests/smoke tests/device_timings \
    tests/controller_scheduling tests/unit_tests tests/test_LPDDR5_params.py tests/test_REFpb.py \
    --basetemp="$TMPDIR/backend" --junitxml="$run_dir/backend-tests.xml"
) > "$run_dir/backend-tests.log" 2>&1
python -m simulator.resource_partition --output "$run_dir/experiment" --diagnostics \
  > "$run_dir/experiment.log" 2>&1
python -m simulator.resource_partition --output "$run_dir/refresh-on" --refresh \
  > "$run_dir/refresh-on.log" 2>&1
printf 'Results: %s\n' "$run_dir"

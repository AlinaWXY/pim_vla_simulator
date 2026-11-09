#!/usr/bin/env bash
set -euo pipefail
: "${PIM_RUN:?}" "${PIM_PHASE:?}"
case "$PIM_PHASE" in repeat|finalize) ;; *) echo "Invalid PIM_PHASE: $PIM_PHASE" >&2; exit 2;; esac
source /opt/rh/gcc-toolset-14/enable
export LD_LIBRARY_PATH="$PIM_RUN/frozen/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIM_VLA_SOURCE_COMMIT="$(cat "$PIM_RUN/frozen/SOURCE_COMMIT")"
python_exec=$(cat "$PIM_RUN/python-path")
(cd "$PIM_RUN" && sha256sum -c frozen/runtime-SHA256SUMS --quiet && sha256sum -c frozen/config-SHA256SUMS --quiet)
(cd "$PIM_RUN/frozen/source" && sha256sum -c ../source-SHA256SUMS --quiet)
cd "$PIM_RUN/frozen/source"
export PYTHONPATH="$PWD/ramulator2/python"
exec "$python_exec" "${PIM_PARALLEL_SCRIPT:-$PIM_RUN/jobs/parallel_cases.py}" \
  --run "$PIM_RUN" --phase "$PIM_PHASE" --index "${SLURM_ARRAY_TASK_ID:-0}"

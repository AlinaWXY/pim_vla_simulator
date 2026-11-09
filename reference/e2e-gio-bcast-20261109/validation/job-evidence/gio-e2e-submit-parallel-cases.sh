#!/usr/bin/env bash
set -euo pipefail
run=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224/results/e2e-gio-bcast-calibrated
test ! -e "$run/cases"
test ! -f "$run/jobs/parallel-cases.id"
case_id=$(sbatch --parsable --partition=dev --ntasks=1 --cpus-per-task=1 --mem=8G --time=11:55:00 \
  --array=0-17%18 --job-name=gio-repeat --output="$run/logs/repeat-%A_%a.log" \
  --export="ALL,PIM_RUN=$run,PIM_PHASE=repeat" "$run/jobs/parallel-worker.sbatch")
printf '%s\n' "$case_id" > "$run/jobs/parallel-cases.id"
printf '%s\n' "$case_id" > "$run/jobs/cases.id"
finalize_id=$(sbatch --parsable --partition=dev --ntasks=1 --cpus-per-task=1 --mem=12G --time=00:40:00 \
  --dependency="afterok:$case_id" --job-name=gio-finalize --output="$run/logs/finalize-%A_%a.log" \
  --export="ALL,PIM_RUN=$run,PIM_PHASE=finalize" "$run/jobs/parallel-worker.sbatch")
printf '%s\n' "$finalize_id" > "$run/jobs/aggregate.id"
audit_id=$(sbatch --parsable --dependency="afterok:$finalize_id" /fact_data/xinyaowang/gio-e2e-final-audit.sbatch)
printf '%s\n' "$audit_id" > "$run/jobs/audit.id"
printf 'REPEATS=%s\nFINALIZE=%s\nAUDIT=%s\n' "$case_id" "$finalize_id" "$audit_id"

#!/usr/bin/env bash
set -euo pipefail
run=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224/results/e2e-gio-bcast-calibrated
test -f "$run/jobs/worker.sbatch"
test ! -f "$run/jobs/solo.id"
submit_job() {
  local phase=$1; shift
  sbatch --parsable --partition=dev --ntasks=1 --cpus-per-task=1 --mem=8G --time=11:55:00 \
    --job-name="gio-$phase" --output="$run/logs/$phase-%A_%a.log" \
    --export="ALL,PIM_RUN=$run,PIM_PHASE=$phase" "$@" "$run/jobs/worker.sbatch"
}
solo_id=$(submit_job solo --array=0-4%5)
printf '%s\n' "$solo_id" > "$run/jobs/solo.id"
calibration_id=$(submit_job calibrate --array=0-5%6)
printf '%s\n' "$calibration_id" > "$run/jobs/calibration.id"
collect_id=$(submit_job collect --dependency="afterok:$solo_id:$calibration_id")
printf '%s\n' "$collect_id" > "$run/jobs/collect.id"
printf 'AE_SOLOS=%s\nVLM_CALIBRATION=%s\nCOLLECT=%s\n' "$solo_id" "$calibration_id" "$collect_id"

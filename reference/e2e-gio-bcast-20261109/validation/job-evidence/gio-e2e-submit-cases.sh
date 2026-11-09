#!/usr/bin/env bash
set -euo pipefail
run=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224/results/e2e-gio-bcast-calibrated
python_exec=$(cat "$run/python-path")
test ! -s "$run/jobs/cases.id"
"$python_exec" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["validation"]["passed"]; print(c["parameters"], c["validation"])' "$run/calibration/calibration.json"
for variant in feature_gpu_before token_gpu_before token_local_before token_gpu_after token_local_after; do
  test -s "$run/solos/ae-solo/$variant/provenance.json"
done
# The completed calibration dependency aged out of Slurm's active job table.
# Verify its frozen artifacts above and submit against the current inputs.
case_id=$(sbatch --parsable --partition=dev --ntasks=1 --cpus-per-task=1 --mem=8G --time=11:55:00 \
  --array=0-8%9 --job-name=gio-case --output="$run/logs/case-%A_%a.log" \
  --export="ALL,PIM_RUN=$run,PIM_PHASE=case" "$run/jobs/worker.sbatch")
printf '%s\n' "$case_id" > "$run/jobs/cases.id"
aggregate_id=$(sbatch --parsable --partition=dev --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
  --dependency="afterok:$case_id" --job-name=gio-aggregate --output="$run/logs/aggregate-%A_%a.log" \
  --export="ALL,PIM_RUN=$run,PIM_PHASE=aggregate" "$run/jobs/worker.sbatch")
printf '%s\n' "$aggregate_id" > "$run/jobs/aggregate.id"
printf 'CASES=%s\nAGGREGATE=%s\n' "$case_id" "$aggregate_id"

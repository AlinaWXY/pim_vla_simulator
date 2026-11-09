#!/usr/bin/env bash
#SBATCH --job-name=gio-final-audit
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/fact_data/xinyaowang/gio-e2e-final-audit-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
root=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224
run="$root/results/e2e-gio-bcast-calibrated"
audit=/fact_data/xinyaowang/gio-e2e-audit-20261109-112224
python_exec=$(cat "$run/python-path")
export LD_LIBRARY_PATH="$run/frozen/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"$python_exec" "$run/frozen/source/tools/audit_campaign.py" "$run" \
  --expected-source 6bcd57b073e86674f5abec24eee56ecde8252f30 --git-repo "$root" --output "$run/audit-campaign.json"
"$python_exec" "$audit/analyze_calibration.py" "$run" --output "$run/calibration-diagnostics"
mkdir -p "$run/validation"
cp "$audit/gio-window-validation.jsonl" "$audit/negative-sequencer.log" "$audit/gpu-arbitration-fix.patch" \
  "$audit/native_gio_concurrency.cpp" "$audit/build-and-run.sh" "$run/validation/"
cp /fact_data/xinyaowang/gio-e2e-validate-1981253.log /fact_data/xinyaowang/gio-e2e-prepare-1981261.log "$run/validation/"
git -C "$root" bundle create "$run/validation/source-6bcd57b.bundle" codex/gio-bcast-e2e-20261109
printf 'FULL_CAMPAIGN_AUDIT_PASSED\n'
archive=/fact_data/xinyaowang/gio-e2e-calibrated-6bcd57b.tar.gz
test ! -e "$archive"
tar -C "$root/results" -czf "$archive.partial" e2e-gio-bcast-calibrated
mv "$archive.partial" "$archive"
sha256sum "$archive" > "$archive.sha256"
printf 'ARCHIVE=%s\n' "$archive"

#!/usr/bin/env bash
#SBATCH --job-name=gio-e2e-prepare
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/fact_data/xinyaowang/gio-e2e-prepare-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
root=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224
audit=/fact_data/xinyaowang/gio-e2e-audit-20261109-112224
cd "$root"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"$audit/native_gio_concurrency" reference/spec/spec.json | tee "$audit/gio-window-validation.jsonl"
git diff --check
git diff HEAD --binary > "$audit/gpu-arbitration-fix.patch"
git add ramulator2/src/ramulator/controller/impl/pim_controller.cpp tests/native_sequencer_tests.cpp
git -c user.name=Codex -c user.email=codex@openai.com commit -m 'Skip owner-blocked GPU bank candidates before native arbitration'
"$PYTHON" programs/gen_pim_program.py --output programs/generated --compact
"$PYTHON" tests/run_pim_smoke.py
"$PYTHON" tests/run_concurrency_smoke.py
PIM_PARTITION=dev bash tools/run-campaign.sh prepare "$PWD/results/e2e-gio-bcast-calibrated"
printf 'FIX_VALIDATED_AND_CAMPAIGN_PREPARED\n'
bash /fact_data/xinyaowang/gio-e2e-submit-initial.sh

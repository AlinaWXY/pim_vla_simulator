#!/usr/bin/env bash
#SBATCH --job-name=gio-e2e-validate
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/fact_data/xinyaowang/gio-e2e-validate-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
root=/fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224
audit=/fact_data/xinyaowang/gio-e2e-audit-20261109-112224
cd "$root"
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
git show 98a6671:ramulator2/src/ramulator/controller/impl/pim_controller.cpp > "$audit/baseline-pim_controller.cpp"
g++ -std=c++20 -O2 -I"$root/ramulator2/src" tests/native_sequencer_tests.cpp \
  "$audit/baseline-pim_controller.cpp" \
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp \
  ramulator2/src/ramulator/plugin/bus_checker.cpp \
  -L"$root/ramulator2" -Wl,-rpath,"$root/ramulator2" -lramulator -o "$audit/negative-sequencer"
if "$audit/negative-sequencer" reference/spec/spec.json > "$audit/negative-sequencer.log" 2>&1; then
  echo 'ERROR: regression did not reproduce on the baseline' >&2
  exit 1
fi
"$PYTHON" -c 'import pathlib,sys; s=pathlib.Path(sys.argv[1]).read_text(); assert "GPU owner-blocked ACT1 hid another bank" in s,s; print("BASELINE_BUG_REPRODUCED:",s.strip())' "$audit/negative-sequencer.log"
cmake --build build -j8
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
build/native_spec_tests reference/spec/spec.json reference/spec/command-pair-matrix.json
build/native_microbench reference/spec/spec.json
build/native_lookahead_tests reference/spec/spec.json
build/native_sequencer_tests reference/spec/spec.json
build/native_refresh_tests reference/spec/spec.json
build/test_pim_resources
"$PYTHON" tests/test_fault_injection.py
"$PYTHON" tests/smoke_native.py
bash "$audit/build-and-run.sh" "$root" "$audit" | tee "$audit/gio-window-validation.jsonl"
git diff --check
git add ramulator2/src/ramulator/controller/impl/pim_controller.cpp tests/native_sequencer_tests.cpp
git commit -m 'Skip owner-blocked GPU bank candidates before native arbitration'
"$PYTHON" programs/gen_pim_program.py --output programs/generated --compact
"$PYTHON" tests/run_pim_smoke.py
"$PYTHON" tests/run_concurrency_smoke.py
PIM_PARTITION=dev bash tools/run-campaign.sh prepare "$PWD/results/e2e-gio-bcast-calibrated"
printf 'FIX_VALIDATED_AND_CAMPAIGN_PREPARED\n'

#!/usr/bin/env bash
#SBATCH --job-name=gio-e2e-build
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/fact_data/xinyaowang/gio-e2e-build-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd /fact_data/xinyaowang/pim_vla_gio_bcast_e2e_20261109-112224
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$PWD/ramulator2/ext/yaml-cpp" -DFETCHCONTENT_SOURCE_DIR_FMT="$PWD/ramulator2/ext/fmt"
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
"$PYTHON" programs/gen_pim_program.py --output programs/generated --compact
"$PYTHON" tests/run_pim_smoke.py
"$PYTHON" tests/run_concurrency_smoke.py
PIM_PARTITION=dev bash tools/run-campaign.sh prepare "$PWD/results/e2e-gio-bcast"
printf 'BUILD_VALIDATED_AND_CAMPAIGN_PREPARED\n'

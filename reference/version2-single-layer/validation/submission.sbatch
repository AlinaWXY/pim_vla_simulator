#!/usr/bin/env bash
#SBATCH --job-name=pim-v2-one-layer
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=version2-single-layer-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd /fact_data/xinyaowang/pim_vla_sim2_20261109
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PIM_VLA_SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=no)"
RUNROOT="results/version2-single-layer-$SLURM_JOB_ID"
mkdir -p "$RUNROOT/validation" "$RUNROOT/frozen/bin" "$RUNROOT/frozen/lib"
git diff --quiet 292cfbc44dd5b1367760329f3a064782763dd3bf HEAD -- ramulator2 CMakeLists.txt tools/benchmark_program_input.cpp
cmp build/pim_vla_sim results/version2-single-layer-1977130/frozen/bin/pim_vla_sim
cmp ramulator2/libramulator.so results/version2-single-layer-1977130/frozen/lib/libramulator.so
echo 292cfbc44dd5b1367760329f3a064782763dd3bf > "$RUNROOT/validation/native-clean-build-source.txt"
git diff --name-only 292cfbc44dd5b1367760329f3a064782763dd3bf HEAD > "$RUNROOT/validation/changed-since-native-build.txt"
echo "Reusing byte-identical native runtime from the validated clean build; only generator dependencies changed."
cmake --build build -j4
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v > "$RUNROOT/validation/python-tests.txt" 2>&1
build/native_spec_tests reference/spec/spec.json reference/spec/command-pair-matrix.json > "$RUNROOT/validation/native-spec.txt"
build/native_microbench reference/spec/spec.json > "$RUNROOT/validation/native-microbench.txt"
build/native_lookahead_tests reference/spec/spec.json > "$RUNROOT/validation/native-lookahead.txt"
build/native_sequencer_tests reference/spec/spec.json > "$RUNROOT/validation/native-sequencer.txt"
build/test_pim_resources > "$RUNROOT/validation/resources.txt"
"$PYTHON" tests/emit_resources_fixture.py "$RUNROOT/validation/resource-fixture.txt"
build/test_pim_resources "$RUNROOT/validation/resource-fixture.txt" >> "$RUNROOT/validation/resources.txt"
"$PYTHON" tests/test_fault_injection.py > "$RUNROOT/validation/fault-injection.txt" 2>&1
"$PYTHON" programs/gen_pim_program.py --output "$RUNROOT/frozen/programs" --compact > "$RUNROOT/validation/program-generation.jsonl"
"$PYTHON" tools/verify_program_input.py --binary build/benchmark_program_input "$RUNROOT/frozen/programs/token_local_after.json" > "$RUNROOT/validation/json-input.json"
cp build/pim_vla_sim "$RUNROOT/frozen/bin/"
cp ramulator2/libramulator.so "$RUNROOT/frozen/lib/"
cp reference/spec/spec.json "$RUNROOT/frozen/spec.json"
export LD_LIBRARY_PATH="$PWD/$RUNROOT/frozen/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PYTHON" experiments/run_single_layer.py --binary "$RUNROOT/frozen/bin/pim_vla_sim" --program-dir "$RUNROOT/frozen/programs" --spec "$RUNROOT/frozen/spec.json" --output "$RUNROOT/experiment" --requirements results/task-documents/version2-rewrite.docx --repeats 2
test -z "$(git status --porcelain --untracked-files=no)"
echo "$RUNROOT"

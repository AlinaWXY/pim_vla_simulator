#!/usr/bin/env bash
#SBATCH --job-name=pim-rank-refresh
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=rank-refresh-single-layer-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd /fact_data/xinyaowang/pim_vla_sim2_20261109
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PIM_VLA_SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=no)"
RUNROOT="results/rank-refresh-single-layer-$SLURM_JOB_ID"
mkdir -p "$RUNROOT/validation" "$RUNROOT/frozen/bin" "$RUNROOT/frozen/lib"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first -j4
echo "$PIM_VLA_SOURCE_COMMIT" > "$RUNROOT/validation/native-clean-build-source.txt"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v > "$RUNROOT/validation/python-tests.txt" 2>&1
build/native_spec_tests reference/spec/spec.json reference/spec/command-pair-matrix.json > "$RUNROOT/validation/native-spec.txt"
build/native_microbench reference/spec/spec.json > "$RUNROOT/validation/native-microbench.txt"
build/native_lookahead_tests reference/spec/spec.json > "$RUNROOT/validation/native-lookahead.txt"
build/native_sequencer_tests reference/spec/spec.json > "$RUNROOT/validation/native-sequencer.txt"
build/native_refresh_tests reference/spec/spec.json > "$RUNROOT/validation/native-refresh.txt"
build/test_pim_resources > "$RUNROOT/validation/resources.txt"
"$PYTHON" tests/test_fault_injection.py > "$RUNROOT/validation/fault-injection.txt" 2>&1
# Hold programs, workload measurements and the spec constant across the refresh correction.
git diff --exit-code f7c7af6eab0655b526773b9b15ed63e8e6f73add HEAD -- programs workloads input-data ramulator2/python/ramulator/dram/lpddr5x_pim.py reference/spec > "$RUNROOT/validation/input-source-diff.txt"
cp -a results/version2-single-layer-1977315/frozen/programs "$RUNROOT/frozen/programs"
# Preserve the original generation log as evidence for these exact reused bytes.
cp results/version2-single-layer-1977315/validation/program-generation.jsonl "$RUNROOT/validation/program-generation.jsonl"
cp build/pim_vla_sim "$RUNROOT/frozen/bin/"
cp ramulator2/libramulator.so "$RUNROOT/frozen/lib/"
cp reference/spec/spec.json "$RUNROOT/frozen/spec.json"
export LD_LIBRARY_PATH="$PWD/$RUNROOT/frozen/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PYTHON" - "$RUNROOT" <<'PY'
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); old=pathlib.Path('results/version2-single-layer-1977315')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((old/'frozen/manifest.json').read_text())
for p in (root/'frozen/programs').glob('*.json'):
    assert sha(p)==manifest['files']['programs/'+p.name]['sha256']
assert sha(root/'frozen/spec.json')==manifest['files']['spec.json']['sha256']
(root/'validation/input-reuse.json').write_text(json.dumps(dict(previous_source='f7c7af6eab0655b526773b9b15ed63e8e6f73add',previous_run=str(old),programs_and_spec_byte_identical=True,generator_workload_spec_source_unchanged=True),indent=2)+'\n')
# Presence of the retired key must fail before any workload is loaded, including null.
for value in (False,True,None,0):
    cfg=root/'validation/retired-config.json';cfg.write_text(json.dumps(dict(refresh_allbank=value)))
    run=subprocess.run([str((root/'frozen/bin/pim_vla_sim').resolve()),'--config',str(cfg),'--output',str(root/'validation/rejected-output')],capture_output=True,text=True)
    assert run.returncode!=0 and 'refresh_allbank is retired' in run.stderr,(run.returncode,run.stderr)
(root/'validation/native-config-rejection.json').write_text(json.dumps(dict(refresh_allbank_values_rejected=[False,True,None,0],checks_passed=True),indent=2)+'\n')
PY
"$PYTHON" experiments/run_single_layer.py --binary "$RUNROOT/frozen/bin/pim_vla_sim" --program-dir "$RUNROOT/frozen/programs" --spec "$RUNROOT/frozen/spec.json" --output "$RUNROOT/experiment" --requirements results/task-documents/version2-rewrite.docx --repeats 2
test -z "$(git status --porcelain --untracked-files=no)"
echo "$RUNROOT"

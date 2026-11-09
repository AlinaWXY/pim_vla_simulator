#!/usr/bin/env bash
#SBATCH --job-name=pim-down-row
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=down-row-single-layer-%j.log
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
cd /fact_data/xinyaowang/pim_vla_sim2_20261109
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONPATH="$PWD/ramulator2/python"
export PIM_VLA_SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=no)"
RUNROOT="results/down-row-single-layer-$SLURM_JOB_ID"
BASELINE="results/rank-refresh-single-layer-1980912"
mkdir -p "$RUNROOT/validation" "$RUNROOT/frozen/bin" "$RUNROOT/frozen/lib"
# Only the physical request program changes. Keep the verified native model,
# spec, measured GPU inputs and all runner accounting byte-identical.
git diff --exit-code ea217bad2fce69c3c5773f690fde64d633041791 HEAD -- ramulator2 CMakeLists.txt tools/benchmark_program_input.cpp experiments input-data reference/spec tests/native_spec_tests.cpp tests/native_sequencer_tests.cpp tests/native_refresh_tests.cpp > "$RUNROOT/validation/native-input-source-diff.txt"
cmp build/pim_vla_sim "$BASELINE/frozen/bin/pim_vla_sim"
cmp ramulator2/libramulator.so "$BASELINE/frozen/lib/libramulator.so"
echo ea217bad2fce69c3c5773f690fde64d633041791 > "$RUNROOT/validation/native-clean-build-source.txt"
cp "$BASELINE/frozen/bin/pim_vla_sim" "$RUNROOT/frozen/bin/"
cp "$BASELINE/frozen/lib/libramulator.so" "$RUNROOT/frozen/lib/"
cp "$BASELINE/frozen/spec.json" "$RUNROOT/frozen/spec.json"
export LD_LIBRARY_PATH="$PWD/$RUNROOT/frozen/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v > "$RUNROOT/validation/python-tests.txt" 2>&1
build/native_spec_tests reference/spec/spec.json reference/spec/command-pair-matrix.json > "$RUNROOT/validation/native-spec.txt"
build/native_sequencer_tests reference/spec/spec.json > "$RUNROOT/validation/native-sequencer.txt"
build/native_refresh_tests reference/spec/spec.json > "$RUNROOT/validation/native-refresh.txt"
"$PYTHON" programs/gen_pim_program.py --output "$RUNROOT/frozen/programs" --compact > "$RUNROOT/validation/program-generation.jsonl"
"$PYTHON" tools/verify_program_input.py --binary build/benchmark_program_input "$RUNROOT/frozen/programs/token_local_after.json" > "$RUNROOT/validation/json-input.json"
"$PYTHON" - "$RUNROOT" "$BASELINE" <<'PY'
import gc,hashlib,json,pathlib,sys
root,old=map(pathlib.Path,sys.argv[1:]);rows=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((old/'frozen/manifest.json').read_text())
for name in ('bin/pim_vla_sim','lib/libramulator.so','spec.json'):
    assert sha(root/'frozen'/name)==manifest['files'][name]['sha256']
for path in sorted((root/'frozen/programs').glob('*.json')):
    baseline=old/'frozen/programs'/path.name
    assert sha(baseline)==manifest['files']['programs/'+path.name]['sha256']
    before=json.loads(baseline.read_text());after=json.loads(path.read_text())
    assert before['expected_dq_bytes_per_layer']==after['expected_dq_bytes_per_layer']
    assert [s['name'] for s in before['stages']]==[s['name'] for s in after['stages']]
    for a,b in zip(before['stages'],after['stages']):
        if a['name']!='down':assert a==b,(path.name,a['name'])
        else:assert a!=b and a['expected_dq_bytes']==b['expected_dq_bytes']
    rows.append(dict(variant=after['variant'],non_down_stages_byte_semantics_unchanged=True,total_dq_bytes=after['expected_dq_bytes_per_layer'],old_program_sha256=sha(baseline),new_program_sha256=sha(path)))
    del before,after;gc.collect()
(root/'validation/controlled-comparison.json').write_text(json.dumps(dict(baseline_run=str(old),native_runtime_and_spec_identical=True,measured_gpu_inputs_unchanged=True,programs=rows),indent=2)+'\n')
PY
"$PYTHON" experiments/run_single_layer.py --binary "$RUNROOT/frozen/bin/pim_vla_sim" --program-dir "$RUNROOT/frozen/programs" --spec "$RUNROOT/frozen/spec.json" --output "$RUNROOT/experiment" --requirements results/task-documents/version2-rewrite.docx --repeats 2
test -z "$(git status --porcelain --untracked-files=no)"
echo "$RUNROOT"

#!/usr/bin/env bash
#SBATCH --job-name=dataflow-postprocess
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/fact_data/xinyaowang/pim_vla_o_projection_20261104/dataflow-history-layout-20261104
#SBATCH --output=review-%j.log
set -euo pipefail
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/experiments/pi05-unified-gemv"
sha256sum experiments/pi05-unified-gemv/*.py experiments/pi05-unified-gemv/validation/*.py > analysis-code.sha256
cp ../dataflow-history-review-20261104/dq-ca-tests.log dq-ca-tests.log
"$PYTHON" experiments/pi05-unified-gemv/compare_dataflow_history.py \
  --direct /fact_data/xinyaowang/pim_vla_localwr_act2_20261103/runs/direct-1903036 \
  --modern /fact_data/xinyaowang/pim_vla_o_projection_20261104/r3/runs/data-1924418 \
  --output report --windows-cache ../dataflow-history-review-20261104/report/windows/dq-ca-windows.json
sha256sum --check analysis-code.sha256
"$PYTHON" - <<'PY'
import hashlib,json,re,tarfile,os
from pathlib import Path
r=Path.cwd();out=r/'report'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
log=(r/'dq-ca-tests.log').read_text();assert log.rstrip().endswith('OK')
report=json.loads((out/'comparison.json').read_text())
for row,wall,traffic in zip(report['rows'],[109.762884,171.284428,180.834206,210.230188],[5472,4392,4392,4776]):
    assert abs(row['layer_step_us']-wall)<1e-8
    assert row['external_dq_KiB']==traffic
    assert row['o_projection_timed']==(row['flow']==4)
assert abs(report['rows'][-1]['pim_o_subflow_us']-27.997424)<1e-8
checks=dict(status='passed',analysis_job=os.environ['SLURM_JOB_ID'],native_simulation_rerun=False,
    source_native_jobs=[1903036,1924418],unit_tests_reused_from_job_1928983=int(re.search(r'Ran (\d+) tests in',log)[1]),
    original_native_latency_and_DQ_totals_unchanged=True,
    analysis_code_sha256={str(p.relative_to(r)):digest(p) for p in (r/'experiments').rglob('*') if p.is_file()},
    output_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
(out/'analysis-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
with tarfile.open(r/'dataflow-history-review.tar.gz','w:gz') as tar:
    for p in [out,r/'dq-ca-tests.log',r/'analysis-code.sha256',r/'dataflow-history-review.sbatch']:
        tar.add(p,arcname=str(p.relative_to(r)))
p=r/'dataflow-history-review.tar.gz'
(r/'archive.json').write_text(json.dumps(dict(bytes=p.stat().st_size,sha256=digest(p)),indent=2)+'\n')
print(json.dumps(dict(status='passed',output=str(out),archive_bytes=p.stat().st_size)),flush=True)
PY

#!/usr/bin/env bash
#SBATCH --job-name=oproj-continuous
#SBATCH --partition=cpu-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/fact_data/xinyaowang/pim_vla_o_projection_20261104/continuous-review-20261104-v2
#SBATCH --output=review-%j.log
set -euo pipefail
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:/fact_data/xinyaowang/pim_vla_o_projection_20261104/r3/experiments/pi05-unified-gemv"
sha256sum continuous_dq.py compare_o_projection.py validation/test_continuous_dq.py > analysis-code.sha256
"$PYTHON" -m unittest discover -s validation -p test_continuous_dq.py -v > continuous-tests.log 2>&1
"$PYTHON" -m unittest discover -s ../r3/experiments/pi05-unified-gemv/validation -p test_channel_analysis.py -v > interval-regression-tests.log 2>&1
"$PYTHON" compare_o_projection.py ../r3/runs/data-1924418 --output report
sha256sum --check analysis-code.sha256
"$PYTHON" - <<'PY'
import hashlib,json,re,tarfile
from pathlib import Path
r=Path.cwd();out=r/'report'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
counts={}
for name in ('continuous-tests.log','interval-regression-tests.log'):
    s=(r/name).read_text();assert s.rstrip().endswith('OK')
    counts[name]=int(re.search(r'Ran (\d+) tests in',s)[1])
old=json.loads((r/'../r3/runs/data-1924418/comparison.json').read_text())
new=json.loads((out/'comparison.json').read_text())
for a,b in zip(old['rows'],new['rows']):
    for key in ('variant','latency_us','dq_busy_us','dq_idle_us','dq_utilization_percent','external_dq_KiB_all_channels'):
        assert a[key]==b[key],(key,a[key],b[key])
checks=dict(status='passed',tests=counts,native_simulation_rerun=False,native_latency_and_DQ_totals_unchanged=True,
    source_native_job=1924418,original_comparison_sha256=digest(r/'../r3/runs/data-1924418/comparison.json'),
    analysis_code_sha256={p.name:digest(p) for p in (r/'continuous_dq.py',r/'compare_o_projection.py',r/'validation/test_continuous_dq.py')},
    native_source_provenance_sha256=digest(r/'../r3/runs/data-1924418/provenance.json'),
    output_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
(out/'continuous-review-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
with tarfile.open(r/'continuous-review.tar.gz','w:gz') as tar:
    for p in [out,r/'continuous-tests.log',r/'interval-regression-tests.log',r/'analysis-code.sha256']:
        tar.add(p,arcname=str(p.relative_to(r)))
p=r/'continuous-review.tar.gz'
(r/'archive.json').write_text(json.dumps(dict(bytes=p.stat().st_size,sha256=digest(p)),indent=2)+'\n')
print(json.dumps(checks['tests']),flush=True)
PY

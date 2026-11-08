#!/usr/bin/env bash
#SBATCH --job-name=1108-ca-duration
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=uniform-ca-duration-%j.log
set -euo pipefail
export PYTHON=/fact_data/xinyaowang/pim_vla_reproduce_xinyao_1024/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
src=experiments/pi05-unified-gemv
root=reference/evaluation-1108
out=$root/uniform-transactions
old=$out/before-ca-duration-fix
mkdir -p "$old"
if [ ! -e "$old/native-reservations.json.gz" ]; then
  cp -a "$out/stream-reference" "$out/arbitration-diagnostic" "$old/"
  cp "$out/native-reservations.json.gz" "$out/REPORT.zh-CN.md" "$out/SHA256SUMS.json" "$old/"
fi
"$PYTHON" "$src/prepare_uniform_bus.py" --runs results/evaluation-1971385 --output "$out/native-reservations.json.gz"
"$PYTHON" - <<'PY'
import gzip,json
from pathlib import Path
r=Path('reference/evaluation-1108/uniform-transactions')
def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
old=load(r/'before-ca-duration-fix/native-reservations.json.gz')
new=load(r/'native-reservations.json.gz')
rows=[]
for key,n in new.items():
    o=old[key]
    assert n['source_sha256']==o['source_sha256']
    assert len(n['events'])==len(o['events'])
    changed=0; two_ck=0; twenty_one_ck=0
    for a,b in zip(o['events'],n['events']):
        if a==b:continue
        assert a[:5]==b[:5] and a[1]==0 and b[5]-a[5] in (15008,20*15008)
        changed+=1
        two_ck+=b[5]-a[5]==15008
        twenty_one_ck+=b[5]-a[5]==20*15008
    rows.append(dict(template=key,extended_CA_events=changed,two_CK=two_ck,twenty_one_CK=twenty_one_ck))
(r/'CA-duration-correction.json').write_text(json.dumps(dict(
    correction='Use native ready_at for CA end; previous exporter truncated two-CK ACTAB and 21-CK MAC_RUN/ROPE_RUN commands to one CK.',
    invariant='All event starts, types, ranks, groups, DQ and refresh reservations unchanged; only multi-CK CA commands regain their missing cycles.',
    templates=rows),indent=2)+'\n')
print('CA_DURATION_AUDIT',sum(x['extended_CA_events'] for x in rows),flush=True)
PY
UNIFORM_LIBRARY="$PWD/$out/libuniform-stream.so" "$PYTHON" -m unittest discover -s "$src/validation" -p test_uniform_transactions.py -v > "$out/unit-tests.log" 2>&1
"$PYTHON" "$src/run_uniform_transactions.py" --reference "$root" --demand "$root/gpu/uniform-memory-demand.json" --bus "$out/native-reservations.json.gz" --library "$out/libuniform-stream.so" --output "$out/stream-reference" --stream-reference
"$PYTHON" "$src/diagnose_uniform_arbitration.py"
"$PYTHON" "$src/publish_uniform_transactions.py" --root "$root" --job "$SLURM_JOB_ID"

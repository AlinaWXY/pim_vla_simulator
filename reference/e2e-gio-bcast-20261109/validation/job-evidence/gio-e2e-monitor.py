"""Read native status, optionally caching the last display in a sidecar."""
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
records = []
paths = sorted(root.glob('cases/*/native/*/repeat-*'))
if '--all' in sys.argv or not paths:
    paths = sorted(root.glob('calibration/candidate-*')) + sorted(root.glob('solos/ae-solo/*')) + paths
for directory in paths:
    record = {'run': str(directory.relative_to(root))}
    assessment = directory / 'assessment.json'
    raw = directory / 'raw.json'
    if assessment.is_file():
        result = json.loads(assessment.read_text())
        record.update({k: result[k] for k in ('passed', 'total_us', 'relative_error')})
    elif raw.is_file() and (directory / 'provenance.json').is_file():
        result = json.loads(raw.read_text())
        record.update(status='complete', total_us=result['metrics']['total_us'])
    else:
        log = directory / 'native.log'
        lines = log.read_text(errors='replace').splitlines() if log.is_file() else []
        progress = next((s for s in reversed(lines) if 'progress ck=' in s), '')
        errors = [s for s in lines if s.startswith('ERROR:')]
        if errors:
            record.update(status='failed', error=errors[-1])
        else:
            record.update(status='running', progress=progress or (lines[-1] if lines else 'loading / before first progress checkpoint'))
    records.append(record)
if '--compact' in sys.argv:
    grouped = {}
    for record in records:
        parts = record.pop('run').split('/')
        case = parts[1] if parts[0] == 'cases' else '/'.join(parts)
        grouped.setdefault(case, []).append(record)
    display = {case: {'case': case, 'repeats': len(runs), 'same_checkpoint': all(x == runs[0] for x in runs), 'state': runs[0], **({'other_states': runs[1:]} if any(x != runs[0] for x in runs) else {})} for case, runs in grouped.items()}
    previous = {}
    if '--changes' in sys.argv:
        cache = root / 'monitor-last.json'
        previous = json.loads(cache.read_text()) if cache.is_file() else {}
        cache.write_text(json.dumps(display))
    changed = [value for case, value in display.items() if previous.get(case) != value]
    for value in changed:
        print(json.dumps(value))
    if not changed:
        print('No new checkpoints.')
else:
    for record in records:
        print(json.dumps(record))

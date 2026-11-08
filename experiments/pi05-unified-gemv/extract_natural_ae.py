"""Prepare ordered natural-cache AE timings from retained traces; no GPU rerun."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cold = json.loads((a.reference/'gpu/cold-ae-kernel-ledger.json').read_text())
    template = [r for r in cold if r['repeat'] == 0]
    assert len(template) == 1033
    db = sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings = {r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
    ranges = [dict(r, label=r['text'] or strings.get(r['textId'], ''))
              for r in db.execute('SELECT * FROM NVTX_EVENTS') if r['end']]
    requests = [r for r in ranges if r['label'].startswith('request/trace/')]
    steps = [r for r in ranges if r['label'].startswith('step/')]
    runtime = {r['correlationId']:dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
    grouped = defaultdict(list)
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        launch = runtime.get(row['correlationId'])
        if launch is None:
            continue
        request = next((r for r in requests if r['globalTid'] == launch['globalTid']
                        and r['start'] <= launch['start'] < r['end']), None)
        if request is None:
            continue
        step = next((r for r in steps if r['globalTid'] == launch['globalTid']
                     and r['start'] <= launch['start'] < r['end']), None)
        if step is None:
            continue
        grouped[request['label'], int(step['label'][5:])].append(
            dict(row, kernel=strings[row['demangledName']]))
    assert len(requests) == 5 and len(grouped) == 50
    overlap_pairs = []
    for key, rows in grouped.items():
        assert len(rows) == len(template), key
        overlap_pairs.extend(dict(request=key[0],step=key[1],ordinal=i,
                                  overlap_ns=x['end']-y['start'])
                             for i,(x,y) in enumerate(zip(rows,rows[1:]))
                             if x['end'] > y['start'])
        for x,t in zip(rows, template):
            assert x['kernel'] == t['kernel'], (key, x['kernel'], t['kernel'])
    units = []
    for i,t in enumerate(template):
        per_step = [statistics.mean((grouped[r['label'],s][i]['end']-
                                    grouped[r['label'],s][i]['start'])/1000
                                    for r in requests) for s in range(10)]
        units.append(dict(ordinal=i, kernel=t['kernel'], operator=t['operator'],
                          category=t['category'], shapes=t['shapes'],
                          mean_us=statistics.mean(per_step), step_mean_us=per_step))
    rope_ms = sum(u['mean_us'] for u in units if u['category'] == 'rope_apply')/1000*10
    warm = json.loads((a.reference/'gpu/warm-summary.json').read_text())
    expected = warm['fine']['category_ms']['rope_apply']['mean']*10
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(status='prepared timing input; no new b-g evaluation',
        scope='Natural-cache profiled device kernel durations; excludes CPU launch gaps, separate DMA, and PIM handoff costs. Trace timestamps contain overlapping kernels, so raw duration sums are not activity unions or a validated sequential replay. Not a fused or unprofiled baseline.',
        samples='Five requests, all ten denoising steps, 1033 ordered kernels per step.',
        validation=dict(ordered_kernel_names_match_cold_ledger=True,
                        samples=50, kernels_per_step=1033, RoPE_raw_duration_sum_ms_AE=rope_ms,
                        RoPE_reported_exclusive_activity_ms_AE=expected,
                        overlapping_adjacent_kernel_pairs=len(overlap_pairs),
                        maximum_adjacent_overlap_ns=max((x['overlap_ns'] for x in overlap_pairs),default=0)),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                       [a.database, a.reference/'gpu/cold-ae-kernel-ledger.json',
                        a.reference/'gpu/warm-summary.json']}, units=units,
        overlap_audit=overlap_pairs)
    (a.output/'natural-ae-kernel-ledger.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['validation']))


if __name__ == '__main__':
    main()

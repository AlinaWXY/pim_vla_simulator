"""Export unmerged native CA, typed DQ, and rank refresh reservations."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import analyze_evaluation as A
from channel_analysis import dq_intervals


def native_ca_span(row):
    start = (int(row['clock'])-1)*15008
    stop = (int(row['ready_at'])-1)*15008
    assert stop > start
    return start, stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {}
    for variant in A.VARIANTS:
        for case in A.CASES:
            root = a.runs / variant / case
            r = A.read(root / 'simulation/result.json')
            ratio = r['config']['frontend_ratio']
            assert ratio == 32 and r['tCK_ns'] == .938
            events, hashes = [], {}
            for ch in range(16):
                paths = [root / f'simulation/{name}.csv.ch{ch}'
                         for name in ('timeline', 'rank_ca_events')]
                hashes.update({str(path.relative_to(root)): A.sha(path) for path in paths})
                with paths[0].open() as f:
                    rows = list(csv.DictReader(f))
                by_clock = {int(row['clock']): row for row in rows
                            if row['command'] in ('RD','RDA','WR','WRA','WRGB','RDRES')}
                for d in dq_intervals(rows, r['resolved_timings_ck'], ratio, r['config']['pim_data_path']):
                    row = by_clock[d['clock']]
                    kind = 2 if d['command'] in ('WR','WRA','WRGB') else 1
                    # Events use CK=15008 in 1/16 ps units; native edge 1 is time zero.
                    events.append([ch, kind, int(row['Rank']), int(row['BankGroup']),
                                   (d['begin']-1)*469, (d['end']-1)*469])
                with paths[1].open() as f:
                    for row in csv.DictReader(f):
                        if row['scope'] != 'global':
                            continue
                        start, stop = native_ca_span(row)
                        rank = int(row['Rank'])
                        events.append([ch, 0, rank, -1, start, stop])
                        if row['command'] in ('REFab', 'REFpb'):
                            assert row['command'] == 'REFab'
                            events.append([ch, 3+rank, rank, -1, start,
                                           start+r['resolved_timings_ck']['nRFC']*15008])
            result[variant+'/'+case] = dict(events=events, source_sha256=hashes,
                timings=r['resolved_timings_ck'], config=r['config'])
            print(variant, case, len(events), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.output, 'wt', encoding='utf-8') as f:
        json.dump(result, f, separators=(',',':'))


if __name__ == '__main__':
    main()

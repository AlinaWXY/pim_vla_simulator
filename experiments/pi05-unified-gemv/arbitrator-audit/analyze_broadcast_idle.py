"""Exact maximal DQ idle intervals for corrected job 1971385, continuous layer."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
parser.add_argument('--output',type=Path)
args=parser.parse_args();ROOT=args.root.resolve()
OUT=args.output.resolve() if args.output else ROOT/'outputs/arbitrator/broadcast-idle'
OUT.mkdir(parents=True,exist_ok=True)
REF = ROOT / 'reference/evaluation-1108/direct-before-corrected'
SOURCE = ROOT / 'experiments/pi05-unified-gemv'
NATIVE = ROOT / 'reference/evaluation-1108/gpu-broadcast-dq-idle/native'
sys.path.insert(0, str(SOURCE))
from channel_analysis import dq_intervals, interval_accounting, read_csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    variant, case = 'feature_gpu_before', 'serial_layer_control'
    provenance = REF / 'native-provenance' / variant / case
    result = json.loads((provenance / 'result.json').read_text())
    manifest = json.loads((provenance / 'manifest.json').read_text())
    config = result['config']
    assert not config['pim_rank_controller'] and not config['pim_rank_staging']
    assert config['pim_stream_inputs'] and config['pim_data_path'] == 'attacc'
    ratio, total_ck, tck = config['frontend_ratio'], result['memory_cycles'], result['tCK_ns']
    wall = total_ck * ratio
    hashes = json.loads((REF / 'raw-source-hashes.json').read_text())[variant + '/' + case]
    accounting = {int(x['channel']): x for x in read_csv(REF / 'bus-accounting.csv')
                  if x['variant'] == variant and x['case'] == case}
    markers = {int(x['marker']): int(x['frontend_tick']) for x in read_csv(provenance / 'barriers.csv')}
    phases = [(markers[s['start']], markers[s['end']], s['name']) for s in manifest['segments']]
    all_rows, summaries, all_gaps, all_busy, observed_hashes = [], [], [], [], {}
    for channel in range(config['channels']):
        path = NATIVE / f'timeline.csv.ch{channel}'
        observed_hashes[path.name] = sha(path)
        assert observed_hashes[path.name] == hashes['simulation/' + path.name], path
        commands = read_csv(path)
        assert all(int(x['Channel']) == channel for x in commands)
        dq = dq_intervals(commands, result['resolved_timings_ck'], ratio, config['pim_data_path'])
        assert all(0 <= x['begin'] < x['end'] <= wall for x in dq)
        busy, gaps = interval_accounting(dq, 0, wall)
        assert busy == 24384 * ratio
        assert sum(b - a for a, b in gaps) == 141529 * ratio
        assert len(dq) * result['transaction_bytes'] == int(accounting[channel]['dq_bytes'])
        assert abs(busy / ratio * tck / 1000 - float(accounting[channel]['dq_busy_us'])) < 1e-10
        rows = []
        for index, (begin, end) in enumerate(gaps, 1):
            overlaps = [name for a, b, name in phases if a < end and b > begin]
            rows.append(dict(channel=channel, window_id=index,
                             start_frontend_tick=begin, end_frontend_tick_exclusive=end,
                             start_ck=begin / ratio, end_ck_exclusive=end / ratio,
                             duration_ck=(end - begin) / ratio,
                             start_us=begin / ratio * tck / 1000,
                             end_us=end / ratio * tck / 1000,
                             duration_ns=(end - begin) / ratio * tck,
                             boundary='leading' if begin == 0 else 'trailing' if end == wall else 'internal',
                             overlapping_phases='; '.join(overlaps)))
        lengths = [r['duration_ck'] for r in rows]
        internal = [r['duration_ck'] for r in rows if r['boundary'] == 'internal']
        summaries.append(dict(channel=channel, total_ck=total_ck, dq_busy_ck=busy / ratio,
                              dq_idle_ck=sum(lengths), window_count=len(rows),
                              longest_ck=max(lengths), mean_ck=statistics.mean(lengths),
                              median_ck=statistics.median(lengths), shortest_ck=min(lengths),
                              internal_window_count=len(internal),
                              internal_idle_ck=sum(internal), internal_mean_ck=statistics.mean(internal),
                              internal_longest_ck=max(internal),
                              leading_idle_ck=next((r['duration_ck'] for r in rows if r['boundary'] == 'leading'), 0),
                              trailing_idle_ck=next((r['duration_ck'] for r in rows if r['boundary'] == 'trailing'), 0)))
        all_rows.extend(rows)
        all_gaps.append(gaps)
        all_busy.append(dq)
    ch0 = [r for r in all_rows if r['channel'] == 0]
    # Independently reconcile the exact intervals with the previously published plot bins.
    bins = [x for x in read_csv(REF / 'DQ-CA-timeline-250ns.csv')
            if x['variant'] == variant and x['resource'] == 'dq']
    max_error = 0
    tick_us = tck / ratio / 1000
    for row in bins:
        left, right = float(row['start_us']), float(row['end_us'])
        busy_us = sum(max(0, min(right, x['end'] * tick_us) - max(left, x['begin'] * tick_us))
                      for x in all_busy[0])
        actual = 100 * busy_us / (right - left)
        max_error = max(max_error, abs(actual - float(row['utilization_percent'])))
    assert max_error < 1e-7, max_error
    write_csv('idle-windows-ch0.csv', ch0)
    write_csv('idle-windows-all-channels.csv', all_rows)
    write_csv('channel-summary.csv', summaries)
    distribution = Counter(x['duration_ck'] for x in ch0)
    distribution_rows = [dict(duration_ck=k, window_count=n, total_idle_ck=k * n,
                              fraction_of_windows=n / len(ch0), fraction_of_idle_ck=k * n / 141529)
                         for k, n in sorted(distribution.items())]
    write_csv('window-length-distribution-ch0.csv', distribution_rows)
    top = sorted(ch0, key=lambda x: (-x['duration_ck'], x['window_id']))[:20]
    write_csv('longest-20-windows-ch0.csv', top)
    summary = dict(variant=variant, case=case, native_job=1971385, tCK_ns=tck,
                   frontend_ticks_per_ck=ratio, observation_interval_ticks=[0, wall],
                   definition='Maximal positive-length complement of DQ burst intervals; adjacent bursts merge; no CA or phase-boundary splitting; includes leading and trailing idle.',
                   timing='RD/RDA +nCL; WR/WRA +nCWL; AttAcc WRGB/RDRES +0; burst nBL_min=2 CK.',
                   coordinate='Native frontend ticks / 32, so first/last windows can have fractional CK; other gap lengths are integral CK.',
                   channels=summaries, all_channel_gap_sequences_identical=all(x == all_gaps[0] for x in all_gaps),
                   longest_20_ch0=top, verified_raw_sha256=observed_hashes,
                   accounting_sha256=sha(REF / 'bus-accounting.csv'),
                   result_sha256=sha(provenance / 'result.json'),
                   plot_bin_max_error_percentage_points=max_error,
                   scope='One continuous native layer/step; GPU arithmetic omitted, DQ-only idle, no GPU availability inference.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k in ('channels', 'all_channel_gap_sequences_identical', 'longest_20_ch0', 'plot_bin_max_error_percentage_points')}, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    fig.suptitle('GPU broadcast baseline: continuous DQ idle windows', fontsize=16, weight='bold')
    ax = axes[0]
    ax.vlines([x['start_ck'] for x in ch0], 0, [x['duration_ck'] for x in ch0], color='#257886', linewidth=.65)
    ax.set(xlabel='Window start (memory CK)', ylabel='Idle window length (CK)', xlim=(0, total_ck))
    s = summaries[0]
    ax.text(.98, .95, f"Channel 0 | {s['window_count']:,} windows\nLongest {s['longest_ck']:,.0f} CK | Mean {s['mean_ck']:.3f} CK\nTotal idle {s['dq_idle_ck']:,.0f} CK", transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=.9, edgecolor='#cccccc'))
    ax.grid(alpha=.2)
    edges = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if max(distribution) >= edges[-1]:
        edges.append(max(distribution) + 1)
    counts = [sum(n for k, n in distribution.items() if a <= k < b) for a, b in zip(edges, edges[1:])]
    labels = [f'{a:g}–<{b:g}' for a, b in zip(edges, edges[1:])]
    ax = axes[1]
    ax.bar(labels, counts, color='#257886')
    ax.set(xlabel='Idle window length (CK)', ylabel='Number of windows (log scale)', yscale='log')
    ax.set_ylim(.7, max(counts) * 2)
    for i, n in enumerate(counts):
        if n:
            ax.text(i, n * 1.08, f'{n:,}', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', labelrotation=25)
    ax.grid(axis='y', alpha=.2)
    fig.supxlabel('Job 1971385 / feature_gpu_before / serial_layer_control | 1 CK = 0.938 ns | DQ only, includes boundary idle', fontsize=9)
    fig.savefig(OUT / 'dq-idle-windows.png', dpi=180)
    fig.savefig(OUT / 'dq-idle-windows.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()

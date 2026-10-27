"""Independent all-channel phase-window accounting from native issued commands.

Zero external-command phases are candidate windows, not guaranteed schedulable
GPU time. Burst-equivalent spare time is not a physical DQ waveform.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

EXTERNAL = {'RD', 'RDA', 'WR', 'WRA', 'WRGB', 'RDRES'}
DEFAULT_CASES = ('chain', 'v', 'pv', 'gate_up_z_store', 'down_tiled_dram')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def frontend_tick(memory_clock, ratio):
    return (memory_clock-1)*ratio+1


def classify_phase(timelines, begin, end, ratio):
    # A command issued at a memory edge belongs to its actual frontend tick;
    # multiplying memory_clock by ratio instead would move boundary commands.
    counts = [sum(begin <= frontend_tick(int(e['clock']), ratio) < end
                  for e in rows if e['command'] in EXTERNAL) for rows in timelines]
    return counts, not any(counts)


def verify_run(directory):
    directory = Path(directory)
    summary_path = directory/'summary.json'
    summary = read_json(summary_path) if summary_path.is_file() else {}
    cases = summary.get('primary_cases', list(DEFAULT_CASES))
    require(tuple(cases) == DEFAULT_CASES, 'Only the five primary stages belong in AE window totals')
    stages, phases, source_hashes = {}, [], {}
    for case in cases:
        root = directory/case
        manifest_path, result_path, barriers_path = (root/'manifest.json', root/'simulation/result.json',
                                                     root/'simulation/barriers.csv')
        manifest, result = read_json(manifest_path), read_json(result_path)
        ratio = result['config']['frontend_ratio']
        channels = result['config']['channels']
        require(channels == 16 and ratio == 32, 'Unexpected channel/clock geometry')
        ck_us, burst_ck = result['tCK_ns']/1000, result['resolved_timings_ck']['nBL_min']
        require(burst_ck == 2 and result['transaction_bytes'] == 32, 'Changed native payload timing')
        barriers = {int(e['marker']): int(e['frontend_tick']) for e in read_csv(barriers_path)}
        timeline_paths = [root/f'simulation/timeline.csv.ch{ch}' for ch in range(channels)]
        timelines = [read_csv(p) for p in timeline_paths]
        external = [[e for e in rows if e['command'] in EXTERNAL] for rows in timelines]
        for channel, rows in enumerate(timelines):
            require(all(int(e['Channel']) == channel for e in rows), 'Timeline channel mismatch')
            require(all(int(a['clock']) <= int(b['clock']) for a, b in zip(rows, rows[1:])),
                    'Nonmonotonic native issue clock')
        actual = Counter(e['command'] for rows in external for e in rows)
        require(actual == Counter({k: v for k, v in result['issued_commands'].items() if k in EXTERNAL and v}),
                'External command count differs from native result counters')
        per_channel_counts = [len(rows) for rows in external]
        require(len(set(per_channel_counts)) == 1, 'Whole-H sweep is expected to balance all16 channels')
        payload_by_channel = [count*burst_ck*ck_us for count in per_channel_counts]
        classified, zero_us, last_end = [0]*channels, 0.0, None
        for segment in manifest['segments']:
            begin, end = barriers[segment['start']], barriers[segment['end']]
            require(begin < end and (last_end is None or begin == last_end), 'Noncontiguous phase windows')
            last_end = end
            counts, no_external = classify_phase(timelines, begin, end, ratio)
            classified = [a+b for a, b in zip(classified, counts)]
            duration = (end-begin)/ratio*ck_us
            native_work = sum(begin <= frontend_tick(int(e['clock']), ratio) < end
                              for rows in timelines for e in rows)
            require(native_work > 0, 'Empty/artificial WAIT-only phase must not inflate zero-transfer windows')
            if no_external:
                zero_us += duration
            phases.append(dict(case=case, phase=segment['name'], begin_tick=begin, end_tick=end,
                duration_us=duration, external_command_counts_per_channel=counts,
                zero_external_command_phase=no_external))
        require(classified == per_channel_counts, 'A native external command falls outside exactly one phase')
        require(manifest['segments'] and barriers[manifest['segments'][0]['start']] == barriers[0]
                and last_end == barriers[max(barriers)], 'Unaudited outer phases')
        stages[case] = dict(stage_us=result['memory_cycles']*ck_us,
            external_commands_per_channel=per_channel_counts, external_dq_bytes=sum(per_channel_counts)*32,
            payload_equivalent_us_per_channel=payload_by_channel[0],
            payload_equivalent_spare_us_per_channel=result['memory_cycles']*ck_us-payload_by_channel[0],
            zero_external_command_phase_us=zero_us)
        for path in [manifest_path, result_path, barriers_path]+timeline_paths:
            source_hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    total_us = sum(s['stage_us'] for s in stages.values())
    payload_us = sum(s['payload_equivalent_us_per_channel'] for s in stages.values())
    report = dict(status='passed', stage_sum_us=total_us, payload_equivalent_us=payload_us,
        payload_equivalent_spare_us=total_us-payload_us,
        payload_equivalent_spare_fraction=1-payload_us/total_us,
        zero_external_command_phase_us=sum(s['zero_external_command_phase_us'] for s in stages.values()),
        actual_GPU_usable_time_us=None, native_clock_conversion='(memory_clock-1)*frontend_ratio+1',
        stages=stages, phases=phases, source_sha256=source_hashes,
        qualification='Five independent stage runs; no invented inter-stage/NL idle time; no mixed GPU traffic')
    comparison = directory/'channel-window-summary.json'
    if comparison.is_file():
        other = read_json(comparison)
        for name in ('stage_sum_us', 'payload_equivalent_us', 'payload_equivalent_spare_us',
                     'payload_equivalent_spare_fraction', 'zero_external_command_phase_us'):
            require(abs(report[name]-other[name]) < 1e-9, f'Channel analysis mismatch: {name}')
        for case in cases:
            require(abs(stages[case]['zero_external_command_phase_us']-
                        other['stages'][case]['zero_external_command_phase_us']) < 1e-9,
                    f'Channel zero-transfer phase mismatch: {case}')
        require(len(phases) == len(other['phases']), 'Channel phase count mismatch')
        for a, b in zip(phases, other['phases']):
            require(a['case'] == b['case'] and a['phase'] == b['phase'] and
                    a['zero_external_command_phase'] == b['zero_external_command_phase'] and
                    a['external_command_counts_per_channel'] == b['external_command_counts_per_channel'],
                    'Channel analysis classifies an external command in a different phase')
        report['independent_channel_summary_match'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = verify_run(args.run)
    print(json.dumps({k: v for k, v in report.items() if k not in ('stages','phases','source_sha256')}, indent=2))

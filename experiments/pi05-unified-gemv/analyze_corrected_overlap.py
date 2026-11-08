"""Corrected native inputs + retained GPU ledger + ordered idle-service replay."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import analyze_evaluation as A
from replay_idle_service import replay


def refresh_blocked(path, stage):
    """Keep GPU rank 0 blackout separate from the one-CK external REF command."""
    ratio = stage['config']['frontend_ratio']
    result = A.read(path / 'simulation/result.json')
    n_rfc = result['resolved_timings_ck']['nRFC'] * ratio
    per_channel = []
    for ch in range(16):
        intervals = []
        with (path / f'simulation/rank_ca_events.csv.ch{ch}').open() as f:
            for row in csv.DictReader(f):
                if row['scope'] != 'global' or row['command'] not in ('REFab', 'REFpb'):
                    continue
                assert row['command'] == 'REFab', 'Per-bank refresh requires bank mapping'
                if int(row['Rank']) == 0:
                    start = (int(row['clock']) - 1) * ratio + 1
                    intervals.append([start, min(stage['wall_ticks'], start + n_rfc)])
        per_channel.append(intervals)
    return A.merge([x for intervals in per_channel for x in intervals])


def schedule(variant, stages, ks, handoffs, accounting):
    after = A.VARIANTS[variant]['after']
    gpu_rope = A.VARIANTS[variant]['gpu_rope']
    feature = A.VARIANTS[variant]['feature']
    blocks = []
    cursor = 0.0
    layer = -1
    seen = set()
    rms = [k['ordinal'] for k in ks if k['category'] == 'rms_pow']
    assert len(rms) == 37
    starts = {rms[2 * i] - 1: i for i in range(18)}
    starts[rms[-1] - 1] = 18

    def gpu(name, us, ordinal=None):
        nonlocal cursor
        blocks.append(dict(kind='gpu', name=name, begin_us=cursor,
                           end_us=cursor + us, layer=layer, source_gpu_ordinal=ordinal))
        cursor += us

    def pim(case, section='all'):
        nonlocal cursor
        s = stages[case]
        pieces = ([(0, s['wall_ticks'])] if accounting == 'explicit_native_io'
                  else A.effective_pieces(s, after)[0])
        split = s['wall_ticks']
        if case == 'chain':
            names = ('gpu_rope_export',) if gpu_rope else ('projection_gemv_release', 'qkv_run_local_spills')
            split = next(p['end'] for p in s['phases'] if p['name'].split('/')[-1] in names)
        if case == 'pv' and not after:
            split = next(p['end'] for p in s['phases'] if p['name'].split('/')[-1] == 'pv_result_gpu_export')
        left, right = ((0, split) if section == 'pre' else
                       (split, s['wall_ticks']) if section == 'post' else (0, s['wall_ticks']))
        for a, b in pieces:
            a, b = max(a, left), min(b, right)
            if a >= b:
                continue
            us = (b - a) * s['tick_us']
            blocks.append(dict(kind='pim', name=case + '_' + section, begin_us=cursor,
                               end_us=cursor + us, layer=layer, source_case=case,
                               source_begin_tick=a, source_end_tick=b))
            cursor += us

    for k in ks:
        cat = k['category']
        if k['ordinal'] in starts:
            layer = starts[k['ordinal']]
        trigger = {'q_projection': ('chain', 'pre'), 'attention_qkt': ('chain', 'post'),
                   'attention_sv': ('pv', 'all'), 'ffn_gate': ('gate_up_gpu', 'all'),
                   'ffn_down': ('down_gpu_input', 'all')}.get(cat)
        if trigger and (layer, trigger) not in seen:
            seen.add((layer, trigger))
            if trigger == ('pv', 'all') and not after:
                pim('pv', 'pre')
                name = ('feature' if feature else 'token') + '_direct_PV_to_O_pack'
                gpu(name, handoffs[name])
                pim('pv', 'post')
            else:
                pim(*trigger)
            if trigger == ('chain', 'post') and feature:
                gpu('feature_score_rank_sum', handoffs['feature_score_rank_sum'])
            if trigger == ('pv', 'all'):
                gpu('common_O_rank_sum_cast', handoffs['common_O_rank_sum_cast'])
            if trigger == ('gate_up_gpu', 'all'):
                gpu('common_GateUp_fp32_to_bf16', handoffs['common_GateUp_fp32_to_bf16'])
            if trigger == ('down_gpu_input', 'all'):
                gpu('common_Down_fp32_to_bf16', handoffs['common_Down_fp32_to_bf16'])
        keep = cat not in A.REMOVED or (gpu_rope and cat == 'rope_apply')
        if cat == 'copies_casts' and k['operator'] == 'aten::contiguous' and k['shapes'] == [[1, 16, 8, 256]]:
            keep = False
        if keep:
            gpu(cat, k['us'], k['ordinal'])
    assert len(seen) == 18 * 5
    expected = sum(s['wall_us'] - (0 if accounting == 'explicit_native_io' else A.effective_pieces(s, after)[1])
                   for s in stages.values()) * 18
    assert math.isclose(sum(b['end_us'] - b['begin_us'] for b in blocks if b['kind'] == 'pim'), expected, abs_tol=1e-6)
    full = [dict(**{**b, 'begin_us': b['begin_us'] + step * cursor,
                        'end_us': b['end_us'] + step * cursor}, step=step)
            for step in range(10) for b in blocks]
    assert all(math.isclose(x['end_us'], y['begin_us'], abs_tol=1e-7) for x, y in zip(full, full[1:]))
    for step in range(10):
        ordinals = [b['source_gpu_ordinal'] for b in full if b['kind'] == 'gpu'
                    and b['step'] == step and b['source_gpu_ordinal'] is not None]
        assert ordinals == sorted(set(ordinals))
    return full


def make_windows(stages):
    cache = {}
    def windows(block):
        key = (block['source_case'], block['source_begin_tick'], block['source_end_tick'])
        if key not in cache:
            case, a, b = key
            s = stages[case]
            cache[key] = [[(max(a, x) - a) * s['tick_us'], (min(b, y) - a) * s['tick_us']]
                          for x, y in s['service_idle'] if max(a, x) < min(b, y)]
        return cache[key]
    return windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--gpu', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ks = A.kernels(args.gpu)
    handoffs = {}
    for file in ('handoff-measurements.json', 'direct-pv-handoff.json'):
        handoffs.update({k: x['cold32MiB']['device_kernel_median_us']
                         for k, x in A.read(args.gpu / file)['operators'].items()})
    raw_units = A.read(args.gpu / 'prefix-kernels.json')['ordered_mean_units']
    warm = A.read(args.gpu / 'warm-summary.json')['baseline']['phase_ms']
    vlm_ms = warm['vit']['mean'] + warm['vlm']['mean']
    factor = vlm_ms * 1000 / sum(u['us'] for u in raw_units)
    units = [{**u, 'us': u['us'] * factor} for u in raw_units]
    A.csvout(out / 'VLM-ordered-units.csv', [dict(ordinal=u['ordinal'], kind=u['kind'],
             measured_us=raw_units[i]['us'], normalized_us=u['us'], kernel=u['kernel'])
             for i, u in enumerate(units)])
    rows = []
    all_stages = {}
    checks = []
    details = []
    letters = dict(feature_gpu_before='c', token_gpu_before='d', token_local_before='e',
                   token_gpu_after='f', token_local_after='g')
    for variant in A.VARIANTS:
        stages = {}
        for case in A.CASES:
            path = args.runs / variant / case
            print('EXTRACT', variant, case, flush=True)
            s = A.extract(path)
            after = A.VARIANTS[variant]['after']
            assert s['config']['pim_rank_controller'] == after
            assert s['config']['pim_rank_staging'] == after
            assert s['config']['pim_stream_inputs'] != after
            s['gpu_rank_refresh'] = refresh_blocked(path, s)
            s['service_idle'] = A.gaps(A.merge(s['joint'] + s['gpu_rank_refresh']), 0, s['wall_ticks'])
            stages[case] = s
            all_stages[variant + '/' + case] = s
        for accounting in ('explicit_native_io', 'shared_tensor_io_dedup'):
            blocks = schedule(variant, stages, ks, handoffs, accounting)
            result = replay(blocks, units, make_windows(stages))
            name = variant + '-' + accounting
            for field, suffix in [('placements', 'VLM-placement'), ('waits', 'AE-GPU-wait'), ('execution', 'execution')]:
                A.csvout(out / (name + '-' + suffix + '.csv'), result.pop(field))
            A.csvout(out / (name + '-solo-schedule.csv'), blocks)
            row = dict(case=letters[variant], variant=variant, accounting=accounting, **result)
            row['AE_GPU_solo_ms'] = sum(b['end_us'] - b['begin_us'] for b in blocks if b['kind'] == 'gpu') / 1000
            row['native_PIM_ms'] = sum(b['end_us'] - b['begin_us'] for b in blocks if b['kind'] == 'pim') / 1000
            if variant == 'feature_gpu_before':
                rows.append(dict(case='b', variant=variant, accounting=accounting,
                                 AE_solo_ms=result['AE_solo_ms'], AE_gpu_wait_ms=0,
                                 AE_with_interference_ms=result['AE_solo_ms'], VLM_ms=vlm_ms,
                                 VLM_progress_during_PIM_ms=0, VLM_progress_during_AE_ms=0,
                                 VLM_remaining_after_AE_ms=vlm_ms, total_ms=result['AE_solo_ms'] + vlm_ms,
                                 AE_GPU_solo_ms=row['AE_GPU_solo_ms'], native_PIM_ms=row['native_PIM_ms']))
            rows.append(row)
            # A separate retained-trace-duration check avoids concealing the tiny
            # difference between separate event phase means and profiling means.
            raw = replay(blocks, raw_units, make_windows(stages))
            details.append(dict(variant=variant, accounting=accounting,
                                raw_profile_VLM_ms=sum(u['us'] for u in raw_units) / 1000,
                                raw_profile_total_ms=raw['total_ms'],
                                normalized_total_ms=result['total_ms']))
            checks.append(dict(variant=variant, accounting=accounting, status='passed',
                               solo_physical_duration_conserved=True, kernel_order_preserved=True,
                               AE_wait_and_VLM_service_conserved=True,
                               no_minimum_window_or_whole_kernel_fit=True))
            print('RESULT', json.dumps(row), flush=True)
    A.csvout(out / 'evaluation-b-g.csv', sorted([r for r in rows if r['accounting'] == 'explicit_native_io'], key=lambda r:r['case']))
    A.csvout(out / 'shared-tensor-io-accounting.csv', sorted([r for r in rows if r['accounting'] != 'explicit_native_io'], key=lambda r:r['case']))
    A.csvout(out / 'profile-duration-check.csv', details)
    with gzip.open(out / 'native-bus-templates.json.gz', 'wt') as f:
        json.dump(all_stages, f, separators=(',', ':'))
    A.dump(out / 'summary.json', dict(model='ordered cross-window idle service approximation',
        native_run=str(args.runs), VLM_measured_ms=vlm_ms, VLM_profile_scale=factor,
        policies=dict(memory_pressure=None, outstanding_requests=None, read_write_ratio=None,
            rank_controller_before=False, input_before='GPU WRGB throughout operators',
            after='DRAM staging + rank controller + coarse commands',
            progression='1x retained kernel service only during all-channel DQ/CA idle and GPU rank 0 not refreshing',
            AE_GPU_priority='finish the in-flight VLM unit, then run consecutive AE GPU units; do not start another VLM unit until PIM',
            idle_threshold_us=0, whole_kernel_fit=False, kernel_reordering=False,
            main_IO='retain every native transfer phase; explicit GPU PV-to-O packing before O on before variants',
            alternate_IO='inherited shared-tensor producer/consumer dedup, reported separately',
            accounting_identity='AE_with_interference + VLM - VLM_progress_during_AE',
            GPU_placement='rank 0 for refresh mask; no addresses or request arrivals synthesized',
            limitations=['conditional service-clock approximation, not hardware E2E or request-level arbitration',
                         'PIM native templates are fixed-priority; GPU does not alter native command scheduling',
                         'independent native stages are concatenated; no globally continuous refresh simulation across inserted GPU operations',
                         'host launch gaps/context switches are not modeled',
                         'full native IO plus retained GPU kernel timings may charge shared memory effects twice; alternate accounting shown',
                         'prefix/setup ordered trace normalized to existing VLM phase mean; exact standalone VLM kernel-only range was not tagged']),
        handoff_us=handoffs, evaluation=rows, checks=checks,
        source_sha256={str(p):A.sha(p) for p in [Path(__file__), Path(__file__).with_name('replay_idle_service.py'),
                        args.gpu/'prefix-kernels.json', args.gpu/'cold-ae-kernel-ledger.json',
                        args.gpu/'warm-summary.json', args.gpu/'handoff-measurements.json', args.gpu/'direct-pv-handoff.json']}))
    A.dump(out / 'checks.json', dict(status='passed', cases=checks))


if __name__ == '__main__':
    main()

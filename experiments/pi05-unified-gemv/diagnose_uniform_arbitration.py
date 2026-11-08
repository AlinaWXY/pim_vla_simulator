"""Counterfactual CA/DQ ablations; diagnostic only, not new evaluation cases."""
import argparse
import gzip
import json
from pathlib import Path
from run_uniform_transactions import read, csvout, templates, simulate
from uniform_transactions import Engine, U, CK


class DiagnosticEngine(Engine):
    mode = 'original'

    def add(self, end, events, channels=16):
        if end and self.mode != 'original':
            original = events
            refresh_starts = {(x[0], x[4]) for x in original if x[1] >= 3}
            events = [x for x in original
                      if not (self.mode in ('remove_PIM_CA', 'remove_PIM_CA_DQ')
                              and x[1] == 0 and (x[0], x[4]) not in refresh_starts)
                      and not (self.mode in ('remove_PIM_DQ', 'remove_PIM_CA_DQ')
                               and x[1] in (1, 2))]
            # Preserve the native clock phase if this slice has no REF CA pulse.
            # This past pulse cannot block any nonnegative GPU issue time.
            for ch in range(channels):
                if not any(x[0] == ch and x[1] == 0 for x in events):
                    first = next((x for x in original if x[0] == ch and x[1] == 0), None)
                    if first is not None:
                        t = first[4] % CK - 1000 * CK
                        events.append((ch, 0, 0, -1, t, t + CK))
        return super().add(end, events, channels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('reference/evaluation-1108'))
    a = p.parse_args()
    out = a.root / 'uniform-transactions/arbitration-diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    units = read(a.root / 'gpu/uniform-memory-demand.json')['units']
    warm = read(a.root / 'gpu/warm-summary.json')['baseline']['phase_ms']
    measured_vlm = warm['vit']['mean'] + warm['vlm']['mean']
    scale = measured_vlm * 1000 / sum(u['us'] for u in units)
    for u in units:
        u['duration'] = round(u['us'] * scale * U)
    units[-1]['duration'] += round(measured_vlm * 1000 * U) - sum(u['duration'] for u in units)
    engine = DiagnosticEngine(a.root / 'uniform-transactions/libuniform-stream.so')
    empty = engine.add(0, [], 16)
    plans, cache = [], {}
    for u in units:
        measured = u['duration']
        u['measured_duration'] = measured
        handles, durations = [], []
        for ch in range(16):
            r = u['read_sectors'] // 16 + int(ch < u['read_sectors'] % 16)
            w = u['write_sectors'] // 16 + int(ch < u['write_sectors'] % 16)
            key = (measured, r, w)
            if key not in cache:
                handle, _ = engine.stream_plan(*key)
                reference = engine.run(empty, handle)
                assert reference.requests == r + w
                cache[key] = handle, reference.finish
            handle, duration = cache[key]
            handles.append(handle)
            durations.append(duration)
        u['duration'] = max(durations)
        plans.append(handles)
    bias = sum(u['duration'] for u in units) / U / 1000 - measured_vlm
    with gzip.open(a.root / 'uniform-transactions/native-reservations.json.gz', 'rt') as f:
        typed = json.load(f)
    rows = []
    for mode in ('original', 'remove_PIM_CA', 'remove_PIM_DQ', 'remove_PIM_CA_DQ'):
        engine.mode = mode
        for case, variant in [('e', 'token_local_before'), ('g', 'token_local_after')]:
            seq = templates(a.root / 'corrected-overlap', typed, engine, variant)
            result = simulate(seq, units, plans, engine)
            result.pop('placements')
            result.pop('AE_waits')
            result['total_ms'] -= bias
            result['VLM_ms'] = measured_vlm
            result['AE_PIM_ms'] = sum(x['duration'] for x in seq if x['kind'] == 'pim') / U / 1000
            result['AE_GPU_ms'] = sum(x['duration'] for x in seq if x['kind'] == 'gpu') / U / 1000
            rows.append(dict(case=case, ablation=mode, **result))
            csvout(out / 'ablation.csv', rows)
            print(json.dumps(rows[-1]), flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(
        qualification='Counterfactual diagnostics of the existing uniform row-hit model, not hardware predictions. Native refresh CA and target-rank blackout, GPU-GPU turnaround and all workload counts remain unchanged. CA and DQ effects are coupled and must not be added.',
        VLM_reference_bias_ms=bias, results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()

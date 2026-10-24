"""Fused-MACAB sweep with frozen, measured Thor GPU timing inputs."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .ffn import FFNConfig, run_ffn
from .hybrid import make_plan, write_json

THOR = Path(__file__).resolve().parents[1] / 'experiments/2026-09-17/thor'
VARIANTS = [(r, ck) for r in (2, 5, 10) for ck in (8, 12)]
SCENARIOS = ('unrestricted', 'floor', 'delta')


def config_for(case, r, ck):
    values = dict(case['config'])
    values.update(reuse_tokens=r, macab_tokens_per_command=r,
                  macab_issue_interval_ck=ck, input_buffer_bytes=None,
                  accumulator_buffer_bytes_per_bank=None)
    return FFNConfig(**values)


def plan(root):
    base = make_plan()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'plan.json', {
        'cases': base['cases'], 'variants': VARIANTS,
        'scope': base['scope'],
        'assumptions': [
            'One enhanced MACAB reads a K block once and applies it to R token contexts',
            'Explicit input and accumulator buffers scale with R; no weight SRAM',
            'Fused issue interval 8 or 12 CK is an architectural assumption',
            'Refresh/input/result/barrier costs are simulated; nonlinear compute excluded',
            'GPU times are cold-cache Thor measurements; channel partitioning is modeled',
            'Delivery/add after max of branches; previous bandwidth corrections unchanged',
            'PIM-only has delivery at 273 GB/s but no partial-sum add',
        ],
        'thor_source': str(THOR),
    })
    for name in ('combined.json', 'measurements.jsonl', 'source.json', 'measurement-meta.json'):
        (root / f'thor-{name}').write_bytes((THOR / name).read_bytes())


def run(root, index):
    data = json.loads((root / 'plan.json').read_text())
    if not 0 <= index < len(data['cases']):
        raise ValueError('Invalid case index')
    case = data['cases'][index]
    for r, ck in data['variants']:
        cfg = config_for(case, r, ck)
        key = f"{case['id']}-r{r}-ck{ck}"
        result = run_ffn(cfg, root / 'cases' / key)
        scans = (cfg.horizon + r - 1) // r
        expected = 3 * cfg.hidden_size * cfg.intermediate_size // 256 * scans
        assert result['issued_commands']['MACAB'] == expected
        assert result['logical_mac_operations'] == (
            cfg.horizon * 3 * cfg.hidden_size * cfg.intermediate_size)
        assert result['pim_issue_interval_ck'] == ck
        print(json.dumps({'id': key, 'latency_us': result['latency_ns'] / 1000}), flush=True)


def summarize(root):
    data = json.loads((root / 'plan.json').read_text())
    prior = json.loads((root / 'thor-combined.json').read_text())
    gpu = {(r['horizon'], r['pim_packages'], r['pim_width']): r for r in prior['rows']}
    full = {r['horizon']: r['full_gpu_measured_us'] for r in prior['rows']}
    rows = []
    for case in data['cases']:
        for r, ck in data['variants']:
            cfg = config_for(case, r, ck)
            key = f"{case['id']}-r{r}-ck{ck}"
            result = json.loads((root / 'cases' / key / 'result.json').read_text())
            if result['config'] != asdict(cfg):
                raise ValueError(f'Configuration mismatch: {key}')
            tp = result['latency_ns'] / 1000
            row = dict(id=key, horizon=cfg.horizon, reuse=r, interval_ck=ck,
                       pim_packages=cfg.packages, pim_width=cfg.intermediate_size,
                       pim_fraction=cfg.intermediate_size / 4096, pim_us=tp,
                       full_gpu_us=full[cfg.horizon],
                       input_buffer_per_rank=cfg.input_buffer_bytes,
                       accumulator_per_bank=cfg.accumulator_buffer_bytes_per_bank,
                       macab_commands=result['issued_commands']['MACAB'])
            if cfg.packages == 8:
                for scenario in SCENARIOS:
                    row[scenario + '_us'] = tp + 4 * cfg.horizon * 1024 / 273 / 1000
            else:
                old = gpu[cfg.horizon, cfg.packages, cfg.intermediate_size]
                for scenario, field in (
                    ('unrestricted', 'gpu_measured_unrestricted_us'),
                    ('floor', 'gpu_floor_scenario_us'), ('delta', 'gpu_delta_scenario_us'),
                ):
                    total = ('hybrid_unrestricted_us' if scenario == 'unrestricted'
                             else f'hybrid_{scenario}_scenario_us')
                    overhead = old[total] - max(old['pim_us'], old[field])
                    row[scenario + '_us'] = max(tp, old[field]) + overhead
            rows.append(row)
    best = []
    for r, ck in data['variants']:
        for h in (10, 20, 30, 40, 50):
            group = [x for x in rows if (x['reuse'], x['interval_ck'], x['horizon']) == (r, ck, h)]
            for scenario in SCENARIOS:
                field = scenario + '_us'
                hybrid = min((x for x in group if x['pim_packages'] < 8), key=lambda x: x[field])
                pim = next(x for x in group if x['pim_packages'] == 8)
                winner = min(group, key=lambda x: x[field])
                best.append(dict(horizon=h, reuse=r, interval_ck=ck, scenario=scenario,
                                 hybrid=hybrid, full_pim_us=pim[field], full_gpu_us=full[h],
                                 winner=('GPU' if full[h] <= winner[field] else winner['id']),
                                 best_us=min(full[h], winner[field])))
    write_json(root / 'summary.json', dict(complete=True, cases=len(rows), best=best,
                                          rows=rows, assumptions=data['assumptions']))
    with (root / 'summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'complete': True, 'cases': len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'plan':
        plan(root)
    elif args.action == 'run':
        if args.index is None:
            parser.error('run requires --index')
        run(root, args.index)
    else:
        summarize(root)


if __name__ == '__main__':
    main()

"""Report fused simulation candidates against frozen measured GPU baselines."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / 'summary.json').read_text())
    best = report['best']
    lines = ['# Fused-MACAB + measured Thor GPU coarse sweep', '',
             '1380 PIM simulations; R=2/5/10, nCCDAB=8/12 CK, H=10/20/30/40/50.',
             'GPU uses frozen Thor cold-cache core measurements from 0917/00.',
             'PIM packages 1/2/4, Fp=256..3840; all-PIM uses 8 packages and Fp=4096.',
             'Nonlinear PIM compute excluded. Enhanced multi-token datapath is hypothetical.',
             'Delivery and merge follow the same prior models; this is not measured hybrid hardware.',
             'Both bandwidth corrections are sensitivity scenarios, not rigorous bounds.', '',
             '| R | CK | H | Model | Best hybrid PIM packages | PIM fraction | Hybrid us | Full PIM us | Full GPU us | Winner including endpoints |',
             '|---:|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for b in best:
        if b['scenario'] == 'unrestricted':
            continue
        h = b['hybrid']
        lines.append(f"| {b['reuse']} | {b['interval_ck']} | {b['horizon']} | {b['scenario']} | "
                     f"{h['pim_packages']} | {h['pim_fraction']:.2%} | "
                     f"{h[b['scenario']+'_us']:.3f} | {b['full_pim_us']:.3f} | "
                     f"{b['full_gpu_us']:.3f} | {b['winner']} |")
    lines.extend(['', 'Resource assumptions: input SRAM per rank = 2*max(1024,Fp)*R bytes;',
                  'FP32 accumulators = 4*R bytes per bank. No weight SRAM is added.',
                  'Full PIM uses 80 KiB input SRAM/rank and 40 B accumulators/bank at R=10.',
                  'Intervals 8/12 CK are independent hardware assumptions, not measured silicon.',
                  'Ordinary MACAB baseline uses 6 CK and one token per command.', ''])
    (root / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, ck in enumerate((8, 12)):
        for i, scenario in enumerate(('floor', 'delta')):
            ax = axes[i, j]
            for r in (2, 5, 10):
                group = sorted((b for b in best if b['interval_ck'] == ck
                                and b['reuse'] == r and b['scenario'] == scenario),
                               key=lambda b: b['horizon'])
                ax.plot([b['horizon'] for b in group],
                        [b['hybrid'][scenario+'_us'] for b in group], marker='o', label=f'Hybrid R={r}')
            ax.plot([b['horizon'] for b in group], [b['full_pim_us'] for b in group],
                    '--', color='tab:green', label='Full PIM R=10')
            ax.plot([b['horizon'] for b in group], [b['full_gpu_us'] for b in group],
                    'k--', label='Measured full GPU')
            ax.set(title=f'{ck} CK | {scenario} bandwidth scenario', ylabel='Latency (us)')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('Horizon H')
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds')
    fig.suptitle('Fused-MACAB simulation + measured Thor GPU core')
    fig.text(.5, .01, f'Plotted {stamp}; GPU collection 2026-09-17 19:22–19:43 +08:00',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(root / f'fused-comparison.{suffix}', dpi=160)


if __name__ == '__main__':
    main()

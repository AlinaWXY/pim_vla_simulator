"""Separate every fused hardware case, including full split curves and optima."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODELS = [('floor', 'Optimistic correction', '#167cb4'),
          ('delta', 'Additive bandwidth penalty', '#df7e22'),
          ('unrestricted', 'No bandwidth correction', '#888888')]
HS = (10, 20, 30, 40, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'summary.json'
    report = json.loads(source.read_text())
    assert report['complete'] and len(report['rows']) == 1380
    now = datetime.now(ZoneInfo('Asia/Hong_Kong'))
    out = root / 'plots' / now.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    best = report['best']
    stamp = now.isoformat(timespec='seconds')
    footer = ('PIM timing simulation + Thor GPU core measurements; PIM nonlinear compute excluded.\n'
              'GPU data: 2026-09-17 19:22–19:43 +08:00 | Plotted: ' + stamp)
    files = []

    def save(fig, name):
        fig.text(.5, .012, footer, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94))
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{suffix}', dpi=170)
        plt.close(fig)
        files.append(name)

    for r in (2, 5, 10):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        for row, ck in enumerate((8, 12)):
            for model, label, color in MODELS:
                group = sorted((b for b in best if b['reuse'] == r and b['interval_ck'] == ck
                                and b['scenario'] == model), key=lambda b: b['horizon'])
                assert len(group) == 5
                values = [b['hybrid'] for b in group]
                style = dict(color=color, marker='o', label=label,
                             ls=':' if model == 'unrestricted' else '-',
                             zorder=1 if model == 'unrestricted' else 3,
                             alpha=.6 if model == 'unrestricted' else 1)
                axes[row, 0].plot(HS, [v[model+'_us'] for v in values], **style)
                axes[row, 1].plot(HS, [v['pim_fraction']*100 for v in values], **style)
                axes[row, 2].plot(HS, [v['pim_packages'] for v in values], **style)
                if model != 'unrestricted':
                    for h, v in zip(HS, values):
                        other = next(b['hybrid']['pim_fraction'] for b in best
                                     if b['reuse'] == r and b['interval_ck'] == ck
                                     and b['horizon'] == h
                                     and b['scenario'] == ('delta' if model == 'floor' else 'floor'))
                        above = v['pim_fraction'] > other or (
                            v['pim_fraction'] == other and model == 'floor')
                        axes[row, 1].annotate(f"{v['pim_fraction']*100:g}%",
                                             (h, v['pim_fraction']*100),
                                             xytext=(0, 8 if above else -13),
                                             textcoords='offset points', ha='center',
                                             fontsize=8, color=color)
            axes[row, 0].plot(HS, [b['full_gpu_us'] for b in group], 'k--', label='Measured full GPU')
            axes[row, 0].plot(HS, [b['full_pim_us'] for b in group], '--', color='#388846',
                              label=f'Full PIM (same R={r}, CK={ck})')
            for col, title in enumerate(('Best hybrid latency', 'Best hybrid PIM width share',
                                         'Best hybrid PIM package count')):
                ax = axes[row, col]
                ax.set_title(f'{ck} CK | {title}')
                ax.set_xticks(HS)
                ax.grid(alpha=.25)
            axes[row, 0].set_ylabel('Latency (us)')
            axes[row, 0].legend(fontsize=7)
            axes[row, 1].set_ylim(-6, 107)
            axes[row, 1].yaxis.set_major_formatter(PercentFormatter())
            axes[row, 2].set_yticks((1, 2, 4))
            axes[row, 2].set_ylim(.5, 4.5)
        for ax in axes[1]:
            ax.set_xlabel('Horizon H')
        fig.suptitle(f'R={r}: latency and selected split for each bandwidth model\n'
                     'Split panels show best HYBRID only; full GPU/PIM may be faster', fontsize=14)
        save(fig, f'R{r}-overview')

        for ck in (8, 12):
            fig, axes = plt.subplots(3, 5, figsize=(19, 11), sharex=True, sharey=True)
            for i, (model, label, _) in enumerate(MODELS):
                for j, h in enumerate(HS):
                    ax = axes[i, j]
                    b = next(b for b in best if (b['reuse'], b['interval_ck'], b['horizon'], b['scenario'])
                             == (r, ck, h, model))
                    for packages, color in ((1, '#167cb4'), (2, '#df7e22'), (4, '#8355b2')):
                        curve = sorted((x for x in report['rows'] if (x['reuse'], x['interval_ck'],
                                       x['horizon'], x['pim_packages']) == (r, ck, h, packages)),
                                       key=lambda x: x['pim_fraction'])
                        assert len(curve) == 15
                        ax.plot([x['pim_fraction']*100 for x in curve],
                                [x[model+'_us']/b['full_gpu_us'] for x in curve],
                                '.-', color=color, label=f'{packages} PIM packages')
                    ax.axhline(1, color='black', ls='--', label='Full GPU')
                    ax.axhline(b['full_pim_us']/b['full_gpu_us'], color='#388846', ls='--',
                               label='Full PIM (8 packages)')
                    winner = b['hybrid']
                    ax.scatter(winner['pim_fraction']*100, winner[model+'_us']/b['full_gpu_us'],
                               marker='*', s=150, color='red', zorder=5, label='Best hybrid')
                    ax.set_title(f'H={h} | {label}', fontsize=9)
                    ax.set_yscale('log')
                    ax.grid(alpha=.2, which='both')
                    ax.set_xticks((0, 25, 50, 75, 100))
                    if j == 0:
                        ax.set_ylabel('Latency / measured full GPU\n(lower is better)')
                    if i == 2:
                        ax.set_xlabel('PIM intermediate width (%)')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=6, fontsize=9)
            fig.suptitle(f'All split candidates: R={r}, MACAB interval={ck} CK', fontsize=15)
            save(fig, f'R{r}-ck{ck}-all-splits')

    with (out / 'best-points.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['R', 'CK', 'H', 'model', 'pim_packages',
                                'pim_width', 'pim_percent', 'hybrid_us', 'full_pim_us',
                                'full_gpu_us', 'winner_including_endpoints'])
        writer.writeheader()
        for b in best:
            v = b['hybrid']
            writer.writerow(dict(R=b['reuse'], CK=b['interval_ck'], H=b['horizon'], model=b['scenario'],
                                 pim_packages=v['pim_packages'], pim_width=v['pim_width'],
                                 pim_percent=v['pim_fraction']*100, hybrid_us=v[b['scenario']+'_us'],
                                 full_pim_us=b['full_pim_us'], full_gpu_us=b['full_gpu_us'],
                                 winner_including_endpoints=b['winner']))
    accounting = list(csv.DictReader((root / 'slurm-accounting.csv').open(), delimiter='|'))
    starts = [x['Start'] for x in accounting if x.get('Start') and x['Start'] != 'Unknown']
    ends = [x['End'] for x in accounting if x.get('End') and x['End'] != 'Unknown']
    meta = dict(plotted_at=stamp, source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                simulation_accounting_start=min(starts), simulation_accounting_end=max(ends),
                thor_metadata=str(root / 'thor-measurement-meta.json'), figures=files,
                note='Replot only; no simulations or hardware measurements rerun')
    (out / 'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    lines = ['# Fused-MACAB: every case plotted separately', '',
             '仅重绘已有数据。蓝色为乐观校正，橙色为附加带宽代价，灰色为不校正带宽。',
             '总览中的占比与颗粒数是最优混合候选；全 GPU 或全 PIM 仍可能更快。',
             '每个 R 都与相同 R、CK 的全 PIM 比较，不再共用 R=10 的全 PIM 曲线。', '',
             '| R | 总览：延迟、维度比例、颗粒数 | 8 CK 全部分片曲线 | 12 CK 全部分片曲线 |',
             '|---:|---|---|---|']
    for r in (2, 5, 10):
        lines.append(f'| {r} | [PNG](R{r}-overview.png) / [PDF](R{r}-overview.pdf) | '
                     f'[PNG](R{r}-ck8-all-splits.png) / [PDF](R{r}-ck8-all-splits.pdf) | '
                     f'[PNG](R{r}-ck12-all-splits.png) / [PDF](R{r}-ck12-all-splits.pdf) |')
    lines.extend(['', '全部分片图：每一行是一个带宽模型，每一列是一个 H；',
                  '横轴为 PIM 中间维度占比，纵轴为相对纯 GPU 的延迟（对数刻度），小于 1 表示更快。',
                  '红星标出最优混合点，绿色虚线是该 R/CK 的全 PIM 基线。',
                  '', '[90 个最优点的完整表格](best-points.csv)', '',
                  'PIM 为增强硬件假设下的仿真，不含非线性计算；GPU 通道缩减未实测。',
                  f'绘图时间：{stamp}；原始来源与时间见 provenance.json。'])
    (out / 'README.md').write_text('\n'.join(lines)+'\n')
    print(out)


if __name__ == '__main__':
    main()

"""Plot native DQ partitions against the frozen xinyao_1102 single-bank reference.

Optional plotting dependency: matplotlib (the simulation itself needs PyYAML).
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).resolve().parents[2]/'reference/baseline-xinyao_1102.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((args.run/'channel-analysis.json').read_text())
    baseline = json.loads(args.baseline.read_text())
    labels = ['Q/K + RoPE + QKT', 'V projection/cache', 'PV', 'Gate/Up + GPU export', 'GPU Z + Down + output']
    fields = ['modelled_dq_occupied_us', 'tail_clear_complete_zero_external_phase_us',
              'modelled_dq_idle_outside_tail_clear_work_us']
    colors = ['#d47450', '#277f92', '#d7e2e7']
    legends = ['DQ occupied', 'Whole work phases without DQ', 'Other DQ idle']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(left=.16, right=.95, top=.76, bottom=.22, wspace=.15)
    max_wall = max([s['stage_us'] for s in report['stages'].values()] +
                   [baseline['cases'][case]['latency_us'] for case in report['primary_cases']])
    for ax, title, rows in zip(axes, ['xinyao_1102 | One staging bank', 'xinyao_1103 | Two staging banks'],
            [[baseline['stage_channel_partitions'][case][0] for case in report['primary_cases']],
             [report['stages'][case]['channels'][0] for case in report['primary_cases']]]):
        for y, row in enumerate(rows):
            left = 0
            for field, color, legend in zip(fields, colors, legends):
                ax.barh(y, row[field], left=left, color=color, height=.6,
                    label=legend if y == 0 else None)
                left += row[field]
            ax.text(left+.4, y, f'{left:.3f}', va='center', fontsize=9)
        ax.set_title(title, fontsize=12, pad=16)
        ax.set_xlim(0, max_wall*1.15)
        ax.set_xlabel('Microseconds per layer/step/channel')
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=.2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    total = report['channels'][0]
    fig.text(.035, .94, 'H16 | Channel bandwidth time breakdown', fontsize=22, weight='bold', color='#183342')
    fig.text(.035, .875, f"Stage sum: {baseline['stage_sum_us']:.3f} -> {total['stage_us']:.3f} us   |   DQ occupied: {total['modelled_dq_occupied_us']:.6f} us", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
    fig.text(.035, .035, 'Internal bank reads and SRF broadcast use rank CA, no DQ. GPU arithmetic and achieved concurrent GPU service remain unmeasured.', fontsize=9, color='#526474')
    output = args.output or args.run/'channel-breakdown.png'
    fig.savefig(output, dpi=180, facecolor='white')


if __name__ == '__main__':
    main()

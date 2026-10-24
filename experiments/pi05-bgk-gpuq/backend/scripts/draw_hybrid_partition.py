"""Explain intermediate-dimension splitting and the required GPU partial-sum merge."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    args = p.parse_args()
    font = FontProperties(fname='/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    fig, ax = plt.subplots(figsize=(20, 13))
    fig.patch.set_facecolor('#fafbfd')
    ax.set(xlim=(0, 120), ylim=(0, 100))
    ax.axis('off')
    blue, orange, ink = '#1971b8', '#d97716', '#23344a'
    pale_blue, pale_orange = '#dceefa', '#ffecd6'

    def text(x, y, s, size=14, color=ink, ha='center', va='center'):
        ax.text(x, y, s, fontproperties=font, fontsize=size, color=color, ha=ha, va=va)

    def box(x, y, w, h, label, color, face, size=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1',
                                   facecolor=face, edgecolor=color, linewidth=1.5))
        text(x+w/2, y+h/2, label, size, color)

    def arrow(a, b, color=ink, style='-', curve='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16,
                                    linewidth=2, color=color, linestyle=style,
                                    connectionstyle=curve))

    text(60, 97, '按 FFN 中间维度切分：为什么 PIM 的结果必须交给 GPU 相加？', 23)
    text(60, 93, '示例：D=1024，F=4096；GPU / PIM 各负责 2048 个中间特征（50% 仅用于示意）', 14)

    text(27, 88, 'Gate / Up：沿输出维度 F 切列', 17)
    for x, c, fc, label in ((7, blue, pale_blue, 'GPU 列分片\n[1024, 2048]'),
                             (27, orange, pale_orange, 'PIM 列分片\n[1024, 2048]')):
        ax.add_patch(Rectangle((x, 73), 20, 11, facecolor=fc, edgecolor=c, linewidth=2))
        text(x+10, 78.5, label, 14, c)
    text(27, 70.5, 'W_gate、W_up 均为 [D,F]，两者使用相同切分', 12)

    text(85, 88, 'Down：沿归约维度 F 切行', 17)
    for y, c, fc, label in ((78.5, blue, pale_blue, 'GPU 行分片 [2048,1024]'),
                             (73, orange, pale_orange, 'PIM 行分片 [2048,1024]')):
        ax.add_patch(Rectangle((65, y), 40, 5.5, facecolor=fc, edgecolor=c, linewidth=2))
        text(85, y+2.75, label, 14, c)
    text(85, 70.5, 'W_down 为 [F,D]；两边都保留全部 D 个输出', 12)
    text(60, 65, '同一组中间特征必须配对：Gate/Up 的列分片 ↔ Down 的行分片', 15)
    ax.plot([3, 117], [61, 61], color='#cbd5df', linewidth=1)

    box(14, 35, 83, 23, '', blue, '#f0f7fd')
    box(14, 8, 83, 23, '', orange, '#fff7ed')
    text(16, 55, 'GPU 分支：从分配给 GPU 的 LPDDR 读取权重并计算', 13, blue, ha='left')
    text(16, 28, 'PIM 分支：在分配给 PIM 的 LPDDR 侧计算', 13, orange, ha='left')
    box(1, 28, 10, 13, '输入 X\n[H,1024]', ink, '#edf1f6', 12)
    arrow((11, 37), (17, 44), blue)
    arrow((11, 32), (17, 20), orange)
    for y, c, fc, side in ((44, blue, pale_blue, 'G'), (20, orange, pale_orange, 'P')):
        box(17, y-5, 24, 10, 'Gate + Up\nGELU × 逐元素乘', c, fc, 13)
        box(46, y-5, 17, 10, f'中间激活 Z_{side}\n[H,2048]', c, fc, 12)
        box(68, y-5, 14, 10, '× Down\n[2048,1024]', c, fc, 12)
        box(87, y-5, 8, 10, f'Y_{side}\n[H,1024]', c, fc, 11)
        arrow((41.5, y), (45.5, y), c)
        arrow((63.5, y), (67.5, y), c)
        arrow((82.5, y), (86.5, y), c)
    text(55, 37, 'Y_G：只累加 GPU 负责的 2048 项', 12, blue)
    text(55, 11, 'Y_P：只累加 PIM 负责的另 2048 项', 12, orange)

    box(101, 34, 17, 17, 'GPU 求和\nY = Y_G + Y_P\n[H,1024]', blue, pale_blue, 13)
    arrow((95.5, 44), (100.5, 44), blue)
    arrow((95.5, 20), (109.5, 20), orange, '--')
    arrow((109.5, 20), (109.5, 33.5), orange, '--')
    text(107, 15.7, 'PIM → GPU\n交付最终部分结果', 12, orange)
    text(109.5, 55, '后续算子在 GPU 执行', 12, blue)
    arrow((109.5, 51.5), (109.5, 53), blue)
    text(105.5, 28, '传输', 12, orange)

    text(60, 4, '关键：切掉的是每个输出的“求和项”，不是最终输出坐标；所以两份 [H,D] 必须逐元素相加，不能拼接。', 15)
    fig.text(.5, .014,
             '当前交付模型：PIM accumulator → RDRES → 共享逻辑 → 写入 GPU 可访问的 LPDDR → GPU 读取并求和。\n'
             'FP32 部分结果大小 = H × 1024 × 4 B；H=50 时为 200 KiB，与 PIM 负责多少中间维度无关。',
             ha='center', fontproperties=font, fontsize=12, color=ink)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.075)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(args.output.with_suffix('.'+suffix), dpi=180, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    main()

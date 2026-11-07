# DQ / CA 易读版

**DQ搬数据；共享CA发命令。** 此处把总量、连续窗口和局部命令事件分开呈现，复用job1966885的原始记录，仿真数据及主报告数值均未改变。

1. [总量与最长连续空闲](bus_summary.png)：上排总量拆分不表示时间顺序，下排比较最长一次连续空闲。DQ和CA下排横轴范围不同。
2. [DQ时间轴](dq_readable.png)：绿色是≥1µs的整段空闲，格内是时长；灰色是存在传输且可能含短空隙的区域。算子标注与前后时间轴对齐。
3. [共享CA局部放大](ca_readable.png)：前后同取Down阶段183.3–187.8µs。120次发射变为5次维护命令，留下3.897µs连续空闲；蓝/橙刻线宽度不代表命令持续时间。
4. [共享CA整层时间轴](ca_full_readable.png)：覆盖约210µs。上图按每1µs计算忙碌比例，前后纵轴均为0–50%；下图显示所有≥1µs的精确连续空闲段，前后分别为1段、50段。灰色含命令或短空隙，并非整段忙碌。

这次16个通道忙闲时间相同，因此图示channel0即可，不再重复画16条相同轨迹。共享CA与rank内部CA是两种独立资源。

可缩放矢量版本：[概览SVG](bus_summary.svg)、[DQ SVG](dq_readable.svg)、[CA局部SVG](ca_readable.svg)、[CA整层SVG](ca_full_readable.svg)。

使用Python、Matplotlib和支持中文的字体复现，原生归档保持只读：

```powershell
python experiments/pi05-unified-gemv/redraw_command_figures.py `
  --archive ../backups/xinyao_1107-native-1966885.tar.gz `
  --reference reference/commands-1107 `
  --output reference/commands-1107/readable
```

Windows默认使用微软雅黑，Linux默认查找Noto Sans CJK，也可用`--font`指定字体文件。脚本核对8份输入文件与既有原始哈希，重新计算并核对忙碌总量和最长空闲，证据保存在[plot-evidence.json](plot-evidence.json)。

整层CA图使用同样参数运行`experiments/pi05-unified-gemv/redraw_ca_full.py`。它另行验证各时间格的忙碌积分等于原始CA busy总量，最后不足1µs的格子按实际时长归一化；各格数值和所有长空闲段保存在[ca-full-evidence.json](ca-full-evidence.json)。

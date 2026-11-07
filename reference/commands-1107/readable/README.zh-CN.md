# DQ / CA 易读版

**DQ搬数据；共享CA发命令。** 此处把总量、连续窗口和局部命令事件分开呈现，复用job1966885的原始记录，仿真数据及主报告数值均未改变。

1. [总量与最长连续空闲](bus_summary.png)：上排总量拆分不表示时间顺序，下排比较最长一次连续空闲。DQ和CA下排横轴范围不同。
2. [DQ时间轴](dq_readable.png)：绿色是≥1µs的整段空闲，格内是时长；灰色是存在传输且可能含短空隙的区域。算子标注与前后时间轴对齐。
3. [共享CA局部放大](ca_readable.png)：前后同取Down阶段183.3–187.8µs。120次发射变为5次维护命令，留下3.897µs连续空闲；蓝/橙刻线宽度不代表命令持续时间。

这次16个通道忙闲时间相同，因此图示channel0即可，不再重复画16条相同轨迹。共享CA与rank内部CA是两种独立资源。

可缩放矢量版本：[概览SVG](bus_summary.svg)、[DQ SVG](dq_readable.svg)、[CA SVG](ca_readable.svg)。

使用Python、Matplotlib和支持中文的字体复现，原生归档保持只读：

```powershell
python experiments/pi05-unified-gemv/redraw_command_figures.py `
  --archive ../backups/xinyao_1107-native-1966885.tar.gz `
  --reference reference/commands-1107 `
  --output reference/commands-1107/readable
```

Windows默认使用微软雅黑，Linux默认查找Noto Sans CJK，也可用`--font`指定字体文件。脚本核对8份输入文件与既有原始哈希，重新计算并核对忙碌总量和最长空闲，证据保存在[plot-evidence.json](plot-evidence.json)。

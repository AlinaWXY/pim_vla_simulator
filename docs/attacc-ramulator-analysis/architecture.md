# 架构、配置和执行链的增量

每个有效文件的逐项记录见 [file_review.csv](file_review.csv)，精确代码证据见 [file_delta.csv](generated/file_delta.csv)。本节解释这些增量怎样影响结果，以及哪些数字是输入/设计选择。

## 1. HBM3 基线也被重写

| 项目 | 原版 HBM3 | ATTACC HBM3 与 HBM3-PIM | 来源判定 |
| --- | --- | --- | --- |
| 层级 | channel/pch/BG/bank/row/column | 增加 rank，在 pch 与 BG 之间 | 作者注释引用 JESD238A；标准全文未核对，PARTIAL |
| 组织预设 | 3 个；其中 8Gb 条目 density 写为 6<<10 | 12 个，1/2/3/4 ranks 对应 4/8/12/16-Hi | 逐个原始值见 organization_presets.csv；PARTIAL |
| DQ、默认 channel_width | 128、64 | 32、32 | 模型表示改变，不能解读为整个 HBM 只有 32 根 DQ |
| internal_prefetch_size | 2 | 8 | 作者接口表示；8×32bit/8=32B 一次事务 |
| 列数 | 64 | 32 | 列地址以 32B 为单位；一个 bank 行 1024B |
| density 校验 | 未计 rank/prefetch | 计入 rank 和 internal_prefetch_size | 算术实现 CODE/DERIVED |
| 速率 | 2.0 Gbps 一档 | 普通 HBM3 五档，PIM 每档分 PC/NPC 共十档 | 作者预设；大部分 DRAM ns 引用 HBM2 配置 |
| tCK 计算 | rate/2 | rate/4，QDR 表示 | 代码公式已确认，DERIVED |
| RRDS、FAW、RREFD、CCDS、RD/WR turnaround | pseudochannel 层约束 | 多项移到 rank；CCDL 仍约束同 BG | 作者时序层级选择；注释参考 JESD235C，不代表标准原文已逐条验证 |
| PREA action | 遍历较浅的层级 | PREab 函数改名 PREA，并支持 rank 层级 | CODE |

12 个组织的行数只有三类：8192/16384/32768。rank=1 的 density 标签为 2/4/8Gb，rank=2 为 4/8/16Gb，rank=3 为 6/12/24Gb，rank=4 为 8/16/32Gb。其余字段统一为 DQ32、channel1、pch2、BG4、bank4、column32。**这里 density 按代码公式是每个 channel 的容量**；默认 `HBM3_8Gb_2R + channel16` 合计 16GiB/HBM，不能将预设名中的 8Gb 误认为整个 HBM 的容量。

刷新表增加 6/12/24/32Gb 项：`tRFC={160,260,310,350,410,450,610,650}ns` 对应 `2/4/6/8/12/16/24/32Gb`。160/260/350/450 已在原版；其他四值没有找到逐项原始出处。`tREFISB={244,122,82,61}ns` 的注释对应四种堆叠高度，但实现直接使用 rank 数作数组下标，有偏移问题。见 [findings.md](findings.md)。

`timing_presets.csv` 保存的是**原始表值**。加载后会重算 `tCK_ps`、`nRFC`、`nREFISB`，随后允许 YAML 中 `nXXX` 或 `tXXX` 覆盖多数参数。默认 5.2Gbps/8Gb/2R 下：tCK=769ps，nRFC=456，nREFISB=107（实际取82ns）。两项刷新周期使用原版 `JEDEC_rounding`：`((t_ps*1000//tCK_ps)+974)//1000`，不是简单浮点 ceil。

## 2. 新增地址映射

四个注册名为 `HBM3-Base`、`HBM3-Custom`、`HBM3-PIM`、`HBM3-PIM-Custom`。共同先移除 32B 事务的低 5 位；没有再次从列位数中扣除 prefetch 位。

| 映射 | 从地址高位到低位（不含 byte offset） | 来源 |
| --- | --- | --- |
| HBM3-Base / HBM3-PIM | Ch, Pch, Rank, BG, Bank, Row, Column | 作者代码，CODE；与 PIM trace 默认位序相配 |
| HBM3-Custom | Row, Bank, Rank, Column, BG, Pch, Ch | 作者普通 HBM3 对照映射，CODE |
| HBM3-PIM-Custom | Pch, Rank, BG, Bank, Row, Column, Ch | 以实际位提取循环为准，CODE；注册描述未完整列 BG |

默认 PIM 地址位宽依次为 `4/1/1/2/2/14/5/5`（最后 5 为 byte offset）。这些是仿真数据布局，未提供与厂商地址扰码、修复映射或实际 PIM 控制接口的对应表。三个 rank 的组织虽然出现在预设中，简单按 `calc_log2(count)` 切位的映射不能自然完整表示 3 个 rank，需另行验证/修改。

## 3. 前端、控制器、调度与完成判断

- `PIMLoadStoreTrace` 读取“操作名 地址”两列，支持十类 PIM 请求和 LD/ST，既可十六进制也可十进制。每 tick 连续注入，直到被 memory system 拒绝或 trace 发完；普通 LoadStoreTrace 也被改成批量注入。此改变会影响排队/到达时间。
- 新增 HBM3 与 HBM3-PIM 控制器，尝试每周期发一个 row 和一个 column 命令。PIM 控制器另有 PIM 队列，调度顺序大体为 active → priority（刷新）→ PIM → 普通读写；PIM 前置 ACT 不像普通请求那样转入 active 队列。
- 水位0.2/0.8、priority buffer 的 `512*3+32=1568`、普通 `ReqBuffer.max_size=32` 都可追溯到原版通用控制器/请求类；**不是 ATTACC 新测量的硬件容量**。ATTACC 沿用了这些值，新增 PIM 队列也沿用 ReqBuffer。原版 enqueue 判断为 `size <= max_size`，所以不能把32直接当成已经严格模拟的实际队列上限。
- PIM scheduler 和修改后的 FRFCFS 均保护待服务 row hit，避免另一请求过早关掉需要的行；还会重排 ready 请求。PIM 队列注释虽写 in-order，实际算法不是完全严格顺序发射。
- BARRIER 到队首就删除；屏障后请求的 ACT/PRE 仍可提前。它是作者的调度机制，不是已测量的全局硬件同步延迟。
- `is_pending()` 扩展到控制器和 memory system；main 在前端发完且队列排空后结束。仅普通 Read 具有未来返回事件，PIM 缺少独立计算完成事件。
- `HBM3TraceRecorder` 为每 channel 记录发射周期/命令/地址，以 `second` 布尔变量每两次 plugin update 增加一周期。它依赖控制器双次回调，不使用新增的 `get_clk()`。计能不能直接拿请求数当 ACT/PRE 发射数。
- `AllBankHBM3` 每 nREFI 给每个 pseudochannel 发刷新请求；变量虽叫 `m_num_ranks`，取的是 pseudochannel 数。`No` 模块关闭刷新，只是显式实验选项。默认刷新 request 地址与 REFab 的 channel scope 是否一致，见 findings。

以上逻辑来源均为作者代码；没有独立的硬件测量数据支撑其队列优先级、广播开销或双发射时序。71 条 HBM3-PIM 约束声明全部列在 [timing_constraints.csv](generated/timing_constraints.csv)；其数值按表达式引用 [timing_sources.csv](timing_sources.csv)。固定2周期 ACT、4次激活窗口和 turnaround 的 ±1/+2 来自作者/继承的控制总线模型，不能外推成 LPDDR5 PIM 实测值。表中“声明了约束”也不保证所有 scope 路径都会执行该约束，详见 findings。

## 4. 三个 trace 生成器增加的数据与假设

| 数据/算法 | 取值或规则 | 来源性质 |
| --- | --- | --- |
| 硬编码组织 | channel16、pch2、rank2、BG4、bank4、row2^14、column2^5、prefetch32B | 作者默认 HBM3 配置；不从 YAML 反读 |
| 规模常量 | n_attacc8、n_hbm5、max_n_hbm8 | 作者实验/地址空间选择；部分只用于地址粒度或未消费 |
| CLI 默认 | dhead128、nhead64、seqlen2048、maxlen4096、dbyte2 | 输入默认值，不需要物理实验来源；以代码为准，README nhead 示例写1 |
| 数据类型 | n_mac=int(32/dbyte)，默认16 | 按32B数据粒度推导；脚本顶部 data_size=16 后被 CLI 的字节值覆盖 |
| K/V 划分 | v_offset=2^23；partition_size 由 maxlen×dhead 与并行因子计算 | 作者地址规划；常量没有物理“latency”含义 |
| BA 的 score MAC 次数/有效 channel/head | ceil(L/16)×ceil(dhead/(4×n_mac)) | 作者循环；context 次数相同（乘法顺序互换） |
| BG 的 MAC 次数/有效 channel/head | score 与 context 均为 ceil(L/16)×ceil(dhead/n_mac) | 作者循环；MACSB，对列/bank/row 进行映射 |
| BUFFER 的 MAC 次数/有效 channel/head | ceil(L/2)×ceil(dhead/n_mac) | 作者循环；MACPB，在 BG/rank/bank 间交错 |
| head 分配 | num_itr=ceil(n_head_per_hbm/16)，每组使用最多16个 channel | 作者软件划分；不是 Ramulator 自动得出的映射 |
| 搬移分批 | 多处每16组 MAC 搬输出，跨 head 重叠；有 dummy MAC | 软件 trace 策略；会影响请求计数，不能与数学最小操作数直接等同 |

执行顺序围绕 WRGB→score MAC→MVSB→SFM→MVGB→context MAC→MVSB，插入 BARRIER 并交错不同 head。脚本中的“8/4/16tCK”注释不生成带时间的请求，因此不能作为 latency 输入。预设 L、head 维度是工作负载输入，扩展 LPDDR5 PIM 时需重新生成地址/广播/并行度，而不是只替换一列 MAC latency。

## 5. 外层 PIM 数据链：不是 C++ Ramulator 新接口

| 文件 | 作用及需核查的数据 |
| --- | --- |
| `src/ramulator_wrapper.py` | 生成 YAML/trace，解析总周期及五类请求计数，缓存结果，换成时间与字节流量；tCK硬编码0.769ns；每HBM取 ceil(num_ops/5)；fast_mode 用64 heads及 ceil 分组外推 |
| `src/config.py` | PIM mem/SRAM/ALU/IO/comm能量表，5 HBMs、8 AttAcc、16GiB/HBM、内部带宽和softmax参数；详见 energy.md |
| `src/devices.py::PIM` | score 调用一次完整 attention trace，context 直接返回0，softmax另算；按字节/FLOPs累加能量，乘 AttAcc 数量 |
| `src/model.py` / `src/type.py` | 构造 score、context、softmax 等 layer；定义 dbyte/FLOPs/操作次数和 BA/BG/BUFFER 类型，是外层输入契约 |
| `src/system.py` / `main.py` | 组合GPU与PIM系统、逐层/逐token结果、CLI功耗开关以及流水/FF优化；属于外层系统模型，不能当作单命令延迟来源 |

直接影响 PIM 的外层常量：基础带宽670.4GB/s/HBM；无功耗限制时 BA/BG/BUFFER 的 BW_SCALE=16/8/1，受限时9/3/1；FLOPS_PER_HBM=MEM_BW×opb（默认opb1）；SOFTMAX_MEM_BW=670.4GB/s×num_hbm，SOFTMAX_FLOPS数值等于该带宽值。它们是作者模型设定；670.4不能由“5.2Gbps×1024/8=665.6GB/s”精确重建，差异需要澄清，不能自动标为标准峰值带宽。

接口配置 NVLink3/4=600/900GB/s，PCIe4/5=64/128GB/s；实际通信耗时使用一半的配置值作为单方向带宽。代码数值已确认，但端点、双向口径、可持续带宽及测量来源未在这些语句中闭合。`MAX_COMPUTE_UTIL=0.8/MAX_OFF_MEM_BW_UTIL=0.85` 用于外层 xPU 模型，PIM 类未按该参数对 Ramulator 周期再缩放。

## 6. 构建和兼容性也属于差异

作者删除了 tests、部分前端、RowHammer插件，以及 DDR3/4/5、LPDDR5、HBM/HBM2 的构建注册；物理源码没有因此从目录删除。原版共有 action helper 的 PREab→PREA 改名也可能影响这些未编译的实现。直接套用安装脚本会失去本项目需要的 LPDDR5 构建入口，因此本分支只保存审计结果，未把 ATTACC 全套补丁施加到工作源码。

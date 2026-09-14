# 缺口、实现疑点与需要的资料

这是来源审计及静态代码检查，以下没有做完整仿真复现或修改作者源码。明确的代码数值/控制流不一致可以直接核对；对运行结果的影响仍需针对性执行验证。排序按对latency/energy解释的重要性。

## 已定位的边界和疑点

| ID | 发现与精确证据位置 | 影响/结论边界 |
| --- | --- | --- |
| F01 | `hbm3_pim_controller.cpp::tick` 仅为Read添加未来depart，PIM最终命令直接remove；`is_pending`只看请求队列 | 没有独立PIM结果完成延迟；总周期末尾不能自动包含ALU pipeline drain |
| F02 | `HBM3-PIM.cpp::set_timing_vals` 重写 tCK_ps；wrapper `__init__` 固定0.769ns | 表内1300并非默认有效tCK；变更速率还需修改wrapper，否则转换错误 |
| F03 | `src/config.py` 的io系数未×8；wrapper流量以32B为粒度；`devices.py::PIM`直接相乘 | 注释与单位链不一致；不能无说明地把算出的数当正确J值 |
| F04 | wrapper `run` 的BUFFER mem_acc×1，`output`命中缓存时×2；fast_mode新运行返回time不乘group，缓存路径乘group | 缓存状态可能改变输出定义；首次df为空时还有先run再走缓存分支，因此不能笼统等同为“所有首次调用都×1” |
| F05 | `PIM.get_time_and_energy` 的score分支跑完整attention trace，但只计score FLOPs；context分支返回0 | context算术能量可能漏记，需核对作者口径 |
| F06 | `HBM3-PIM.cpp`及HBM3补丁：tREFISB数组{244,122,82,61}，用rank数直接索引 | 注释按1/2/3/4R顺序，代码却取122/82/61/0。默认2R取82ns；4R取隐式初始化0。该字段未被默认AllBank刷新使用，不应夸大为必然改变默认实验 |
| F07 | action补丁中 `ACTSB`跨全部rank/BG，`PRESB`却只关匹配rank/BG的bank，且与PREPB代码相同 | same-bank状态范围不对称；混合开闭/换行时需要验证，不能把命名当正确广播的证明 |
| F08 | action补丁 `BankGroup::PIMPerBankActions` 在BG node上比较rank与 `node->parent->parent->id`，BG与 `node->parent->id` | 对七层组织而言这分别是pch ID和rank ID，与ACTPB使用的祖先层级不一致；目标bank时序传播可能错误 |
| F09 | 两个HBM3 controller的tick无条件在第二次调度调用中读取 `req_it->command`；此前可能未找到请求或已remove该iterator | 存在未定义行为风险；需sanitizer/小trace验证后方可把运行周期当稳定基准 |
| F10 | `HBM3-PIM.cpp` 将SFM scope设为channel，而SFM的搬移约束声明在pch/rank/BG；原版 `src/dram/node.h::check_ready` 到scope即返回 | SFM不会检查这些下层incoming约束。数据表列的是声明，不足以证明固定SFM latency已经生效；update_timing仍可影响后续命令 |
| F11 | AllBankHBM3按pch发REQ，REFab scope为channel；PREA状态动作关闭整个channel；REFab等待约束主要在pch | 刷新请求粒度与状态/ready检查范围需要联调确认；不能把名字“AllBank”当成作用域完全一致 |
| F12 | 所有trace注释MACAB8tCK/SFM16tCK；C++表PC6/NPC4且没有SFM算术latency | 注释不是生效参数；此前据此推算延迟的结论应更正 |
| F13 | 三rank组织使用简单log2位切分；`NoTranslation.max_addr=2^31`与16GiB默认总容量也不是同一口径 | 非默认组织/地址范围应做映射验证。前端未实际使用声明的Translation配置来限制这些PIM地址，不能说整个HBM只能访问2GiB |
| F14 | bank trace `run_attention` 的“Head1: Context”段仍索引 `cmd_context_mac[i]`和`cmd_context_mvsb[i]` | 按控制流会再次取Head0列表；未执行数值结果校验，因此不能据此保证多head trace正确 |
| F15 | dummy controller `is_pending`恒true；dummy memory bool函数无return；多个旧DRAM被移出build且共享PRE helper改名 | 安装补丁偏重作者专用运行路径；不要认为所有原版前端/设备仍兼容 |

代码路径均相对本目录 [sources/attacc](sources/attacc/)；修改原版文件的证据在 [patches](sources/attacc/pim_ramulator_src/patches/) 或 [完整差异](generated/effective.diff)。F10的继承行为可核对[固定原版 node.h](https://github.com/CMU-SAFARI/ramulator2/blob/b7c70275f04126c647edb989270cc429776955d1/src/dram/node.h)。

这些发现不等价于“论文结论错误”；它们限定了当前公开实现能支持哪些逐命令latency/energy解释，哪些还缺验证。此次没有替作者补齐或修改这些行为。

## 向作者或资料提供方索取的具体材料

| 优先级 / 资料 | 必须包含的字段 | 能解决的问题 |
| --- | --- | --- |
| P0：MAC功耗节流计算原表/脚本 | 各速率、BA/BG/PB拓扑、激活/读取次数和pJ、单位覆盖范围、功耗预算W、VDD/IDD来源、峰值或平均窗口、取整规则、NPC最小间隔 | 唯一复算nCCDAB/nCCDSB的6/7/8等周期；确认预算是每stack还是每AttAcc |
| P0：GEMV/accumulator RTL及综合时序报告 | 位宽/精度、每命令元素数、PE数、pipeline stages、启动间隔II、结果完成latency、时钟/电压/工艺corner、SDC、DC版本 | 区分666MHz、命令吞吐与结果完成时间；建立MAC结果依赖 |
| P0：数据路径与控制命令规范 | WRGB/MVSB/MVGB/SFM/SETM/SETH每次传输量、广播范围、前置条件、buffer地址/容量/端口、busy和依赖、可重叠命令矩阵 | 给每个新增操作建立真正的完成时间与资源约束 |
| P0：PIM/HBM能耗工作表 | 0.44/1.01/1.23/0.5等值的原始输入、路径长度、缩放、活动率、pJ/bit或pJ/B；ALU0.32计什么操作；context如何计能 | 闭合每个系数来源，解释io单位差异和是否遗漏能量 |
| P1：采用的标准/厂商timing表 | JESD238A和相关JESD235C版本；具体表号、速度档、density/rank/stack定义；tCCD、tRRD、tRFC/tRFCSB/tREFISB/tRREFD | 验证普通HBM3时序、刷新新增值以及rank作用域 |
| P1：修改后的FinCACTI工程及输出 | SRAM容量/宽度/端口/bank、技术参数、电压、读写能量、访问时间、面积和校准目标 | 复算0.0034及SRAM搬移/softmax延迟 |
| P1：有效命令轨迹及预期结果 | 冷行/命中/换行/刷新/多head/PB跨rank用例；每周期已发命令；有效最终输出和总周期；原始程序提交与构建环境 | 检查广播、barrier、trace索引、pipeline drain和cache一致性 |
| P1：softmax与互连模型 | 算法近似、长度依赖、吞吐/latency、功耗数据；670.4GB/s来源；NVLink/PCIe单双向口径和S6原文 | 核实外层time/energy与C++请求的分工 |

如果拿到的是Samsung LPDDR5X-PIM资料，应索取相同类别，但以该器件真实命令和组织为准。已有Ramulator LPDDR5基础timing只能覆盖普通访存，不会自动补出PIM计算、buffer、广播、能量和功耗预算。

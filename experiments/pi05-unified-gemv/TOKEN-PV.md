# QKV 融合、V 本地写回与 KV token 分片的 PV

本版以 `810e01e` 为对照。QKV共同按奇偶token分rank，每份X仅经DQ暂存一次，每个SRF输入片段消费48个Q/K/V权重列。V cache/PV也按token分片，以本地cast、打包、读改写替代V的GPU往返。RoPE、QKT、Gate/Up、Down保留各自算术与数据流。

## 张量与地址

每 head 使用不同 channel 上的两个 PIM rank，均为各自 channel 的物理 rank1。head 为 h、KV token 为 k 时，channel 为 `2*h+k%2`，本地 KV 编号为 `k//2`。每 rank 的 16 个 bank 都参与计算。

| 阶段 | 每 rank 输入 | 每 rank 驻留矩阵 | 每 rank 输出 |
|---|---|---|---|
| QKV融合 | X[8,1024]，16 KiB | Wqkv[1024,768]，1.5 MiB | Q、K、V各[8,256]，本地写回 |
| PV | P[16,496]，15.5 KiB | Vcache[496,256]，248 KiB | partial O[16,256]，FP32 16 KiB |

整批P[16,496]共15.5 KiB先写完，再开始PV，一次计算全部16个query。496个本地KV位置包括484个prefix、8个新token、4个padding。两rank对应的P取自GPU已完成softmax的同一head矩阵的不同KV列，全部query都要计算。

每个bank的feature映射相同：`pair=f%128; bank=pair//8; lane=2*(pair%8)+f//128`。融合Wqkv区域为每bank `[393216,491520)`，按`[input fragment64][Q16/K16/V16 columns][input lane16]`排列，共96 KiB/bank、24 MiB/全部rank。新cache区域为每bank `[163840,180224)`，不覆盖Q/K cache、FFN或暂存区。独立V诊断仍使用旧Wv `[131072,163840)`，不与融合路径重复计算。

cache中一个元素的bank内地址为：

```text
local_k = k//2
cache_offset = 163840 + ((local_k//16)*16 + feature_lane)*32 + (local_k%16)*2
```

即 `[本地KV块31][feature lane16][KV lane16] BF16`。实际使用15.5 KiB/bank，保留16 KiB；prefix必须在初始cache构建时采用该布局，本轮逐步latency不计一次性旧cache迁移。

## V 本地写回

每bank的共享FP32 output buffer容量为2 KiB。QKV融合同时保留 `[8 token][48 Q/K/V feature]`，占1.5 KiB，其中V占512 B。V输出需要按feature收集8个新token并转换为BF16，写入本地KV尾块480–495。该块包含4个旧prefix、8个新token和4个padding。

通过rank描述符展开为真实单bank操作：读取目标32 B列、保留旧prefix、cast/gather新值并设置padding、写回32 B列。旧prefix保护由内部读改写实现，不能直接用新V覆盖整列。每bank共16列，所有rank合计内部读取128 KiB、写入128 KiB；只有64 KiB是新V的有效BF16数据。

cast及本地端口按现有配置计时，LOCALRD/LOCALWR保留rank CA竞争、DRAM行准备、ACT1/ACT2 owner/deadline与写恢复。新cache的准备状态依赖完整V计算和全部本地写完成。独立PV实验显式声明cache已准备；串联实验必须经过实际V写回。此路径是时序和数据映射模型，不执行神经网络数值推理。

这里的2 KiB约束是MAC算子共享输出空间，不是全部片上存储总预算；32 B本地读改写暂存和原有RoPE/score辅助结构另行保留。GATHER使用rank端口2 CK、CAST处理8元素1 CK，属于配置的架构假设，尚未细化输出SRAM端口或标定电路成本。

## 整块SRF、P暂存与GPU归约

每bank独立配置1 KiB输入SRF和2 KiB可复用FP32 output buffer。SRF按整批装入、整批消费，所有有效子片段计算完成后才允许任何输入覆盖；没有两套流式交替缓冲。QKV的每个子片段为8 token×16 hidden BF16=256 B，一批装入4片，使用全部1 KiB；PV每片为16 query×16 KV BF16=512 B，一批装入2片，末批只有一个有效片段。子片段是同一物理SRF内的不重叠寻址视图，不是额外SRAM。

P整批暂存采用31个512 B片段，编号就是本地KV块。片段交错存入bank0、bank4，分别8 KiB和7.5 KiB，沿用整批写入、seal、分片LOCALRD和同rank广播。32 B物理列与rankCA、DRAM时序约束不变。

整块SRF的等待只限制实际LOCALRD及输入覆盖。MAC发出时权重已锁存，源bank的PRE/ACT仍可按原DRAM时序在PE计算尾部准备下一行；这些命令继续经过rank CA及ACT1/ACT2 owner/deadline检查。

PV每片段执行16列MAC，每列服务16 CK。每bank的16 query×16 feature FP32输出占1 KiB，所有31个KV块累加完成后导出16 KiB/rank。相比旧布局的62片×8列，新布局31片×16列保持MAC数量、scalar MAC量和有效cache读取量不变。QKV使用1.5 KiB、PV1 KiB、Gate/Up2 KiB、Down256 B，均受独立2 KiB输出容量约束。

```text
partial_A = P[:, even_KV] @ Vcache_A
partial_B = P[:, odd_KV]  @ Vcache_B
O = partial_A + partial_B  # GPU求和
```

GPU接收两rank的FP32部分结果，并按query和feature索引相加。P已在GPU做完整softmax，无需在rank内重新softmax。当前对P输入与两边结果的DQ传输计时；GPU侧P分片打包、加法核、调度和内存访问尚未计时。

## 流量与比较口径

全部16channel、每层每step：

| 项目 | 810e01e | 新路径 |
|---|---:|---:|
| Q/K与V的X输入合计 | 512 KiB | 256 KiB |
| V FP32导出 | 128 KiB | 0 |
| V GPU prefix读取 | 64 KiB | 0 |
| V GPU cache写回 | 128 KiB | 0 |
| P输入 | 496 KiB | 248 KiB |
| PV FP32结果导出 | 128 KiB | 256 KiB |
| 主路径外部DQ总量 | 5216 KiB | 4520 KiB |

主路径总量减少696 KiB。修改前五阶段总和与修改后四阶段（融合attention、PV、Gate/Up、Down）总和比较；独立V诊断、combined FFN、serial control不重复计入。对attention块单独比较时，修改前Q/K链与V相加，修改后取融合chain。DQ空闲按原始数据传输时间线计算，进一步区分完整无外部传输工作窗口、其它空隙和DQ/总CA同时空闲。未注入并发Thor请求，资源空闲不能直接解释为实测GPU可用带宽。

默认runner启用本路径；`--legacy-pv-feature-sharding`保留810e01e的独立V GPU回写和feature分片PV；`--legacy-v-feature-sharding`或`--direct-input-baseline`还会关闭V token投影。`--no-qkv-fusion`提供token cache/PV但分开投影的诊断模式。API的`pv_token_sharding`和`qkv_fusion`默认false以保留旧调用语义。其它旧模式需明确组合布局开关，不能仅靠旧dispatch开关复现历史布局。

QKT仍对每个K token的feature向量与全部Q行点积，得到score的一列；K的逻辑转置不需要整矩阵DRAM转置。K cache的feature连续布局与PV所需的KV连续布局不同。

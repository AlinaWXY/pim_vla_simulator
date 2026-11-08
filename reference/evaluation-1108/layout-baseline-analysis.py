"""Full-attention accounting; never turn analytical bytes/MACs into simulated latency."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PACKAGE = ROOT / 'publish-xinyao_1104_O_Projection'
REF = PACKAGE / 'reference/o-projection'
H, M, S, D, X, R, B = 8, 16, 992, 256, 1024, 16, 16
BF16, FP32, BURST, TCK = 2, 4, 32, 0.938
KIB, MIB = 1024, 1024**2

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

current = {stage: read(REF / 'both' / stage / 'summary.json') for stage in ('chain', 'pv')}
stream = {stage: read(REF / 'oproj_only' / stage / 'summary.json') for stage in ('chain', 'pv')}
channel_report = read(REF / 'both/channel-analysis.json')

# Baseline: two ranks/head; both KV caches and QKV outputs split by RoPE-paired
# feature subsets. Each rank holds all M queries. O is input-row sharded so
# both designs keep PV on-rank and export sixteen [M,X] O partials.
dq = [
    ('QKV input X', R*M*X*BF16, R*(M//2)*X*BF16),
    ('Q forwarding: read source', 0, H*M*D*BF16),
    ('Q forwarding: write peer', 0, H*M*D*BF16),
    ('QKT FP32 score export', 2*H*M*S*FP32, H*M*S*FP32),
    ('PV BF16 probability input', 2*H*M*S*BF16, H*M*S*BF16),
    ('PV to O external transfer', 0, 0),
    ('O BF16 partial export', R*M*X*BF16, R*M*X*BF16),
]
base_dq = sum(a for _, a, _ in dq)
token_dq = sum(b for _, _, b in dq)
assert (base_dq, token_dq) == (2512*KIB, 1640*KIB)
assert token_dq == sum(s['external_DQ_bytes'] for s in current.values())
assert token_dq == sum(s['external_DQ_bytes'] for s in stream.values())
assert current['chain']['traffic_bytes']['input_srf_multicast_source'] == 256*KIB
assert current['chain']['traffic_bytes']['input_srf_multicast_destination'] == 4096*KIB
assert current['chain']['traffic_bytes']['pim_internal'] == 24*MIB
assert current['pv']['traffic_bytes']['pim_internal'] == H*S*D*BF16 + 8*MIB

weights = []
for name in ('Wq', 'Wk', 'Wv', 'Wo'):
    unique = X*H*D*BF16
    weights.append(dict(matrix=name, unique_bytes=unique,
        baseline_active_bytes=unique, current_active_bytes=2*unique,
        current_declared_resident_bytes=(16 if name=='Wo' else 2)*unique,
        baseline_copies=1, current_active_copies=2,
        current_declared_resident_copies=16 if name=='Wo' else 2))
assert sum(w['baseline_active_bytes'] for w in weights) == 16*MIB
assert sum(w['current_active_bytes'] for w in weights) == 32*MIB
assert sum(w['current_declared_resident_bytes'] for w in weights) == 88*MIB

macs = [
    ('QKV', 3*M*X*H*D, 3*M*X*H*D),
    ('QKT', H*M*S*D, H*M*S*D),
    ('PV', H*M*S*D, H*M*S*D),
    ('O', H*M*D*X, 2*H*M*D*X),
]

def csv_write(name, fields, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

dq_rows = [dict(transfer=n, baseline_KiB=a/KIB, current_KiB=b/KIB,
               saved_KiB=(a-b)/KIB) for n,a,b in dq]
csv_write('dq-breakdown.csv', list(dq_rows[0]), dq_rows)
csv_write('weight-capacity.csv', list(weights[0]), weights)
mac_rows = [dict(operator=n, baseline_MACs=a, current_MACs=b,
                baseline_ideal_full_utilization_us=a/(R*B*16)*TCK/1000,
                current_ideal_full_utilization_us=b/(R*B*16)*TCK/1000) for n,a,b in macs]
csv_write('mac-service-demand.csv', list(mac_rows[0]), mac_rows)

chain_phases, pv_phases = current['chain']['phase_us'], current['pv']['phase_us']
qkv_names = ['projection_gpu_input_stage', 'projection_local_input_compute',
    'v_local_cache_spill', 'projection_local_spill', 'projection_gemv_release', 'rope_local']
groups = [
    ('QKV + local cache/RoPE', sum(chain_phases[n] for n in qkv_names)),
    ('Q forwarding', sum(v for k,v in chain_phases.items() if k.startswith(('q_export', 'q_remote')))),
    ('QKT + score spill/export', sum(v for k,v in chain_phases.items() if k.startswith('qk_'))),
    ('PV + local O input staging', sum(v for k,v in pv_phases.items() if k.startswith('pv_'))),
    ('O input/compute/export', sum(v for k,v in pv_phases.items() if k.startswith('o_projection'))),
    ('Phase boundary overhead', sum(s['phase_boundary_us'] for s in current.values())),
]
native_total = sum(s['latency_us'] for s in current.values())
assert math.isclose(sum(v for _,v in groups), native_total, abs_tol=1e-9)
assert math.isclose(native_total, 104.410656, abs_tol=1e-9)
stream_total = sum(s['latency_us'] for s in stream.values())
assert math.isclose(stream_total, 94.860878, abs_tol=1e-9)
latency_rows = [dict(stage=n, current_model_us=v, baseline_model_us='NOT_SIMULATED') for n,v in groups]
csv_write('current-native-latency.csv', list(latency_rows[0]), latency_rows)

# Same-dimensional FP64 check of the two algebraic decompositions, including
# feature subsets that keep each (j,j+128) RoPE pair on the same rank.
# This does NOT validate BF16 rounding, physical address packing or native timing.
rng = np.random.default_rng(20261106)
f0 = np.r_[np.arange(D//4), np.arange(D//2,3*D//4)]
f1 = np.setdiff1d(np.arange(D), f0)
assert len(f0)==len(f1)==128 and len(set(f0)&set(f1))==0
checks = []
for head in range(H):
    q = rng.normal(0, .2, (M,D))
    k = rng.normal(0, .2, (S,D))
    v = rng.normal(0, .2, (S,D))
    wo = rng.normal(0, .2, (D,X))
    score = q @ k.T / np.sqrt(D)
    fs = (q[:,f0] @ k[:,f0].T + q[:,f1] @ k[:,f1].T) / np.sqrt(D)
    ts = np.empty_like(score)
    for parity in (0,1):
        ts[:,parity::2] = q @ k[parity::2].T / np.sqrt(D)
    prob = np.exp(score-score.max(axis=1,keepdims=True))
    prob /= prob.sum(axis=1,keepdims=True)
    y = (prob @ v) @ wo
    fy = (prob @ v[:,f0]) @ wo[f0] + (prob @ v[:,f1]) @ wo[f1]
    ty = sum((prob[:,p::2] @ v[p::2]) @ wo for p in (0,1))
    errors = dict(head=head, feature_score_max_abs=float(np.max(np.abs(fs-score))),
        token_score_max_abs=float(np.max(np.abs(ts-score))),
        feature_output_max_abs=float(np.max(np.abs(fy-y))),
        token_output_max_abs=float(np.max(np.abs(ty-y))))
    assert max(v for k,v in errors.items() if k!='head') < 1e-12
    checks.append(errors)

# DQ busy is sum of payload bursts per balanced channel, not wall latency.
all_dq = channel_report['channels'][0]
burst_ck = all_dq['modelled_dq_occupied_us']*1000/TCK/(all_dq['external_DQ_bytes']/BURST)
assert math.isclose(burst_ck, 2, abs_tol=1e-9)
dq_busy = dict(baseline_us_per_channel=base_dq/R/BURST*burst_ck*TCK/1000,
               current_us_per_channel=token_dq/R/BURST*burst_ck*TCK/1000,
               interpretation='Payload burst occupancy only, NOT transfer completion or total latency')

sources = [REF/'both'/s/'summary.json' for s in ('chain','pv')]
sources += [REF/'oproj_only'/s/'summary.json' for s in ('chain','pv')]
sources += [REF/'both/channel-analysis.json']
sources += [PACKAGE/'experiments/pi05-unified-gemv/source'/n for n in
    ('qkv_token_layout.py','o_projection_layout.py','rank_staging.py')]
sources += [PACKAGE/'experiments/pi05-bgk-gpuq/backend/ramulator2/src/ramulator/controller/impl'/n for n in
    ('unified_gemv_extension.inc','rank_rope_extension.inc','rank_staging_extension.inc','lpddr5_controller.cpp')]
result = dict(
    scope='One layer, one denoising step, QKV/RoPE/QKT/softmax boundary/PV/O; no FFN',
    config=dict(heads=H,query_tokens=M,valid_kv_tokens=984,padded_kv_tokens=S,
                head_dim=D,hidden=X,active_ranks=R,banks_per_rank=B,tCK_ns=TCK,
                physical_ranks='rank1 in each of sixteen channels; rank0 reserved for GPU',
                transfer_dtypes='BF16 input/weights/P/context/O; FP32 scores'),
    baseline=dict(kind='Analytical RoPE-paired K/V feature-sharded baseline; native timing not implemented',
        rank_qkv_output=[16,128],rank_kv_cache=[992,128],bank_kv_cache=[992,8],
        rank_o_input=[16,128],rank_o_active_weight=[128,1024],rank_o_output=[16,1024],
        intra_rank_broadcast=True,pv_stays_local=True,all_ranks_compute=True),
    current=dict(rank_qkv_output=[8,256],rank_kv_cache=[496,256],bank_kv_cache=[496,16],
        rank_o_input=[16,256],rank_o_active_weight=[256,1024],rank_o_output=[16,1024],
        intra_rank_broadcast=True,pv_stays_local=True,all_ranks_compute=True),
    dq=dict(baseline_bytes=base_dq,current_bytes=token_dq,saved_bytes=base_dq-token_dq,
        reduction_percent=100*(1-token_dq/base_dq),rows=dq_rows,
        formula_baseline='R*M*X*2 + 2*H*M*S*4 + 2*H*M*S*2 + R*M*X*2',
        formula_current='R*M/2*X*2 + 2*H*M*D*2 + H*M*S*4 + H*M*S*2 + R*M*X*2',
        general_saving_bytes='2*H*M*X + 6*H*M*S - 4*H*M*D',
        excluded='One-time weights/KV initialization, GPU-internal memory traffic; final O reduction on GPU'),
    weights=weights,
    initialization=dict(baseline_minimum_weight_upload_MiB=16,current_active_only_upload_MiB=32,
        current_declared_full_residency_upload_MiB=88,
        active_only_extra_bytes_break_even_reuses=16*MIB/(base_dq-token_dq),
        full_residency_extra_bytes_break_even_reuses=72*MIB/(base_dq-token_dq),
        caveat='Byte costs if each resident weight is externally loaded once; not per-step traffic; no initialization latency measured'),
    macs=mac_rows,total_MACs=dict(baseline=sum(a for _,a,_ in macs),current=sum(b for _,_,b in macs)),
    gpu_reduction=dict(common_final_O_additions=(R-1)*M*X,baseline_extra_score_additions=H*M*S,
        arithmetic_latency_us=None),
    dq_payload_busy=dq_busy,
    native_latency=dict(current_bulk_score_stage_sum_us=native_total,
        current_stream_score_stage_sum_us=stream_total,baseline_stage_sum_us=None,
        current_complete_latency_us=None,baseline_complete_latency_us=None,groups=latency_rows,
        interpretation='Existing independent-stage native model sums, not a fresh continuous full-attention run',
        unknowns=['GPU score reduction for baseline','GPU scaling/softmax','GPU final O reduction',
                  'GPU Q forwarding/dispatch overhead','GPU packing/dispatch','concurrent VLM interference'],
        baseline_implementation_gaps=['QKV columns/rows fixed at48/8 for current fused mapping',
            'RoPE/Q ownership hard-coded to token parity',
            'QKT validates496 tokens and all16 feature lanes per bank',
            'O input validates8192B per rank; feature baseline needs4096B',
            'No current native binary or configured WSL distribution available locally']),
    analytical_service_caveat='MAC/4096*CK assumes full16-lane utilization. Feature baseline QKT has8 useful features/bank; a one-dot16-lane unit without two-dot packing can take twice the ideal QKT service. No service-demand number is a latency prediction.',
    broadcast=dict(current_qkv_source_bytes=256*KIB,current_qkv_bank_delivery_bytes=4096*KIB,
        baseline_qkv_source_bytes=512*KIB,baseline_qkv_bank_delivery_bytes=8192*KIB,
        multicast_delivery_factor=16,baseline_uses_same_facility=True,
        source='same-rank multicast after local source read; multicast itself adds no separate service in current model'),
    validation=dict(dq_matches_current_native_counters=True,weight_reads_match_native_counters=True,
        stage_groups_close=True,algebra='FP64 at current attention shapes, eight heads',algebra_checks=checks,
        BF16_accuracy_validated=False,baseline_native_simulated=False),
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

lines = [
    '完整 Attention：feature-wise baseline 与当前 token-wise 布局公平核算',
    '统计单位：单层、单个 denoising step；全部16个PIM rank参与，16 banks/rank。',
    '8 heads；16 query tokens；KV有效984/padded992；head_dim256；hidden1024。',
    '两者均允许rank内输入广播、QKV融合、PV本地接O；最终16份O部分和交GPU归约。',
    '两者输入/权重/P/PV/O为BF16，score为FP32；同dtype不代表逐位数值等价。',
    '', '一、每步外部DQ（KiB，16 channels合计）',
    '项目 | feature baseline | 当前 | 当前节省',
]
lines += [f"{r['transfer']} | {r['baseline_KiB']:g} | {r['current_KiB']:g} | {r['saved_KiB']:g}" for r in dq_rows]
lines += [f'合计 | {base_dq/KIB:g} | {token_dq/KIB:g} | {(base_dq-token_dq)/KIB:g}',
    f'当前减少 {result["dq"]["reduction_percent"]:.6f}%。不含一次性权重/KV预装，不含GPU内部显存流量。',
    '跨rank无直接互连或跨channel免费广播；Q转发的读出与写入均计入。',
    'PV->O两者都是0外部DQ；O输出两者都是512KiB，不能只给当前布局融合收益。',
    '', '二、BF16权重容量（MiB）',
    '矩阵 | baseline有效 | 当前有效 | 当前声明常驻',
]
lines += [f"{w['matrix']} | {w['baseline_active_bytes']/MIB:g} | {w['current_active_bytes']/MIB:g} | {w['current_declared_resident_bytes']/MIB:g}" for w in weights]
lines += ['合计 | 16 | 32 | 88',
    '当前88MiB中有56MiB是W_O未访问的额外常驻副本，不是token分片的必要代价。',
    '当前有效32MiB相对baseline16MiB，多16MiB；缓存K/V逻辑总量两者相同，为7.75MiB，另有暂存/对齐。',
    '若权重通过外部接口逐份预装一次，上传容量分别为16/32/88MiB；与每步DQ分开。',
    '计入一次权重加载，单层复用N次的DQ：baseline=16MiB+N*2512KiB；当前有效副本=32MiB+N*1640KiB；当前全常驻=88MiB+N*1640KiB。',
    '额外权重上传字节的摊平点约为19次/85次复用同一层权重；这是字节数分界，不是latency分界，且不含双方共同的KV初始化。',
    '', '三、运算量（MAC）',
    '算子 | baseline | 当前',
]
lines += [f'{n} | {a:,} | {b:,}' for n,a,b in macs]
lines += [f'合计 | {sum(a for _,a,_ in macs):,} | {sum(b for _,_,b in macs):,}',
    '当前O MAC为baseline两倍；完整四算子MAC增加约16.84%。',
    'baseline在GPU上额外归约126,976个score元素；两者最终O均需245,760次两数加法（数学计数，不是GPU时延）。',
    '', '四、latency：已测与未测必须区分',
    '当前已建模阶段 | microseconds',
]
lines += [f'{n} | {v:.9f}' for n,v in groups]
lines += [f'当前集中score导出阶段和 | {native_total:.6f}',
    f'当前流式score导出阶段和（调度消融，字节数不变） | {stream_total:.6f}',
    'baseline原生阶段和 | 未测，未实现对应完整路径',
    '两者完整端到端attention latency | 均未测',
    '现有阶段和不含GPU softmax、最终归约及dispatch；baseline还需score归约。不能将未知项写为0。',
    '现有legacy feature开关只覆盖部分V/PV路径，不是这里定义的完整feature-wise QKV/QKT/PV/O baseline。',
    f'纯DQ payload占用/每channel：baseline {dq_busy["baseline_us_per_channel"]:.6f} us；当前 {dq_busy["current_us_per_channel"]:.6f} us。',
    '上述DQ时间只含32B/2CK的burst占用，不含排队、ACT/PRE、读写转换、cast/gather、计算等，不能当总latency。',
    'MAC数量也不能直接缩放latency：baseline QKT只有8个有效feature/bank，而当前16个；需明确8-lane打包/归约及命令实现。',
    '', '五、rank内输入广播',
    '当前QKV外部X=256KiB；rank内送达16 banks合计4MiB；额外16倍bank接收不重复计外部DQ。',
    'baseline同样使用广播：外部X=512KiB；bank接收8MiB。当前的相对收益来自两rank分担token，少了一份跨rank输入。',
    '广播本身在当前模型中不另加service，但source DRAM读取、SRF填充、行切换和CA占用均非免费。',
    'PV的概率输入也由496降到248KiB；O的本地输入基线较小，两者仍可广播。',
    '', '六、校验与范围',
    '当前DQ和活跃权重读取量与既有原生summary闭合；分组阶段时间与原生阶段和闭合。',
    '已用当前矩阵尺寸、8个head的FP64随机数据验证两种attention/O分解；没有验证BF16准确率。',
    '本报告没有声称新运行了baseline原生仿真，没有改动发布包。',
    '完整公式、来源SHA256、数值检查误差见comparison.json。',
]
(OUT/'REPORT.zh-CN.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(dq_KiB=[base_dq/KIB,token_dq/KIB],
    reduction_percent=result['dq']['reduction_percent'],weights_MiB=[16,32,88],
    current_model_us=native_total,current_stream_model_us=stream_total,
    dq_payload_busy=dq_busy,native_latency_groups=groups,
    fp64_max_abs=max(v for x in checks for k,v in x.items() if k!='head'),
    baseline_native_simulated=False),ensure_ascii=False,indent=2))

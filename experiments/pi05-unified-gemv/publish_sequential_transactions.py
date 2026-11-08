"""Reconcile the ACT/PRE ablation with its traffic and LPDDR configuration."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=2e-7),(a,b)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    p.add_argument('--root',type=Path,default=Path('reference/evaluation-1108'));a=p.parse_args()
    out=a.root/'sequential-rows';src=Path('experiments/pi05-unified-gemv')
    old=read(a.root/'online-arbitration-preparation/evaluation-isolated/summary.json')
    control=read(out/'row-hit-control/summary.json');rows=read(out/'with-rows/summary.json')
    demand=read(a.root/'gpu/uniform-memory-demand.json')
    expected=dict(tCK_ps=938,nBL_min=2,nBL_max=4,nCL=23,nCWL=12,nRCD=20,nRCDWR=9,
                  nRP=20,nRPab=23,nRAS=45,nRC=65,nWR=37,nRTP=8,nRRDS=4,nRRDL=4,
                  nFAW=16,nPPD=2,nAAD=8,nRFC=224,nREFI=4164)
    provenance=list((a.root/'direct-before-corrected/native-provenance').glob('*/*/result.json'))
    for f in provenance:
        data=read(f)
        assert all(data['resolved_timings_ck'][k]==v for k,v in expected.items()),f
        assert data['transaction_bytes']==32 and data['config']['model']=='lpddr5x-8533'
    assert not control['GPU_row_state'] and rows['GPU_row_state']
    counts=rows['GPU_solo_command_counts'];units=demand['units']
    assert counts['reads']==sum(u['read_sectors'] for u in units)
    assert counts['writes']==sum(u['write_sectors'] for u in units)
    assert counts['act1']==counts['act2']>0
    assert counts['prepb']+counts['preab']>0
    assert counts['row_hits']+counts['row_misses']==counts['reads']+counts['writes']
    indexed=[{x['case']:x for x in d['evaluation']} for d in (old,control,rows)]
    old_by,control_by,new_by=indexed
    checks=[]
    for kind,data in [('row-hit-control',control),('with-rows',rows)]:
        for r in data['evaluation']:
            close(r['AE_solo_ms'],old_by[r['case']]['AE_solo_ms'])
            if r['case']=='b':continue
            close(r['raw_model_total_ms']-data['VLM_reference_bias_ms'],r['total_ms'])
            close(r['AE_solo_ms']+data['VLM_retained_ms']-r['net_overlap_ms'],r['total_ms'])
            placements=list(csv.DictReader((out/kind/(r['case']+'-VLM-placement.csv')).open()))
            assert [int(x['ordinal']) for x in placements]==list(range(r['completed_units']))
            close(sum(float(x['memory_stall_us']) for x in placements)/1000,r['VLM_memory_stall_ms'])
            if kind=='with-rows':
                selected=units[:r['completed_units']]
                assert r['GPU_reads']==sum(x['read_sectors'] for x in selected)
                assert r['GPU_writes']==sum(x['write_sectors'] for x in selected)
                assert r['GPU_act1']==r['GPU_act2']>0
                assert r['GPU_row_hits']+r['GPU_row_misses']==r['GPU_reads']+r['GPU_writes']
            checks.append(dict(model=kind,case=r['case'],ordered_units=r['completed_units']))
    audit=dict(job=a.job,native_configs_verified=len(provenance),timings_ck=expected,
        row_bytes=2048,transaction_bytes=32,sectors_per_row=64,banks_per_rank=16,
        GPU_solo_counts=counts,checks=checks,
        status='Traffic, timing provenance, order and accounting verified; not hardware validation')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 连续地址下的 GPU ACT/PRE 修正', '',
        '**这是连续地址近似的模型对照，不是真实 Thor 逐请求 trace，也不是硬件端到端实测。**', '',
        f'分支 `xinyao_1108`；作业 `{a.job}`。PIM、AE GPU 计时、Nsight 每 kernel 访存量均沿用已有证据；不使用 ρ、在途请求上限或人为读写比。', '',
        '## 结果（ms）', '',
        '| 配置 | 旧轮转 BG、全 row-hit | 连续地址、全 row-hit 对照 | 连续地址、加入行状态 | 行状态相对对照 | 新模型原始总时长 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in 'bcdefg':
        o,v,r=old_by[c],control_by[c],new_by[c]
        raw=r.get('raw_model_total_ms',r['total_ms']+rows['VLM_reference_bias_ms'])
        lines.append(f"| {c} | {o['total_ms']:.6f} | {v['total_ms']:.6f} | {r['total_ms']:.6f} | {r['total_ms']-v['total_ms']:+.6f} | {raw:.6f} |")
    lines+=['', '前三列延续原报告的 VLM 参考偏差抵消口径，最后一列保留模型原始时长。这些值不是置信区间。', '',
        f"已有 VLM 时长 {rows['VLM_retained_ms']:.6f} ms；连续地址 row-hit 对照的独立 VLM 模型为 {control['VLM_model_reference_ms']:.6f} ms；加入行状态后为 {rows['VLM_model_reference_ms']:.6f} ms。共同参考偏差能否在并发场景中抵消尚未经验证。", '',
        '**上述独立模型与已有计时的偏差很大，说明请求排序、地址局部性及 L2 写流量代理尚未校准。前三列仅用于模型诊断，不应作为可信的 Thor b–g 主表，更不能据此确定硬件 e/g 的排序。**', '',
        '## e/g 的竞争和命令证据', '',
        '| 指标 | e | g |','|---|---:|---:|']
    for key,label in [('VLM_memory_stall_ms','VLM 额外访存等待/ms'),('PIM_interference_ms','PIM 被推迟/ms'),
                      ('AE_GPU_wait_ms','后续 AE GPU 等待/ms'),('GPU_act1','GPU ACT1 次数'),
                      ('GPU_act2','GPU ACT2 次数'),('GPU_prepb','GPU PREpb 次数'),('GPU_preab','GPU PREab 次数')]:
        vals=[new_by[c][key] for c in 'eg']
        fmt=(lambda x:f'{x:,}') if key.startswith('GPU_') else (lambda x:f'{x:.6f}')
        lines.append(f'| {label} | {fmt(vals[0])} | {fmt(vals[1])} |')
    lines+=['', f"e→g 总延迟变化：{new_by['g']['total_ms']-new_by['e']['total_ms']:+.6f} ms。命令计数只包含 AE 期间已执行的 VLM 前缀；两者推进的 kernel 数不同，不能直接把计数差当作效率差。", '',
        '## 新增的执行约束', '',
        '每 channel/rank/bank 维护开行状态。关闭行先 ACT1/ACT2；同 bank 不同行先 PREpb，等待 nRP 后开行；行命中才直接 RD/WR。ACT1/ACT2 采用连续两个 CK 的合法发射方式，一旦 ACT1 已发出，第二个 CK 已被预约，以保证 nAAD=8 CK 的 ACT2 截止时间。', '',
        'ACT/PRE 占用同一条外部 CA，并与已就绪 PIM 命令竞争；RD/WR 仍预约 CL/CWL 后的 DQ burst。行控制使用配置中的 nRCD/nRCDWR、nRAS、nRP、nRC、nRTP、写恢复、nRRD、nFAW 和 nPPD。已打开 GPU 行在固定 refresh 前通过 PREab 关闭，等待 nRPab，再进入目标 rank 的 nRFC；GPU 未开行时不虚增 PREab，不改变 PIM 独立基线。', '',
        'kernel 依赖顺序、跨窗口推进及后续 AE GPU 等待沿用原模型。f/g 继续采用原模型的完整外部 I/O 阶段独占规则；这是一项保留的调度策略，并非“私有细命令隔离”必然要求整段 I/O 独占。', '',
        '## 地址假设与解释边界', '',
        '采用当前模拟器的 x16、4 BG × 4 bank、1024 column、BL16 配置：每 bank 一行 2048 B，每个 32 B 请求占一个 burst，共 64 个 burst/行。连续流跨行可先打开另一个 bank；只有替换同 bank 已打开的另一行时才发 PREpb，refresh 则会关闭已有行。不能简单地给每 64 次访问固定加一个 PRE。', '',
        '每 kernel/channel 的请求按原读写发射顺序映射为 rank0 分区内从 sector0 开始的单一连续地址流。column=i%64，BG=(i//64)%4，bank=(i//256)%4，row=i//1024；这是现有 RoBaRaCoCh 映射限制在 rank0 的结果。PIM 使用 rank1。bank 状态在同一连续 PIM 阶段内保留，阶段间重建；真实跨 kernel 分配、复用和访问先后未被恢复。', '',
        'Nsight 的总量约束请求数量，无法唯一给出实际 row hit/miss。时间到达仍采用此前允许的均匀近似；空间局部性仍偏理想。写侧仍是 L2 write sectors 的全写回代理，不是 DRAM 写回实测。该修正补齐了原先漏掉的行操作，但没有声称获得真实突发性或完整 GPU/MC 模型。', '',
        '独立 GPU 参考也加入同样的行操作，避免把 GPU 本来就需要的 ACT/PRE 全算成 PIM 干扰。保留相同地址映射的 row-hit 对照，避免把 BG 映射变化误归因于 ACT/PRE。', '',
        '## 验证', '',
        f"核验 {len(provenance)} 份原生配置，完整 VLM 请求量与 Nsight 账本一致；ACT1/ACT2 配对、行命中/未命中计数、b–g 总时长及 kernel 顺序通过独立对账。见 `audit.json`、`unit-tests.log` 和各配置逐 kernel CSV。", '']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[f for f in out.rglob('*') if f.is_file() and f.name not in ('SHA256SUMS.json','libsequential-test.so')]
    files+=list(src.glob('sequential*'))+[src/'run_sequential_transactions.py',Path(__file__),
        src/'uniform_transactions.cpp',src/'uniform_transactions.py',src/'online_transactions.py',
        src/'validation/test_sequential_transactions.py',a.root/'gpu/uniform-memory-demand.json',
        a.root/'online-arbitration-preparation/native-programs.json.gz',
        a.root/'online-arbitration-preparation/natural-ae-kernel-ledger.json',
        a.root/'uniform-transactions/native-reservations.json.gz',a.root/'gpu/warm-summary.json']+provenance
    files+=list((a.root/'corrected-overlap').glob('*-explicit_native_io-solo-schedule.csv'))
    backend=Path('experiments/pi05-bgk-gpuq/backend/ramulator2')
    files+=[backend/'python/ramulator/dram/lpddr5.py',backend/'python/ramulator/dram/lpddr5x.py',
            backend/'src/ramulator/controller/addr_mapper/impl/ro_ba_ra_co_ch.cpp',
            backend/'src/ramulator/controller/addr_mapper/addr_mapper_base.cpp',
            backend/'src/ramulator/controller/impl/lpddr_controller_base.cpp']
    hashes={str(f.resolve().relative_to(Path.cwd())).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(job=a.job,verified_files=len(hashes),e_ms=new_by['e']['total_ms'],g_ms=new_by['g']['total_ms'])))

if __name__=='__main__':main()

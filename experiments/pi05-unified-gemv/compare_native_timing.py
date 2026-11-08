"""Compare archived and corrected causal arbitration runs without changing either."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def csvout(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reference/evaluation-1108'))
    args=parser.parse_args();root=args.root
    old_dir=root/'sequential-rows/with-rows';new_root=root/'native-timing-fix';new_dir=new_root/'with-rows'
    old=json.loads((old_dir/'summary.json').read_text());new=json.loads((new_dir/'summary.json').read_text())
    before={r['case']:r for r in old['evaluation']};after={r['case']:r for r in new['evaluation']}
    metrics=('PIM_interference_ms','total_ms','raw_model_total_ms','AE_GPU_wait_ms',
             'VLM_memory_stall_ms','VLM_completed_model_ms','VLM_remaining_model_ms',
             'net_overlap_ms','completed_units')
    rows=[];stage_deltas=[]
    for case in 'cdefg':
        a=before[case];b=after[case]
        assert a['variant']==b['variant']
        assert a['AE_solo_ms']==b['AE_solo_ms']
        assert a['AE_native_schedule_ms']==b['AE_native_schedule_ms']
        row=dict(case=case,variant=b['variant'])
        for m in metrics:
            row[m+'_old']=a[m];row[m+'_new']=b[m];row[m+'_delta']=b[m]-a[m]
        row['PIM_interference_percent_change']=100*(b['PIM_interference_ms']/a['PIM_interference_ms']-1)
        row['total_percent_change']=100*(b['total_ms']/a['total_ms']-1)
        rows.append(row)
        astages=read_csv(old_dir/f'{case}-PIM-and-AE-waits.csv')
        bstages=read_csv(new_dir/f'{case}-PIM-and-AE-waits.csv')
        assert len(astages)==len(bstages)
        for x,y in zip(astages,bstages):
            assert (x['block'],x['name'],x['native_us'],x['isolated_bus_model_us']) == (
                    y['block'],y['name'],y['native_us'],y['isolated_bus_model_us'])
            stage_deltas.append(dict(case=case,block=x['block'],name=x['name'],
                PIM_wait_old_us=float(x['PIM_GPU_interference_us']),
                PIM_wait_new_us=float(y['PIM_GPU_interference_us']),
                PIM_wait_delta_us=float(y['PIM_GPU_interference_us'])-float(x['PIM_GPU_interference_us']),
                AE_GPU_wait_delta_us=float(y['AE_GPU_wait_us'])-float(x['AE_GPU_wait_us'])))
        actual=sum(x['PIM_wait_new_us'] for x in stage_deltas if x['case']==case)/1000
        assert abs(actual-b['PIM_interference_ms'])<1e-9
        assert abs(b['raw_model_total_ms']-b['AE_solo_ms']-b['PIM_interference_ms']-
                   b['AE_GPU_wait_ms']-b['VLM_remaining_model_ms'])<1e-9
    for key in ('VLM_retained_ms','VLM_model_reference_ms','VLM_reference_bias_ms'):
        assert old[key]==new[key],(key,old[key],new[key])
    paired=[]
    for unfused,fused in (('d','f'),('e','g')):
        paired.append(dict(before_case=unfused,after_case=fused,
            old_fused_minus_unfused_ms=before[fused]['total_ms']-before[unfused]['total_ms'],
            new_fused_minus_unfused_ms=after[fused]['total_ms']-after[unfused]['total_ms']))
    report=dict(job='1974571',comparison=rows,fusion_pairs=paired,
        validation=dict(transaction_tests_passed=51,online_only_tests_passed=14,
            old_library_regressions_reproduced=3,slurm_state='COMPLETED',elapsed='00:04:02'),
        microprobes=dict(warm_row_WRGB_at_100=dict(old_GPU_RD_CK=101,new_GPU_RD_CK=102,
            old_PIM_wait_CK=4,new_PIM_wait_CK=5),
            ordinary_rank1_RD_at_100=dict(old_rank0_WR_CK=101,new_rank0_WR_CK=115)),
        invariants=dict(native_PIM_stage_and_solo_times_unchanged=len(stage_deltas),
            VLM_retained_ms=new['VLM_retained_ms'],VLM_model_reference_ms=new['VLM_model_reference_ms'],
            VLM_reference_bias_ms=new['VLM_reference_bias_ms'],
            total_definition='AE_solo + PIM_interference + AE_GPU_wait + remaining_VLM - VLM_reference_bias'),
        timing_audit=dict(
            native='LPDDR5XPIM, LPDDR5X_8533, attacc, x16, two ranks, WCK always_on',
            fixed='Independent transfer issue history: all transfers >=2 CK channel-wide; same BG >=4 CK; ordinary RD/WR retain native rank/BG/sibling constraints. Pending PIM priority includes these issue constraints.',
            transfer_matrix_cases=144,
            ordinary_cross_rank_issue_CK={'RD_to_RD':4,'RD_to_WR':15,'WR_to_RD':2,'WR_to_WR':3},
            CA='Controller-to-DRAM command/address bus, not a bidirectional read/write data bus. Native CA widths preserved; ACT1/ACT2 reserve consecutive 1 CK commands.',
            DQ='BL=2 CK; ordinary RD latency=23 CK, WR latency=12 CK; buffer transfers have zero CL/CWL in native AttAcc. Committed return non-overlap and direction/rank guards remain checked.',
            rows='Native RCD/RCDWR, RAS/RC, RP/RPab, RTP/write recovery, RRD/FAW, PPD, RFC/REFI checked against resolved spec and emitted traces.',
            conservative_extensions=['Mixed WRGB/RDRES versus ordinary RD/WR cross-rank DQ end-to-start guards remain 1 CK after writes and 2 CK after reads; native AttAcc has no such explicit mixed-command guard.',
                'Fixed refresh admission drains outstanding data, and consecutive ACT1/ACT2 policy can be stricter than the native minimum timings.']),
        limitations=['This remains an independent causal replay model, not a mixed-traffic native controller run or hardware measurement.',
            'GPU mover kernel/SM/copy-engine contention, cache residency/refill, actual request addresses and warp-level progress remain unmodeled.',
            'After configurations retain whole external-I/O epoch ownership; differences are conditional on that policy.',
            'The retained solo VLM bias subtraction is large and unvalidated for concurrent execution; raw totals are included.'])
    csvout(new_root/'comparison.csv',rows);csvout(new_root/'stage-deltas.csv',stage_deltas)
    (new_root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source=Path('experiments/pi05-unified-gemv')
    paths=[source/x for x in ('sequential_transactions.cpp','sequential_timing.h','sequential_rows.h',
        'uniform_transactions.cpp','online_transactions.cpp','run_sequential_transactions.py',
        'native-timing-fix.sbatch','compare_native_timing.py','validation/native_timing_probe.cpp',
        'validation/test_native_timing_transactions.py','validation/test_online_transactions.py',
        'validation/test_sequential_transactions.py')]
    paths += [root/'online-arbitration-preparation/native-programs.json.gz',
              root/'uniform-transactions/native-reservations.json.gz',root/'gpu/uniform-memory-demand.json',
              root/'online-arbitration-preparation/natural-ae-kernel-ledger.json',root/'gpu/warm-summary.json',
              old_dir/'summary.json']
    paths += [p for p in new_root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json']
    hashes={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (new_root/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for row in rows:
        print(row['case'],*[f'{row[m+"_old"]:.9f} -> {row[m+"_new"]:.9f} ({row[m+"_delta"]:+.9f})'
                           for m in ('PIM_interference_ms','total_ms')])
    print(json.dumps(report['invariants'],indent=2));print(json.dumps(paired,indent=2))


if __name__=='__main__':main()

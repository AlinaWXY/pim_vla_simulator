"""Native directed checks; run in the isolated remote backend via PYTHONPATH."""
import argparse
import csv
import json
from pathlib import Path
from simulator.backend import run_trace
from simulator.config import MemoryConfig

OUT=Path(__file__).resolve().parent/'native-tests'

def address(rank=0,bg=0,bank=0,row=0,col=0,ch=0):
    return (row<<20)|(bank<<18)|(bg<<16)|(rank<<15)|(col<<9)|(ch<<5)
def line(op,rank=0,bg=0,bank=0,row=0,col=0,slot=None):
    return f'{op} {address(rank,bg,bank,row,col)}' + ('' if slot is None else f' {slot}')
def simulate(name, lines, **overrides):
    trace=OUT/f'{name}.trace'
    trace.write_text('\n'.join(lines)+'\n')
    cfg=MemoryConfig(model='lpddr5x-8533',channels=16,ranks=2,refresh=False,pim=True,
                     frontend_ratio=16,pim_input_broadcast=True,pim_mac_scope='channel',
                     pim_data_path='attacc',timing_overrides={'nCCDAB':20,**overrides})
    result=run_trace(cfg,trace,OUT/name)
    rows=list(csv.DictReader((OUT/name/'bgk_events.csv.ch0').open()))
    for row in rows:
        for key in row.keys()-{'operation'}: row[key]=int(row[key])
    return result,rows

def main():
    x=[line('PIM_WR_BG',r,g,slot=0) for r in range(2) for g in range(4)]
    q=[line('PIM_WR_Q',bank=b,slot=0) for b in range(4)]
    results={}
    r,e=simulate('dispatch_mac',x+[line('PIM_MAC_K',slot=0),line('PIM_MAC_K',col=1,slot=0),'BARRIER 1'])
    fills=[v for v in e if v['operation']=='PIM_WR_BG'];mac=[v for v in e if v['operation']=='PIM_MAC_K']
    assert len(fills)==8 and all(v['target_banks']==4 for v in fills)
    assert mac[0]['clock']>=max(v['ready_at'] for v in fills)
    assert mac[1]['clock']-mac[0]['clock']>=r['resolved_timings_ck']['nCCDL']
    assert all(v['service_cycles']==2 and v['target_banks']==32 for v in mac)
    results['BG_dispatch_and_K_array_cadence']=r['latency_ns']
    r,e=simulate('q_multicast_result',q+[line('PIM_MAC_QK',slot=0),line('PIM_RD_BG'),'BARRIER 1'])
    fills=[v for v in e if v['operation']=='PIM_WR_Q'];mac=[v for v in e if v['operation']=='PIM_MAC_QK'][0]
    result=[v for v in e if v['operation']=='PIM_RD_BG'][0]
    assert all(v['target_banks']==8 for v in fills)
    assert mac['clock']>=max(v['ready_at'] for v in fills)
    assert result['clock']>=mac['ready_at'] and mac['service_cycles']==20
    results['Q_multicast_and_BG_result_readiness']=r['latency_ns']
    # Same-slot refills are held until computation ends; inactive slot may fill.
    r,e=simulate('buffer_ownership',q+[line('PIM_MAC_QK',slot=0),line('PIM_WR_Q',bank=0,slot=1),
                                      line('PIM_WR_Q',bank=1,slot=0),'BARRIER 1'])
    mac=[v for v in e if v['operation']=='PIM_MAC_QK'][0]
    inactive=[v for v in e if v['operation']=='PIM_WR_Q' and v['buffer']==1][0]
    same=[v for v in e if v['operation']=='PIM_WR_Q' and v['buffer']==0][-1]
    assert inactive['clock']<mac['ready_at'] and same['clock']>=mac['ready_at']
    results['double_buffer_overlap_and_overwrite_guard']=r['latency_ns']
    # Open every bank together with a completed MAC, then exercise independent
    # local writes on the same row. Bank ports are not CA or DQ transactions.
    writes=[line('PIM_LOCAL_WR',r,g,b,col=c) for c in range(8) for r in range(2) for g in range(4) for b in range(4)]
    for port in (4,8):
        r,e=simulate(f'local_write_{port}',x+[line('PIM_MAC_K',slot=0),'BARRIER 1']+writes+['BARRIER 2'],nLOCALWR=port)
        local=[v for v in e if v['operation']=='PIM_LOCAL_WR']
        assert len(local)==256
        assert r['issued_commands']['WR']==0 and r['traffic_bytes']['dram_write']==0
        same=[v for v in local if v['Rank']==0 and v['BankGroup']==0 and v['Bank']==0]
        assert all(b['clock']-a['clock']>=port for a,b in zip(same,same[1:]))
        assert all(v['ready_at']-v['clock']==port+r['resolved_timings_ck']['nWR'] for v in local)
        assert len(set(v['clock'] for v in local)) < len(local), 'bank ports serialized globally'
        results[f'independent_local_ports_{port}CK']=r['latency_ns']
    # Changing rows must cause native precharge and both ACT phases.
    r,e=simulate('local_write_row_conflict',x+[line('PIM_MAC_K',slot=0),'BARRIER 1',
                 line('PIM_LOCAL_WR',row=1),line('PIM_LOCAL_WR',row=2),'BARRIER 2'])
    assert r['issued_commands']['PREpb']>=2 and r['issued_commands']['ACT1']>=2 and r['issued_commands']['ACT2']>=2
    local=[v for v in e if v['operation']=='PIM_LOCAL_WR']
    assert local[1]['clock']>=local[0]['ready_at']+r['resolved_timings_ck']['nRP']
    results['local_write_native_row_recovery']=r['latency_ns']
    (OUT/'summary.json').write_text(json.dumps({'passed':len(results),'cases':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    main()

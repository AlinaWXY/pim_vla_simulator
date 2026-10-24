"""Fault injection against completed native logs, without modifying results."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import audit_native as A


def run(case):
    cache={}
    real_read=A.read_csv
    def cached(path):
        path=Path(path)
        if path not in cache:
            cache[path]=real_read(path)
        return copy.deepcopy(cache[path])
    event_path=case/'simulation/bgk_events.csv.ch0'
    native_path=case/'simulation/timeline.csv.ch0'
    events=cached(event_path)
    require=lambda c,m:A.require(c,m)
    require(any(e['operation']=='PIM_LOCAL_WR' for e in events),'Use complete K case')
    results=[]
    for fault in ('wrong_multicast_fanout','missing_X_vector','MAC_before_complete_X','missing_write_recovery','wrong_ACT_row'):
        def injected(path,fault=fault):
            rows=cached(path)
            if Path(path)==event_path:
                if fault=='wrong_multicast_fanout':
                    next(e for e in rows if e['operation']=='PIM_WR_BG')['target_banks']=3
                elif fault=='missing_X_vector':
                    index=next(i for i,e in enumerate(rows) if e['operation']=='PIM_WR_BG')
                    rows.pop(index)
                elif fault=='missing_write_recovery':
                    e=next(e for e in rows if e['operation']=='PIM_LOCAL_WR')
                    e['ready_at']=e['clock']+e['service_cycles']
                elif fault=='MAC_before_complete_X':
                    e=next(e for e in rows if e['operation']=='PIM_MAC_K')
                    inputs=[v for v in rows if v['operation']=='PIM_WR_BG' and v['buffer']==0][:10]
                    e['clock']=max(v['ready_at'] for v in inputs)-1
                    e['ready_at']=e['clock']+e['service_cycles']
            if Path(path)==native_path and fault=='wrong_ACT_row':
                first_store=next(e for e in events if e['operation']=='PIM_LOCAL_WR')
                candidates=[e for e in rows if e['command']=='ACT2'
                    and e['Rank']==first_store['Rank'] and e['BankGroup']==first_store['BankGroup']
                    and e['Bank']==first_store['Bank'] and e['clock']<first_store['clock']]
                require(bool(candidates),'No bank activation found for injection')
                candidates[-1]['Row']+=1
            return rows
        try:
            with patch.object(A,'read_csv',side_effect=injected):
                A.audit_run(case,write=False)
        except AssertionError as error:
            results.append(dict(fault=fault,detected=True,reason=str(error)[:220]))
        else:
            raise AssertionError(f'Audit failed to detect {fault}')
    output=dict(case=str(case),status='passed',faults=results,
        result_files_modified=False,method='in-memory CSV read interception only')
    (case.parent/'audit-fault-detection.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.case.resolve()),indent=2))

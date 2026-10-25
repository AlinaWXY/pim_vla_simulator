"""Inject timing/dataflow faults into in-memory native logs; files stay intact."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import audit_native as A


def run(root):
    cache = {}
    real_read = A.read_csv
    def cached(path):
        path = Path(path)
        if path not in cache:
            cache[path] = real_read(path)
        return copy.deepcopy(cache[path])
    results = []
    faults = [('qk', f) for f in ('wrong_multicast_fanout','missing_Q_read','relay_before_read_return',
        'missing_relay_completion','wrong_Q_source_bank','wrong_ACT_row')]
    faults += [('qkv','MAC_before_complete_X'),('qkv','missing_X_vector')]
    for name, fault in faults:
        case = root/name
        ep, np = case/'simulation/bgk_events.csv.ch0', case/'simulation/timeline.csv.ch0'
        original_events = cached(ep)
        def injected(path):
            rows = cached(path)
            if Path(path) == ep:
                if fault == 'wrong_multicast_fanout':
                    next(e for e in rows if e['operation']=='PIM_RELAY_Q')['target_banks'] = 7
                elif fault == 'missing_Q_read':
                    rows.pop(next(i for i,e in enumerate(rows) if e['operation']=='PIM_RD_Q'))
                elif fault == 'relay_before_read_return':
                    read = next(e for e in rows if e['operation']=='PIM_RD_Q')
                    relay = next(e for e in rows if e['operation']=='PIM_RELAY_Q')
                    relay['clock'] = read['ready_at']-1
                    relay['ready_at'] = relay['clock']+relay['service_cycles']
                elif fault == 'missing_relay_completion':
                    relay = next(e for e in rows if e['operation']=='PIM_RELAY_Q')
                    relay['ready_at'] = relay['clock']
                elif fault == 'wrong_Q_source_bank':
                    next(e for e in rows if e['operation']=='PIM_RD_Q')['Bank'] ^= 1
            if Path(path) == np:
                if fault == 'wrong_ACT_row':
                    read = next(e for e in original_events if e['operation']=='PIM_RD_Q')
                    candidates = [e for e in rows if e['command']=='ACT2'
                        and (e['Rank'],e['BankGroup'],e['Bank']) == (read['Rank'],read['BankGroup'],read['Bank'])
                        and e['clock'] < read['clock']]
                    A.require(bool(candidates), 'No activation for mutation')
                    candidates[-1]['Row'] += 1
                elif fault == 'MAC_before_complete_X':
                    mac = next(e for e in rows if e['command']=='MACAB')
                    fills = [e for e in rows if e['command']=='WRGB'][:A.HORIZON]
                    mac['clock'] = fills[-1]['clock']+1
                elif fault == 'missing_X_vector':
                    rows.pop(next(i for i,e in enumerate(rows) if e['command']=='WRGB'))
            return rows
        try:
            with patch.object(A, 'read_csv', side_effect=injected):
                A.audit_run(case, write=False)
        except AssertionError as error:
            results.append(dict(case=name, fault=fault, detected=True, reason=str(error)[:220]))
        else:
            raise AssertionError(f'Audit failed to detect {name}/{fault}')
    output = dict(status='passed', faults=results, result_files_modified=False,
                  method='in-memory CSV read interception only')
    (root/'audit-fault-detection.json').write_text(json.dumps(output,indent=2)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    print(json.dumps(run(parser.parse_args().run_root.resolve()), indent=2))

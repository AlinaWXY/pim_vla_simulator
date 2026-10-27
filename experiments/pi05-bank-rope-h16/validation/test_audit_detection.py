"""Mutate actual native logs in memory and require the audit to reject them."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import audit_run as A


def run(directory):
    directory = Path(directory).resolve()
    A.audit_run(directory, write=False)
    cache = {}
    real_read = A.read_csv

    def cached(path):
        path = Path(path)
        if path not in cache:
            cache[path] = real_read(path)
        return copy.deepcopy(cache[path])

    ep = directory / 'simulation/bgk_events.csv.ch0'
    np = directory / 'simulation/timeline.csv.ch0'
    original = cached(ep)
    scenarios = ('wrong_active_rank', 'remote_before_peer_completion', 'zero_rope_completion',
                 'missing_k_burst', 'wrong_working_row_activation', 'duplicate_score_offset')
    reports = []
    for fault in scenarios:
        def mutated(path):
            path = Path(path)
            rows = cached(path)
            if path == ep:
                if fault == 'wrong_active_rank':
                    next(e for e in rows if e['operation'] == 'PIM_MAC_QK')['Rank'] = 0
                elif fault == 'remote_before_peer_completion':
                    remote = next(e for e in rows if e['operation'] == 'PIM_Q_REMOTE')
                    peer_events = cached(directory / 'simulation/bgk_events.csv.ch1')
                    peer = next(e for e in peer_events if e['operation'] == 'PIM_Q_EXPORT'
                                and A.location(e)[1:] == A.location(remote)[1:])
                    remote['clock'] = peer['ready_at'] - 1
                    remote['ready_at'] = remote['clock'] + remote['service_cycles']
                elif fault == 'zero_rope_completion':
                    rope = next(e for e in rows if e['operation'] == 'PIM_ROPE_EXEC')
                    rope.update(ready_at=rope['clock'], service_cycles=0)
                elif fault == 'missing_k_burst':
                    rows.pop(next(i for i, e in enumerate(rows) if e['operation'] == 'PIM_MAC_QK'))
                elif fault == 'duplicate_score_offset':
                    scores = [e for e in rows if e['operation'] == 'PIM_RD_BG']
                    scores[1]['Row'], scores[1]['Column'] = scores[0]['Row'], scores[0]['Column']
            if path == np:
                if fault == 'remote_before_peer_completion':
                    remote = next(e for e in original if e['operation'] == 'PIM_Q_REMOTE')
                    peer_events = cached(directory / 'simulation/bgk_events.csv.ch1')
                    peer = next(e for e in peer_events if e['operation'] == 'PIM_Q_EXPORT'
                                and A.location(e)[1:] == A.location(remote)[1:])
                    command = next(e for e in rows if e['clock'] == remote['clock'] and e['command'] == 'WRGB')
                    command['clock'] = peer['ready_at'] - 1
                elif fault == 'wrong_working_row_activation':
                    activates = [e for e in rows if e['command'] in ('ACT2', 'ACTAB') and e['Row'] == A.WORK_ROW]
                    A.require(activates, 'No work row activation available for fault injection')
                    activates[0]['Row'] -= 1
            return rows
        try:
            with patch.object(A, 'read_csv', side_effect=mutated):
                A.audit_run(directory, write=False)
        except AssertionError as error:
            reports.append(dict(fault=fault, detected=True, reason=str(error)[:300]))
        else:
            raise AssertionError(f'Independent audit accepted injected fault: {fault}')
    report = dict(status='passed', faults=reports, native_result_files_modified=False,
                  method='Read interception mutates in-memory CSV copies only; valid baseline audited first')
    (directory / 'audit-fault-detection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    print(json.dumps(run(parser.parse_args().path), indent=2))

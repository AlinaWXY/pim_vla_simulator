"""Native Z dependency rejections and a bounded overlapping-row progress smoke."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def address(bank=0, column=0, rank=1, row=176):
    return (row << 20) | ((bank % 4) << 18) | ((bank // 4) << 16) | (rank << 15) | (column << 9)


def progress_trace():
    # No inter-fragment fence: a future Z row must not repeatedly cancel the
    # older MAC's weight-row activation. Four fragments also reuse both slots.
    lines = []
    for fragment in range(4):
        lines += [f'PIM_Z_LOAD {address(bank, fragment)} {fragment % 2}' for bank in range(16)]
        lines += [f'PIM_MAC_AB {address(row=160, column=16*fragment+slot)} {fragment % 2}' for slot in range(16)]
    return '\n'.join(lines+['BARRIER 0', 'WAIT 16', 'BARRIER 1'])+'\n'


CASES = {
    'overlapping_rows_progress_smoke': (progress_trace(), True, None),
    'load_without_store_or_preinstall': (
        f'PIM_Z_LOAD {address()} 0\n', False,
        'Z_LOAD has no completed Z_STORE or explicit preinstalled Z'),
    'duplicate_source_bank': (
        f'PIM_Z_LOAD {address()} 0\nBARRIER 0\nPIM_Z_LOAD {address()} 0\n', True,
        'Duplicate Z_LOAD source bank'),
    'mixed_fragment_columns': (
        f'PIM_Z_LOAD {address()} 0\nBARRIER 0\nPIM_Z_LOAD {address(1,1)} 0\n', True,
        'Z_LOAD mixed columns'),
    'incomplete_sixteen_bank_input': (
        ''.join(f'PIM_Z_LOAD {address(bank)} 0\n' for bank in range(15)) +
        f'BARRIER 0\nPIM_MAC_AB {address(row=160)} 0\n', True,
        'Down MAC requires all16 source banks'),
    'rank0_z_request': (
        f'PIM_Z_STORE {address(rank=0)}\n', False,
        'rank0 is reserved for GPU'),
    'wrong_z_row': (
        f'PIM_Z_LOAD {address(row=175)} 0\n', True,
        'PIM_Z address must be row176'),
}


def worker(backend, output, name):
    sys.path[:0] = [str(backend/'ramulator2/python'), str(backend)]
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig

    output.mkdir(parents=True, exist_ok=False)
    trace = output/'invalid.trace'
    source, preinstalled, message = CASES[name]
    trace.write_text(source, encoding='ascii')
    config = MemoryConfig(model='lpddr5x-8533', channels=16, ranks=2,
        pim=True, refresh=False, pim_data_path='attacc', pim_mac_scope='rank',
        pim_input_broadcast=False, pim_rank_rope=False, pim_ffn_dram=True,
        ffn_z_preinstalled=preinstalled, pim_local_transfer_cycles=2,
        frontend_ratio=32, max_cycles=5_000,
        timing_overrides={'nCCDAB': 16, 'nLOCALWR': 4})
    try:
        native = run_trace(config, trace, output/'simulation')
    except Exception as error:
        result = dict(case=name, passed=message is not None and isinstance(error, RuntimeError) and message in str(error),
            compiled_backend_invoked=True, exception_type=type(error).__name__,
            message=str(error), expected_exception_type='RuntimeError', expected_message=message,
            ffn_z_preinstalled=preinstalled)
    else:
        if message is None:
            with (output/'simulation/timeline.csv.ch0').open(newline='') as stream:
                events = list(csv.DictReader(stream))
            macs = [e for e in events if e['command'] == 'MACAB']
            last_mac = max(int(e['clock']) for e in macs) if macs else -1
            passed = (native['issued_commands'].get('LOCALRD') == 64 and
                      native['issued_commands'].get('MACAB') == 64 and len(macs) == 64 and
                      native['memory_cycles'] >= last_mac+16 and native['memory_cycles'] < 5_000 and
                      all(e['Rank'] == '1' and e['pim_rank_mask'] == '2' for e in macs))
            result = dict(case=name, passed=passed, compiled_backend_invoked=True,
                positive_progress_smoke=True, memory_cycles=native['memory_cycles'],
                bounded_max_cycles=5_000, expected_reads=64, expected_macs=64,
                last_mac_clock=last_mac, message='Four unfenced Z/MAC fragments must complete with full service tail')
        else:
            result = dict(case=name, passed=False, compiled_backend_invoked=True,
                          message='Invalid trace completed without native rejection')
    (output/'case-result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if args.case:
        return worker(backend, output, args.case)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(backend/'ramulator2/python'), str(backend), env.get('PYTHONPATH', '')])
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(backend/'ramulator2'), env.get('LD_LIBRARY_PATH', '')])
    env['PYTHONNOUSERSITE'] = '1'
    results = []
    for name in CASES:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--backend', str(backend), '--output', str(output/name), '--case', name],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        (output/f'{name}.log').write_text(completed.stdout, encoding='utf-8')
        path = output/name/'case-result.json'
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(
            case=name, passed=False, message='No native rejection evidence from worker')
        result['worker_returncode'] = completed.returncode
        result['passed'] = bool(result.get('passed')) and completed.returncode == 0
        results.append(result)
    summary = dict(passed=all(r['passed'] for r in results), cases=results,
                   case_count=len(results), subprocess_isolation=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

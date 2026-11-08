"""Validate recorded GPU effective addresses; no DRAM timing inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def addresses(path, prefix, strided):
    text = path.read_text()
    base = re.search(prefix+r'_BASE a=(0x[0-9a-f]+) b=(0x[0-9a-f]+) c=(0x[0-9a-f]+) bytes=256', text)
    assert base, path
    a, b, c = [int(x, 16) for x in base.groups()]
    assert prefix+'_PASS' in text, path
    records = []
    pattern = r'MEMTRACE: CTX .* - grid_launch_id (\d+) - CTA (\d+),(\d+),(\d+) - warp (\d+) - (\S+) - (.*)'
    for match in re.finditer(pattern, text):
        launch, x, y, z, warp = map(int, match.groups()[:5])
        opcode = match[6]
        if not opcode.startswith(('LDG', 'STG')):
            continue
        values = [int(x, 16) for x in re.findall(r'0x[0-9a-f]+', match[7])]
        assert len(values) == 32
        start = x*32 if strided else warp*32
        if opcode.startswith('STG'):
            expected = [c+4*(start+lane) for lane in range(32)]
        elif a <= values[0] < a+256:
            expected = [a+4*(start+lane) for lane in range(32)]
        else:
            expected = [b+4*((17*(start+lane)) % 64 if strided else start+lane) for lane in range(32)]
        assert values == expected, (path, opcode, start)
        records.append((opcode, values))
    assert len(records) == 6
    reads = Counter(v for op, vs in records if op.startswith('LDG') for v in vs)
    writes = Counter(v for op, vs in records if op.startswith('STG') for v in vs)
    assert reads == Counter([a+4*i for i in range(64)]+[b+4*i for i in range(64)])
    assert writes == Counter(c+4*i for i in range(64))
    return dict(status='passed', warp_memory_records=6, lane_reads=128, lane_writes=64,
                exact_effective_addresses_and_lane_order_verified=True,
                scope='Tiny correctness probe; GPU effective/virtual addresses, not physical DRAM requests.')


def samples(path, metric):
    text = path.read_text()
    assert 'Result verification passed.' in text
    times = [(int(a), int(b)) for a, b in re.findall(r'Timestamps -> Start: \[(\d+)\]\s+End: \[(\d+)\]', text)]
    vals = [float(v.replace(',', '')) for v in re.findall(re.escape(metric)+r'\s+([\d,.]+)', text)]
    assert len(times) == len(vals) > 0
    first = dict(begin_ns=times[0][0], end_ns=times[0][1], value=vals[0])
    # NVIDIA documents a possible outlier in the first PM sample and
    # recommends discarding it. Keep the rejected record in the audit.
    times, vals = times[1:], vals[1:]
    assert all(a < b for a, b in times)
    assert any(x > 0 for x in vals)
    return dict(status='passed', metric=metric, retained_printed_samples=len(times),
                first_sample_excluded_per_CUPTI_guidance=first,
                min_interval_ns=min(b-a for a,b in times), max_interval_ns=max(b-a for a,b in times),
                min_value=min(vals), max_value=max(vals),
                scope='Official CUPTI vector-add probe; printed sample subset, not a Pi0.5 or DRAM-address trace.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    metrics = (a.root/'pm-metrics.log').read_text().splitlines()
    dram = [x.strip() for x in metrics if 'dram__' in x]
    result = dict(status='feasibility_probes_passed; Pi0.5 full capture and e/g replay not completed',
                  platform='NVIDIA Thor, sm_110, aarch64; native LPDDR target retained',
                  CUDA_address_probe=addresses(a.root/'injected-shared.log', 'PROBE', True),
                  PyTorch_address_probe=addresses(a.root/'torch-address.log', 'TORCH', False),
                  PM_L2=samples(a.root/'pm-l2-sectors.log', 'lts__t_sectors.sum'),
                  PM_cycles=samples(a.root/'pm-cycles.log', 'gpc__cycles_elapsed.avg'),
                  exposed_dram_base_metrics=dram,
                  limitations=[
                      'NVBit instrumentation changes execution time; its recording order is not a native MC arrival trace.',
                      'Effective addresses are not physical addresses or rank/bank/row IDs.',
                      'L2 sector activity includes hits and does not identify individual DRAM reads or writebacks.',
                      'Only small CUDA and PyTorch addition probes have been validated; full Pi0.5 and TMA coverage remain unverified.',
                      'No calibrated Thor GPU timing model or physical-address mapping has been established.',
                      'No HBM GPU configuration was substituted for Thor/LPDDR; no new e/g latency was computed.'])
    (a.root/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    files = [f for f in a.root.iterdir() if f.is_file() and f.name not in ('SHA256SUMS.json', 'validation-run.log')]
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (a.root/'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

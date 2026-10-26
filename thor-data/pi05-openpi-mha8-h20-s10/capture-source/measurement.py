"""Model-call and phase timing with no profiler and GPU occupancy guards."""
import time
import torch
from common import gpu_guard, stats


def measure(call, label, meter=None, warmup=20, iterations=100):
    gpu_guard()
    print(time.strftime('%H:%M:%S'), label, f'warmup={warmup} n={iterations}', flush=True)
    for i in range(warmup):
        if meter:
            meter.reset()
        call(i)
        torch.cuda.synchronize()
    rows = []
    for i in range(iterations):
        if meter:
            meter.reset()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        result = call(i)
        end.record()
        torch.cuda.synchronize()
        row = dict(wall_ms=(time.perf_counter()-wall)*1000, gpu_span_ms=start.elapsed_time(end))
        if meter:
            row.update(meter.read(start, end))
        rows.append(row)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite output')
    gpu_guard()
    record = dict(label=label, samples=rows,
                  wall_ms=stats([r['wall_ms'] for r in rows]),
                  gpu_span_ms=stats([r['gpu_span_ms'] for r in rows]))
    if meter:
        record['components_ms'] = {k: stats([r[k+'_ms'] for r in rows]) for k in ('vit','vlm','ae','other')}
        record['shares_pct'] = {k: 100*v['mean']/record['gpu_span_ms']['mean'] for k,v in record['components_ms'].items()}
    print('TIMING', label, record['wall_ms'], flush=True)
    return record

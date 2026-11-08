"""Ordered, cross-window GPU service model; no synthetic memory requests.

This implements the user-selected approximation. A VLM work unit makes one
microsecond of progress per microsecond of joint bus/rank availability. Busy
intervals suspend its service clock. At the next AE GPU operation, finish any
already-started VLM work unit first. Neither kernel fitting nor a minimum idle
window threshold is used. This is NOT request-level hardware simulation.
"""
import math


def replay(blocks, units, windows):
    """windows(block) returns local [start,end) availability intervals in us."""
    eps = 1e-9
    index = 0
    remaining = 0.0
    now = 0.0
    progress_pim = 0.0
    wait_gpu = 0.0
    placements = []
    waits = []
    execution = []
    current = None
    available = 0.0
    segments = 0

    def begin(at):
        nonlocal remaining, current
        remaining = units[index]['us']
        assert remaining > 0
        current = dict(ordinal=index, kind=units[index].get('kind', 'kernel'),
                       start_us=at, solo_us=remaining, progress_during_pim_us=0.0,
                       tail_wait_us=0.0, service_intervals=0)

    def finish(at):
        nonlocal index, current, remaining
        current['end_us'] = at
        assert math.isclose(current['solo_us'], current['progress_during_pim_us']
                            + current['tail_wait_us'], abs_tol=1e-7)
        placements.append(current)
        index += 1
        remaining = 0.0
        current = None

    for number, block in enumerate(blocks):
        elapsed = block['end_us'] - block['begin_us']
        assert elapsed >= 0
        if block['kind'] == 'gpu':
            if current is not None:
                delay = remaining
                current['tail_wait_us'] += delay
                wait_gpu += delay
                waits.append(dict(block=number, name=block['name'],
                                  step=block.get('step'), layer=block.get('layer'),
                                  vlm_ordinal=index, ready_us=now,
                                  gpu_begin_us=now + delay, wait_us=delay))
                now += delay
                finish(now)
        start = now
        if block['kind'] == 'pim':
            previous = 0.0
            for a, b in windows(block):
                assert -eps <= a < b <= elapsed + eps
                assert a >= previous - eps
                previous = b
                available += b - a
                at = start + a
                limit = start + b
                while at < limit - eps and index < len(units):
                    if current is None:
                        begin(at)
                    work = min(remaining, limit - at)
                    remaining -= work
                    current['progress_during_pim_us'] += work
                    current['service_intervals'] += 1
                    progress_pim += work
                    segments += 1
                    at += work
                    if remaining <= eps:
                        finish(at)
        now += elapsed
        execution.append(dict(block=number, kind=block['kind'], name=block['name'],
                              step=block.get('step'), layer=block.get('layer'),
                              begin_us=start, end_us=now))

    ae_end = now
    completed_service = progress_pim + wait_gpu
    total_service = sum(x['us'] for x in units)
    assert completed_service <= total_service + 1e-6
    # Any work unit crossing the final AE completion is finished outside AE.
    if current is not None:
        current['remaining_at_AE_end_us'] = remaining
        current['end_us'] = now + remaining
        placements.append(current)
    remaining_vlm = max(0.0, total_service - completed_service)
    total = ae_end + remaining_vlm
    solo_ae = sum(b['end_us'] - b['begin_us'] for b in blocks)
    assert math.isclose(ae_end, solo_ae + wait_gpu, abs_tol=1e-6)
    assert math.isclose(total, solo_ae + total_service - progress_pim, abs_tol=1e-6)
    assert [p['ordinal'] for p in placements] == list(range(len(placements)))
    assert all(x['end_us'] <= y['start_us'] + eps
               for x, y in zip(placements, placements[1:]))
    return dict(AE_solo_ms=solo_ae / 1000, AE_gpu_wait_ms=wait_gpu / 1000,
                AE_with_interference_ms=ae_end / 1000,
                VLM_ms=total_service / 1000,
                VLM_progress_during_PIM_ms=progress_pim / 1000,
                VLM_progress_during_AE_ms=completed_service / 1000,
                VLM_remaining_after_AE_ms=remaining_vlm / 1000,
                total_ms=total / 1000, available_window_ms=available / 1000,
                unused_window_ms=max(0.0, available - progress_pim) / 1000,
                started_VLM_units=len(placements), completed_VLM_units=index,
                cross_window_units=sum(p['service_intervals'] > 1 for p in placements),
                service_intervals=segments, placements=placements, waits=waits,
                execution=execution)

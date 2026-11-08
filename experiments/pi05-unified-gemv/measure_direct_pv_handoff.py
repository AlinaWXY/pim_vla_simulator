"""Measure an explicit GPU packing implementation for the corrected PV->O path.

The native RD_RES stream is query-major, then 16-element feature packet,
then channel. WR_GB consumes feature-packet-major, then query, then channel.
Materialize that permutation once and reuse it for both O tiles. This is an
implementation choice, not a claim that every driver must perform a copy.
Only these two new boundary kernels are measured; no VLM/model rerun.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1108)
    flush = torch.empty(32 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    result = dict(
        scope='Only the newly required direct PV-to-O GPU packing implementation',
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_axes=['query', 'feature_packet', 'channel', 'bf16_lane'],
        destination_axes=['feature_packet', 'query', 'channel', 'bf16_lane'],
        implementation='transpose(0,1).contiguous(); reuse for both O tiles',
        input_output_dtype='bfloat16',
        timing='device kernel time; CUDA profiler CPU parent aggregation',
        cold_policy='32 MiB write before each invocation; same supplemental handoff policy as existing measurements',
        operators={},
    )
    for layout, fragments in [('feature', 8), ('token', 16)]:
        name = layout + '_direct_PV_to_O_pack'
        x = torch.randn(16, fragments, 16, 16, device='cuda', dtype=torch.bfloat16)
        def operation():
            return x.transpose(0, 1).contiguous()
        for _ in range(20):
            y = operation()
        torch.cuda.synchronize()
        modes = {}
        for mode in ('warm', 'cold32MiB'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(60):
                    if mode != 'warm':
                        flush.zero_()
                    with torch.profiler.record_function('boundary/' + name):
                        y = operation()
                torch.cuda.synchronize()
            events = [e for e in prof.events() if e.name == 'boundary/' + name
                      and e.device_type == torch.autograd.DeviceType.CPU]
            values = [e.device_time_total for e in events]
            assert len(values) == 60 and min(values) > 0
            modes[mode] = dict(device_kernel_median_us=statistics.median(values),
                               device_kernel_mean_us=statistics.mean(values),
                               device_kernel_samples_us=values)
        # Validate the actual permutation, including all elements, not just shape.
        assert y.is_contiguous() and torch.equal(y.transpose(0, 1), x)
        modes['validation'] = dict(input_shape=list(x.shape), output_shape=list(y.shape),
                                    payload_bytes=x.numel() * x.element_size(),
                                    permutation_exact=True, reused_O_tiles=2)
        result['operators'][name] = modes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v['cold32MiB']['device_kernel_median_us']
                      for k, v in result['operators'].items()}), flush=True)


if __name__ == '__main__':
    main()

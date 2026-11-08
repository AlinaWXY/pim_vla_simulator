"""Reconstruct the retained prefix; collect counters without replacing its timings.

Run in the isolated, pinned OpenPI/transformers overlay. NCU capture is delimited
by cudaProfilerStart/Stop; --mode trace instead writes a kernel-name alignment
trace. No AE latency baseline is measured by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['trace', 'counters', 'logical'], required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.support))
    from common import gpu_guard, load_weights
    from mha8 import convert
    gpu_guard()
    import torch
    import transformers
    from transformers.modeling_utils import no_init_weights
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import openpi.models_pytorch.pi0_pytorch as pi
    import openpi.models_pytorch.gemma_pytorch as gemma
    import transformers.models.gemma.modeling_gemma as patched
    assert torch.__version__ == '2.11.0+cu130'
    assert transformers.__version__ == '4.53.2'
    expected = ['dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16',
                '225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb',
                '17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde']
    hashes = {m.__file__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for m in (pi, gemma, patched)}
    assert list(hashes.values()) == expected, hashes
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    config = Pi0Config(pi05=True, action_horizon=16, pytorch_compile_mode=None)
    with no_init_weights():
        model = PI0Pytorch(config)
    model.paligemma_with_expert.paligemma.tie_weights()
    model.eval()
    checkpoint = '/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors'
    log('Loading retained checkpoint')
    weights = load_weights(model, checkpoint, log)
    model.to('cuda')
    tokens = json.loads((a.support / 'pi_tokens.json').read_text())['pi05']
    gen = torch.Generator().manual_seed(17)
    images = {k: (torch.rand((1, 3, 224, 224), generator=gen)*2-1).to('cuda')
              for k in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')}
    obs = Observation(images=images,
        image_masks={k: torch.ones(1, dtype=torch.bool, device='cuda') for k in images},
        state=torch.zeros((1,32), device='cuda'),
        tokenized_prompt=torch.tensor([tokens['ids']], device='cuda'),
        tokenized_prompt_mask=torch.tensor([tokens['mask']], dtype=torch.bool, device='cuda'))
    noise = torch.randn((1,16,32), generator=gen).to('cuda')
    class PrefixComplete(Exception):
        pass
    def stop_before_ae(*args, **kwargs):
        raise PrefixComplete
    model.denoise_step = stop_before_ae
    def prefix():
        try:
            model.sample_actions('cuda', obs, noise=noise, num_steps=10)
        except PrefixComplete:
            return
        raise AssertionError('Prefix stop hook was not reached')
    with torch.inference_mode():
        conversion = convert(model)
        for i in range(3):
            gpu_guard()
            prefix()
            torch.cuda.synchronize()
            log('Warm prefix ' + str(i))
        gpu_guard()
        if a.mode == 'trace':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                prefix()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(a.output / 'prefix-trace.json'))
        elif a.mode == 'counters':
            torch.cuda.cudart().cudaProfilerStart()
            prefix()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            from torch.utils._python_dispatch import TorchDispatchMode
            from torch.utils._pytree import tree_flatten
            records=[]
            def tensors(value):
                found={}
                for x in tree_flatten(value)[0]:
                    if isinstance(x,torch.Tensor) and x.is_cuda:
                        key=(x.data_ptr(),tuple(x.shape),tuple(x.stride()))
                        found[key]=dict(shape=list(x.shape),dtype=str(x.dtype),
                                        logical_bytes=x.numel()*x.element_size())
                return list(found.values())
            class FlashIO(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    kwargs=kwargs or {}
                    result=func(*args,**kwargs)
                    if 'flash_attention' in str(func):
                        inputs=tensors((args,kwargs));outputs=tensors(result)
                        records.append(dict(operator=str(func),inputs=inputs,outputs=outputs,
                            read_bytes=sum(x['logical_bytes'] for x in inputs),
                            write_bytes=sum(x['logical_bytes'] for x in outputs)))
                    return result
            with FlashIO():
                prefix()
                torch.cuda.synchronize()
            assert len(records)==81,len(records)
            (a.output/'flash-logical-io.json').write_text(json.dumps(dict(
                scope='Logical tensor footprint, not measured DRAM bytes',records=records),indent=2)+'\n')
    report = dict(source_sha256=hashes, checkpoint=checkpoint, weights=weights,
                  conversion=conversion, mode=a.mode, torch=torch.__version__,
                  transformers=transformers.__version__, pid=os.getpid(),
                  policy='Memory evidence only; retained baseline durations remain unchanged',
                  input=dict(batch=1, cameras=3, image_hw=[224,224], prompt_tokens=200,
                             query_heads=8, kv_heads=8, H=16, seed=17))
    (a.output / ('capture-' + a.mode + '.json')).write_text(json.dumps(report, indent=2)+'\n')
    log('Capture completed')


if __name__ == '__main__':
    main()

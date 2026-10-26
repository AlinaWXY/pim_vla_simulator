"""Reuse yesterday's cached OpenPI baseline; only attention head geometry changes."""
import argparse
from collections import Counter
from contextlib import contextmanager
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import gpu_guard, load_weights, compare
from bench import ROOT, Meter
from mha8 import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('validate','coarse','labels','fine'), required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--validated')
    ap.add_argument('--labels')
    ns = ap.parse_args()
    out = ROOT/'runs'/ns.tag
    out.mkdir(exist_ok=False)
    success = False
    counter = 0
    def emit(kind, **values):
        nonlocal counter
        counter += 1
        (out/f'{counter:03d}_{kind}.json').write_text(json.dumps(dict(kind=kind, **values), indent=2)+'\n')
        print(time.strftime('%H:%M:%S'), kind, flush=True)
    def phase(text):
        print(time.strftime('%H:%M:%S'), text, flush=True)

    try:
        gpu_guard()
        import torch
        import transformers
        from transformers.modeling_utils import no_init_weights
        from transformers.models.gemma import modeling_gemma
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.model import Observation
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from adarms_cache import Pi05ConditioningCache
        from measurement import measure
        from fine import run_fine, kv_metadata
        import fine
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        cfg = Pi0Config(pi05=True, action_horizon=20, pytorch_compile_mode=None)
        sources = [Path(__file__), Path(__file__).with_name('mha8.py'),
                   Path(inspect.getfile(PI0Pytorch)), Path(inspect.getfile(modeling_gemma)),
                   Path(inspect.getfile(Pi05ConditioningCache)), Path(inspect.getfile(fine)),
                   ROOT/'code/bench.py', ROOT/'code/common.py',
                   ROOT/'code/baselines_20261021/measurement.py',
                   ROOT/'runtime/openpi/src/openpi/models_pytorch/gemma_pytorch.py']
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        if ns.mode != 'validate':
            validation = json.loads(Path(ns.validated).read_text())
            if not validation['cache_bitwise_equal'] or validation['source_hashes'] != hashes:
                raise RuntimeError('Missing or stale validation receipt')
        phase('Loading existing Thor checkpoint with sequential readinto')
        with no_init_weights():
            model = PI0Pytorch(cfg)
        model.paligemma_with_expert.paligemma.tie_weights()
        model.eval()
        checkpoint = Path('/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors')
        loaded = load_weights(model, checkpoint, phase)
        gpu_guard()
        model.to('cuda')
        tokens = json.loads((ROOT/'code/pi_tokens.json').read_text())['pi05']
        values = []
        for seed in (17,23,41,97):
            gen = torch.Generator().manual_seed(seed)
            images = {name:(torch.rand((1,3,224,224), generator=gen)*2-1).to('cuda')
                      for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
            obs = Observation(images=images,
                              image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in images},
                              state=torch.zeros((1,32),device='cuda'),
                              tokenized_prompt=torch.tensor([tokens['ids']],device='cuda',dtype=torch.long),
                              tokenized_prompt_mask=torch.tensor([tokens['mask']],device='cuda',dtype=torch.bool))
            values.append((obs, torch.randn((1,20,32), generator=gen).to('cuda')))
        def call(i=0):
            obs, noise = values[i % len(values)]
            return model.sample_actions('cuda', obs, noise=noise, num_steps=10)

        with torch.inference_mode():
            original = [call(i).clone() for i in range(4)] if ns.mode == 'validate' else None
            conversion = convert(model)
            cache = Pi05ConditioningCache(model)
            dimensions = {}
            for name, tr in [('vlm',model.paligemma_with_expert.paligemma.language_model),
                             ('expert',model.paligemma_with_expert.gemma_expert.model)]:
                dimensions[name] = dict(layers=len(tr.layers), residual_width=tr.config.hidden_size,
                                        ffn_width=tr.config.intermediate_size,
                                        query_heads=tr.config.num_attention_heads,
                                        kv_heads=tr.config.num_key_value_heads, head_dim=tr.config.head_dim,
                                        q_weight=list(tr.layers[0].self_attn.q_proj.weight.shape),
                                        k_weight=list(tr.layers[0].self_attn.k_proj.weight.shape),
                                        v_weight=list(tr.layers[0].self_attn.v_proj.weight.shape),
                                        o_weight=list(tr.layers[0].self_attn.o_proj.weight.shape))
            emit('environment', mode=ns.mode, torch=torch.__version__, transformers=transformers.__version__,
                 source_commit='215abfb217dbac7d5f1273282331b9b1866c0479', source_hashes=hashes,
                 python=sys.executable, checkpoint=str(checkpoint), load=loaded,
                 constructor_config=dataclasses.asdict(cfg), actual_dimensions=dimensions,
                 parameter_dtype_elements=dict(Counter({d:sum(p.numel() for p in model.parameters() if str(p.dtype)==d)
                                                        for d in {str(p.dtype) for p in model.parameters()}})),
                 cpu_threads=1, interop_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
                 matmul_precision=torch.get_float32_matmul_precision(), cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                 conditioning_cache=cache.metadata(), conversion=conversion,
                 backend='Same native OpenPI eager attention as yesterday, selective BF16/FP32, no compile/CUDA Graph or cache flush')

            if ns.mode == 'validate':
                uncached = [call(i).clone() for i in range(4)]
                change_checks = [compare(x,y) for x,y in zip(original,uncached)]
                records, prefix_kv = [], []
                native_attn = modeling_gemma.eager_attention_forward
                native_step = model.denoise_step
                def attention(module,q,k,v,mask,scaling,**kwargs):
                    if q.shape[1]!=8 or k.shape[1]!=8 or v.shape[1]!=8 or q.shape[-1]!=256:
                        raise RuntimeError('Actual attention shape does not match MHA8')
                    if module.num_key_value_groups!=1 or min(k.stride()[1],v.stride()[1])<=0:
                        raise RuntimeError('KV heads are broadcast rather than physically expanded')
                    records.append(dict(q=list(q.shape),k=list(k.shape),v=list(v.shape),
                                        q_stride=list(q.stride()),k_stride=list(k.stride()),v_stride=list(v.stride())))
                    return native_attn(module,q,k,v,mask,scaling,**kwargs)
                def step(*args,**kwargs):
                    if not prefix_kv:
                        prefix_kv.append(kv_metadata(args[2]))
                    return native_step(*args,**kwargs)
                def guard_joint(module,args,kwargs):
                    embeds=kwargs.get('inputs_embeds')
                    if embeds is not None and all(x is not None for x in embeds):
                        raise RuntimeError('Joint training path is outside the validated inference conversion')
                hook=model.paligemma_with_expert.register_forward_pre_hook(guard_joint,with_kwargs=True)
                with cache.applied():
                    cached = [call(i).clone() for i in range(4)]
                    equality = [torch.equal(a,b) for a,b in zip(uncached,cached)]
                    if not all(equality):
                        raise RuntimeError('AdaRMS cache changed MHA8 outputs')
                    modeling_gemma.eager_attention_forward = attention
                    model.denoise_step = step
                    try:
                        checked = call(0)
                        torch.cuda.synchronize()
                        if not torch.equal(checked,cached[0]):
                            raise RuntimeError('Shape hooks changed outputs')
                    finally:
                        modeling_gemma.eager_attention_forward = native_attn
                        del model.denoise_step
                hook.remove()
                if len(records)!=198 or cache.dense_hits!=370 or Counter(r['q'][2] for r in records)!=Counter({968:18,20:180}):
                    raise RuntimeError('Unexpected attention or cache count')
                if prefix_kv[0]['unique_storage_bytes'] != 18*2*1*8*968*256*2:
                    raise RuntimeError('Unexpected physical prefix KV storage')
                result=dict(cache_bitwise_equal=True, cache_checks=equality, source_hashes=hashes,
                            reparameterization_checks=change_checks, actual_dimensions=dimensions,
                            prefix_kv=prefix_kv[0], attention_calls=records,
                            cache=cache.metadata(), original_model_accuracy_claim=False,
                            note='Inference-only synthetic MHA8. No trained-head diversity; original joint training branch is not modified or used.')
                (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
                emit('validation_complete', result=result)
            else:
                with cache.applied():
                    if ns.mode=='coarse':
                        reference=call(0).clone()
                        emit('clean_before',result=measure(call,'MHA8_H20_clean_before'))
                        meter=Meter(model)
                        meter.install()
                        try:
                            marked=call(0)
                            if not torch.equal(reference,marked):
                                raise RuntimeError('Phase instrumentation changed output')
                            emit('phase',result=measure(call,'MHA8_H20_phases',meter))
                        finally:
                            meter.restore()
                        emit('clean_after',result=measure(call,'MHA8_H20_clean_after'))
                    else:
                        result=run_fine(model,cache,call,out,ns.mode,ns.labels,phase)
                        result.update(query_heads=8,head_dim=256,actual_dimensions=dimensions)
                        emit(ns.mode+'_complete',result=result)
                emit('cache_audit',metadata=cache.metadata())
        gpu_guard()
        success=True
    except Exception:
        emit('failed',error=traceback.format_exc())
        raise
    finally:
        (out/'exit.json').write_text(json.dumps(dict(success=success,finished_at=time.time()))+'\n')


if __name__=='__main__':
    main()

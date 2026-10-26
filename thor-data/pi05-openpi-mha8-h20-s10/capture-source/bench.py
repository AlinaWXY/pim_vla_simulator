"""Benchmark pinned official OpenPI without modifying its model equations."""
import argparse
import collections
import dataclasses
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from common import compare,gpu_guard,load_weights,stats

ROOT=Path('/var/tmp/shiyuhuang-vla-baseline')
CHECKPOINT=Path('/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('inspect','threads','eager','compiled'),required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--horizons',nargs='+',type=int,default=[50])
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--compile-mode',default='max-autotune')
    parser.add_argument('--preserve-casts',action='store_true')
    args=parser.parse_args()
    out=ROOT/'runs'/args.tag
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    def emit(kind,**data):
        record=dict(kind=kind,**data)
        records.append(record)
        path=out/f'{len(records):03d}_{kind}.json'
        path.write_text(json.dumps(record,indent=2)+'\n')
        print('RECORD',kind,str(path),flush=True)
    def phase(text):
        print(time.strftime('%H:%M:%S'),text,flush=True)
        with (out/'phases.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),message=text))+'\n')
    try:
        if args.mode!='inspect':gpu_guard()
        if args.preserve_casts:
            os.environ['TORCHINDUCTOR_EMULATE_PRECISION_CASTS']='1'
        import torch
        import transformers
        from transformers.modeling_utils import no_init_weights
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.model import Observation
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        cfg=Pi0Config(pi05=True,action_horizon=50,pytorch_compile_mode=None)
        phase('Constructing official OpenPI, selective BF16/FP32')
        with no_init_weights():model=PI0Pytorch(cfg)
        model.paligemma_with_expert.paligemma.tie_weights()
        model.eval()
        param_counts=collections.Counter()
        for name,p in model.named_parameters():param_counts[str(p.dtype)]+=p.numel()
        vision=model.paligemma_with_expert.paligemma.vision_tower
        for name,p in vision.named_parameters():
            expected=torch.float32 if 'embeddings.patch_embedding' in name or 'embeddings.position_embedding' in name else torch.bfloat16
            if p.dtype!=expected:raise RuntimeError('Unexpected vision precision: '+name+' '+str(p.dtype))
        source_files=[ROOT/'runtime/openpi/src/openpi/models_pytorch'/f for f in ('pi0_pytorch.py','gemma_pytorch.py','preprocessing_pytorch.py')]
        emit('environment',arguments=vars(args),python=sys.executable,torch=torch.__version__,
             transformers=transformers.__version__,transformers_path=transformers.__file__,
             source_commit='215abfb217dbac7d5f1273282331b9b1866c0479',
             source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
             config=dataclasses.asdict(cfg),parameter_dtype_elements=dict(param_counts),
             matmul_precision=torch.get_float32_matmul_precision(),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
             torch_threads=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),
             checkpoint=str(CHECKPOINT),cpu_affinity=sorted(os.sched_getaffinity(0)),
             gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv'],text=True))
        if args.mode=='inspect':
            emit('inspect_complete',cuda_initialized=torch.cuda.is_initialized(),parameter_names=list(model.state_dict()))
            return
        phase('Loading existing local checkpoint with strict name/shape coverage')
        audit=load_weights(model,CHECKPOINT/'model.safetensors',phase)
        emit('load_complete',audit=audit)
        gpu_guard()
        model=model.to('cuda')
        torch.cuda.synchronize()
        native=model.sample_actions
        tokens=json.loads((ROOT/'code/pi_tokens.json').read_text())['pi05']

        def make_inputs(h,seed):
            gen=torch.Generator().manual_seed(seed)
            images={name:(torch.rand((1,3,224,224),generator=gen)*2-1).to('cuda') for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
            obs=Observation(images=images,image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in images},
                state=torch.zeros((1,32),device='cuda'),
                tokenized_prompt=torch.tensor([tokens['ids']],device='cuda',dtype=torch.long),
                tokenized_prompt_mask=torch.tensor([tokens['mask']],device='cuda',dtype=torch.bool))
            noise=torch.randn((1,h,32),generator=gen).to('cuda')
            return obs,noise

        def invoke(fn,inputs,compiled=False):
            if compiled:torch.compiler.cudagraph_mark_step_begin()
            return fn('cuda',inputs[0],noise=inputs[1],num_steps=10)

        def measure(fn,inputs,label,compiled=False,meter=None):
            gpu_guard()
            phase(f'{label}: warmup={args.warmup}, samples={args.iterations}')
            for i in range(args.warmup):
                if meter:meter.reset()
                result=invoke(fn,inputs[i%len(inputs)],compiled)
                torch.cuda.synchronize()
            samples=[]
            for i in range(args.iterations):
                if meter:meter.reset()
                if compiled:torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                begin.record()
                result=fn('cuda',inputs[i%len(inputs)][0],noise=inputs[i%len(inputs)][1],num_steps=10)
                end.record()
                torch.cuda.synchronize()
                row={'wall_ms':(time.perf_counter()-start)*1000,'gpu_span_ms':begin.elapsed_time(end)}
                if meter:row.update(meter.read(begin,end))
                samples.append(row)
            if not torch.isfinite(result).all():raise RuntimeError('Non-finite actions')
            gpu_guard()
            record=dict(label=label,threads=torch.get_num_threads(),samples=samples,
                        wall_ms=stats([r['wall_ms'] for r in samples]),gpu_span_ms=stats([r['gpu_span_ms'] for r in samples]))
            if meter:
                record['components_ms']={k:stats([r[k+'_ms'] for r in samples]) for k in ('vit','vlm','ae','other')}
                record['shares_pct']={k:100*v['mean']/record['gpu_span_ms']['mean'] for k,v in record['components_ms'].items()}
            print('TIMING',label,json.dumps({'p50_ms':record['wall_ms']['p50'],'p95_ms':record['wall_ms']['p95']}),flush=True)
            return record

        with torch.inference_mode():
            if args.mode=='threads':
                values=[make_inputs(50,s) for s in (17,23,41,97)]
                trials=[]
                for round_id,order in enumerate(((1,4,14),(14,4,1))):
                    for n in order:
                        torch.set_num_threads(n)
                        r=measure(native,values,f'eager_threads_{n}_round_{round_id}')
                        trials.append(r)
                        emit('thread_trial',result=r)
                scores={n:statistics.mean(r['wall_ms']['p50'] for r in trials if r['threads']==n) for n in (1,4,14)}
                best=min(scores.values())
                selected=min(n for n,v in scores.items() if v<=best*1.05)
                emit('thread_selection',p50_mean_by_threads=scores,selected=selected,
                     rule='Fewest threads within 5% of the lowest mean of two round p50s; reverse order in second round')
                return

            for h in args.horizons:
                model.config=dataclasses.replace(cfg,action_horizon=h)
                values=[make_inputs(h,s) for s in (17,23,41,97)]
                phase(f'H={h}: reference outputs on four fixed synthetic observations')
                references=[]
                for item in values:
                    references.append(invoke(native,item).clone())
                before=measure(native,values,f'eager_H{h}_before')
                emit('eager_clean',horizon=h,result=before)
                chosen=native; compiled=False
                if args.mode=='compiled':
                    phase(f'H={h}: compiling official sample_actions, mode={args.compile_mode}; excluded from timing')
                    started=time.monotonic()
                    candidate=torch.compile(native,mode=args.compile_mode)
                    checks=[]
                    for reference,item in zip(references,values):
                        candidate_out=invoke(candidate,item,True).clone()
                        torch.cuda.synchronize()
                        checks.append(compare(reference,candidate_out))
                    elapsed=time.monotonic()-started
                    emit('compile_check',horizon=h,seconds=elapsed,checks=checks,
                         dynamo_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()})
                    if not all(c['accepted'] for c in checks):raise RuntimeError('Compiled output failed fixed numerical gate')
                    chosen=candidate;compiled=True
                    timed=measure(chosen,values,f'compiled_H{h}_clean',True)
                    emit('compiled_clean',horizon=h,result=timed)

                meter=Meter(model)
                meter.install()
                try:
                    def instrumented(device,observation,noise=None,num_steps=10):
                        meter.reset()
                        return native(device,observation,noise=noise,num_steps=num_steps)
                    timed_fn=torch.compile(instrumented,mode=args.compile_mode) if compiled else instrumented
                    phase(f'H={h}: validating phase-event variant')
                    checks=[]
                    for ref,item in zip(references,values):
                        value=invoke(timed_fn,item,compiled).clone()
                        torch.cuda.synchronize()
                        checks.append(compare(ref,value))
                    if not all(c['accepted'] for c in checks):raise RuntimeError('Phase markers changed output beyond fixed gate')
                    coarse=measure(timed_fn,values,f'phases_H{h}',compiled,meter)
                    emit('phase_timing',horizon=h,result=coarse,numerical_checks=checks,
                         boundary='ViT towers only (3 calls); VLM prefix transformer; AE from prefix completion through final action including Euler and loop control; Other is preprocessing/embeddings/connector/remaining gaps')
                except Exception:
                    emit('phase_failed',horizon=h,error=traceback.format_exc())
                finally:
                    meter.restore()
                after=measure(chosen,values,f'clean_H{h}_after',compiled)
                emit('clean_after',horizon=h,compiled=compiled,result=after)
            phase('Requested measurements finished')
    except Exception:
        emit('failed',error=traceback.format_exc())
        raise
    finally:
        (out/'exit.json').write_text(json.dumps(dict(finished_at=time.time(),record_kinds=[r['kind'] for r in records]))+'\n')


class Meter:
    def __init__(self,model):
        import torch
        self.model=model
        self.starts=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(3)]
        self.ends=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(3)]
        self.vlm_start=torch.cuda.Event(enable_timing=True,external=True)
        self.vlm_end=torch.cuda.Event(enable_timing=True,external=True)
        self.restores=[]
        self.reset()
    def reset(self):
        self.cursor=0
        self.prefix_count=0
    def install(self):
        vision=self.model.paligemma_with_expert.paligemma.vision_tower
        language=self.model.paligemma_with_expert.paligemma.language_model
        original=vision.forward
        self.restores.append((vision,'forward',original,'forward' in vision.__dict__))
        def vision_forward(*args,**kwargs):
            i=self.cursor;self.cursor+=1
            self.starts[i].record()
            result=original(*args,**kwargs)
            self.ends[i].record()
            return result
        vision.forward=vision_forward
        prefix=language.forward
        self.restores.append((language,'forward',prefix,'forward' in language.__dict__))
        def language_forward(*args,**kwargs):
            self.prefix_count+=1
            self.vlm_start.record()
            result=prefix(*args,**kwargs)
            self.vlm_end.record()
            return result
        language.forward=language_forward
    def read(self,start,end):
        if self.cursor!=3 or self.prefix_count!=1:raise RuntimeError('Phase coverage mismatch')
        for a,b in list(zip(self.starts,self.ends))+[(self.vlm_start,self.vlm_end)]:
            if start.elapsed_time(a)<-.01 or b.elapsed_time(end)<-.01:raise RuntimeError('Stale or out-of-request CUDA event')
        vit=sum(a.elapsed_time(b) for a,b in zip(self.starts,self.ends))
        vlm=self.vlm_start.elapsed_time(self.vlm_end)
        ae=self.vlm_end.elapsed_time(end)
        other=start.elapsed_time(end)-vit-vlm-ae
        if other<-.05:raise RuntimeError('Overlapping phase intervals')
        return dict(vit_ms=vit,vlm_ms=vlm,ae_ms=ae,other_ms=other)
    def restore(self):
        for obj,key,value,had_local in reversed(self.restores):
            if had_local:setattr(obj,key,value)
            else:delattr(obj,key)


if __name__=='__main__':main()

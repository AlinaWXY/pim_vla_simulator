"""NCU cold-kernel diagnostics, not request/step wall latency."""
import argparse
from collections import Counter
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import load_weights
from warm_analysis import classify

ROOT=Path('/var/tmp/shiyuhuang-vla-baseline')
CHECKPOINT=Path('/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors')
PINNED={
    'runtime/openpi/src/openpi/models_pytorch/pi0_pytorch.py':'dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16',
    'runtime/overlay/transformers/models/gemma/modeling_gemma.py':'17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde',
    'runtime/openpi/src/openpi/models_pytorch/gemma_pytorch.py':'225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('probe','model'),required=True)
    parser.add_argument('--cache-policy',choices=('all','none'),required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum() or (args.mode=='model' and args.cache_policy!='all'):
        raise ValueError('Invalid bounded experiment arguments')
    out=ROOT/'runs'/args.tag
    out.mkdir(exist_ok=False)
    report=dict(status='running',arguments=vars(args),host=os.uname().nodename,pid=os.getpid(),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),occupancy=[],operations=[],
        methodology='NCU isolated kernel duration; cache policy applied by launcher; no wall-time latency claim')
    def save():
        (out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
    def guard(label):
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True)
        others=[int(line.split(',')[0]) for line in raw.splitlines() if line.split(',')[0].strip().isdigit() and int(line.split(',')[0])!=os.getpid()]
        fs=os.statvfs(ROOT)
        free=fs.f_bavail*fs.f_frsize
        report['occupancy'].append(dict(label=label,monotonic_ns=time.monotonic_ns(),raw=raw,others=others,free_disk_bytes=free))
        if others or free<512*1024**2:
            save()
            raise RuntimeError('Other GPU process or low disk reserve: '+str((others,free)))
    def log(text):
        print(time.strftime('%H:%M:%S'),text,flush=True)
    active_profiler=False
    try:
        guard('before_import')
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        report['environment']=dict(torch=torch.__version__,python=sys.executable,
            ncu_version=subprocess.check_output(['/usr/local/cuda/bin/ncu','--version'],text=True),
            code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            cpu_threads=1,interop_threads=1,clock_control='none',pipeline_boost_state='stable (NCU default)',
            import_sass=False,cpu_affinity=sorted(os.sched_getaffinity(0)))
        if args.mode=='probe':
            x=torch.arange(256*1024,device='cuda',dtype=torch.float32)
            y=torch.empty_like(x)
            def operation():
                return torch.add(x,1,out=y)
            with torch.inference_mode():
                for _ in range(50):
                    operation()
                torch.cuda.synchronize()
                report['probe']=dict(input_bytes=x.numel()*x.element_size(),output_bytes=y.numel()*y.element_size(),
                    l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,repeats=4,warmup=50)
                torch.cuda.profiler.start()
                active_profiler=True
                for i in range(4):
                    guard('probe_'+str(i))
                    with torch.cuda.nvtx.range('cold_step'),torch.cuda.nvtx.range(f'op_{i:05d}'):
                        operation()
                    torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                active_profiler=False
                report['output_bitwise_equal']=torch.equal(y,x+1)
                if not report['output_bitwise_equal']:
                    raise RuntimeError('Probe changed output')
        else:
            import transformers
            from transformers.modeling_utils import no_init_weights
            from openpi.models.pi0_config import Pi0Config
            from openpi.models.model import Observation
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
            from torch.utils._python_dispatch import TorchDispatchMode
            from adarms_cache import Pi05ConditioningCache
            from mha8 import convert
            from attribution import scopes_for,kv_metadata
            actual={key:sha(ROOT/key) for key in PINNED}
            if actual!=PINNED or torch.__version__!='2.11.0+cu130' or transformers.__version__!='4.53.2':
                raise RuntimeError('Pinned model/runtime mismatch')
            report['environment'].update(source_sha256=actual,transformers=transformers.__version__,precision='accepted selective BF16/FP32',no_compile=True,no_graph=True)
            report['configuration']=dict(model='OpenPI pi0.5',H=16,steps=10,batch=1,q_heads=8,kv_heads=8,head_dim=256,
                adarms_cache=True,physical_mha_scope='VLM and AE',cameras=3,image_hw=[224,224],prefix_tokens=968,
                seed=17,synthetic_fixture=True,selected_step_zero_based=4,repeat_requests=3,warmup_requests=20)
            config=Pi0Config(pi05=True,action_horizon=16,pytorch_compile_mode=None)
            log('Load existing weights once using sequential readinto')
            with no_init_weights():
                model=PI0Pytorch(config)
            model.paligemma_with_expert.paligemma.tie_weights()
            model.eval()
            report['load']=load_weights(model,CHECKPOINT,log)
            guard('after_load')
            model.to('cuda')
            torch.cuda.synchronize()
            tokens=json.loads((ROOT/'code/pi_tokens.json').read_text())['pi05']
            gen=torch.Generator().manual_seed(17)
            images={key:(torch.rand((1,3,224,224),generator=gen)*2-1).to('cuda') for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
            obs=Observation(images=images,image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in images},
                state=torch.zeros((1,32),device='cuda'),tokenized_prompt=torch.tensor([tokens['ids']],device='cuda'),
                tokenized_prompt_mask=torch.tensor([tokens['mask']],device='cuda',dtype=torch.bool))
            noise=torch.randn((1,16,32),generator=gen).to('cuda')
            def call():
                return model.sample_actions('cuda',obs,noise=noise,num_steps=10)
            with torch.inference_mode():
                report['mha_conversion']=convert(model)
                reference=call().clone()
                torch.cuda.synchronize()
                report['dimensions']={name:dict(layers=len(tr.layers),D=tr.config.hidden_size,F=tr.config.intermediate_size,
                    q_heads=tr.config.num_attention_heads,kv_heads=tr.config.num_key_value_heads,head_dim=tr.config.head_dim,
                    active_attention=tr.config._attn_implementation)
                    for name,tr in [('vlm',model.paligemma_with_expert.paligemma.language_model),('ae',model.paligemma_with_expert.gemma_expert.model)]}
                if any((d['layers'],d['q_heads'],d['kv_heads'],d['head_dim'],d['active_attention'])!=(18,8,8,256,'eager') for d in report['dimensions'].values()):
                    raise RuntimeError('Executed attention configuration mismatch')
                cache=Pi05ConditioningCache(model,num_steps=10,batch_size=1)
                state=dict(step=0,rep=0,active=False)
                native=model.denoise_step
                roles=[]
                module_counts=Counter()
                @contextmanager
                def role_scope(role):
                    roles.append(role)
                    module_counts[(state['rep'],role)]+=1
                    try:
                        yield
                    finally:
                        roles.pop()
                class Dispatch(TorchDispatchMode):
                    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                        kwargs=kwargs or {}
                        tensors=[]
                        def visit(value):
                            if isinstance(value,torch.Tensor):
                                tensors.append(value)
                            elif isinstance(value,(list,tuple)):
                                for v in value:
                                    visit(v)
                            elif isinstance(value,dict):
                                for v in value.values():
                                    visit(v)
                        visit(args)
                        visit(kwargs)
                        shapes=[list(t.shape) for t in tensors]
                        operator=func._schema.name
                        category,family=classify(operator,shapes,list(reversed(roles)))
                        i=len(report['operations'])
                        if i%200==0:
                            guard('dispatch_'+str(i))
                        row=dict(id=i,marker=f'op_{i:05d}',repeat=state['rep'],step=4,operator=operator,
                            overload=str(func),shapes=shapes,dtypes=[str(t.dtype) for t in tensors],
                            strides=[list(t.stride()) for t in tensors],roles=list(reversed(roles)),category=category,family=family)
                        report['operations'].append(row)
                        with torch.cuda.nvtx.range(row['marker']):
                            return func(*args,**kwargs)
                def denoise(*args,**kwargs):
                    nonlocal active_profiler
                    index=state['step']
                    state['step']+=1
                    if index!=4:
                        return native(*args,**kwargs)
                    if 'prefix_kv' not in report:
                        report['prefix_kv']=kv_metadata(args[2])
                    torch.cuda.synchronize()
                    guard('selected_step_'+str(state['rep']))
                    log('NCU selected step, repeat '+str(state['rep']+1))
                    torch.cuda.profiler.start()
                    active_profiler=True
                    state['active']=True
                    try:
                        with torch.cuda.nvtx.range('cold_step'),scopes_for(model,lambda:state['active'],role_scope),Dispatch():
                            value=native(*args,**kwargs)
                        torch.cuda.synchronize()
                        return value
                    finally:
                        state['active']=False
                        torch.cuda.profiler.stop()
                        active_profiler=False
                with cache.applied():
                    if not torch.equal(call(),reference):
                        raise RuntimeError('Cache changed output')
                    log('20 complete warmups before selected-step cold-kernel collection')
                    for i in range(20):
                        guard('warmup_'+str(i))
                        call()
                    torch.cuda.synchronize()
                    model.denoise_step=denoise
                    try:
                        for rep in range(3):
                            state.update(rep=rep,step=0)
                            value=call()
                            torch.cuda.synchronize()
                            if state['step']!=10 or not torch.equal(value,reference) or cache.dense_hits!=370:
                                raise RuntimeError('Trajectory, actions or cache coverage changed')
                            save()
                            guard('repeat_finished_'+str(rep))
                    finally:
                        del model.denoise_step
                    report['cache_audit']=cache.metadata()
                report['module_counts']={str(rep):{role:count for (r,role),count in module_counts.items() if r==rep} for rep in range(3)}
                for counts in report['module_counts'].values():
                    for role in ('q_projection','k_projection','v_projection','o_projection','ffn_gate','ffn_up','ffn_down'):
                        if counts[role]!=18:
                            raise RuntimeError('Incomplete projection scopes')
                    if counts['rms_norm']!=37 or counts['cached_adarms']!=37:
                        raise RuntimeError('Incomplete AdaRMS scopes')
                report['output_bitwise_equal']=True
                report['reference_actions']=reference.cpu().tolist()
        guard('finished')
        report.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        log('CAPTURE_COMPLETE '+args.mode)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        save()
        raise
    finally:
        if active_profiler:
            torch.cuda.profiler.stop()
        (out/'exit.json').write_text(json.dumps(dict(status=report['status']))+'\n')


if __name__=='__main__':
    main()

"""Real fifth denoising step: separate attribution and CUDA-only timing passes."""
import collections
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from common import gpu_guard, stats


LINEARS = {'q_proj':'q_projection','k_proj':'k_projection','v_proj':'v_projection',
           'o_proj':'o_projection','gate_proj':'ffn_gate','up_proj':'ffn_up','down_proj':'ffn_down'}


def category(name, shapes, scopes):
    if scopes and scopes[0] in {*LINEARS.values(), 'action_in_proj','action_out_proj','cached_adarms'}:
        return scopes[0]
    if name in ('aten::_softmax','aten::softmax'):
        return 'softmax'
    if name == 'aten::gelu':
        return 'gelu'
    if name == 'aten::silu':
        return 'silu'
    if scopes and scopes[0] in ('rms_norm','rope_generation','rope_apply','residual_gate'):
        return scopes[0]
    if name == 'aten::bmm' and len(shapes) >= 2 and len(shapes[0]) == len(shapes[1]) == 3:
        if shapes[0][-1] == 256:
            return 'attention_qkt'
        if shapes[1][-1] == 256:
            return 'attention_sv'
        return 'other_bmm'
    if name in ('aten::copy_','aten::_to_copy','aten::clone','aten::contiguous'):
        return 'copy_or_cast'
    if scopes and scopes[0] == 'attention':
        if name == 'aten::cat':
            return 'kv_concat'
        return 'attention_'+name.removeprefix('aten::')
    if scopes and scopes[0] == 'ffn' and name == 'aten::mul':
        return 'ffn_gate_multiply'
    return 'setup_'+name.removeprefix('aten::')


@contextmanager
def scopes_for(model, active, enter):
    restores = []
    def wrap(obj, name, role):
        original = getattr(obj, name)
        had_local = name in obj.__dict__
        def wrapped(*args, **kwargs):
            if not active():
                return original(*args, **kwargs)
            with enter(role):
                return original(*args, **kwargs)
        restores.append((obj,name,original,had_local))
        setattr(obj,name,wrapped)
    try:
        expert = model.paligemma_with_expert.gemma_expert.model
        for name, sub in expert.named_modules():
            leaf = name.rsplit('.',1)[-1]
            if isinstance(sub,torch.nn.Linear):
                wrap(sub,'forward',LINEARS.get(leaf,'cached_adarms' if leaf=='dense' else 'other_linear'))
            elif leaf in ('input_layernorm','post_attention_layernorm','norm'):
                wrap(sub,'forward','rms_norm')
            elif leaf == 'self_attn':
                wrap(sub,'forward','attention')
            elif leaf == 'mlp':
                wrap(sub,'forward','ffn')
            elif leaf == 'rotary_emb':
                wrap(sub,'forward','rope_generation')
        for name in ('action_in_proj','action_out_proj'):
            wrap(getattr(model,name),'forward',name)
        from transformers.models.gemma import modeling_gemma
        wrap(modeling_gemma,'apply_rotary_pos_emb','rope_apply')
        wrap(modeling_gemma,'_gated_residual','residual_gate')
        yield
    finally:
        for obj,name,value,had_local in reversed(restores):
            if had_local:
                setattr(obj,name,value)
            else:
                delattr(obj,name)


def intervals_union(spans):
    merged = []
    for start,end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],end)
        else:
            merged.append([start,end])
    return merged


def summarize(items):
    by_op = collections.defaultdict(list)
    for item in items:
        by_op[(item['external_id'],item['category'],item['operator'])].append((item['start_ns'],item['end_ns']))
    inclusive = collections.Counter()
    operators = collections.Counter()
    bounds = collections.defaultdict(collections.Counter)
    for (op,cat,name),spans in by_op.items():
        for start,end in intervals_union(spans):
            inclusive[cat] += (end-start)/1e6
            operators[cat+'|'+name] += (end-start)/1e6
            bounds[start][cat] += 1
            bounds[end][cat] -= 1
    totals = collections.Counter()
    active = collections.Counter()
    previous = min(bounds)
    for stamp,changes in sorted(bounds.items()):
        cats = [k for k,v in active.items() if v>0]
        if cats:
            totals[cats[0] if len(cats)==1 else 'shared_overlap'] += (stamp-previous)/1e6
        active.update(changes)
        previous = stamp
    return dict(exclusive_ms=dict(totals), inclusive_ms=dict(inclusive), operator_ms=dict(operators),
                gpu_active_union_ms=sum(totals.values()), cuda_activity_count=len(items),
                streams=sorted({i['stream'] for i in items}),
                raw_kernel_sum_ms=sum(i['end_ns']-i['start_ns'] for i in items)/1e6)


def cuda_kernels(prof):
    return sorted([e for e in prof.profiler.kineto_results.events()
                   if str(e.device_type()).endswith('CUDA') and not e.is_user_annotation() and not e.is_hidden_event()],
                  key=lambda e:e.start_ns())


def label_trace(prof):
    events = prof.events()
    cpu = {e.id:e for e in events if str(e.device_type).endswith('CPU') and e.name.startswith('aten::')}
    marker_counts = collections.Counter(e.name[3:] for e in events if str(e.device_type).endswith('CPU') and e.name.startswith('op/'))
    items = []
    for kernel in cuda_kernels(prof):
        e = cpu.get(kernel.linked_correlation_id())
        if e is None:
            raise RuntimeError('Unattributed GPU activity: '+kernel.name())
        parent = e
        scopes = []
        while parent is not None:
            if parent.name.startswith('op/'):
                scopes.append(parent.name[3:])
            parent = getattr(parent,'cpu_parent',None)
        cat = category(e.name,e.input_shapes,scopes)
        items.append(dict(operator=e.name, category=cat, shapes=e.input_shapes, scopes=scopes,
                          kernel=kernel.name(), external_id=e.id, start_ns=kernel.start_ns(),
                          end_ns=kernel.end_ns(), stream=kernel.device_resource_id()))
    for role in LINEARS.values():
        if marker_counts[role] != 18:
            raise RuntimeError('Expected 18 calls per projection: '+str(marker_counts))
    if marker_counts['rms_norm'] != 37 or marker_counts['cached_adarms'] != 37:
        raise RuntimeError('Incomplete cached normalization coverage')
    if any(i['category'] == 'cached_adarms' for i in items):
        raise RuntimeError('Cached conditioning unexpectedly launched CUDA work')
    return items,dict(marker_counts)


def match_trace(prof, labels):
    kernels = cuda_kernels(prof)
    expected = labels['kernel_sequence']
    if len(kernels) != len(expected):
        raise RuntimeError(f'Kernel count changed: {len(kernels)} != {len(expected)}')
    items = []
    for index,(kernel,label) in enumerate(zip(kernels,expected,strict=True)):
        if kernel.name() != label['kernel']:
            raise RuntimeError(f'Kernel order/name changed at {index}: {kernel.name()} != {label["kernel"]}')
        items.append(dict(label,start_ns=kernel.start_ns(),end_ns=kernel.end_ns(),stream=kernel.device_resource_id()))
    return items


def kv_metadata(past):
    tensors = [t for layer in past for t in layer[:2]]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return dict(tensor_shapes=[list(t.shape) for t in tensors], dtype=str(tensors[0].dtype),
                logical_bytes=sum(t.numel()*t.element_size() for t in tensors),
                unique_storage_bytes=sum(storages.values()), unique_storages=len(storages))


def run_fine(model,cache,call,out,mode,labels_path,phase):
    if cache is None or model.config.action_horizon != 16:
        raise ValueError('This fine experiment requires cached pi0.5 H=16')
    if mode == 'micro':
        from micro import run_micro
        return run_micro(model,cache,call,out,phase)
    labels = json.loads(Path(labels_path).read_text()) if labels_path else None
    heads = model.paligemma_with_expert.gemma_expert.model.config.num_key_value_heads
    if mode == 'fine' and (labels is None or labels['horizon']!=16 or labels['kv_heads']!=heads):
        raise ValueError('Need labels matching H=16 and KV heads')
    state = dict(index=0,active=False,profiling=False,prof=None)
    native = model.denoise_step
    had_local = 'denoise_step' in model.__dict__
    kv = None
    def selected(*args,**kwargs):
        nonlocal kv
        index = state['index']
        state['index'] += 1
        if index != 4:
            return native(*args,**kwargs)
        if kv is None:
            kv = kv_metadata(args[2])
        if state['profiling']:
            torch.cuda.synchronize()
            state['prof'] = profile(activities=([ProfilerActivity.CPU,ProfilerActivity.CUDA] if mode=='labels' else [ProfilerActivity.CUDA]),
                                    record_shapes=mode=='labels',with_stack=False,profile_memory=False)
            state['prof'].__enter__()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        state['active'] = state['profiling'] and mode=='labels'
        start.record()
        try:
            value = native(*args,**kwargs)
            end.record()
            state['events'] = (start,end)
            return value
        finally:
            state['active'] = False
            if state['profiling']:
                torch.cuda.synchronize()
                state['prof'].__exit__(None,None,None)
    model.denoise_step = selected
    samples = []
    passes = {}
    reference = None
    try:
        with scopes_for(model,lambda:state['active'],lambda role:record_function('op/'+role)) if mode=='labels' else nullcontext():
            for name in ('clean_before','profiled','clean_after'):
                gpu_guard()
                state['profiling'] = False
                for _ in range(20):
                    state['index']=0
                    value=call(0)
                    torch.cuda.synchronize()
                if reference is None:
                    reference=value.clone()
                state['profiling'] = name=='profiled'
                count = (1 if mode=='labels' else 20) if state['profiling'] else 100
                phase(f'{mode} H=16 KVheads={heads}: {name}, n={count}')
                times=[]
                for i in range(count):
                    state['index']=0
                    value=call(0)
                    torch.cuda.synchronize()
                    if state['index']!=10 or not torch.equal(reference,value):
                        raise RuntimeError('Instrumentation changed output or trajectory length')
                    start,end=state['events']
                    elapsed=start.elapsed_time(end)
                    times.append(elapsed)
                    if state['profiling']:
                        if mode=='labels':
                            items,counts=label_trace(state['prof'])
                            label_data=dict(horizon=16,kv_heads=heads,module_calls=counts,
                                            kernel_sequence=[{k:v for k,v in item.items() if k not in ('start_ns','end_ns','stream')}
                                                             for item in items])
                            (out/'labels.json').write_text(json.dumps(label_data,indent=2)+'\n')
                        else:
                            items=match_trace(state['prof'],labels)
                        result=summarize(items)
                        result.update(step_interval_ms=elapsed,gaps_ms=elapsed-result['gpu_active_union_ms'])
                        samples.append(result)
                        if i==0:
                            (out/'first_sample_kernels.json').write_text(json.dumps(items,indent=2)+'\n')
                            state['prof'].export_chrome_trace(str(out/'trace.json'))
                        state['prof']=None
                passes[name]=stats(times)
                gpu_guard()
    finally:
        if had_local:
            model.denoise_step=native
        else:
            del model.denoise_step
    result=dict(horizon=16,steps=10,step_index=4,layers=18,kv_heads=heads,prefix_kv=kv,
                passes=passes,samples=samples,bitwise_equal=True,conditioning_linear_online_ms=0,
                profile_overhead_pct=100*(passes['profiled']['mean']/passes['clean_before']['mean']-1),
                post_profile_drift_pct=100*(passes['clean_after']['mean']/passes['clean_before']['mean']-1),
                boundary='Real fifth denoise_step, all 18 layers, input/output projections included, Euler excluded',
                interpretation='GPU kernel/activity times include memory access. Not DRAM-subtracted; profile perturbation is reported.')
    for field in ('exclusive_ms','inclusive_ms','operator_ms'):
        keys=sorted({k for s in samples for k in s[field]})
        result[field]={k:stats([s[field].get(k,0) for s in samples]) for k in keys}
    result['gpu_active_union_ms']=stats([s['gpu_active_union_ms'] for s in samples])
    return result

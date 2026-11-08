"""Capture actual fifth-step tensors and build complete, non-fused GPU blocks."""
from collections import Counter,defaultdict
from contextlib import contextmanager
import json
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from native_support import ROOT,CHECKPOINT,PINNED,sha
from common import load_weights
from timing import equal


def describe(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,(tuple,list)):return [describe(x) for x in value]
    if isinstance(value,dict):return {k:describe(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(type(value).__name__)


def prepare(log,guard):
    import transformers
    from transformers.modeling_utils import no_init_weights
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from adarms_cache import Pi05ConditioningCache
    from mha8 import convert
    hashes={k:sha(ROOT/k) for k in PINNED}
    if hashes!=PINNED or torch.__version__!='2.11.0+cu130' or transformers.__version__!='4.53.2':
        raise RuntimeError('Pinned runtime changed')
    config=Pi0Config(pi05=True,action_horizon=16,pytorch_compile_mode=None)
    log('Read existing checkpoint once via sequential readinto')
    with no_init_weights():model=PI0Pytorch(config)
    model.paligemma_with_expert.paligemma.tie_weights();model.eval()
    load=load_weights(model,CHECKPOINT,log)
    guard('after_load');model.to('cuda');torch.cuda.synchronize()
    conversion=convert(model)
    tokens=json.loads((ROOT/'code/pi_tokens.json').read_text())['pi05']
    gen=torch.Generator().manual_seed(17)
    images={key:(torch.rand((1,3,224,224),generator=gen)*2-1).cuda() for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
    obs=Observation(images=images,image_masks={k:torch.ones(1,device='cuda',dtype=torch.bool) for k in images},
        state=torch.zeros((1,32),device='cuda'),tokenized_prompt=torch.tensor([tokens['ids']],device='cuda'),
        tokenized_prompt_mask=torch.tensor([tokens['mask']],device='cuda',dtype=torch.bool))
    noise=torch.randn((1,16,32),generator=gen).cuda()
    def call():return model.sample_actions('cuda',obs,noise=noise,num_steps=10)
    reference=call().clone();torch.cuda.synchronize()
    cache=Pi05ConditioningCache(model,num_steps=10,batch_size=1)
    dims={name:dict(layers=len(tr.layers),D=tr.config.hidden_size,F=tr.config.intermediate_size,
        q_heads=tr.config.num_attention_heads,kv_heads=tr.config.num_key_value_heads,head_dim=tr.config.head_dim,
        attention=tr.config._attn_implementation)
        for name,tr in [('vlm',model.paligemma_with_expert.paligemma.language_model),('ae',model.paligemma_with_expert.gemma_expert.model)]}
    if any((d['layers'],d['q_heads'],d['kv_heads'],d['head_dim'],d['attention'])!=(18,8,8,256,'eager') for d in dims.values()):
        raise RuntimeError('Unexpected model geometry')
    return model,cache,call,reference,dict(configuration=dict(model='OpenPI pi0.5',H=16,steps=10,batch=1,q_heads=8,kv_heads=8,
        head_dim=256,adarms_cache=True,selected_step=4,synthetic_inputs=True,precision='selective BF16/FP32',
        physical_mha_scope='VLM and AE',cameras=3,prefix_tokens=968),load=load,model_source_sha256=hashes,
        mha_conversion=conversion,dimensions=dims,transformers=transformers.__version__)


def build_cases(model,cache,call,reference):
    from transformers.models.gemma import modeling_gemma as gm
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    records={};matmuls={};concats=defaultdict(list);residuals=defaultdict(list)
    state=dict(active=False,layer=None,step=0)
    frame={};restores=[]
    expert=model.paligemma_with_expert.gemma_expert.model
    def install(obj,name,fn):
        restores.append((obj,name,getattr(obj,name),name in obj.__dict__))
        setattr(obj,name,fn)
    def record_module(obj,key):
        native=obj.forward
        def forward(*args,**kwargs):
            value=native(*args,**kwargs)
            if state['active']:
                if key in records:raise RuntimeError('Duplicate record '+key)
                records[key]=dict(fn=native,args=args,kwargs=kwargs,out=value,module=obj)
            return value
        install(obj,'forward',forward)
    for i,layer in enumerate(expert.layers):
        native=layer.forward
        def wrapped(*args,native=native,index=i,**kwargs):
            old=state['layer']
            if state['active']:state['layer']=index
            try:return native(*args,**kwargs)
            finally:state['layer']=old
        install(layer,'forward',wrapped)
        for role,module in [('norm1',layer.input_layernorm),('norm2',layer.post_attention_layernorm),
            ('q',layer.self_attn.q_proj),('k',layer.self_attn.k_proj),('v',layer.self_attn.v_proj),('o',layer.self_attn.o_proj),
            ('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('down',layer.mlp.down_proj)]:
            record_module(module,f'{i}/{role}')
    record_module(expert.norm,'final_norm')
    record_module(expert.rotary_emb,'rope_generation')
    record_module(model.action_in_proj,'action_in')
    record_module(model.action_out_proj,'action_out')
    native_rope=gm.apply_rotary_pos_emb
    native_residual=gm._gated_residual
    native_attention=gm.eager_attention_forward
    def rope(*args,**kwargs):
        value=native_rope(*args,**kwargs)
        if state['active']:records[f'{state["layer"]}/rope']=dict(fn=native_rope,args=args,kwargs=kwargs,out=value)
        return value
    def residual(*args,**kwargs):
        value=native_residual(*args,**kwargs)
        if state['active']:residuals[state['layer']].append(dict(args=args,kwargs=kwargs,out=value))
        return value
    def attention(*args,**kwargs):
        value=native_attention(*args,**kwargs)
        if state['active']:records[f'{state["layer"]}/attention']=dict(args=args,kwargs=kwargs,out=value)
        return value
    install(gm,'apply_rotary_pos_emb',rope)
    install(gm,'_gated_residual',residual)
    install(gm,'eager_attention_forward',attention)
    class Dispatch(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            value=func(*args,**kwargs)
            i=state['layer']
            if i is not None:
                name=func._schema.name
                if name in ('aten::matmul','aten::bmm') and len(args)>=2 and args[0].ndim==4 and args[1].ndim==4:
                    role='qkt' if args[0].shape[-1]==256 else 'sv'
                    key=f'{i}/{role}'
                    if key in matmuls:raise RuntimeError('Duplicate attention matmul')
                    matmuls[key]=dict(fn=func,args=args,kwargs=kwargs,out=value)
                if name=='aten::cat':
                    tensors=args[0];dim=kwargs.get('dim',args[1] if len(args)>1 else 0)
                    if len(tensors)==2 and tensors[0].ndim==4 and dim%4==2:
                        concats[i].append(dict(fn=func,args=args,kwargs=kwargs,out=value))
            return value
    native_step=model.denoise_step
    def step(*args,**kwargs):
        i=state['step'];state['step']+=1
        if i==5:frame['next_x']=args[3]
        if i!=4:return native_step(*args,**kwargs)
        frame['args']=args;frame['kwargs']=kwargs
        state['active']=True
        try:
            with Dispatch():value=native_step(*args,**kwargs)
            frame['out']=value
            return value
        finally:state['active']=False
    install(model,'denoise_step',step)
    try:
        value=call();torch.cuda.synchronize()
        if not equal(value,reference) or state['step']!=10:raise RuntimeError('Capture changed full inference')
    finally:
        for obj,name,native,had in reversed(restores):
            if had:setattr(obj,name,native)
            else:delattr(obj,name)
    cases=[]
    def add(family,fn,expected,inputs,layer=None,suite='native_blocks',note=''):
        cases.append(dict(id=f'{suite}/{family}/{layer if layer is not None else "step"}',family=family,fn=fn,
            expected=expected,layer=layer,suite=suite,inputs=describe(inputs),outputs=describe(expected),note=note))
    def rec_case(family,r,layer=None,suite='native_blocks'):
        add(family,lambda r=r:r['fn'](*r['args'],**r['kwargs']),r['out'],(r['args'],r['kwargs']),layer,suite)
    def norm_call(r,x):return r['fn'](x,*r['args'][1:],**r['kwargs'])
    def pack_sv(x):return x.transpose(1,2).contiguous().reshape(1,16,2048).contiguous()
    rec_case('action_in',records['action_in'])
    action=records['action_in']['out']
    first_norm=records['0/norm1']['args'][0]
    add('action_embedding_prepare',lambda:torch.cat([action],dim=1).to(first_norm.dtype),first_norm,[action])
    rec_case('rope_generation',records['rope_generation'])
    suffix_pad=torch.ones((1,16),device='cuda',dtype=torch.bool)
    suffix_att=torch.tensor([[1]+[0]*15],device='cuda',dtype=action.dtype)
    prefix_pad=frame['args'][1]
    def masks():
        tail=make_att_2d_masks(suffix_pad,suffix_att)
        full=torch.cat((prefix_pad[:,None,:].expand(1,16,prefix_pad.shape[1]),tail),dim=2)
        positions=prefix_pad.sum(dim=-1)[:,None]+torch.cumsum(suffix_pad,dim=1)-1
        return model._prepare_attention_masks_4d(full),positions
    mask_ref=records['0/attention']['args'][4]
    pos_ref=records['rope_generation']['args'][1]
    add('mask_position_setup',masks,(mask_ref,pos_ref),[prefix_pad,suffix_pad,suffix_att],
        note='Static suffix-pad/attention constants preallocated; CPU/H2D creation excluded.')
    for i,layer in enumerate(expert.layers):
        r={k:records[f'{i}/{k}'] for k in ('norm1','norm2','q','k','v','o','gate','up','down','rope','attention')}
        if len(residuals[i])!=2 or len(concats[i])!=2:raise RuntimeError('Incomplete layer boundary coverage')
        ares,fres=residuals[i]
        qkt,sv=matmuls[f'{i}/qkt'],matmuls[f'{i}/sv']
        next_norm=records[f'{i+1}/norm1'] if i<17 else records['final_norm']
        rec_case('norm_input',r['norm1'],i)
        def qkv(r=r):return tuple(r[k]['fn'](*r[k]['args'],**r[k]['kwargs']) for k in ('q','k','v'))
        add('qkv_projection',qkv,tuple(r[k]['out'] for k in ('q','k','v')),[r['q']['args'][0]],i,
            note='Three native linears; shared input can stay cached between them. No projection fusion.')
        rec_case('rope_qk',r['rope'],i)
        cats=concats[i]
        add('kv_concat_pair',lambda cats=cats:tuple(c['fn'](*c['args'],**c['kwargs']) for c in cats),tuple(c['out'] for c in cats),[c['args'] for c in cats],i)
        rec_case('qkt',qkt,i)
        raw=qkt['out'];mask=r['attention']['args'][4];q=r['attention']['args'][1]
        scaling=r['attention']['kwargs']['scaling']
        def probability(raw=raw,mask=mask,scaling=scaling,dtype=q.dtype):
            score=raw*scaling
            if mask is not None:score=score+mask[:,:,:,:raw.shape[-1]]
            return F.softmax(score,dim=-1,dtype=torch.float32).to(dtype)
        add('scale_mask_softmax_cast',probability,sv['args'][0],[raw,mask],i)
        rec_case('sv',sv,i)
        add('sv_output_pack',lambda v=sv['out']:pack_sv(v),r['o']['args'][0],[sv['out']],i)
        rec_case('o_projection',r['o'],i)
        add('residual_attention',lambda ares=ares:native_residual(*ares['args'],**ares['kwargs']),ares['out'],ares['args'],i)
        rec_case('norm_ffn',r['norm2'],i)
        def gateup(r=r):return tuple(r[k]['fn'](*r[k]['args'],**r[k]['kwargs']) for k in ('gate','up'))
        add('ffn_gate_up',gateup,(r['gate']['out'],r['up']['out']),r['gate']['args'],i,
            note='Independent gate/up projections grouped before activation; no kernel fusion.')
        gate,up,act=r['gate']['out'],r['up']['out'],layer.mlp.act_fn
        add('gelu_multiply',lambda gate=gate,up=up,act=act:act(gate)*up,r['down']['args'][0],[gate,up],i)
        rec_case('ffn_down',r['down'],i)
        add('residual_ffn',lambda fres=fres:native_residual(*fres['args'],**fres['kwargs']),fres['out'],fres['args'],i)
        add('post_attention_residual_norm',lambda ares=ares,n=r['norm2']:norm_call(n,native_residual(*ares['args'],**ares['kwargs'])),
            r['norm2']['out'],[ares['args'],r['norm2']['args'][1:]],i,'handoff_composites',
            'Use when O projection is on PIM. Replaces residual_attention + norm_ffn; do not add both.')
        add('post_ffn_residual_nextnorm',lambda fres=fres,n=next_norm:norm_call(n,native_residual(*fres['args'],**fres['kwargs'])),
            next_norm['out'],[fres['args'],next_norm['args'][1:]],i,'handoff_composites',
            'Replaces residual_ffn + next input norm (or final norm in layer17).')
        def gpu_o_exit(v=sv['out'],op=r['o'],res=ares,n=r['norm2']):
            o=op['fn'](pack_sv(v),**op['kwargs'])
            out=native_residual(res['args'][0],o,res['args'][2])
            return norm_call(n,out)
        add('sv_pack_o_residual_norm',gpu_o_exit,r['norm2']['out'],[sv['out'],ares['args'][0],ares['args'][2]],i,'handoff_composites',
            'Use when O projection stays on GPU. Overlaps pack/O/residual/norm component rows.')
        g32,u32=gate.float(),up.float()
        def pim_ffn(g=g32,u=u32,act=act):return (act(g)*u).to(torch.bfloat16)
        add('pim_gelu_multiply_bf16',pim_ffn,pim_ffn(),[g32,u32],i,'pim_interface_candidates',
            'Logical unpacked FP32 Gate/Up buffers -> GELU/multiply -> BF16 Z. Synthetic values, physical bank packing excluded.')
        q0,k0,cos,sin=r['rope']['args'][:4]
        q32,k32=q0.float(),k0.float()
        def pim_rope(q=q32,k=k32,cos=cos,sin=sin):
            a,b=native_rope(q,k,cos,sin)
            return a.to(torch.bfloat16),b.to(torch.bfloat16)
        add('pim_rope_fp32_bf16',pim_rope,pim_rope(),[q32,k32,cos,sin],i,'pim_interface_candidates',
            'Alternative GPU-RoPE partition only; do NOT charge when RoPE runs on PIM. Logical buffers, no bank unpacking.')
        scores32=raw.float()
        def pim_probability(scores=scores32,mask=mask,scaling=scaling):
            score=scores*scaling
            if mask is not None:score=score+mask[:,:,:,:scores.shape[-1]]
            return F.softmax(score,dim=-1,dtype=torch.float32).to(torch.bfloat16)
        add('pim_scale_mask_softmax_bf16',pim_probability,pim_probability(),[scores32,mask],i,'pim_interface_candidates',
            'Logical FP32 QK scores -> scaling/mask/softmax -> BF16 P. P rank/token packing excluded.')
    rec_case('norm_final',records['final_norm'])
    last=records['final_norm']['out'][0]
    add('action_output_cast',lambda:last.to(torch.float32),records['action_out']['args'][0],[last])
    rec_case('action_out',records['action_out'])
    xt,velocity=frame['args'][3],frame['out']
    dt=torch.tensor(-1/10,dtype=torch.float32,device='cuda')
    add('euler_update',lambda:xt+dt*velocity,frame['next_x'],[xt,dt,velocity])
    audit=dict(full_output_bitwise_equal=True,selected_step=4,layers=18,cases=len(cases),
        counts=dict(Counter(c['suite']+'/'+c['family'] for c in cases)),
        native_blocks='Logical non-overlapping math blocks; summed isolated intervals are NOT full-model elapsed latency.',
        composites='Alternatives overlapping native blocks; never add both tables.',
        pim_candidates='FP32 logical tensors derived from native BF16 outputs; validate GPU implementation only, not numerical PIM hardware.',
        host_setup_excluded='Static suffix mask constants are preallocated; host/H2D setup and external coherency/transfer remain outside segment timing.')
    return cases,audit

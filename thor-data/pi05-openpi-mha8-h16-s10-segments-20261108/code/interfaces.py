"""Extra logical FP32 handoff blocks grounded in the token-parity PIM contract."""
import inspect
import torch
import torch.nn.functional as F
from builder import build_cases,describe
from timing import equal


def defaults(fn):
    return {name:p.default for name,p in inspect.signature(fn).parameters.items()}


def build_interfaces(model,cache,call,reference):
    originals,_=build_cases(model,cache,call,reference)
    cases=[]
    cache.active=True
    try:
        for c in originals:
            family=c['family'];fn=c['fn'];layer=c['layer']
            if family in ('post_attention_residual_norm','post_ffn_residual_nextnorm'):
                field='ares' if family=='post_attention_residual_norm' else 'fres'
                r=defaults(fn)[field]
                x,y,gate=r['args'];y32=y.float()
                name='pim_o_fp32_cast_resnorm' if field=='ares' else 'pim_down_fp32_cast_resnorm'
                def perform(fn=fn,r=r,field=field,x=x,y=y,y32=y32,gate=gate):
                    replacement=dict(r,args=(x,y32.to(y.dtype),gate))
                    return fn(**{field:replacement})
                inputs=[x,y32,gate]
                expected=c['expected']
                note='Logical FP32 PIM output -> native BF16 rounding -> gated residual + following cached AdaRMS. Includes cast; physical transport/visibility excluded.'
            elif family=='sv_pack_o_residual_norm':
                v=defaults(fn)['v']
                a=(v.float()*.5).contiguous();b=a.clone()
                def perform(fn=fn,a=a,b=b,dtype=v.dtype):return fn(v=(a+b).to(dtype))
                name='pim_pv_sum_cast_pack_o_resnorm'
                inputs=[a,b]
                expected=c['expected']
                note='Two distinct FP32 partial buffers, constructed as half of the native SV value, sum -> BF16 -> pack/O/residual/norm. Not a numerical PIM simulation.'
            elif family=='pim_scale_mask_softmax_bf16':
                def perform(fn=fn):
                    p=F.pad(fn(),(0,8))
                    return p[...,0::2].contiguous(),p[...,1::2].contiguous()
                name='pim_softmax_bf16_parity_pack'
                inputs=c['inputs']
                expected=perform()
                p=fn();rebuilt=torch.empty((*p.shape[:-1],992),device=p.device,dtype=p.dtype)
                rebuilt[...,0::2]=expected[0];rebuilt[...,1::2]=expected[1]
                if not equal(rebuilt[...,:984],p) or not bool((rebuilt[...,984:]==0).all()):
                    raise RuntimeError('Parity packing failed round-trip')
                note='FP32 scores -> scale/mask/softmax -> BF16 -> 8 zero padding columns -> even/odd KV shards. Each head/rank output [16,496]. Logical packing only, no external DMA.'
            else:
                continue
            value=perform();torch.cuda.synchronize()
            if not equal(value,expected):raise RuntimeError('Interface differs from native/declared reference: '+name)
            cases.append(dict(id=f'pim_interface_candidates/{name}/{layer}',suite='pim_interface_candidates',family=name,layer=layer,
                fn=perform,expected=expected,inputs=inputs if family=='pim_scale_mask_softmax_bf16' else describe(inputs),
                outputs=describe(expected),note=note))
    finally:cache.active=False
    if len(cases)!=72:raise RuntimeError('Expected four interface cases for each of 18 layers')
    return cases

"""1108 controlled layout/RoPE/input-command matrix, with explicit DQ payloads."""
from collections import Counter
from pathlib import Path
import tempfile
import workload
import coarse_workload
import horizon_attention_layout as L
from horizon_attention_workload import Writer

VARIANTS = {
    'feature_gpu_before': dict(feature=True,gpu_rope=True,after=False),
    'token_gpu_before': dict(feature=False,gpu_rope=True,after=False),
    'token_local_before': dict(feature=False,gpu_rope=False,after=False),
    'token_gpu_after': dict(feature=False,gpu_rope=True,after=True),
    'token_local_after': dict(feature=False,gpu_rope=False,after=True),
}

def addr(ch,bank,offset):
    return L.physical_address(ch,1,bank//4,bank%4,offset)

def gpu_rope_records(op):
    return [f'{op} {addr(ch,b,81920+column*32)}'
            for column in range(8,24) for b in range(16) for ch in range(16)]

def finish(path,case,segments,records,variant,**extra):
    Path(path).write_text('\n'.join(records)+'\n',encoding='ascii')
    counts=Counter(r.split()[0] for r in records)
    dq={name:counts[op]*32 for name,op in (
        ('dram_read','LD'),('dram_write','ST'),('pim_input_write','PIM_WR_GB'),
        ('pim_result_read','PIM_RD_RES'),('score_export','PIM_RD_BG'),
        ('q_export','PIM_Q_EXPORT'),('q_remote','PIM_Q_REMOTE'))}
    if VARIANTS[variant]['gpu_rope']:dq['q_export']=0 # GPU already holds Q; readiness token carries no DQ.
    dq['total']=sum(dq.values())
    return dict(case=case,variant=variant,segments=segments,counts=dict(counts),
        expected_dq_bytes=dq,configuration=VARIANTS[variant],numerical_compute=False,
        dimensions=dict(H=16,heads=8,head_dim=256,hidden=1024,FFN=4096,valid_KV=984,padded_KV=992),
        **extra)

def token(path,case,variant):
    v=VARIANTS[variant]
    opts={**coarse_workload.OPTIONS,'rank_descriptors':False,
          'score_dram_spill':v['after']}
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)/'source.trace'
        old=coarse_workload.generate(tmp,case) if v['after'] else workload.generate(tmp,case,**opts)
        raw=tmp.read_text().splitlines()
    out=['BARRIER 0'];segments=[]
    def emit(name,items,kind):
        first=len(out)+1;i=len(segments)
        out.extend(items);out.append(f'BARRIER {i+1}')
        segments.append(dict(name=name,start=i,end=i+1,line_start=first,line_end=len(out),kind=kind))
    for s in old['segments']:
        name=s['name'];short=name.split('/')[-1]
        items=[r for r in raw[s['line_start']-1:s['line_end']] if not r.startswith('BARRIER')]
        if not v['after'] and s.get('kind')=='input_staging':continue
        if v['gpu_rope'] and short in ('rope_local','rope_run'):
            emit('gpu_rope_export',gpu_rope_records('LD'),'external_read')
            emit('gpu_rope_import',gpu_rope_records('ST'),'external_write')
            continue
        if not v['after'] and not short.startswith('o_projection'):
            rewritten=[]
            for r in items:
                f=r.split()
                if f[0]=='PIM_STAGE_FRAGMENT':
                    ch=L.decode_address(int(f[1],0))[0]
                    rows=8 if short.startswith('projection') else 16
                    rewritten.extend(f'PIM_WR_GB {addr(ch,0,q*32)} {f[2]}' for q in range(rows))
                else:rewritten.append(r)
            items=rewritten
        emit('gpu_q_ready' if v['gpu_rope'] and short=='q_export_to_gpu' else name,
             items,'producer_ready' if v['gpu_rope'] and short=='q_export_to_gpu' else s.get('kind','coarse_work'))
    return finish(path,case,segments,out,variant,
        source_generator='coarse_workload' if v['after'] else 'workload with direct external inputs',
        coarse_commands=v['after'],feature_geometry=None)

def feature(path,case,variant):
    if case not in ('chain','pv','gate_up_gpu','down_gpu_input','serial_layer_control'):
        raise ValueError(case)
    if case in ('gate_up_gpu','down_gpu_input'):
        m=token(path,case,'token_gpu_before');m['variant']=variant;m['configuration']=VARIANTS[variant];return m
    if case=='serial_layer_control':
        out=['BARRIER 0'];segments=[]
        for part in ('chain','pv','gate_up_gpu','down_gpu_input'):
            with tempfile.TemporaryDirectory() as td:
                tmp=Path(td)/'p.trace';m=feature(tmp,part,variant);raw=tmp.read_text().splitlines()
            for s in m['segments']:
                i=len(segments);first=len(out)+1
                out.extend(r for r in raw[s['line_start']-1:s['line_end']] if not r.startswith('BARRIER'))
                out.append(f'BARRIER {i+1}')
                segments.append(dict(name=part+'/'+s['name'],kind=s['kind'],start=i,end=i+1,line_start=first,line_end=len(out)))
        return finish(path,case,segments,out,variant)
    w=Writer(path,rank_descriptors=False)
    def allbanks(op,offset):w.banks(op,offset)
    def release(views=2):
        for s in range(views):w.ranks('PIM_GEMV_RELEASE',0,s)
    def gemm(base,fragments,columns):
        for first in range(0,fragments,2):
            for f in range(first,min(first+2,fragments)):
                for q in range(16):w.ranks('PIM_WR_GB',q*32,f%2)
            for f in range(first,min(first+2,fragments)):
                for col in range(columns):w.ranks('PIM_MAC_AB',base+(f*columns+col)*32,f%2)
    if case=='chain':
        w.phase('projection_input_compute',lambda:gemm(393216,64,24),kind='projection')
        def spills():
            for column in range(8,24):allbanks('PIM_SPILL',81920+column*32)
            for packed in range(16):allbanks('PIM_SPILL',163840+30*512+packed*32)
        w.phase('projection_qkv_local_spill',spills,kind='local_write')
        w.phase('projection_gemv_release',release,kind='gemv_release')
        for op,name in (('LD','gpu_rope_export'),('ST','gpu_rope_import')):
            def transfer(op=op):
                for r in gpu_rope_records(op):
                    f=r.split();w.emit(f[0],int(f[1]))
            w.phase(name,transfer,kind='external_read' if op=='LD' else 'external_write')
        for tile in range(62):
            def qk(tile=tile):
                for t in range(tile*16,(tile+1)*16):
                    off=65536+(t//2)*32 if t<968 else 81920+(16+(t-968)//2)*32 if t<984 else 81920+(24+(t-984)//2)*32
                    w.ranks('PIM_MAC_QK',off,0)
            w.phase(f'qk_tile_{tile}_compute',qk,kind='qk')
            def export(tile=tile):
                for packet in range(32):w.ranks('PIM_RD_BG',(tile*32+packet)*32)
            w.phase(f'qk_tile_{tile}_export',export,kind='external_read')
        w.phase('qk_gemv_release',lambda:release(1),kind='gemv_release')
    else:
        w.phase('pv_input_compute',lambda:gemm(163840,62,8),kind='pipeline')
        w.phase('pv_gemv_release',release,kind='gemv_release')
        from rank_staging import input_offset,source_bank
        def spill():
            w.ranks('PIM_STAGE_BEGIN',524288,'4096 16')
            for pair in range(0,8,2):
                for q in range(16):
                    for f in (pair,pair+1):w.ranks('PIM_PV_SPILL',input_offset(f,q,16,2),bank=source_bank(f,2))
            w.ranks('PIM_STAGE_SEAL',524288)
        w.phase('pv_result_dram_spill',spill,kind='local_write')
        for tile in range(2):
            if tile:w.phase('o_projection_input_rewind',lambda:w.ranks('PIM_O_REWIND',524288),kind='gemv_release')
            def compute(tile=tile):
                for first in range(0,8,2):
                    for f in (first,first+1):w.ranks('PIM_STAGE_FRAGMENT',input_offset(f,0,16,2),f%2,bank=source_bank(f,2))
                    for f in (first,first+1):
                        for col in range(32):w.ranks('PIM_MAC_AB',786432+tile*8192+(f*32+col)*32,f%2)
            w.phase(f'o_projection_tile_{tile}_compute',compute,kind='local_compute')
            w.phase(f'o_projection_tile_{tile}_release',release,kind='gemv_release')
            def export():
                for off in range(0,16384,32):w.ranks('PIM_RD_RES',off)
            w.phase(f'o_projection_tile_{tile}_export',export,kind='external_read')
    w.close()
    raw=Path(path).read_text().splitlines()
    return finish(path,case,w.segments,raw,variant,coarse_commands=False,
        feature_geometry=dict(rank_qkv=[16,128],rank_kv=[992,128],bank_kv=[992,8],rank_o_input=[16,128],
            QK_two_tokens_per_32B=True,QK_useful_lanes_per_dot=8,QK_two_dot_packing=False,
            rank_feature_subset='r0=[0:64,128:192], r1=[64:128,192:256]',
            feature_bank='bank=(feature_in_rank//8); RoPE computed by GPU',
            provenance='outputs/attention-full-comparison/comparison.json'))

def generate(path,case,variant):
    return feature(path,case,variant) if VARIANTS[variant]['feature'] else token(path,case,variant)

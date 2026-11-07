"""Compile the current fixed layout into RUN descriptors and direct DQ requests.

Legacy manifests supply only external data addresses. No expanded compute trace
is passed to the controller: it derives each internal request from the 8 words.
STAGE_BEGIN/SEAL and BARRIER are simulator producer/dependency metadata, not CA.
"""
from collections import Counter
from pathlib import Path
import tempfile
import workload
from simulator.coarse import example,line

OPTIONS=dict(rank_staging=True,staging_banks=2,rank_descriptors=True,
    channel_independent_dispatch=True,v_token_sharding=True,pv_token_sharding=True,
    qkv_fusion=True,score_dram_spill=True,o_projection=True,pv_bf16_output=True)


def generate(path,case):
    path=Path(path)
    with tempfile.TemporaryDirectory() as td:
        source=Path(td)/'legacy.trace'
        legacy=workload.generate(source,case,**OPTIONS)
        raw=source.read_text().splitlines()
    out=['BARRIER 0']; segments=[]; marker=0; event=0
    def emit(name,records):
        nonlocal marker
        start=len(out)+1
        out.extend(records);out.append(f'BARRIER {marker+1}')
        segments.append(dict(name=name,start=marker,end=marker+1,line_start=start,line_end=len(out)))
        marker+=1
    def run(name):
        nonlocal event
        event+=1
        return [line(0x8000+(ch<<5),example(name,done=event)) for ch in range(16)]
    def records(seg):
        return [r for r in raw[seg['line_start']-1:seg['line_end']] if not r.startswith('BARRIER')]
    def direct(seg):
        result=[]
        for r in records(seg):
            f=r.split()
            if f[0].startswith('PIM_RANK_'):
                op='PIM_'+f[0][9:];addr=int(f[1],0)
                for bank in range(16):
                    result.append(' '.join([op,hex(addr|((bank//4)<<16)|((bank%4)<<18)),*f[2:]]))
            else: result.append(r)
        return result
    allseg=legacy['segments']; seen=set()
    for s in allseg:
        name=s['name']; short=name.split('/')[-1]
        if s.get('kind')=='input_staging': emit(name,direct(s))
        elif short=='projection_local_input_compute': emit('qkv_run_local_spills',run('qkv'))
        elif short=='rope_local': emit('rope_run',run('rope'))
        elif short in ('q_export_to_gpu','q_remote_buffer_fill','qk_scores_dram_export'):
            emit(name,direct(s))
        elif short=='qk_tile_0_compute': emit('qkt_run_all_31_tiles',run('qkt'))
        elif short=='pv_local_input_compute': emit('pv_run_local_spill',run('pv'))
        elif short=='o_projection_tile_0_compute':
            exports=[r for x in allseg if 'o_projection_tile_' in x['name'] and x['name'].endswith('_export') for r in direct(x)]
            emit('o_run_two_tiles_with_direct_export',run('o')+exports)
        elif short in ('gate_up_local_input_and_compute','down_local_input_and_compute'):
            op='gate_up' if short.startswith('gate_up') else 'down'
            if op in seen: raise ValueError('Duplicate operator')
            seen.add(op)
            exports=[r for x in allseg if x['name'].split('/')[-1]==op+'_result_export' for r in direct(x)]
            emit(op+'_run_with_direct_export',run(op)+exports)
    path.write_text('\n'.join(out)+'\n',encoding='ascii')
    return dict(case=case,segments=segments,counts=dict(Counter(x.split()[0] for x in out)),
        expected_dq_bytes=legacy['expected_dq_bytes'],buffers=legacy['buffers'],
        coarse_commands=True,descriptor_bits=256,wire_bits=294,ca_cycles=21,decode_cycles=1,
        local_rope=True,fixed_layout=workload.layouts(**{k:v for k,v in OPTIONS.items() if k not in ('rank_descriptors','channel_independent_dispatch')}),
        numerical_compute=False,complete_ae_latency=False)

"""Independent before/after coverage, bounded storage, wire and native timing audits."""
from collections import Counter
from pathlib import Path
import audit_native as A
from dispatch_audit import expand_rank_records
from act2_audit import audit_act2
from audit_o_projection import (audit_stages,audit_score_lifecycle,audit_o_lifecycle,
    audit_new_local_write_precharge,audit_o_packet_scratch,audit_staging_generation_reuse)

TYPES={name:i for i,name in enumerate(('LD','ST','PIM_MAC_AB','PIM_WR_GB','PIM_RD_RES',
    'PIM_MAC_HB','PIM_LOCAL_GB','PIM_WR_BG','PIM_WR_Q','PIM_MAC_K','PIM_MAC_QK',
    'PIM_RD_BG','PIM_LOCAL_WR','PIM_RD_Q','PIM_SPILL','PIM_ROPE_COEF','PIM_ROPE_LOAD',
    'PIM_ROPE_EXEC','PIM_ROPE_STORE','PIM_Q_EXPORT','PIM_Q_REMOTE','PIM_Z_STORE',
    'PIM_Z_LOAD','PIM_GEMV_RELEASE','PIM_STAGE_BEGIN','PIM_STAGE_SEAL','PIM_STAGE_LOAD',
    'PIM_STAGE_FRAGMENT','PIM_SCORE_SPILL','PIM_PV_SPILL','PIM_O_REWIND'))}
INTERNAL={2,10,14,15,16,17,18,23,26,27,28,29,30}
DATA={'MACAB','MACQK','LOCALRD','LOCALWR','RD','RDA','WR','WRA','RDRES','WRGB'}


def audit_pair(before,after):
    before,after=Path(before),Path(after);sim=after/'simulation'
    result=A.read_json(sim/'result.json');manifest=A.read_json(after/'manifest.json')
    def collect(name,optional=False):
        paths=[sim/f'{name}.ch{ch}' for ch in range(16)]
        if optional and not paths[0].exists(): return []
        return [e for p in paths for e in A.read_csv(p)]
    native=collect('timeline.csv');gemv=collect('unified_gemv.csv');stage=collect('staging_events.csv')
    runs=collect('coarse_runs.csv');ca=collect('rank_ca_events.csv')
    prior=[e for ch in range(16) for e in A.read_csv(before/f'simulation/timeline.csv.ch{ch}')]
    def key(e):return ({'RDA':'RD','WRA':'WR'}.get(e['command'],e['command']),*(e[x] for x in A.GEOMETRY))
    lhs=Counter(key(e) for e in prior if e['command'] in DATA)
    rhs=Counter(key(e) for e in native if e['command'] in DATA)
    A.require(lhs==rhs,'Before/after data address coverage differs: '+A.counter_difference(lhs,rhs))
    for ch in range(16):
        for command in ('MACAB','MACQK'):
            A.require([key(e) for e in prior if e['Channel']==ch and e['command']==command]==
                      [key(e) for e in native if e['Channel']==ch and e['command']==command], 'MAC reduction order differs')
    old=expand_rank_records(A.trace_records(before/'workload.trace'))
    expected=Counter()
    for e in old:
        t=TYPES.get(e['operation'],-1)
        if t not in INTERNAL:continue
        ch,rank,bg,b,row,col=A.decode(e['address'])
        expected[t,ch,4*bg+b,row*2048+col*32,e['buffer']]+=1
    def child_key(e):return tuple(e[k] for k in ('type','Channel','bank','offset','buffer'))
    actual=Counter(child_key(e) for e in runs if e['event']=='CHILD_ENQUEUE' and e['type'] in INTERNAL)
    A.require(actual==expected,'Descriptor-generated micro addresses differ: '+A.counter_difference(expected,actual))
    queued=Counter(child_key(e) for e in runs if e['event']=='CHILD_ENQUEUE')
    # A staged fragment advances its native cursor over every token row; its
    # completion address is the final row, not the submitted first row.
    completed=Counter(child_key(dict(e,offset=e['offset']-(7 if e['B_base']==393216 else 15)*32))
        if e['type']==27 else child_key(e) for e in runs if e['event']=='CHILD_COMPLETE')
    A.require(queued==completed,'Uncompleted internal requests')
    accepts=Counter((e['Channel'],e['run_id']) for e in runs if e['event']=='ACCEPT')
    finishes=Counter((e['Channel'],e['run_id']) for e in runs if e['event']=='COMPLETE')
    A.require(accepts==finishes and all(n==1 for n in accepts.values()),'RUN completion ownership mismatch')
    wire=[e for e in ca if e['scope']=='global' and e['command'] in ('MAC_RUN','ROPE_RUN')]
    A.require(len(wire)==len(accepts) and all(e['ready_at']-e['clock']==21 for e in wire),'RUN must occupy exactly 21 CA CK')
    allowed=(DATA-{'MACAB','MACQK'})|{'MAC','ACT1','ACT2','ACTAB','PRE','PREpb','PREab','REFab','CASRD','CASWR','CAS','ACT','REFpb'}
    A.require(all(e['command'] in allowed|{'MAC_RUN','ROPE_RUN'} for e in ca),'Local control leaked onto CA')
    context={(e['Channel'],e['clock']):e['Row'] for e in native if e['command']=='MACAB'}
    report=dict(status='passed',data_commands_checked=sum(rhs.values()),descriptor_children_checked=sum(actual.values()),
        descriptor_count=len(accepts),descriptor_global_CA_CK=sum(e['ready_at']-e['clock'] for e in wire),
        exact_data_address_multiset=True,exact_MAC_order=True,
        array_open_row_checks=A.audit_array_rows(native),split_activation=audit_act2(native,result['resolved_timings_ck'],staging_events=stage),
        shared_SRF=A.audit_slot_events(gemv,manifest['case'],mac_context=context,v_token_sharding=True,
            pv_token_sharding=True,qkv_fusion=True,whole_srf=True),
        local_write_recovery=audit_new_local_write_precharge(native,result['resolved_timings_ck']))
    op=collect('o_projection.csv',True);score=collect('score_spill.csv',True)
    report['staging']=audit_stages(stage,op,native,gemv,result)
    report['packet_scratch']=audit_o_packet_scratch(op,result['resolved_timings_ck'])
    report['storage_reuse']=audit_staging_generation_reuse(stage,op)
    if score:report['score']=audit_score_lifecycle(score,native,result)
    if op:report['output']=audit_o_lifecycle(op,native,gemv,result,result['config'].get('pim_o_projection',False))
    return report

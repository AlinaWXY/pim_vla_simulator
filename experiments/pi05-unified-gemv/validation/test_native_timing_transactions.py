"""Differential checks against the resolved native LPDDR5XPIM spec, not a copied table."""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
sys.path.insert(0,str(SRC.parent/'pi05-bgk-gpuq/backend/ramulator2/python'))
from online_transactions import Job,CK
from sequential_transactions import SequentialEngine
from ramulator.dram.lpddr5xpim import LPDDR5XPIM


def native_spec():
    return LPDDR5XPIM(org_preset='LPDDR5X_8Gb_x16',timing_preset='LPDDR5X_8533',
                     rank=2,pim_data_path='attacc',nCCDQK=16)


def native_gap(spec,a,b,relation):
    config=spec.to_config();ids={c:i for i,c in enumerate(spec.commands)}
    same_rank=relation!='different_rank'
    same_bg=relation in ('same_bg','broadcast_bg')
    gap=0
    for tc in config['timing_constraints']:
        level,preceding,following,latency=tc[:4]
        sibling=tc[5] if len(tc)>5 else False
        if ids[a] not in preceding or ids[b] not in following:continue
        applies=(level==0 or level==1 and same_rank!=sibling or
                 level==2 and same_rank and same_bg and not sibling)
        if applies:gap=max(gap,latency)
    return gap


def job(command,rank=1,bg=0,time=0,ca=1):
    write=command in ('WR','WRA','WRGB')
    array=command not in ('WRGB','RDRES')
    latency=(12 if write else 23) if array else 0
    return Job(0,rank,bg,write,array,time*CK,ca*CK,latency*CK,2*CK)


class NativeTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        library=Path(cls.tmp.name)/'probe.so'
        subprocess.run(['g++','-std=c++20','-O2','-shared','-fPIC',
                        str(SRC/'validation/native_timing_probe.cpp'),'-o',str(library)],check=True)
        cls.lib=C.CDLL(str(library));cls.spec=native_spec()
        cls.lib.probe_gap.argtypes=[C.POINTER(Job),C.POINTER(Job)];cls.lib.probe_gap.restype=C.c_int64
        cls.lib.probe_history.argtypes=[C.POINTER(Job),C.c_int,C.POINTER(Job),C.c_int64]
        cls.lib.probe_history.restype=C.c_int64
        cls.lib.probe_constant.argtypes=[C.c_int];cls.lib.probe_constant.restype=C.c_int64
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_all_transfer_pairs_against_native_hierarchy(self):
        records=[]
        for relation in ('same_bg','different_bg','different_rank','broadcast_bg'):
            for a in ('RD','WR','RDA','WRA','WRGB','RDRES'):
                for b in ('RD','WR','RDA','WRA','WRGB','RDRES'):
                    first=job(a,bg=-1 if relation=='broadcast_bg' else 0)
                    second=job(b,rank=0 if relation=='different_rank' else 1,
                               bg=1 if relation=='different_bg' else 0)
                    expected=native_gap(self.spec,a,b,relation)
                    with self.subTest(relation=relation,a=a,b=b):
                        actual=self.lib.probe_gap(C.byref(first),C.byref(second))//CK
                        self.assertEqual(actual,expected)
                        issue=self.lib.probe_history(C.byref(first),1,C.byref(second),0)//CK
                        self.assertGreaterEqual(issue,expected)
                    records.append(dict(preceding=a,following=b,relation=relation,
                        native_issue_gap_CK=expected,arbiter_issue_gap_CK=actual,
                        earliest_with_existing_DQ_guards_CK=issue))
        if os.environ.get('NATIVE_TIMING_AUDIT'):
            Path(os.environ['NATIVE_TIMING_AUDIT']).write_text(json.dumps(records,indent=2)+'\n')

    def test_older_transfer_constraint_survives_intervening_transfer(self):
        history=(Job*2)(job('RD',time=0),job('WRGB',time=4))
        next_job=job('WR',rank=0)
        self.assertEqual(self.lib.probe_history(history,2,C.byref(next_job),5*CK),15*CK)

    def test_CA_occupancy_is_independent_of_transfer_issue_gap(self):
        first=Job(0,1,-1,0,0,0,21*CK,0,0)
        second=job('RD',rank=0)
        self.assertEqual(self.lib.probe_history(C.byref(first),1,C.byref(second),0),21*CK)

    def test_row_refresh_and_data_constants_match_native(self):
        t=self.spec.resolve()[1]
        expected=[t[k] for k in ('nBL_min','nCL','nCWL','nRFC','nREFI','nRCD','nRCDWR',
                                'nRP','nRPab','nRAS','nRC','nRTP')]
        expected += [t['nCWL']+t['nBL_min']+1+t['nWR'],t['nRRDS'],t['nFAW'],t['nPPD']]
        self.assertEqual([self.lib.probe_constant(i) for i in range(len(expected))],expected)
        self.assertEqual(t['nRRDS'],t['nRRDL'])

    def test_emitted_GPU_CA_trace_satisfies_native_row_and_refresh_rules(self):
        engine=SequentialEngine(os.environ['UNIFORM_LIBRARY'])
        events=[(0,0,0,-1,100*CK,101*CK),(0,3,0,-1,100*CK,324*CK)]
        cal=engine.add(400*CK,events,16);program=engine.program(400*CK,0,cal,[])
        context=engine.begin(program,0);engine.trace(context)
        issued=[];seen=0;now=0
        names={1:'ACT1',2:'ACT2',3:'PREpb',4:'PREab'}
        for write in (False,True,False):
            plan,_=engine.stream_plan(800*CK,0 if write else 200,200 if write else 0)
            # Deliberate row conflicts across all banks, plus fixed and
            # periodic refresh; this validates emitted times, not constants.
            for i in range(200):engine.address(plan,i,(i%16)*64+(i//16)*1024)
            result=engine.execute(context,plan,now);now=result.finish
            trace=engine.trace(context)
            for x in trace[seen:]:
                issued.append((x['time']//CK,names.get(x['command'],'WR' if write else 'RD'),x['bank']))
            seen=len(trace)
        issued.extend((t,'REFab',-1) for t in range(100,now//CK+1,4164))
        issued.sort()
        spec=self.spec;cfg=spec.to_config();ids={c:i for i,c in enumerate(spec.commands)}
        history=[];checked=0
        for time,command,bank in issued:
            for tc in cfg['timing_constraints']:
                level,preceding,following,latency=tc[:4]
                window=tc[4] if len(tc)>4 else 1
                sibling=tc[5] if len(tc)>5 else False
                if sibling or ids[command] not in following:continue
                matches=[]
                for t,c,b in reversed(history):
                    if ids[c] not in preceding:continue
                    if level==2 and b>=0 and bank>=0 and b%4!=bank%4:continue
                    if level>=3 and b>=0 and bank>=0 and b!=bank:continue
                    matches.append((t,c,b))
                    if len(matches)==window:break
                if len(matches)==window:
                    self.assertGreaterEqual(time-matches[-1][0],latency,
                        (command,time,bank,matches[-1],tc))
                    checked+=1
            history.append((time,command,bank))
        self.assertGreater(checked,1000)


if __name__=='__main__':unittest.main()

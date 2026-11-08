import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sequential_transactions import SequentialEngine,CK

class SequentialTests(unittest.TestCase):
    def setUp(self):self.e=SequentialEngine(os.environ['UNIFORM_LIBRARY'])
    def program(self,jobs=(),refs=(),end=0,mode=0):
        cal=self.e.add(end*CK,refs,16)
        js=[[0,rank,bg,wr,array,ready*CK,ca*CK,lat*CK,burst*CK]
            for rank,bg,wr,array,ready,ca,lat,burst in jobs]
        return self.e.program(end*CK,mode,cal,js)
    def run_trace(self,n,write=False,span=None,program=None,addresses=None):
        p=self.program() if program is None else program
        h,_=self.e.stream_plan((span or n*4+60)*CK,0 if write else n,n if write else 0)
        if addresses:
            for i,sector in enumerate(addresses):self.e.address(h,i,sector)
        c=self.e.begin(p,0);self.e.trace(c);r=self.e.execute(c,h,0)
        return r,self.e.stats(c),self.e.trace(c)
    def test_cold_read_ACT1_ACT2_RCD(self):
        r,s,t=self.run_trace(1,span=60)
        self.assertEqual([x['command'] for x in t],[1,2,0])
        self.assertEqual([x['time']//CK for x in t],[0,1,21])
        self.assertEqual(s['act2'],1);self.assertEqual(s['row_misses'],1)
    def test_WRGB_to_warm_GPU_RD_has_two_CK_issue_spacing(self):
        p=self.program([(1,-1,1,0,t,1,0,2) for t in range(100,200,4)],end=200)
        c=self.e.begin(p,0);self.e.trace(c)
        warm,_=self.e.stream_plan(60*CK,1,0)
        self.e.execute(c,warm,0)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(c,h,100*CK)
        self.assertEqual(r.first_issue,102*CK)
        self.assertEqual(self.e.drain(c).pim_wait,5*CK)
    def test_native_cross_rank_RD_to_WR_cannot_reorder_DQ_to_evade_15_CK(self):
        # Warm rank0, then ordinary rank1 RD@100 would previously admit
        # rank0 WR@101: its earlier return escaped the DQ-only check.
        p=self.program([(1,0,0,1,100,1,23,2)],end=200)
        c=self.e.begin(p,0)
        warm,_=self.e.stream_plan(60*CK,1,0);self.e.execute(c,warm,0)
        h,_=self.e.stream_plan(30*CK,0,1)
        r=self.e.execute(c,h,100*CK)
        self.assertEqual(r.first_issue,115*CK)
    def test_write_uses_native_shorter_RCD(self):
        r,s,t=self.run_trace(1,True,60)
        self.assertEqual([x['time']//CK for x in t],[0,1,10])
    def test_one_full_row_has_64_sectors(self):
        r,s,t=self.run_trace(64,span=400)
        self.assertEqual(s['act1'],1);self.assertEqual(s['act2'],1)
        self.assertEqual(s['prepb'],0);self.assertEqual(s['row_hits'],63)
    def test_next_contiguous_row_uses_next_BG(self):
        r,s,t=self.run_trace(65,span=400)
        self.assertEqual(s['act2'],2);self.assertEqual(s['prepb'],0)
        self.assertEqual([x['bank'] for x in t if x['command']==2],[0,1])
    def test_conflict_after_all_16_banks(self):
        # Addresses 0 and 1024 are different rows of bank0 in rank0.
        r,s,t=self.run_trace(2,span=200,addresses=[0,1024])
        acts=[x for x in t if x['command']==2]
        pres=[x for x in t if x['command']==3]
        self.assertEqual(len(acts),2)
        self.assertEqual(len(pres),1)
        self.assertEqual(acts[-1]['row'],1)
        self.assertGreaterEqual(acts[-1]['time']-pres[-1]['time'],20*CK)
    def test_refresh_has_PREab_and_reopens(self):
        refs=[(0,0,0,-1,100*CK,101*CK),(0,3,0,-1,100*CK,324*CK)]
        p=self.program(refs=refs,end=400)
        r,s,t=self.run_trace(2,span=400,program=p)
        self.assertEqual(s['preab'],1);self.assertEqual(s['act2'],2)
        self.assertIn(77*CK,[x['time'] for x in t if x['command']==4])
        for x in t:
            if x['command']!=4:self.assertFalse(100*CK<=x['time']<324*CK)
    def test_PIM_blocks_GPU_ACT(self):
        p=self.program([(1,-1,0,0,0,21,0,0)],end=80)
        r,s,t=self.run_trace(1,span=80,program=p)
        self.assertEqual(t[0]['time'],21*CK)
        self.assertEqual(r.pim_wait,0)
    def test_ACT_pair_cannot_be_starved_past_AAD_by_PIM(self):
        p=self.program([(1,-1,0,0,1,21,0,0)],end=80)
        r,s,t=self.run_trace(1,span=80,program=p)
        self.assertEqual([x['time']//CK for x in t[:2]],[0,1])
        self.assertEqual(r.pim_wait,CK)
        self.assertEqual(s['act1'],s['act2'])
    def test_solo_normalization_with_rows_and_refresh(self):
        p=self.program()
        for rd,wr,span in [(10,10,100),(1025,0,30),(1000,800,10000)]:
            h,_=self.e.stream_plan(span*CK,rd,wr)
            ref,info=self.e.reference(p,h)
            r=self.e.execute(self.e.begin(p,0),ref,0)
            self.assertEqual(r.finish,info.finish);self.assertEqual(r.memory_stall,0)
    def test_write_recovery_before_precharge(self):
        r,s,t=self.run_trace(2,True,30,addresses=[0,1024])
        self.assertEqual(s['prepb'],1)
        last_write={}
        for x in t:
            if x['command']==0:last_write[x['bank']]=x['time']
            elif x['command']==3 and x['bank'] in last_write:
                self.assertGreaterEqual(x['time']-last_write[x['bank']],52*CK)
    def test_rank0_PIM_array_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'disjoint'):
            self.program([(0,0,0,1,0,1,23,2)])
    def test_no_synthetic_GPU_PREab_without_GPU_traffic(self):
        refs=[(0,0,0,-1,100*CK,101*CK),(0,3,0,-1,100*CK,324*CK)]
        p=self.program([(1,-1,0,0,77,1,0,0)],refs=refs,end=400)
        c=self.e.begin(p,0);r=self.e.drain(c)
        self.assertEqual(r.pim_wait,0);self.assertEqual(self.e.stats(c)['preab'],0)

if __name__=='__main__':unittest.main()

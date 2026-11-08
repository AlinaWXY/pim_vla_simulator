import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_transactions import OnlineEngine,CK

class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.e=OnlineEngine(os.environ['UNIFORM_LIBRARY'])
    def program(self,jobs=(),mode=0,refs=(),end=100):
        cal=self.e.add(end*CK,refs,16)
        js=[[0,rank,bg,wr,array,ready*CK,ca*CK,lat*CK,burst*CK]
            for rank,bg,wr,array,ready,ca,lat,burst in jobs]
        return self.e.program(end*CK,mode,cal,js)
    def test_gpu_does_not_see_unreleased_PIM_return(self):
        p=self.program([(1,-1,1,0,23,1,0,2)])
        h,_=self.e.stream_plan(30*CK,1,0)
        c=self.e.begin(p,0);r=self.e.execute(c,h,0)
        self.assertEqual(r.first_issue,0)
        # RD return23..25 must finish, then two-CK cross-rank R->W guard.
        self.assertEqual(r.pim_wait,4*CK)
        self.assertEqual(self.e.drain(c).pim_end,104*CK)
    def test_PIM_wins_ready_tie(self):
        p=self.program([(1,-1,1,0,0,1,0,2)])
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,2*CK)
        self.assertEqual(r.pim_wait,0)
    def test_new_GPU_issue_cannot_extend_pending_PIM_via_issue_spacing(self):
        # First RD@0 makes released WRGB@23 wait until27. The second RD
        # is ready at26: its late DQ fits, but issuing it would move WRGB
        # to28 through nBL_min. PIM must retain its27 CK issue slot.
        p=self.program([(1,-1,1,0,23,1,0,2)])
        h,_=self.e.stream_plan(51*CK,2,0)
        c=self.e.begin(p,0);r=self.e.execute(c,h,0)
        self.assertEqual(r.memory_stall,3*CK)
        self.assertEqual(self.e.drain(c).pim_wait,4*CK)
    def test_CA_duration_21_CK(self):
        p=self.program([(1,-1,0,0,0,21,0,0)],mode=1)
        h,_=self.e.stream_plan(50*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,21*CK)
    def test_refresh_rejects_incomplete_return(self):
        refs=[(0,0,0,-1,20*CK,21*CK),(0,3,0,-1,20*CK,244*CK)]
        p=self.program(refs=refs,end=300)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,244*CK)
    def test_other_rank_refresh_does_not_stop_GPU(self):
        refs=[(0,0,1,-1,20*CK,21*CK),(0,4,1,-1,20*CK,244*CK)]
        p=self.program(refs=refs,end=300)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,0)
    def test_private_compute_has_no_external_CA_jobs(self):
        p=self.program(mode=1,end=100)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.finish,30*CK)
        self.assertEqual(r.pim_wait,0)
    def test_boundary_handoff_waits_for_committed_GPU_return(self):
        p=self.program([(1,-1,1,0,23,0,0,2)],mode=1)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,0)
        self.assertEqual(r.pim_wait,4*CK)
    def test_PIM_delay_propagates_to_later_command(self):
        p=self.program([(1,-1,1,0,23,1,0,2),(1,-1,1,0,33,1,0,2)])
        h,_=self.e.stream_plan(30*CK,1,0)
        c=self.e.begin(p,0);self.e.execute(c,h,0);r=self.e.drain(c)
        self.assertEqual(r.pim_end,104*CK)
        self.assertEqual(r.pim_requests,2)
    def test_committed_GPU_write_blocks_PIM_read_with_turnaround(self):
        p=self.program([(1,-1,0,0,13,1,0,2)])
        h,_=self.e.stream_plan(30*CK,0,1)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,0)
        self.assertEqual(r.pim_wait,2*CK)
    def test_unreleased_future_program_does_not_change_first_GPU_issue(self):
        h,_=self.e.stream_plan(30*CK,1,0)
        for future in [23,60,90]:
            p=self.program([(1,-1,1,0,future,1,0,2)])
            r=self.e.execute(self.e.begin(p,0),h,0)
            self.assertEqual(r.first_issue,0)
    def test_after_IO_epoch_is_exclusive_between_bursts(self):
        p=self.program([(-2,-1,0,0,0,0,0,0),
                        (1,-1,1,0,10,1,0,2),(1,-1,1,0,20,1,0,2),
                        (-3,-1,0,0,30,0,0,0)],mode=1)
        h,_=self.e.stream_plan(30*CK,1,0)
        r=self.e.execute(self.e.begin(p,0),h,0)
        self.assertEqual(r.first_issue,30*CK)
    def test_cloned_channel_states_are_independent(self):
        p=self.program([(1,-1,1,0,23,1,0,2)])
        a=self.e.begin(p,0);b=self.e.clone(a)
        h,_=self.e.stream_plan(30*CK,1,0)
        self.e.execute(a,h,0)
        self.assertEqual(self.e.drain(a).pim_end,104*CK)
        self.assertEqual(self.e.drain(b).pim_end,100*CK)
    def test_solo_reference_accounts_for_all_committed_turnarounds(self):
        p=self.program(end=0)
        for reads,writes,span in [(10,10,100),(100,80,1000),(1000,800,10000),(100,300,1000)]:
            h,_=self.e.stream_plan(span*CK,reads,writes)
            normalized,ref=self.e.reference(p,h)
            check=self.e.execute(self.e.begin(p,0),normalized,0)
            self.assertEqual(check.finish,ref.finish)
            self.assertEqual(check.requests,reads+writes)
            self.assertEqual(check.memory_stall,0)

if __name__=='__main__':unittest.main()

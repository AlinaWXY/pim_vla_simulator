"""Resource and scheduler tests for continuous kernel replay, requiring the shared library."""
import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from request_model import Replay,simulate,ticks,CK

@unittest.skipUnless(os.environ.get('REQUEST_REPLAY_LIBRARY'),'set REQUEST_REPLAY_LIBRARY to run native request tests')
class RequestReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r=Replay(os.environ['REQUEST_REPLAY_LIBRARY'])
    def test_single_burst_uses_small_gap_not_kernel_sized_gap(self):
        h=self.r.add(1000*CK,[(0,0,23*CK,25*CK)],channels=1)
        r=self.r.probe(h,0,1)
        self.assertEqual(r.finish,27*CK)
        self.assertEqual(r.dq_wait,2*CK)
    def test_channels_arbitrate_independently(self):
        h=self.r.add(1000*CK,[(0,0,23*CK,200*CK)],channels=2)
        self.assertEqual(self.r.probe(h,0,1,0).finish,202*CK)
        self.assertEqual(self.r.probe(h,0,1,1).finish,25*CK)
    def test_rank1_refresh_only_blocks_its_CA_pulse(self):
        h=self.r.add(1000*CK,[(0,1,0,CK),(0,3,0,224*CK)],channels=1)
        r=self.r.probe(h,0,1);self.assertEqual(r.finish,26*CK);self.assertEqual(r.refresh_wait,0)
    def test_rank0_refresh_blocks_gpu_request(self):
        h=self.r.add(1000*CK,[(0,1,0,CK),(0,2,0,224*CK)],channels=1)
        self.assertEqual(self.r.probe(h,0,1).finish,249*CK)
    def test_CA_and_DQ_may_be_served_at_distinct_times(self):
        h=self.r.add(1000*CK,[(0,1,20*CK,26*CK)],channels=1)
        # CA issued at zero, data returned at 23..25; occupied CA during data is legal.
        self.assertEqual(self.r.probe(h,0,1).finish,25*CK)
    def test_request_queue_and_solo_calibration(self):
        h=self.r.add(ticks(100),[],channels=1)
        p=dict(pressure=.9,outstanding=2,write_percent=20,guard_ck=2)
        r=self.r.kernel(h,0,ticks(5),p)
        self.assertEqual(r.finish,ticks(5));self.assertLessEqual(r.max_inflight,2)
        self.assertGreater(r.requests,0);self.assertGreater(r.calibration_trim,0)
    def test_kernel_crosses_PIM_boundary_and_AE_waits(self):
        h=self.r.add(ticks(1),[],channels=1)
        seq=[dict(kind='pim',name='p',duration=ticks(1),handle=h),dict(kind='gpu',name='AE',duration=ticks(3))]
        ks=[dict(kind='kernel',duration=ticks(2)),dict(kind='kernel',duration=ticks(2))]
        p=dict(pressure=0,outstanding=32,write_percent=20,guard_ck=2,dispatch_us=0)
        r=simulate(seq,ks,self.r,p,collect=True)
        self.assertAlmostEqual(r['AE_wait_ms'],.001)
        self.assertAlmostEqual(r['total_ms'],.007)
        self.assertEqual(r['completed_VLM_units'],1) # AE priority: second VLM kernel waits.
        self.assertTrue(r['placements'][0]['crossed_AE_GPU_boundary'])
    def test_dependency_order_and_net_savings_identity(self):
        h=self.r.add(ticks(4),[],channels=1)
        seq=[dict(kind='pim',name='p',duration=ticks(4),handle=h),dict(kind='gpu',name='AE',duration=ticks(2))]
        ks=[dict(kind='kernel',duration=ticks(x)) for x in (1,10,1)]
        p=dict(pressure=0,outstanding=32,write_percent=20,guard_ck=2)
        r=simulate(seq,ks,self.r,p,collect=True)
        self.assertEqual([p['ordinal'] for p in r['placements']],[0,1])
        self.assertAlmostEqual(r['effective_saving_ms'],r['VLM_advanced_ms']-r['AE_wait_ms'])
        self.assertAlmostEqual(r['total_ms'],.014)
    def test_busy_DQ_does_not_stop_compute_only_kernel(self):
        h=self.r.add(ticks(10),[(0,0,0,ticks(10))],channels=1)
        p=dict(pressure=0,outstanding=32,write_percent=20,guard_ck=2)
        self.assertEqual(self.r.kernel(h,0,ticks(3),p).finish,ticks(3))
    def test_GPU_byte_demand_is_identical_across_layouts_and_start_times(self):
        h=self.r.add(ticks(10),[(0,0,0,ticks(10)),(0,2,0,224*CK)],channels=1)
        empty=self.r.add(ticks(10),[],channels=1)
        p=dict(pressure=.6,outstanding=8,write_percent=20,guard_ck=2)
        first=self.r.kernel(h,0,ticks(3),p);second=self.r.kernel(empty,ticks(2),ticks(3),p)
        self.assertEqual(first.requests,second.requests)
        self.assertEqual(first.calibration_trim,second.calibration_trim)

if __name__=='__main__':unittest.main()

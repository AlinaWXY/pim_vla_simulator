"""Physical reservation and dependency checks, using an externally built library."""
import os
import random
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uniform_transactions import Engine, CK, U
from run_uniform_transactions import simulate


class UniformTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(os.environ['UNIFORM_LIBRARY'])

    def calendar(self, events=(), end=200, channels=1):
        return self.engine.add(end*CK, [[ch,k,r,b,a*CK,z*CK]
                               for ch,k,r,b,a,z in events],channels)

    def test_idle_now_is_not_permission_for_delayed_read(self):
        c=self.calendar([(0,2,1,-1,25,27)])
        r=self.engine.probe(c,3*CK)
        self.assertEqual(r.first_issue,5*CK)
        self.assertEqual(r.finish,30*CK)  # WRGB end27 + W->R guard1, read starts28.

    def test_CA_and_future_DQ_must_both_fit(self):
        c=self.calendar([(0,2,1,-1,25,27),(0,0,1,-1,5,8)])
        r=self.engine.probe(c,3*CK)
        self.assertEqual(r.first_issue,8*CK)
        self.assertGreater(r.ca_delay,0)
        self.assertGreater(r.dq_delay,0)

    def test_exported_native_CA_holds_the_entire_command(self):
        from prepare_uniform_bus import native_ca_span
        for command, cycles in [('RD',1), ('ACTAB',2), ('MAC_RUN',21)]:
            with self.subTest(command=command):
                a,b=native_ca_span(dict(command=command,clock=34,ready_at=34+cycles))
                c=self.engine.add(200*CK,[(0,0,1,-1,a,b)],1)
                # An RD offered in the final occupied cycle must wait for
                # ready_at, including the tails formerly lost by the exporter.
                r=self.engine.probe(c,(32+cycles)*CK)
                self.assertEqual(r.first_issue,(33+cycles)*CK)

    def test_R_to_W_guard_against_future_PIM_write(self):
        c=self.calendar([(0,2,1,-1,25,27)])
        r=self.engine.probe(c,0)
        # A read at0 would occupy23..25, but its turnaround collides with WRGB25.
        self.assertEqual(r.first_issue,5*CK)

    def test_same_rank_write_read_turnaround_is_longer(self):
        c=self.calendar([(0,2,0,0,0,2)])
        r=self.engine.probe(c,0,bg=0)
        self.assertEqual(r.first_issue,17*CK)  # DQ starts40 = 2+38.

    def test_refresh_blocks_only_target_rank(self):
        other=self.calendar([(0,0,1,-1,10,11),(0,4,1,-1,10,234)],end=300)
        target=self.calendar([(0,0,0,-1,10,11),(0,3,0,-1,10,234)],end=300)
        self.assertEqual(self.engine.probe(other,11*CK).first_issue,11*CK)
        self.assertEqual(self.engine.probe(target,11*CK).first_issue,234*CK)

    def test_read_tail_cannot_cross_into_refresh(self):
        c=self.calendar([(0,3,0,-1,20,244)],end=300)
        self.assertEqual(self.engine.probe(c,0).first_issue,244*CK)

    def test_write_recovery_before_refresh(self):
        c=self.calendar([(0,3,0,-1,50,274)],end=300)
        self.assertEqual(self.engine.probe(c,0,write=True).first_issue,274*CK)

    def test_channels_arbitrate_independently(self):
        c=self.calendar([(0,0,1,-1,0,80)],channels=2)
        self.assertEqual(self.engine.probe(c,channel=0).first_issue,80*CK)
        self.assertEqual(self.engine.probe(c,channel=1).first_issue,0)

    def test_preserve_fractional_slice_clock_phase(self):
        c=self.engine.add(200*CK,[(0,0,1,-1,469,469+CK)],1)
        self.assertEqual(self.engine.probe(c,0).first_issue,469+CK)

    def test_isolated_plan_reproduces_retained_time_and_all_bytes(self):
        c=self.calendar(end=0)
        for reads,writes in ((0,0),(17,3),(1000,200)):
            p,info=self.engine.plan(10*U,reads,writes)
            r=self.engine.run(c,p)
            self.assertEqual(r.requests,reads+writes)
            self.assertEqual(r.finish,10*U)
            self.assertEqual(r.backpressure,0)
            self.assertLessEqual(info.calibrated_span,10*U)

    def test_memory_delay_propagates_to_virtual_progress(self):
        p,_=self.engine.plan(100*CK,12,0)
        c=self.calendar([(0,0,1,-1,0,300)],end=300)
        r=self.engine.run(c,p)
        self.assertGreaterEqual(r.finish,400*CK)
        self.assertGreaterEqual(r.backpressure,300*CK)

    def test_impossible_volume_is_rejected_not_trimmed(self):
        with self.assertRaisesRegex(ValueError,'cannot fit'):
            self.engine.plan(10*CK,1000,1000)

    def test_ordered_kernel_crosses_PIM_and_delays_AE_GPU(self):
        c=self.calendar([(ch,0,1,-1,0,10) for ch in range(16)],end=50,channels=16)
        p,_=self.engine.plan(100*CK,1,0)
        units=[dict(duration=100*CK,kind='kernel',read_sectors=16,write_sectors=0)]*2
        sequence=[dict(kind='pim',name='pim',calendar=c,duration=50*CK),
                  dict(kind='gpu',name='nonlinear',duration=20*CK)]
        r=simulate(sequence,units,[[p]*16]*2,self.engine)
        self.assertEqual(r['completed_units'],1)
        self.assertAlmostEqual(r['AE_GPU_wait_ms'],60*CK/U/1000)
        self.assertAlmostEqual(r['total_ms'],230*CK/U/1000)
        self.assertEqual(r['placements'][0]['ordinal'],0)
        self.assertTrue(r['placements'][0]['extends_past_PIM'])

    def test_no_legal_CA_before_PIM_end_does_not_launch(self):
        c=self.calendar([(ch,0,1,-1,0,100) for ch in range(16)],end=100,channels=16)
        p,_=self.engine.plan(100*CK,1,0)
        unit=dict(duration=100*CK,kind='kernel',read_sectors=16,write_sectors=0)
        r=simulate([dict(kind='pim',name='pim',calendar=c,duration=100*CK)],
                   [unit],[[p]*16],self.engine)
        self.assertEqual(r['completed_units'],0)
        self.assertEqual(r['AE_GPU_wait_ms'],0)

    def test_cross_kernel_write_read_turnaround_is_preserved(self):
        c=self.calendar(end=0)
        w,_=self.engine.plan(14*CK,0,1)
        r,_=self.engine.plan(25*CK,1,0)
        first=self.engine.run(c,w)
        second=self.engine.run(c,r,start=first.finish,previous=first)
        self.assertEqual(second.first_issue,29*CK)
        self.assertEqual(second.finish,54*CK)

    def test_fixed_uniform_stream_exposes_physical_reference_bias(self):
        c=self.calendar(end=0)
        p,info=self.engine.stream_plan(10*CK,0,20)
        r=self.engine.run(c,p)
        self.assertEqual(r.requests,20)
        self.assertGreater(info.finish,10*CK)
        self.assertEqual(r.finish,info.finish)

    def test_earliest_slot_matches_independent_brute_force_oracle(self):
        rng=random.Random(1108)
        def guard(w0,rank0,bg0,w1,rank1,bg1):
            if rank0!=rank1:return 1 if w0 else 2
            same=bg0<0 or bg1<0 or bg0==bg1
            if w0 and not w1:return 23+(13 if same else 7)+(2 if same else 0)
            if not w0 and w1:return 4 if same else 2
            return 2 if same else 0
        for _ in range(50):
            dq=[];ca=[]
            for t in range(0,160,8):
                if rng.randrange(2):
                    k=rng.choice([1,2]);rank=rng.randrange(2);bg=rng.randrange(4)
                    dq.append((0,k,rank,bg,t,t+2))
                if rng.randrange(2):ca.append((0,0,1,-1,t+3,t+4))
            ref=[(0,3,0,-1,180,404),(0,0,0,-1,180,181)]
            cal=self.calendar(ca+dq+ref,end=450)
            for write in (False,True):
                ready=rng.randrange(160);lat=12 if write else 23;bg=rng.randrange(4)
                def legal(t):
                    if t<404 and t+lat+2+(37 if write else 0)>180:return False
                    if any(t<z and t+1>a for _,_,_,_,a,z in ca+ref[1:]):return False
                    for _,kind,rank,b,a,z in dq:
                        old=kind==2
                        if not (t+lat+2+guard(write,0,bg,old,rank,b)<=a or
                                z+guard(old,rank,b,write,0,bg)<=t+lat):return False
                    return True
                expected=next(t for t in range(ready,500) if legal(t))
                self.assertEqual(self.engine.probe(cal,ready*CK,write,bg).first_issue,expected*CK)


if __name__=='__main__':
    unittest.main()

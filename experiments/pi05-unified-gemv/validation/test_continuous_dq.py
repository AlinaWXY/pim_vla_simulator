"""Independent boundary/oracle tests for continuous external-DQ windows."""
import csv
import gzip
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import continuous_dq as C


def phase(name,begin,end,meaningful=True,kind="compute"):
    return dict(name=name,kind=kind,begin=begin,end=end,meaningful=meaningful)


def spans(windows):
    return [(w["begin"],w["end"]) for w in windows]


class PhysicalWindowsTests(unittest.TestCase):
    def test_half_open_union_and_scope_clipping(self):
        busy,idle=C.maximal_idle_windows([
            dict(begin=-5,end=2),dict(begin=4,end=6),dict(begin=6,end=8),
            dict(begin=12,end=20)],0,15)
        self.assertEqual((busy,idle),(9,[(2,4),(8,12)]))

    def test_one_burst_splits_idle_not_a_sum(self):
        phases=[phase("work",0,20)]
        row,windows,complete=C.window_record([dict(issue=10,begin=10,end=12)],0,20,
            variant="fixture",case="chain",stage="chain",scope="stage",channel=0,tick_us=.5,phases=phases)
        self.assertEqual(row["dq_idle_total_us"],9)
        self.assertEqual(row["longest_duration_us"],5)
        self.assertEqual(row["window_count"],2)
        self.assertEqual(row["window_duration_p50_us"],4.5)
        self.assertEqual(row["window_duration_p95_us"],4.95)
        self.assertEqual([(w["begin_tick"],w["end_tick"]) for w in windows],[(0,10),(12,20)])
        self.assertEqual(complete,[])

    def test_delayed_native_read_and_write_tails_use_burst_end(self):
        intervals=C.dq_intervals([dict(clock=2,command="RD"),dict(clock=20,command="WR")],
            dict(nBL_min=2,nCL=5,nCWL=3),4,"attacc")
        self.assertEqual(C.maximal_idle_windows(intervals,28,93),(9,[(33,89)]))
        self.assertEqual(C.maximal_idle_windows(intervals,5,25),(0,[(5,25)]))
        self.assertEqual(C.maximal_idle_windows(intervals,33,89),(0,[(33,89)]))

    def test_all_channels_idle_is_complement_of_busy_union(self):
        first=[dict(issue=2,begin=2,end=5)]
        second=[dict(issue=4,begin=4,end=8)]
        self.assertEqual(C.maximal_idle_windows(first+second,0,10),(6,[(0,2),(8,10)]))
        self.assertEqual(C.maximal_idle_windows([],3,3),(0,[]))

    def test_integer_grid_oracle_for_overlaps_and_clipped_edges(self):
        rng=random.Random(20261104)
        for _ in range(200):
            intervals=[]
            for _ in range(rng.randrange(12)):
                left=rng.randrange(-5,35)
                intervals.append(dict(begin=left,end=left+rng.randrange(10)))
            busy,idle=C.maximal_idle_windows(intervals,0,30)
            occupied={i for i in range(30) if any(e["begin"]<=i<e["end"] for e in intervals)}
            actual={i for a,b in idle for i in range(a,b)}
            self.assertEqual(busy,len(occupied))
            self.assertEqual(actual,set(range(30))-occupied)
            self.assertTrue(all(a[1]<b[0] for a,b in zip(idle,idle[1:])))


class CompleteWorkTests(unittest.TestCase):
    def test_adjacent_compute_and_spill_merge_without_phase_break(self):
        work=[phase("compute0",0,10),phase("spill0",10,15,kind="local_write"),phase("compute1",15,25)]
        result=C.complete_work_windows([],0,25,work)
        self.assertEqual(spans(result),[(0,25)])
        self.assertEqual([p["name"] for p in result[0]["phases"]],["compute0","spill0","compute1"])

    def test_transfer_phase_idle_holes_never_bridge_complete_work(self):
        work=[phase("compute0",0,10),phase("export",10,20,kind="external_read"),phase("compute1",20,30)]
        dq=[dict(issue=12,begin=12,end=14)]
        self.assertEqual(spans(C.complete_work_windows(dq,0,30,work)),[(0,10),(20,30)])
        self.assertEqual(C.maximal_idle_windows(dq,0,30)[1],[(0,12),(14,30)])

    def test_control_bridge_only_between_work_and_not_at_ends(self):
        work=[phase("leading",0,3,False,"gemv_release"),phase("compute",3,10),
            phase("release",10,12,False,"gemv_release"),phase("rewind",12,14,False,"gemv_release"),
            phase("spill",14,20),phase("trailing",20,25,False,"gemv_release")]
        result=C.complete_work_windows([],0,25,work)
        self.assertEqual(spans(result),[(3,20)])
        self.assertEqual([p["name"] for p in result[0]["phases"]],["compute","release","rewind","spill"])
        self.assertEqual(C.complete_work_windows([],0,25,[phase("only_control",0,25,False)]),[])

    def test_delayed_burst_inside_control_prevents_bridge(self):
        work=[phase("compute0",0,10),phase("control",10,15,False),phase("compute1",15,25)]
        dq=[dict(issue=-3,begin=11,end=13)]
        self.assertEqual(spans(C.complete_work_windows(dq,0,25,work)),[(0,10),(15,25)])

    def test_incoming_tail_removes_preburst_gap_and_breaks_adjacency(self):
        work=[phase("compute0",0,10),phase("compute1",10,25)]
        dq=[dict(issue=-2,begin=12,end=15)]
        self.assertEqual(spans(C.complete_work_windows(dq,0,25,work)),[(0,10),(15,25)])
        self.assertEqual(spans(C.complete_work_windows(dq,10,14,[phase("covered",10,14)])),[])

    def test_all_channels_union_keeps_original_issue_time_for_tail(self):
        dq=[dict(issue=2,begin=12,end=15)]
        work=[phase("compute",10,20)]
        result=C.complete_work_windows(C.union_intervals(dq),10,20,work,issue_intervals=dq)
        self.assertEqual(spans(result),[(15,20)])

    def test_exact_issue_boundary_and_missing_time_breaks(self):
        work=[phase("first",0,10),phase("second",10,20)]
        dq=[dict(issue=10,begin=15,end=17)]
        self.assertEqual(spans(C.complete_work_windows(dq,0,20,work)),[(0,10)])
        self.assertEqual(spans(C.complete_work_windows([],0,20,[phase("first",0,9),phase("second",10,20)])),[(0,9),(10,20)])

    def test_scope_clipping_does_not_promote_incomplete_work_phase(self):
        work=[phase("partial",0,10),phase("complete",10,20)]
        self.assertEqual(spans(C.complete_work_windows([],5,20,work)),[(10,20)])


class ScopeTests(unittest.TestCase):
    @staticmethod
    def manifest(case,segments):
        return dict(case=case,segments=[dict(name=name,stage=stage,kind=kind,start=i,end=i+1,
            counts={"PIM_MAC_QK" if kind=="qk" else "PIM_GEMV_RELEASE":1})
            for i,(name,stage,kind) in enumerate(segments)])

    def test_qkt_includes_export_and_final_release(self):
        manifest=self.manifest("chain",[("before","chain","compute"),
            ("qk_tile_0_compute","chain","qk"),("qk_tile_0_export","chain","external_read"),
            ("qk_tile_1_compute","chain","qk"),("qk_tile_1_export","chain","external_read"),
            ("qk_gemv_release","chain","gemv_release")])
        qkt=C.semantic_scopes(manifest,dict(enumerate(range(0,70,10))),70)[1]
        self.assertEqual((qkt["begin"],qkt["end"]),(10,60))
        self.assertEqual(qkt["metadata"]["qkt_compute_phase_count"],2)
        self.assertEqual(qkt["metadata"]["qkt_export_phase_count"],2)
        self.assertEqual(qkt["metadata"]["first_export_phase_begin_tick"],20)

    def test_serial_stage_scopes_stop_at_boundary_full_run_uses_actual_clock(self):
        manifest=self.manifest("serial_layer_control",[("chain/compute","chain","qk"),("pv/compute","pv","qk")])
        scopes=C.semantic_scopes(manifest,{0:1,1:11,2:31},40)
        self.assertEqual([(s["scope"],s["stage"],s["begin"],s["end"]) for s in scopes],
            [("stage","chain",1,11),("stage","pv",11,31),("serial_layer","serial_layer_control",0,40)])
        self.assertEqual([spans(C.complete_work_windows([],s["begin"],s["end"],s["phases"])) for s in scopes],
            [[(1,11)],[(11,31)],[(1,31)]])


class NativeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="continuous-dq-test-",dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"simulation").mkdir()
        self.manifest=dict(case="chain",expected_dq_bytes={"total":16*2*32},segments=[])
        for i,(name,kind,operation) in enumerate([
            ("qk_tile_0_compute","qk","PIM_MAC_QK"),("qk_tile_0_export","external_read","PIM_RD_RES"),
            ("qk_tile_1_compute","qk","PIM_MAC_QK"),("qk_tile_1_export","external_read","PIM_RD_RES"),
            ("qk_gemv_release","gemv_release","PIM_GEMV_RELEASE")]):
            self.manifest["segments"].append(dict(name=name,kind=kind,start=i,end=i+1,counts={operation:1}))
        result=dict(config=dict(frontend_ratio=1,channels=16,pim_data_path="attacc"),
            resolved_timings_ck=dict(nBL_min=2,nCL=3,nCWL=2),memory_cycles=40,latency_ns=40000,tCK_ns=1000)
        self.dump("manifest.json",self.manifest)
        self.dump("simulation/result.json",result)
        self.csv("simulation/barriers.csv",("marker","frontend_tick"),
            [dict(marker=i,frontend_tick=t) for i,t in enumerate((1,9,17,25,33,37))])
        for channel in range(16):
            self.csv(f"simulation/timeline.csv.ch{channel}",("clock","command","Channel"),
                [dict(clock=11,command="RDRES",Channel=channel),dict(clock=27+(channel==1),command="RDRES",Channel=channel)])

    def dump(self,name,value):
        (self.root/name).write_text(json.dumps(value),encoding="utf-8")

    def csv(self,name,fields,rows):
        with (self.root/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)

    def test_real_native_parser_all_channels_qkt_and_prefix(self):
        rows,windows,complete,hashes=C.analyze_case(self.root,"neither")
        self.assertEqual(len(rows),51)
        self.assertEqual(len(hashes),19)
        qkt=next(r for r in rows if r["scope"]=="qkt" and r["channel"]==0)
        self.assertEqual((qkt["dq_busy_us"],qkt["dq_idle_total_us"]),(4,32))
        self.assertEqual((qkt["longest_duration_us"],qkt["longest_complete_work_window_us"]),(14,8))
        self.assertEqual(qkt["complete_work_window_count"],2)
        simultaneous=next(r for r in rows if r["scope"]=="qkt" and r["channel"]=="all_channels")
        self.assertEqual(simultaneous["dq_busy_us"],5)
        prefix=next(r for r in rows if r["scope"]=="qkt_before_first_export" and r["channel"]==0)
        self.assertEqual((prefix["scope_begin_tick"],prefix["scope_end_tick"]),(1,11))
        self.assertEqual(prefix["first_export_phase_begin_tick"],9)
        self.assertEqual(prefix["longest_complete_work_window_us"],8)

    def test_count_mismatch_rejected(self):
        self.manifest["expected_dq_bytes"]["total"]+=32
        self.dump("manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"byte contract"):
            C.analyze_case(self.root,"neither")

    def test_incoming_burst_is_not_mislabelled_first_score_export(self):
        # The QKT envelope now begins at tick5, with an earlier RD landing at6.
        self.csv("simulation/barriers.csv",("marker","frontend_tick"),
            [dict(marker=i,frontend_tick=t) for i,t in enumerate((5,9,17,25,33,37))])
        self.csv("simulation/timeline.csv.ch0",("clock","command","Channel"),
            [dict(clock=3,command="RD",Channel=0),dict(clock=11,command="RDRES",Channel=0),dict(clock=27,command="RDRES",Channel=0)])
        self.manifest["expected_dq_bytes"]["total"]+=32
        self.dump("manifest.json",self.manifest)
        rows,_,_,_=C.analyze_case(self.root,"neither")
        prefix=next(r for r in rows if r["scope"]=="qkt_before_first_export" and r["channel"]==0)
        self.assertEqual(prefix["scope_end_tick"],11)
        self.assertEqual(prefix["dq_busy_us"],2)
        self.assertEqual(prefix["longest_complete_work_window_us"],1)

    def test_write_guard_and_explicit_fresh_nested_output(self):
        with self.assertRaisesRegex(ValueError,"outside the immutable"):
            C.write_report(self.root,self.root/"derived")
        with patch.object(C,"VARIANTS",(".",)),patch.object(C,"CASES",(".",)):
            report=C.write_report(self.root,self.root/"derived",allow_nested_output=True)
        self.assertEqual(len(report["records"]),51)
        for name in ("continuous-dq.json",*report["files"].values()):
            self.assertTrue((self.root/"derived"/name).is_file())
        self.assertEqual(len(report["qkt_windows"]),3)
        self.assertTrue(all(w["case"]=="chain" and w["channel"]==0 for w in report["qkt_windows"]))
        self.assertEqual(report["files"]["physical_windows"],"continuous-dq-windows.csv.gz")
        with gzip.open(self.root/"derived"/report["files"]["physical_windows"],"rt",encoding="utf-8",newline="") as stream:
            physical=list(csv.DictReader(stream))
        self.assertEqual(len(physical),119)
        self.assertEqual({w["channel"] for w in physical},{str(i) for i in range(16)}|{"all_channels"})
        self.assertEqual(len(report["complete_work_windows"]),85)


if __name__=="__main__":
    unittest.main()

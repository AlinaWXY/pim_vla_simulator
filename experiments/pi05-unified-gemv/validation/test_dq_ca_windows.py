"""Resource attribution, strict-work, reconstruction and artifact tests."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import dq_ca_windows as C


def phase(name,begin,end,meaningful=True):
    return dict(name=name,begin=begin,end=end,kind="compute" if meaningful else "gemv_release",meaningful=meaningful)


class ResourceTests(unittest.TestCase):
    @staticmethod
    def ca(clock,ready,scope="global",rank=1,channel=0,command="ACT1"):
        return dict(clock=clock,ready_at=ready,scope=scope,Rank=rank,Channel=channel,command=command)

    def test_global_and_rank_mirror_are_distinct_resources(self):
        rows=[self.ca(2,4),self.ca(2,4,"rank"),self.ca(4,5,"rank",0)]
        result=C.ca_intervals(rows,4,0)
        self.assertEqual([(e["begin"],e["end"]) for e in result["global_ca"]],[(5,13)])
        self.assertEqual(len(result["rank1_ca"]),1)
        self.assertEqual(len(result["rank0_ca"]),1)

    def test_wildcard_rank_reserves_each_rank_without_global_charge(self):
        result=C.ca_intervals([self.ca(2,3,"rank",-1)],1,0)
        self.assertEqual(result["global_ca"],[])
        self.assertEqual(result["rank0_ca"],result["rank1_ca"])

    def test_same_resource_overlap_and_bad_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Overlapping global_ca"):
            C.ca_intervals([self.ca(1,3),self.ca(2,4)],1,0)
        with self.assertRaisesRegex(ValueError,"channel label"):
            C.ca_intervals([self.ca(1,2,channel=1)],1,0)
        with self.assertRaisesRegex(ValueError,"Unknown CA"):
            C.ca_intervals([self.ca(1,2,scope="bank")],1,0)

    def test_exact_native_ca_width_and_half_open_touch(self):
        result=C.ca_intervals([self.ca(1,3),self.ca(3,4)],32,0)
        self.assertEqual(C.maximal_idle_windows(result["global_ca"],0,128),(96,[(0,1),(97,128)]))

    def test_resource_specific_strict_phase_qualification(self):
        phases=[phase("work0",0,10),phase("release",10,12,False),phase("work1",12,22)]
        dq=[]
        ca=[dict(issue=1,begin=1,end=2)]
        self.assertEqual([(w["begin"],w["end"]) for w in C.complete_work_windows(dq,0,22,phases)],[(0,22)])
        self.assertEqual([(w["begin"],w["end"]) for w in C.complete_work_windows(ca,0,22,phases)],[(12,22)])
        # A pulse never becomes a primary phase suffix. The suffix is diagnostic.
        pieces=C.diagnostic_joint_windows(C.complete_work_windows(dq,0,22,phases),ca,phases)
        self.assertEqual([(p["begin"],p["end"]) for p in pieces],[(0,1),(2,22)])

    def test_joint_issue_union_rejects_each_resource_issuing_phase(self):
        phases=[phase("ca_issue",0,10),phase("dq_issue",10,20),phase("free",20,30)]
        ca=[dict(issue=1,begin=1,end=2)]
        dq=[dict(issue=11,begin=15,end=17)]
        joint=C.complete_work_windows(C.union_intervals(ca+dq),0,30,phases,issue_intervals=ca+dq)
        self.assertEqual([(w["begin"],w["end"]) for w in joint],[(20,30)])

    def test_incoming_ca_tail_and_control_bridge_are_not_free(self):
        phases=[phase("first",0,10),phase("bridge",10,15,False),phase("second",15,25)]
        ca=[dict(issue=9,begin=9,end=12)]
        joint=C.complete_work_windows(ca,0,25,phases)
        self.assertEqual([(w["begin"],w["end"]) for w in joint],[(15,25)])
        tail=C.complete_work_windows([dict(issue=-1,begin=-1,end=3)],0,10,[phase("work",0,10)])
        self.assertEqual([(w["begin"],w["end"]) for w in tail],[(3,10)])

    def test_diagnostic_pulse_in_control_does_not_count_control_suffixes(self):
        phases=[phase("first",0,10),phase("bridge",10,20,False),phase("second",20,30)]
        work=C.complete_work_windows([],0,30,phases)
        pieces=C.diagnostic_joint_windows(work,[dict(begin=12,end=15)],phases)
        self.assertEqual([(p["begin"],p["end"]) for p in pieces],[(0,10),(20,30)])

    def test_verified_flat_widths_exclude_internal_ports_and_reject_unknowns(self):
        rows=[dict(clock=1,command="ACTAB"),dict(clock=2,command="LOCALWR"),dict(clock=3,command="MACAB")]
        result=C.reconstruct_legacy_ca(rows,4,{"ACTAB":2,"MACAB":1},{"LOCALWR","LOCALRD"})
        self.assertEqual([(e["begin"],e["end"]) for e in result],[(1,9),(9,13)])
        with self.assertRaisesRegex(ValueError,"no verified width"):
            C.reconstruct_legacy_ca(rows,4,{"ACTAB":2},{"LOCALWR"})


class NativeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="dq-ca-test-",dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.case=self.root/"pv"
        (self.case/"simulation").mkdir(parents=True)
        segments=[]
        for i,(name,kind,op) in enumerate((("compute0","compute","PIM_MAC_AB"),
                ("release_bridge","gemv_release","PIM_GEMV_RELEASE"),("compute1","compute","PIM_MAC_AB"),
                ("export","external_read","PIM_RD_RES"),("release_tail","gemv_release","PIM_GEMV_RELEASE"))):
            segments.append(dict(name=name,kind=kind,start=i,end=i+1,counts={op:1}))
        self.manifest=dict(case="pv",segments=segments,expected_dq_bytes={"total":16*32})
        self.result=dict(config=dict(frontend_ratio=1,channels=16,pim_data_path="attacc",pim_rank_controller=True),
            resolved_timings_ck=dict(nBL_min=2,nCL=3,nCWL=2),memory_cycles=40,latency_ns=40000,tCK_ns=1000,transaction_bytes=32)
        self.dump(self.case/"manifest.json",self.manifest)
        self.dump(self.case/"simulation/result.json",self.result)
        self.csv(self.case/"simulation/barriers.csv",("marker","frontend_tick"),
            [dict(marker=i,frontend_tick=t) for i,t in enumerate((1,9,11,21,31,35))])
        for ch in range(16):
            self.csv(self.case/f"simulation/timeline.csv.ch{ch}",("clock","command","Channel"),[dict(clock=23,command="RDRES",Channel=ch)])
            self.csv(self.case/f"simulation/rank_ca_events.csv.ch{ch}",("clock","ready_at","Channel","Rank","command","scope"),[
                ResourceTests.ca(1,2,channel=ch,command="PIM_DISPATCH"),
                ResourceTests.ca(4,5,"rank",channel=ch,command="MACAB"),
                ResourceTests.ca(12,13,"rank",channel=ch,command="LOCALRD"),
                ResourceTests.ca(23,24,channel=ch,command="RDRES")])

    @staticmethod
    def dump(path,value):
        path.write_text(json.dumps(value),encoding="utf-8")

    @staticmethod
    def csv(path,fields,rows):
        with path.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)

    def remove_ca(self):
        for ch in range(16):
            (self.case/f"simulation/rank_ca_events.csv.ch{ch}").unlink()

    def contract(self):
        self.result["config"]["pim_rank_controller"]=False
        self.dump(self.case/"simulation/result.json",self.result)
        source=self.root/"historical-source.txt"
        source.write_text("Verified fixture source: shared CA RDRES=1CK; LOCALRD/LOCALWR internal.",encoding="utf-8")
        contract=dict(model="legacy_flat_shared_ca",source_root=".",source_sha256={source.name:C.sha256(source)},
            source_evidence="Independent fixture source contract",command_cycles_ck={"RDRES":1},no_ca_commands=["LOCALRD","LOCALWR"],
            cases={"pv":dict(manifest_sha256=C.sha256(self.case/"manifest.json"),result_sha256=C.sha256(self.case/"simulation/result.json"),
                barriers_sha256=C.sha256(self.case/"simulation/barriers.csv"),
                timeline_sha256={f"timeline.csv.ch{ch}":C.sha256(self.case/f"simulation/timeline.csv.ch{ch}") for ch in range(16)})})
        path=self.root/"contract.json"
        self.dump(path,contract)
        return path

    def test_recorded_primary_differs_from_ca_sliced_diagnostic(self):
        result=C.analyze_case(self.case,"staged")
        self.assertEqual(len(result["records"]),51)
        rows={r["resource"]:r for r in result["records"] if r["channel"]==0}
        self.assertEqual(rows["dq"]["longest_complete_work_window_us"],20)
        self.assertEqual(rows["global_ca"]["longest_complete_work_window_us"],10)
        self.assertEqual(rows["dq_global_ca"]["longest_complete_work_window_us"],10)
        self.assertEqual(rows["dq_global_ca"]["diagnostic_joint_longest_us"],19)
        self.assertEqual(rows["dq_global_ca"]["resource_busy_us"],3)
        self.assertEqual(result["rank_ca_accounting"][0]["rank1_ca_busy_us"],2)
        self.assertEqual(result["ca_provenance"]["status"],"recorded")

    def test_missing_ca_is_unavailable_not_zero(self):
        self.remove_ca()
        report=C.analyze_case(self.case,"missing")
        ca=[r for r in report["records"] if r["resource"]!="dq"]
        self.assertEqual(len(ca),34)
        self.assertTrue(all(not r["available"] and r["longest_complete_work_window_us"] is None for r in ca))
        self.assertTrue(all(r["available"] for r in report["records"] if r["resource"]=="dq"))

    def test_partial_ca_logs_fail_instead_of_inventing_other_channels(self):
        (self.case/"simulation/rank_ca_events.csv.ch15").unlink()
        with self.assertRaisesRegex(ValueError,"Partial CA"):
            C.analyze_case(self.case,"missing_one")

    def test_disabled_controller_logs_are_not_misread_as_free_ca(self):
        self.result["config"]["pim_rank_controller"]=False
        self.dump(self.case/"simulation/result.json",self.result)
        with self.assertRaisesRegex(ValueError,"Flat CA trace contains private rank commands"):
            C.analyze_case(self.case,"disabled")

    def test_flat_global_logs_are_counted_without_a_rank_controller(self):
        self.result['config']['pim_rank_controller']=False
        self.dump(self.case/'simulation/result.json',self.result)
        for ch in range(16):
            self.csv(self.case/f'simulation/rank_ca_events.csv.ch{ch}',
                ('clock','ready_at','Channel','Rank','command','scope'),
                [ResourceTests.ca(23,24,channel=ch,command='RDRES')])
        report=C.analyze_case(self.case,'flat')
        self.assertEqual(report['ca_provenance']['model'],'flat_shared_ca')
        self.assertEqual(report['rank_ca_accounting'],[])
        row=next(r for r in report['records'] if r['channel']==0 and r['resource']=='global_ca')
        self.assertEqual(row['resource_busy_us'],1)

    def test_verified_portable_legacy_contract_rebuilds_actual_trace(self):
        self.remove_ca()
        report=C.analyze_case(self.case,"direct",self.contract())
        self.assertEqual(report["ca_provenance"]["status"],"reconstructed_verified")
        joint=next(r for r in report["records"] if r["channel"]==0 and r["resource"]=="dq_global_ca")
        self.assertEqual(joint["longest_complete_work_window_us"],20)
        self.assertEqual(report["rank_ca_accounting"],[])

    def test_legacy_source_tampering_is_rejected(self):
        self.remove_ca();contract=self.contract()
        (self.root/"historical-source.txt").write_text("changed",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"source hash mismatch"):
            C.analyze_case(self.case,"direct",contract)

    def test_legacy_raw_tampering_is_rejected(self):
        self.remove_ca();contract=self.contract()
        path=self.case/"simulation/timeline.csv.ch0"
        path.write_text(path.read_text()+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"timeline hash mismatch"):
            C.analyze_case(self.case,"direct",contract)

    def test_hierarchical_missing_logs_cannot_use_flat_contract(self):
        self.remove_ca();contract=self.contract()
        self.result["config"]["pim_rank_controller"]=True
        self.dump(self.case/"simulation/result.json",self.result)
        with self.assertRaisesRegex(ValueError,"hierarchical controller"):
            C.analyze_case(self.case,"bad",contract)

    def test_all_channel_union_preserves_asymmetry_and_issues(self):
        path=self.case/"simulation/rank_ca_events.csv.ch1"
        rows=C.read_csv(path)
        rows.append(ResourceTests.ca(14,15,channel=1,command="PIM_DISPATCH"))
        self.csv(path,("clock","ready_at","Channel","Rank","command","scope"),rows)
        report=C.analyze_case(self.case,"asymmetric")
        rows={r["channel"]:r for r in report["records"] if r["resource"]=="dq_global_ca"}
        self.assertEqual(rows[0]["longest_complete_work_window_us"],10)
        self.assertEqual(rows[1]["longest_complete_work_window_us"],0)
        self.assertEqual(rows["all_channels"]["longest_complete_work_window_us"],0)

    def test_output_guard_and_gzip_records(self):
        with self.assertRaisesRegex(ValueError,"outside every immutable"):
            C.write_report([dict(flow="fixture",directory=self.case)],self.case/"derived")
        output=self.root/"external-derived"
        result=C.write_report([dict(flow="fixture",directory=self.case)],output)
        self.assertEqual(len(result["records"]),51)
        self.assertEqual(len(result["evidence"]),1)
        self.assertNotIn("physical_windows",result)
        with gzip.open(output/result["files"]["physical_windows"],"rt",encoding="utf-8",newline="") as stream:
            windows=list(csv.DictReader(stream))
        self.assertEqual({w["resource"] for w in windows},set(C.RESOURCES))
        self.assertEqual({w["channel"] for w in windows},{str(i) for i in range(16)}|{"all_channels"})


if __name__=="__main__":
    unittest.main()

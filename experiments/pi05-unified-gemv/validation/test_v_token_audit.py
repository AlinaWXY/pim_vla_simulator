"""Token ownership, full Wv replicas, cache permutation and SRF fault cases."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source"))
import workload
import v_token_layout as producer
from audit_native import audit_trace, audit_slot_events, trace_records
from v_token_audit import audit_v_token_mapping, output_coordinates, physical_address
from test_audit import event


class TokenVTests(unittest.TestCase):
    def test_token_X_coverage_is_complete_without_pair_duplication(self):
        for channel in range(16):
            actual = Counter()
            for local in range(8):
                for fragment in range(64):
                    for lane in range(16):
                        item = producer.input_coordinates(channel, local, fragment, lane)
                        self.assertEqual(item["tensor"], "X")
                        actual[item["token"],item["feature"]] += 1
            self.assertEqual(actual, Counter((t,k) for t in range(channel%2,16,2) for k in range(1024)))

    def test_weight_copies_cover_every_BF16_cell_and_all_features(self):
        # Exhaustive two-rank replicas for one head; other head ownership is
        # sampled independently at every feature and both reduction endpoints.
        cells = defaultdict(set)
        for parity in range(2):
            for feature in range(256):
                expected_bank = feature%128//8
                expected_lane = 2*(feature%8)+feature//128
                for reduction in range(1024):
                    owner = producer.weight_location(0, parity, reduction, feature)
                    self.assertEqual(owner[:4], (parity,1,expected_bank//4,expected_bank%4))
                    offset = 131072+(reduction//16*16+expected_lane)*32+reduction%16*2
                    self.assertEqual(owner[4], offset)
                    cells[parity,expected_bank].add(owner[4])
        for offsets in cells.values():
            self.assertEqual(offsets, set(range(131072,163840,2)))
        self.assertEqual(len(cells),32)
        for head in range(1,8):
            for parity in range(2):
                for feature in range(256):
                    for reduction in (0,1023):
                        value = producer.weight_location(head,parity,reduction,feature)
                        base = producer.weight_location(0,parity,reduction,feature)
                        self.assertEqual(value,(2*head+parity,*base[1:]))

    def test_export_scatter_preserves_original_cache_prefix_and_padding(self):
        actual, source_bytes = {}, set()
        prefix = {(h,t,d):(h,t,d) for h in range(8) for t in range(960,968) for d in range(256)}
        cache = dict(prefix)
        for item in producer.iter_cache_scatter():
            src,dst = item["source"],item["destination"]
            h,t,d = src["head"],src["token"],src["feature"]
            expected = output_coordinates(h,t,d)
            self.assertEqual(src["trace_address"],expected["trace_address"])
            self.assertEqual(src["result_offset_bytes"],expected["result_offset_bytes"])
            self.assertEqual(dst["physical_address"],expected["cache_address"])
            self.assertEqual(dst["token"],968+t)
            actual[h,t,d]=dst["physical_address"]
            source_bytes.add((src["channel"],src["result_offset_bytes"]))
            cache[h,dst["token"],d]=(h,t,d)
        self.assertEqual(len(actual),32768)
        self.assertEqual(len(set(actual.values())),32768)
        self.assertEqual(source_bytes,{(c,o) for c in range(16) for o in range(0,8192,4)})
        for h in range(8):
            for t in range(984,992):
                for d in range(256): cache[h,t,d]=0
        self.assertEqual({k:cache[k] for k in prefix},prefix)
        self.assertTrue(all(cache[h,968+t,d]==(h,t,d) for h in range(8) for t in range(16) for d in range(256)))
        self.assertTrue(all(cache[h,t,d]==0 for h in range(8) for t in range(984,992) for d in range(256)))

    def test_new_V_and_serial_trace_match_independent_geometry_and_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            for case,banks in (("v",1),("v",2),("serial_layer_control",2)):
                path=Path(temp)/f"{case}{banks}.trace"
                manifest=workload.generate(path,case,staging_banks=banks,v_token_sharding=True)
                records=trace_records(path)
                audit_trace(records,case,staging_banks=banks,v_token_sharding=True)
                result=audit_v_token_mapping(manifest,records)
                self.assertEqual(result["logical_input_elements_checked"],131072)
                self.assertEqual(result["export_bytes"],131072)

    def test_mapping_faults_fail_even_when_trace_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"v.trace"
            original=workload.generate(path,"v",v_token_sharding=True)
            records=trace_records(path)
            for fault,pattern in (("tokens","token parity"),("features","projection feature"),("export","export"),("prefix","prefix")):
                manifest=deepcopy(original); contract=manifest["v_token_mapping"]
                if fault=="tokens":contract["input"]["rank_tokens"][1]["logical_tokens"][0]=0
                elif fault=="features":contract["feature_lanes_by_bank"][0][1]=1
                elif fault=="export":contract["trace_ranges"]["export_packets"]["line_start"]+=1
                else:contract["cache_scatter"]["preserve_prefix"]=[960,967]
                with self.subTest(fault=fault),self.assertRaisesRegex(AssertionError,pattern):
                    audit_v_token_mapping(manifest,records)

    def test_PV_trace_remains_byte_identical_under_new_V_option(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=[Path(temp)/f"pv{i}.trace" for i in range(2)]
            for path,flag in zip(paths,(False,True)):
                workload.generate(path,"pv",staging_banks=2,v_token_sharding=flag)
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())

    def fragment(self, rows=8, columns=16):
        fills=[event("GEMV_MULTICAST",1+2*t,duration=0,tokens=1<<t) for t in range(rows)]
        macs=[event("GEMV_MAC",40+8*c,duration=8,tokens=255,columns=c+1) for c in range(columns)]
        return fills+macs+[event("GEMV_RELEASE",40+8*columns,duration=0,tokens=255,columns=columns)]

    def test_new_V_requires_sixteen_columns_before_release(self):
        audit_slot_events(self.fragment(),"v",v_token_sharding=True)
        with self.assertRaisesRegex(AssertionError,"incomplete input/weight"):
            audit_slot_events(self.fragment(columns=8),"v",v_token_sharding=True)

    def test_new_V_rejects_sixteen_input_rows_and_missing_eighth_row(self):
        with self.assertRaisesRegex(AssertionError,"extra token rows"):
            audit_slot_events(self.fragment(rows=16),"v",v_token_sharding=True)
        with self.assertRaisesRegex(AssertionError,"missing bank/token/lane"):
            audit_slot_events(self.fragment(rows=7),"v",v_token_sharding=True)


if __name__ == "__main__":
    unittest.main()

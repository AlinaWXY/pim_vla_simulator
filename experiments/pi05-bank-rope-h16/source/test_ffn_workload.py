"""Single-active-rank FFN coverage, buffer and traffic contracts."""

from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import ffn_layout as L
from ffn_workload import generate


def read_trace(path):
    return [(f[0], int(f[1]), int(f[2]) if len(f) == 3 else None)
            for f in (line.split() for line in Path(path).read_text().splitlines())]


class FFNLayoutTests(unittest.TestCase):
    def test_active_rank_dimensions_buffers_and_nonoverlap(self):
        layout = L.build_layout()
        self.assertEqual((L.TOTAL_RANKS,L.TOTAL_BANKS,L.LOCAL_INTERMEDIATE), (16,256,256))
        gu, down = layout["regions"]["gate_up"], layout["regions"]["down_tiled"]
        self.assertEqual((gu["base"],gu["bytes_per_bank"]), (262144,65536))
        self.assertEqual((down["base"],down["bytes_per_bank"]), (327680,32768))
        self.assertEqual(gu["base"]+gu["bytes_per_bank"], down["base"])
        self.assertEqual((gu["accumulator_bytes_per_bank"],down["accumulator_bytes_per_bank"]), (2048,1024))
        self.assertEqual(down["rank_local_z_sram_bytes"], 8192)
        self.assertEqual(down["partial_result_fp32_bytes"], 1024**2)
        self.assertEqual(layout["ffn_weight_bytes"], 24*1024**2)

    def test_configurable_attention_reservation_only_translates_offsets(self):
        reserve = 128*1024
        alt = L.build_layout(reserve)
        self.assertEqual(alt["regions"]["gate_up"]["base"], reserve)
        self.assertEqual(alt["regions"]["down_tiled"]["base"], reserve+65536)
        for op,k,n in (("gate_up",31,5000),("down_tiled",4095,1023)):
            default = L.weight_location(op,k,n)
            shifted = L.weight_location(op,k,n,reserve)
            self.assertEqual(default[:4],shifted[:4])
            self.assertEqual(default[-1]-shifted[-1],128*1024)
        with self.assertRaises(ValueError):
            L.build_layout(123)

    def test_GU_all_features_have_exactly_one_active_rank_owner(self):
        owners = Counter()
        for output in range(4096):
            for projection in ("gate","up"):
                for reduction in (0,15,16,511,1023):
                    loc = L.gate_up_location(projection,reduction,output)
                    self.assertEqual(loc[1],1)
                    self.assertEqual(loc[0],output//256)
                    bank = loc[2]*4+loc[3]
                    self.assertEqual(bank,output%16)
                    col,input_lane = divmod((loc[-1]-L.GU_BASE)//2,16)
                    fragment,slot_projection = divmod(col,32)
                    slot,p = divmod(slot_projection,2)
                    self.assertEqual((fragment,input_lane,slot,p),
                        (reduction//16,reduction%16,(output%256)//16,("gate","up").index(projection)))
                    self.assertEqual(L.decode_address(L.physical_address(*loc)),loc)
            owners[L.gate_up_location("gate",0,output)[:4]] += 1
        self.assertEqual(len(owners),256)
        self.assertEqual(set(owners.values()),{16})

    def test_Down_full_and_tiled_reconstruct_same_unique_weight(self):
        for reduction in range(4096):
            for output in (0,15,16,255,256,511,768,1023):
                for tiled in (False,True):
                    loc = L.down_location(reduction,output,tiled)
                    self.assertEqual(loc[:4],L.rank_bank_coords(reduction//256,output%16))
                    col,lane = divmod((loc[-1]-L.DOWN_BASE)//2,16)
                    if tiled:
                        tile_fragment,slot = divmod(col,16)
                        tile,fragment = divmod(tile_fragment,16)
                        reconstructed_output = tile*256+slot*16+output%16
                    else:
                        fragment,slot = divmod(col,64)
                        reconstructed_output = slot*16+output%16
                    self.assertEqual(fragment*16+lane,reduction%256)
                    self.assertEqual(reconstructed_output,output)

    def test_sixteen_rank_partials_sum_to_full_Down_dot(self):
        for query in (0,15):
            for output in (0,257,1023):
                z = lambda k: (query+1)*(k%13-6)
                weight = lambda k: output%17-k%7
                expected = sum(z(k)*weight(k) for k in range(4096))
                partials = []
                for rank in range(16):
                    terms = range(rank*256,(rank+1)*256)
                    self.assertTrue(all(L.down_location(k,output,True)[0] == rank for k in terms))
                    partials.append(sum(z(k)*weight(k) for k in terms))
                self.assertEqual(sum(partials),expected)


class FFNWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cases = {}
        for case in ("gate_up","down_tiled","down_full"):
            path = Path(cls.temp.name)/(case+".trace")
            cls.cases[case] = generate(path,case),read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_payload_counts_and_expectedDQ(self):
        expected = {
            "gate_up":dict(PIM_WR_GB=16384,PIM_MAC_AB=32768),
            "down_tiled":dict(PIM_LOCAL_GB=16384,PIM_MAC_AB=16384,PIM_RD_RES=32768),
            "down_full":dict(PIM_LOCAL_GB=4096,PIM_MAC_AB=16384,PIM_RD_RES=32768),
        }
        for case,(manifest,trace) in self.cases.items():
            counts = Counter(op for op,_,_ in trace)
            self.assertEqual(dict(counts),manifest["counts"])
            self.assertEqual({op:n for op,n in counts.items() if op not in ("BARRIER","WAIT")},expected[case])
            dq = manifest["expected_dq_bytes"]
            self.assertEqual(dq["total"],512*1024 if case == "gate_up" else 1024**2)
            self.assertEqual((dq["dram_read"],dq["dram_write"]),(0,0))
            self.assertEqual(manifest["local_input_external_dq_bytes"],0)
            self.assertEqual(manifest["required_backend"],dict(
                pim_rank_rope=False,pim_input_broadcast=False,pim_mac_scope="rank",nCCDAB=16))

    def test_every_request_targets_rank1_and_X_is_shared_for_GU(self):
        for manifest,trace in self.cases.values():
            for op,address,_ in trace:
                if op not in ("BARRIER","WAIT"):
                    self.assertEqual(L.decode_address(address)[1],1)
                    self.assertEqual(L.decode_address(address)[2:4],(0,0))
            for record in manifest["input_fragments"]:
                inputs = trace[record["line_start"]-1:record["line_end"]]
                self.assertEqual(len(inputs),256)
                self.assertEqual(Counter((L.decode_address(a)[0],L.decode_address(a)[-1]//32)
                                         for _,a,_ in inputs),Counter((c,q) for c in range(16) for q in range(16)))
                self.assertEqual({b for _,_,b in inputs},{record["fragment"]%2})
        self.assertEqual(len(self.cases["gate_up"][0]["input_fragments"]),64)

    def test_each_weight_column_is_scanned_once_per_channel(self):
        for case,(manifest,trace) in self.cases.items():
            per_channel = defaultdict(list)
            for op,address,_ in trace:
                if op == "PIM_MAC_AB":
                    c,_,_,_,off = L.decode_address(address)
                    per_channel[c].append(off)
            region = manifest["matrix"]
            expected = list(range(region["base"],region["base"]+region["bytes_per_bank"],32))
            for channel in range(16):
                self.assertEqual(per_channel[channel],expected)

    def test_GU_gather_and_nonlinear_are_not_conflated(self):
        manifest,trace = self.cases["gate_up"]
        gather = next(s for s in manifest["segments"] if s["kind"] == "local_gather")
        self.assertEqual(gather["bytes_per_rank"],32768)
        self.assertEqual(gather["bytes_system"],512*1024)
        self.assertEqual(gather["wait_cycles"],2048)
        self.assertTrue(gather["analytical_wait"])
        self.assertIn(("WAIT",2048,None),trace)
        self.assertIsNone(manifest["nonlinear_latency_ck"])
        self.assertFalse(manifest["integrated_chain"])
        self.assertEqual(manifest["accumulator_bytes_per_bank"],2048)

    def test_Down_output_tiles_cover_full_rank_partial_once(self):
        manifest,trace = self.cases["down_tiled"]
        self.assertEqual(len(manifest["exports"]),4)
        for tile,record in enumerate(manifest["exports"]):
            self.assertEqual((record["output_start"],record["output_count_per_rank"]),(tile*256,256))
            self.assertEqual(record["bytes_fp32"],256*1024)
            exports = trace[record["line_start"]-1:record["line_end"]]
            for channel in range(16):
                offsets = [L.decode_address(a)[-1] for op,a,_ in exports
                           if op == "PIM_RD_RES" and L.decode_address(a)[0] == channel]
                self.assertEqual(offsets,list(range(0,16*256*4,32)))
        self.assertEqual(manifest["gpu_partial_count"],16)
        self.assertEqual(manifest["rank_local_z_sram_bytes"],8192)
        self.assertEqual(manifest["local_input_bytes"],512*1024)
        self.assertIsNone(manifest["gpu_partial_reduction_latency_ck"])

    def test_legacy_compute_and_export_tails_are_explicit(self):
        for manifest,trace in self.cases.values():
            for segment in manifest["segments"]:
                body = trace[segment["line_start"]-1:segment["line_end"]]
                if segment["kind"] == "pipeline":
                    self.assertEqual(body[-3][0],"BARRIER")
                    self.assertEqual(body[-2],("WAIT",16,None))
                if segment["kind"] == "export":
                    self.assertEqual(body[-3][0],"BARRIER")
                    self.assertEqual(body[-2],("WAIT",4,None))
            markers = [a for op,a,_ in trace if op == "BARRIER"]
            self.assertEqual(markers,list(range(len(markers))))
            for index,segment in enumerate(manifest["segments"]):
                self.assertEqual(segment["start"],0 if index == 0 else manifest["segments"][index-1]["end"])


if __name__ == "__main__":
    unittest.main()

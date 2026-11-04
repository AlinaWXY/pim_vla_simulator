"""Independent coverage and producer/consumer checks for the four-way study."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"source"))
import horizon_attention_layout as L
import o_projection_layout as O
import pv_token_layout as P
import score_dram_layout as S
import workload

COMMON=dict(rank_staging=True,staging_banks=2,rank_descriptors=True,
    channel_independent_dispatch=True,v_token_sharding=True,pv_token_sharding=True,qkv_fusion=True)


def records(path):
    return [line.split() for line in path.read_text(encoding="ascii").splitlines()]


class OutputProjectionWorkloadTests(unittest.TestCase):
    def test_legacy_trace_hashes_still_match_frozen_native_inputs(self):
        with tempfile.TemporaryDirectory(prefix="op-test-",dir=HERE) as tmp:
            for case in workload.CASES:
                with self.subTest(case=case):
                    path=Path(tmp)/f"{case}.trace"
                    workload.generate(path,case,**COMMON)
                    expected=json.loads((HERE.parents[1]/"reference"/case/"summary.json").read_text())["trace_sha256"]
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)

    def test_scores_cover_all_queries_tokens_once_and_only_after_spill(self):
        cells=set(); addresses=set(); bank_counts=Counter()
        for channel in range(16):
            for tile in range(31):
                for packet in range(32):
                    location=S.packet_location(channel,tile,packet)
                    address=L.physical_address(*location)
                    self.assertNotIn(address,addresses);addresses.add(address)
                    bank_counts[location[2]*4+location[3]]+=1
                    self.assertTrue(655360<=location[4]<655360+16384)
                    for element in S.packet_elements(channel,tile,packet):
                        label=(element["head"],element["query"],element["token"])
                        self.assertNotIn(label,cells);cells.add(label)
        self.assertEqual(len(cells),8*16*992)
        self.assertEqual(bank_counts,{0:16*31*16,4:16*31*16})
        with tempfile.TemporaryDirectory(prefix="op-test-",dir=HERE) as tmp:
            path=Path(tmp)/"scores.trace"
            m=workload.generate(path,"chain",**COMMON,score_dram_spill=True,pv_bf16_output=True)
            rows=records(path)
            writes=[i for i,r in enumerate(rows) if r[0]=="PIM_SCORE_SPILL"]
            reads=[i for i,r in enumerate(rows) if r[0]=="LD"]
            self.assertEqual(len(writes),31*16)
            self.assertEqual(len(reads),31*32*16)
            self.assertLess(max(writes),min(reads))
            self.assertFalse(any(r[0]=="PIM_RD_BG" for r in rows))
            self.assertEqual({int(rows[i][1]) for i in reads},addresses)
            qk=[s for s in m["segments"] if s["name"].startswith("qk_tile_")]
            self.assertEqual([s["kind"] for s in qk],["qk","local_write"]*31)
            self.assertEqual(m["expected_dq_bytes"]["score_pim_to_gpu"],8*16*992*4)

    def test_two_bank_PV_spill_matches_every_source_feature_and_query(self):
        for channel in range(16):
            destinations=set(); source_cells=set();banks=Counter()
            for fragment in range(16):
                for query in range(16):
                    packet=O.input_packet(channel,fragment,query)
                    dst=packet["destination"]
                    self.assertEqual(dst[0:2],(channel,1))
                    self.assertNotIn(packet["destination_address"],destinations)
                    destinations.add(packet["destination_address"]);banks[dst[2]*4+dst[3]]+=1
                    self.assertTrue(524288<=dst[4]<524288+4096)
                    for e in packet["elements"]:
                        owner,lane=L.feature_owner(channel//2,channel%2,e["feature"])
                        self.assertEqual(owner[2]*4+owner[3],packet["source_flat_bank"])
                        self.assertEqual(e["source_accumulator_index"],query*16+lane)
                        source_cells.add((query,e["feature"]))
            self.assertEqual(len(source_cells),16*256)
            self.assertEqual(banks,{0:128,4:128})

    def test_Wo_has_unique_complete_cells_and_correct_feature_reduction(self):
        # An independent feature-labelled dot product detects a permuted input
        # fragment paired with the wrong weight rows, even when shapes agree.
        all_inputs=[O.input_feature(f,l) for f in range(16) for l in range(16)]
        self.assertEqual(sorted(all_inputs),list(range(256)))
        weights=set()
        for feature in range(256):
            for output in range(1024):
                location=O.weight_location(3,1,feature,output)
                self.assertEqual(location[0:2],(7,1))
                self.assertNotIn(location,weights);weights.add(location)
                self.assertEqual(O.decode_weight(location),(3,1,feature,output))
        self.assertEqual(len(weights),256*1024)
        for output in (0,31,32,511,512,1023):
            expected=sum((f%11-5)*((f*17+output*3)%13-6) for f in range(256))
            observed=0
            for fragment in range(16):
                for lane in range(16):
                    feature=O.input_feature(fragment,lane)
                    location=O.weight_location(0,0,feature,output)
                    relative=location[-1]-786432
                    # Actual DRAM lane must select this SRF lane.
                    self.assertEqual(relative%32//2,lane)
                    self.assertEqual((relative%16384)//(32*32),fragment)
                    observed+=(feature%11-5)*((feature*17+output*3)%13-6)
            self.assertEqual(observed,expected)

    def test_BF16_packet_labels_and_O_output_capacity(self):
        pv_cells=set()
        for offset in range(0,8192,32):
            packet=P.result_packet(5,0,offset,16,"BF16")
            self.assertEqual(len(packet),16)
            self.assertEqual([e["byte_in_packet"] for e in packet],list(range(0,32,2)))
            for e in packet:pv_cells.add((e["query"],e["feature"]))
        self.assertEqual(len(pv_cells),16*256)
        output_cells=set()
        for tile in range(2):
            positions=set()
            for query in range(16):
                for output in range(tile*512,(tile+1)*512):
                    e=O.result_element(15,tile,query,output)
                    self.assertLess(e["accumulator_index"]*4,2048)
                    self.assertLess(e["result_offset_bytes"],16384)
                    self.assertNotIn(e["result_offset_bytes"],positions)
                    positions.add(e["result_offset_bytes"]);output_cells.add((query,output))
            self.assertEqual(len(positions),16*512)
        self.assertEqual(len(output_cells),16*1024)

    def test_four_way_traces_have_independent_switches_and_correct_DQ(self):
        with tempfile.TemporaryDirectory(prefix="op-test-",dir=HERE) as tmp:
            for score,projection in ((False,False),(True,False),(False,True),(True,True)):
                with self.subTest(score=score,projection=projection):
                    path=Path(tmp)/f"{score}-{projection}.trace"
                    m=workload.generate(path,"serial_layer_control",**COMMON,
                        score_dram_spill=score,o_projection=projection,pv_bf16_output=True)
                    rows=records(path);counts=Counter(r[0] for r in rows)
                    self.assertEqual(counts["PIM_SCORE_SPILL"],496 if score else 0)
                    self.assertEqual(counts["PIM_PV_SPILL"],4096 if projection else 0)
                    self.assertEqual(counts["PIM_O_REWIND"],16 if projection else 0)
                    self.assertEqual(m["expected_dq_bytes"]["total"],4890624 if projection else 4497408)
                    actual=sum(counts[op]*32 for op in ("LD","ST","PIM_WR_GB","PIM_RD_RES","PIM_RD_BG"))
                    actual+=(m["expanded_counts"].get("PIM_Q_EXPORT",0)+m["expanded_counts"].get("PIM_Q_REMOTE",0))*32
                    self.assertEqual(actual,m["expected_dq_bytes"]["total"])
                    if projection:
                        phases={s["name"]:s for s in m["segments"]}
                        self.assertLess(phases["pv/pv_result_dram_spill"]["end"],
                                        phases["pv/o_projection_tile_0_compute"]["end"])
                        self.assertLess(phases["pv/o_projection_tile_0_export"]["end"],
                                        phases["pv/o_projection_input_rewind"]["end"])
                        self.assertNotIn("pv/pv_result_export",phases)
                        o_mac=[r for r in rows if r[0]=="PIM_MAC_AB" and
                               786432<=L.decode_address(int(r[1]))[-1]<819200]
                        self.assertEqual(len(o_mac),16*2*16*32)
                        self.assertEqual(m["o_projection_mapping"]["output"]["total_bytes"],524288)

    def test_unsupported_precision_and_broadcast_combinations_reject(self):
        with tempfile.TemporaryDirectory(prefix="op-test-",dir=HERE) as tmp:
            path=Path(tmp)/"invalid.trace"
            with self.assertRaisesRegex(ValueError,"BF16 PV"):
                workload.generate(path,"pv",**COMMON,o_projection=True)
            bad=dict(COMMON,staging_banks=1)
            with self.assertRaisesRegex(ValueError,"two-bank"):
                workload.generate(path,"pv",**bad,o_projection=True,pv_bf16_output=True)


if __name__=="__main__":
    unittest.main()

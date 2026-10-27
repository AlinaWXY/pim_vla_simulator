"""Independent ownership, shape, routing and byte-conservation checks."""

from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import attention_layout as L
from attention_workload import generate


def read_trace(path):
    return [(fields[0], int(fields[1]), int(fields[2]) if len(fields) == 3 else None)
            for fields in (line.split() for line in Path(path).read_text().splitlines())]


class BankRopeLayoutTests(unittest.TestCase):
    def test_shape_topology_and_balanced_token_parity(self):
        self.assertEqual((L.ACTIVE_RANKS, L.ACTIVE_BANKS, L.TOTAL_BANKS), (16, 256, 512))
        for head in range(8):
            prefix = Counter(L.token_owner(head, t)[0] for t in range(968))
            suffix = Counter(L.token_owner(head, t)[0] for t in range(968, 984))
            padding = Counter(L.token_owner(head, t)[0] for t in range(984, 992))
            self.assertEqual(set(prefix.values()), {484})
            self.assertEqual(set(suffix.values()), {8})
            self.assertEqual(set(padding.values()), {4})
            self.assertEqual(set(prefix), {(2*head, 1), (2*head+1, 1)})

    def test_pair_major_features_cover_head_once_per_rank(self):
        for channel_in_head in range(2):
            seen = []
            for b in range(16):
                features = [L.feature_from_lane(b, lane) for lane in range(16)]
                for pair in range(8):
                    self.assertEqual(features[2*pair+1]-features[2*pair], 128)
                for lane, feature in enumerate(features):
                    owner, got_lane = L.feature_owner(0, channel_in_head, feature)
                    self.assertEqual(owner, (channel_in_head, 1, b//4, b%4))
                    self.assertEqual(got_lane, lane)
                seen.extend(features)
            self.assertEqual(sorted(seen), list(range(256)))

    def test_weight_copies_and_input_vs_output_lane_axes(self):
        for projection in ("q", "k"):
            for feature in range(256):
                for reduction in (0, 15, 16, 511, 1023):
                    a = L.projection_weight_location(projection, 3, 0, reduction, feature)
                    b = L.projection_weight_location(projection, 3, 1, reduction, feature)
                    self.assertEqual((a[0], b[0]), (6, 7))
                    self.assertEqual(a[1:], b[1:])
                    col, input_lane = divmod(a[-1]//2, 16)
                    fragment, p_output = divmod(col, 32)
                    p, output_lane = divmod(p_output, 16)
                    self.assertEqual((fragment, p, input_lane),
                                     (reduction//16, ("q", "k").index(projection), reduction%16))
                    self.assertEqual(L.feature_from_lane(a[2]*4+a[3], output_lane), feature)
                    self.assertEqual(L.decode_address(L.physical_address(*a)), a)
        self.assertEqual(L.manifest()["qk_weight_system_bytes"], 16*1024**2)

    def test_projection_results_directly_target_local_pair_slots(self):
        for q in range(16):
            for d in range(256):
                owner, slot = L.projection_owner(2, q, d)
                for p in ("q", "k"):
                    loc = L.projection_spill_location(p, 2, q, d)
                    self.assertEqual(loc[:4], owner)
                    self.assertEqual((loc[-1] % 32)//2, slot % 16)
                    self.assertEqual(loc[0] % 2, q % 2)

    def test_single_work_row_and_no_remote_Q_DRAM_allocation(self):
        ranges = list(L.WORK_COLUMNS.values())
        cols = [c for first, stop in ranges for c in range(first, stop)]
        self.assertEqual(cols, list(range(28)))
        self.assertEqual(L.REGIONS["work"]["used_bytes_per_bank"], 896)
        self.assertEqual(L.REGIONS["work"]["reserved_bytes_per_bank"], 2048)
        self.assertEqual(L.REGIONS["w_qk"]["reserved_bytes_per_bank"], 65536)
        for q in range(16):
            for d in range(256):
                qloc = L.query_location(0, q, d)
                kloc = L.key_location(0, 968+q, d)
                cloc = L.coefficient_location(0, q, d % 128, "cos")
                self.assertEqual(qloc[:4], kloc[:4])
                self.assertEqual(qloc[:4], cloc[:4])
                self.assertEqual({qloc[-1]//2048, kloc[-1]//2048, cloc[-1]//2048}, {40})

    def test_K_two_segments_have_no_new_attention_tokens(self):
        offsets = [L.qk_column_offset(t) for t in range(496)]
        self.assertEqual(len(set(offsets)), 496)
        self.assertEqual(offsets[:484], list(range(65536, 65536+484*32, 32)))
        self.assertEqual(offsets[484:], list(range(81920+16*32, 81920+28*32, 32)))
        self.assertEqual(len({off//2048 for off in offsets}), 9)
        self.assertNotIn(65536+484*32, offsets)  # unused address gap is not a K token
        for token in range(992):
            self.assertEqual(L.key_location(0, token, 0)[-1], offsets[token//2])

    def test_Q_forward_same_feature_bank_and_query_slot(self):
        slots = defaultdict(set)
        for q in range(16):
            for d in range(256):
                source = L.query_location(0, q, d)
                dest, slot = L.q_forward_destination(0, q, d)
                self.assertEqual(dest, (source[0]^1, *source[1:4]))
                self.assertEqual(slot//32, q)
                self.assertEqual((slot%32)//2, (source[-1]%32)//2)
                local_slot = L.query_buffer_slot(q, d)
                slots[source[:4]].add(local_slot)
                slots[dest].add(slot)
        self.assertEqual(len(slots), 32)
        self.assertEqual({len(s) for s in slots.values()}, {256})
        self.assertEqual({max(s)+2 for s in slots.values()}, {512})

    def test_coefficient_table_is_shared_by_Q_and_K(self):
        values = set()
        for q in range(16):
            for pair in range(128):
                c = L.coefficient_location(0, q, pair, "cos")
                s = L.coefficient_location(0, q, pair, "sin")
                self.assertEqual(c[:-1], s[:-1])
                self.assertEqual(s[-1]-c[-1], 2)
                values.update((c, s))
        by_bank = Counter(v[:4] for v in values)
        self.assertEqual(set(by_bank.values()), {128})
        self.assertEqual(L.manifest()["initialization_bytes"]["coefficients"], 65536)

    def test_numeric_dot_matches_sixteen_bank_two_level_reduction(self):
        qvalue = lambda q, d: (q+1)*(d%11-5)
        kvalue = lambda t, d: (t%13-6)*(d%7+1)
        for query in (0, 1, 15):
            for token in (0, 1, 967, 968, 983):
                expected = sum(qvalue(query,d)*kvalue(token,d) for d in range(256))
                bg_sums = []
                for bg in range(4):
                    parts = []
                    for bank in range(4):
                        features = [L.feature_from_lane(4*bg+bank,lane) for lane in range(16)]
                        parts.append(sum(qvalue(query,d)*kvalue(token,d) for d in features))
                    bg_sums.append(sum(parts))
                self.assertEqual(sum(bg_sums), expected)


class BankRopeWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name)/"chain.trace"
        cls.manifest = generate(path, case="chain", schedule="double")
        cls.trace = read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_trace_counts_and_880KiB_external_conservation(self):
        actual = Counter(op for op, _, _ in self.trace)
        expected = dict(PIM_WR_GB=8192, PIM_MAC_AB=32768, PIM_SPILL=4096,
                        PIM_ROPE_COEF=2048, PIM_ROPE_LOAD=4096,
                        PIM_ROPE_EXEC=4096, PIM_ROPE_STORE=4096,
                        PIM_Q_EXPORT=2048, PIM_Q_REMOTE=2048,
                        PIM_MAC_QK=7936, PIM_RD_BG=15872)
        self.assertEqual({op:n for op,n in actual.items() if op != "BARRIER"}, expected)
        self.assertEqual(dict(actual), self.manifest["counts"])
        self.assertEqual(self.manifest["expected_dq_bytes"], dict(
            x_gpu_to_pim=256*1024, q_pim_to_gpu=64*1024,
            q_gpu_to_remote_buffer=64*1024, score_pim_to_gpu=496*1024,
            total=880*1024))
        self.assertEqual(self.manifest["expected_local_bytes"]["qk_k_read"], 4063232)

    def test_no_command_uses_GPU_rank_or_remote_Q_DRAM_store(self):
        for op, address, _ in self.trace:
            if op == "BARRIER":
                continue
            self.assertEqual(L.decode_address(address)[1], 1)
            self.assertNotIn(op, ("ST", "LD", "PIM_LOCAL_WR"))
        self.assertEqual(self.manifest["q_remote_dram_writes"], 0)

    def test_projection_X_reuse_and_complete_weight_scan(self):
        self.assertEqual(len(self.manifest["input_fragments"]), 64)
        for item in self.manifest["input_fragments"]:
            x = self.trace[item["line_start"]-1:item["line_end"]]
            self.assertEqual(len(x), 128)
            self.assertEqual({b for _,_,b in x}, {item["fragment"]%2})
            self.assertEqual(Counter((L.decode_address(a)[0],L.decode_address(a)[-1]//32)
                                     for _,a,_ in x), Counter((c,t) for c in range(16) for t in range(8)))
            macs = self.trace[item["mac_line_first"]-1:item["mac_line_last"]]
            self.assertEqual(len(macs), 512)
            for channel in range(16):
                cols = [L.decode_address(a)[-1]//32 for _,a,_ in macs
                        if L.decode_address(a)[0] == channel]
                self.assertEqual(cols, list(range(item["fragment"]*32,(item["fragment"]+1)*32)))

    def test_RoPE_pairing_output_dependencies_and_no_token_global_barrier(self):
        s = next(s for s in self.manifest["segments"] if s["name"] == "rope_local")
        trace = self.trace[s["line_start"]-1:s["line_end"]-1]
        self.assertNotIn("BARRIER", [op for op,_,_ in trace])
        perbank = defaultdict(list)
        for op, address, buffer in trace:
            c,r,g,b,off = L.decode_address(address)
            perbank[c,r,g,b].append((op,(off-81920)//32,buffer))
        self.assertEqual(len(perbank), 256)
        for ops in perbank.values():
            for t in range(8):
                self.assertEqual(ops[t*7:(t+1)*7], [
                    ("PIM_ROPE_COEF",t,None), ("PIM_ROPE_LOAD",8+t,0),
                    ("PIM_ROPE_EXEC",8+t,0), ("PIM_ROPE_LOAD",16+t,1),
                    ("PIM_ROPE_EXEC",16+t,1), ("PIM_ROPE_STORE",8+t,None),
                    ("PIM_ROPE_STORE",16+t,None)])

    def test_Q_forward_has_source_completion_and_no_self_echo(self):
        reads = {a for op,a,_ in self.trace if op == "PIM_Q_EXPORT"}
        writes = {a for op,a,_ in self.trace if op == "PIM_Q_REMOTE"}
        self.assertEqual(len(reads), 2048)
        self.assertEqual(writes, {a^32 for a in reads})
        lastread = max(i for i,(op,_,_) in enumerate(self.trace) if op == "PIM_Q_EXPORT")
        firstwrite = min(i for i,(op,_,_) in enumerate(self.trace) if op == "PIM_Q_REMOTE")
        self.assertIn("BARRIER", [op for op,_,_ in self.trace[lastread+1:firstwrite]])
        self.assertLess(max(i for i,(op,_,_) in enumerate(self.trace) if op == "PIM_Q_REMOTE"),
                        min(i for i,(op,_,_) in enumerate(self.trace) if op == "PIM_MAC_QK"))

    def test_single_K_scan_complete_score_export_and_markers(self):
        macs, exports = defaultdict(list), defaultdict(list)
        for op, address, _ in self.trace:
            if op in ("PIM_MAC_QK", "PIM_RD_BG"):
                c,r,g,b,off = L.decode_address(address)
                self.assertEqual((r,g,b), (1,0,0))
                (macs if op == "PIM_MAC_QK" else exports)[c].append(off)
        for channel in range(16):
            self.assertEqual(macs[channel], [L.qk_column_offset(t) for t in range(496)])
            self.assertEqual(exports[channel], list(range(0,496*16*4,32)))
        markers = [a for op,a,_ in self.trace if op == "BARRIER"]
        self.assertEqual(markers, list(range(len(markers))))
        for i, segment in enumerate(self.manifest["segments"]):
            self.assertEqual(segment["start"], 0 if i == 0 else self.manifest["segments"][i-1]["end"])
        self.assertEqual(self.manifest["buffers"]["qk_query_per_bank"], 512)


if __name__ == "__main__":
    unittest.main()

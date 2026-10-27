"""Whole-batch layout/traffic checks and exact H16 trace regression."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
import tempfile
import unittest

import horizon_ffn as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"validation"))
import audit_horizon_ffn as A


class HorizonFFNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cases = {}
        for h in F.HORIZONS:
            for case in F.CASES:
                path = Path(cls.temp.name)/f"{h}-{case}.trace"
                manifest = F.generate(path, h, case)
                trace = [(p[0], *map(int, p[1:])) for p in (s.split() for s in path.read_text().splitlines())]
                cls.cases[h, case] = manifest, trace, path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_H16_is_byte_identical_to_frozen_workload(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"pi05-bank-rope-h16/source"))
        from ffn_dram_workload import generate
        for case in F.CASES:
            path = Path(self.temp.name)/f"frozen-{case}.trace"
            generate(path, case)
            self.assertEqual(path.read_bytes(), self.cases[16, case][2].read_bytes())

    def test_Z_address_ownership_packing_and_padding(self):
        for h in F.HORIZONS:
            seen, per_bank = set(), Counter()
            b = (h+15)//16
            for feature in range(4096):
                for token in range(h):
                    address = F.z_location(h, token, feature)
                    channel, rank, bg, bank, row, col = A.decode(address)
                    self.assertEqual((channel, rank, bg*4+bank, row), (feature//256, 1, feature%16, 176))
                    self.assertEqual(col, ((feature%256)//16)*b+token//16)
                    self.assertEqual(address%32, (token%16)*2)
                    seen.add(address)
                    per_bank[channel,bg,bank] += 2
            self.assertEqual(len(seen), h*4096)
            self.assertEqual(set(per_bank.values()), {h*32})
            info = F.layout(h)
            self.assertEqual(info["z_bytes_per_bank"], b*512)
            self.assertLessEqual(info["z_bytes_per_bank"], 2048)
            self.assertEqual(info["buffers"]["rank_input_double"], 64*h)
            self.assertEqual(info["buffers"]["GU_accumulator_per_bank"], 128*h)
            self.assertEqual(info["buffers"]["Down_tiled_accumulator_per_bank"], 64*h)

    def test_trace_addresses_match_independent_oracle_for_every_horizon(self):
        for (h, case), (manifest, trace, _) in self.cases.items():
            actual = defaultdict(Counter)
            for op, value, *buffer in trace:
                if op in ("BARRIER", "WAIT"):
                    continue
                actual[op][A.address_key(op, A.decode(value), buffer[0] if buffer else -1)] += 1
            expected = A.expected_addresses(case, h)
            self.assertEqual(dict(actual), dict(expected))
            self.assertEqual(Counter({op: sum(values.values()) for op, values in actual.items()}), A.required_counts(case, h))
            self.assertEqual(Counter(row[0] for row in trace), Counter(manifest["counts"]))

    def test_whole_H_compute_only_Z_bursts_pad(self):
        for (h, case), (manifest, trace, _) in self.cases.items():
            gu, down = case != "down_tiled_dram", case != "gate_up_z_store"
            self.assertEqual(manifest["expected_dq_bytes"], dict(pim_input_write=gu*h*32768,
                pim_result_read=down*h*65536, dram_read=0, dram_write=0, total=gu*h*32768+down*h*65536))
            self.assertEqual(manifest["weight_array_bytes"], (gu*16+down*8)*1024**2)
            self.assertEqual(manifest["token_microbatches"], 1)
            self.assertEqual(manifest["padded_compute_tokens"], h)
            self.assertEqual(manifest["counts"]["PIM_MAC_AB"], gu*32768+down*16384)
            self.assertEqual({r[1] for r in trace if r[0] == "WAIT"}, ({h,4} if down else {h}))
            self.assertFalse(manifest["numeric_compute"])
            self.assertIsNone(manifest["nonlinear_latency_ck"])

    def test_tail_port_padding_counts(self):
        # KiB independently derived from full4096 features and H valid tokens.
        expected = {16:(512,128,0,0), 20:(640,256,0,96), 30:(1024,256,64,16), 40:(1280,384,0,64)}
        for h, (gather,zwrite,gatherpad,zpad) in expected.items():
            m = self.cases[h,"combined_timing_chain"][0]
            internal = m["expected_internal_bytes"]
            self.assertEqual(internal["GU_gather_to_nonlinear"], gather*1024)
            self.assertEqual(internal["GU_gather_padding"], gatherpad*1024)
            self.assertEqual(internal["Z_dram_write"], zwrite*1024)
            self.assertEqual(internal["Z_write_padding"], zpad*1024)
            self.assertEqual(internal["Z_dram_read"], 4*zwrite*1024)
            self.assertEqual(internal["Z_read_padding"], 4*zpad*1024)

    def test_each_fragment_loads_all_token_blocks_before_MAC(self):
        for h in F.HORIZONS:
            m, trace, _ = self.cases[h,"combined_timing_chain"]
            blocks = (h+15)//16
            for rec in m["input_fragments"]:
                inputs = trace[rec["input_line_first"]-1:rec["input_line_last"]]
                self.assertEqual(rec["input_line_last"]+1, rec["mac_line_first"])
                if rec["input_source"] == "bank_dram_z":
                    self.assertEqual(len(inputs), 256*blocks)
                    self.assertEqual({A.decode(row[1])[-1] for row in inputs},
                        {rec["fragment"]*blocks+b for b in range(blocks)})
                    self.assertEqual(rec["useful_input_bytes_per_rank"], h*32)
                else:
                    self.assertEqual(len(inputs), h*16)
                self.assertEqual({row[2] for row in inputs}, {rec["fragment"]%2})


if __name__ == "__main__":
    unittest.main()

"""DRAM Z ownership, transpose and external/internal traffic contracts."""

from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import ffn_layout as SRAM
import ffn_dram_layout as L
from ffn_dram_workload import OPERATORS, generate


def read_trace(path):
    return [(fields[0], int(fields[1]), int(fields[2]) if len(fields) == 3 else None)
            for fields in (line.split() for line in Path(path).read_text().splitlines())]


class DRAMZLayoutTests(unittest.TestCase):
    def test_weight_ownership_unchanged_and_Z_does_not_overlap(self):
        layout = L.build_layout()
        self.assertEqual((L.Z_BASE, L.Z_BASE//L.ROW_BYTES), (360448, 176))
        self.assertEqual(L.DOWN_BASE+L.DOWN_BYTES_PER_BANK, L.Z_BASE)
        self.assertEqual((L.Z_BYTES_PER_BANK, L.Z_RESERVED_BYTES_PER_BANK), (512, 2048))
        self.assertEqual(L.LAYER_STRIDE_BYTES_PER_BANK, 362496)
        self.assertEqual(layout["ffn_weight_bytes"], 24*1024**2)
        for op in ("gate_up", "down_tiled", "down_full"):
            for k, n in ((0, 0), (15, 15), (31, 300), (511, 1023)):
                self.assertEqual(L.weight_location(op, k, n), SRAM.weight_location(op, k, n))
        buffers = layout["buffers"]
        self.assertEqual(buffers, dict(GU_accumulator_per_bank=2048,
            Down_tiled_accumulator_per_bank=1024, rank_input_double=1024,
            bank_column_staging=32, rank_nonlinear_input_staging=128,
            rank_nonlinear_output_staging=32, rank_full_Z=0, rank_full_GU=0))
        self.assertEqual(layout["regions"]["down_tiled"]["rank_local_z_sram_bytes"], 0)
        self.assertEqual(SRAM.build_layout()["regions"]["down_tiled"]["rank_local_z_sram_bytes"], 8192)

    def test_all_Z_scalars_have_unique_producer_bank_addresses(self):
        addresses, per_bank = set(), Counter()
        for feature in range(4096):
            producer = L.gate_up_location("gate", 0, feature)[:4]
            for query in range(16):
                location = L.z_location(query, feature)
                self.assertEqual(location[:4], producer)
                self.assertEqual(location[1], 1)
                self.assertEqual(location[-1], 360448+((feature%256)//16)*32+query*2)
                address = L.physical_address(*location)
                self.assertEqual(L.decode_address(address), location)
                addresses.add(address)
                per_bank[producer] += 1
        self.assertEqual(len(addresses), 16*4096)
        self.assertEqual(len(per_bank), 256)
        self.assertEqual(set(per_bank.values()), {256})

    def test_column_gather_transpose_matches_Down_input_lanes(self):
        # Values encode both token and feature, so a token/feature transpose
        # error cannot pass merely because every byte was covered once.
        for rank in (0, 7, 15):
            for fragment in range(16):
                rank_buffer = {}
                for bank in range(16):
                    feature = rank*256+fragment*16+bank
                    column = L.z_column_location(rank, fragment, bank)
                    for query in range(16):
                        location = L.z_location(query, feature)
                        self.assertEqual(location, (*column[:4], column[-1]+query*2))
                        rank_buffer[L.z_fragment_buffer_offset(query, bank)] = query*4096+feature
                self.assertEqual(set(rank_buffer), set(range(0, 512, 2)))
                for query in (0, 3, 15):
                    for output in (0, 17, 255, 256, 1023):
                        # All output banks consume the same gathered lanes,
                        # independent of which bank originally produced Z.
                        terms = []
                        for lane in range(16):
                            feature = L.z_fragment_feature(rank, fragment, lane)
                            weight = L.down_location(feature, output, True)
                            self.assertEqual(weight[0], rank)
                            self.assertEqual((weight[-1]-L.DOWN_BASE)%32, lane*2)
                            terms.append(rank_buffer[query*32+lane*2]*(output+lane))
                        expected = sum((query*4096+rank*256+fragment*16+lane)*(output+lane)
                                       for lane in range(16))
                        self.assertEqual(sum(terms), expected)


class DRAMZWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cases = {}
        for case in OPERATORS:
            path = Path(cls.temp.name)/(case+".trace")
            cls.cases[case] = generate(path, case), read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_request_counts_and_DQ_are_separate_from_local_Z(self):
        expected = {
            "gate_up_z_store": Counter(PIM_WR_GB=16384, PIM_MAC_AB=32768, PIM_Z_STORE=4096),
            "down_tiled_dram": Counter(PIM_Z_LOAD=16384, PIM_MAC_AB=16384, PIM_RD_RES=32768),
        }
        expected["combined_timing_chain"] = sum(expected.values(), Counter())
        for case, (manifest, trace) in self.cases.items():
            counts = Counter(op for op, _, _ in trace)
            self.assertEqual(dict(counts), manifest["counts"])
            self.assertEqual(Counter({op: n for op, n in counts.items() if op not in ("WAIT", "BARRIER")}), expected[case])
            gu, down = case != "down_tiled_dram", case != "gate_up_z_store"
            dq = manifest["expected_dq_bytes"]
            self.assertEqual(dq, dict(pim_input_write=gu*524288,
                pim_result_read=down*1048576, dram_read=0, dram_write=0,
                total=gu*524288+down*1048576))
            self.assertEqual(manifest["expected_internal_bytes"], dict(
                GU_gather_to_nonlinear=gu*524288, Z_return_to_producer_bank=gu*131072,
                Z_dram_write=gu*131072, Z_dram_read=down*524288,
                Z_transpose_to_rank_input=down*524288))

    def test_Z_columns_have_exact_store_once_and_load_four_times_coverage(self):
        for case, (manifest, trace) in self.cases.items():
            seen = defaultdict(Counter)
            for op, address, buffer in trace:
                if op in ("WAIT", "BARRIER"):
                    continue
                channel, rank, bg, bank, offset = L.decode_address(address)
                self.assertEqual(rank, 1)
                if op in ("PIM_Z_STORE", "PIM_Z_LOAD"):
                    self.assertEqual(offset//2048, 176)
                    fragment = (offset-360448)//32
                    self.assertTrue(0 <= fragment < 16)
                    self.assertEqual(offset%32, 0)
                    self.assertEqual(buffer, fragment%2 if op == "PIM_Z_LOAD" else None)
                    seen[op][(channel, bg*4+bank, fragment)] += 1
                else:
                    self.assertEqual((bg, bank), (0, 0))
            expected_keys = {(r, b, f) for r in range(16) for b in range(16) for f in range(16)}
            if case != "down_tiled_dram":
                self.assertEqual(set(seen["PIM_Z_STORE"]), expected_keys)
                self.assertEqual(set(seen["PIM_Z_STORE"].values()), {1})
            if case != "gate_up_z_store":
                self.assertEqual(set(seen["PIM_Z_LOAD"]), expected_keys)
                self.assertEqual(set(seen["PIM_Z_LOAD"].values()), {4})

    def test_each_Down_fragment_reads_all_banks_before_consumption(self):
        for case in ("down_tiled_dram", "combined_timing_chain"):
            manifest, trace = self.cases[case]
            fragments = [r for r in manifest["input_fragments"] if r["input_source"] == "bank_dram_z"]
            self.assertEqual(len(fragments), 64)
            for record in fragments:
                fragment, tile = record["fragment"], record["tile"]
                inputs = trace[record["input_line_first"]-1:record["input_line_last"]]
                macs = trace[record["mac_line_first"]-1:record["mac_line_last"]]
                self.assertEqual(record["input_line_last"]+1, record["mac_line_first"])
                self.assertEqual(len(inputs), 256)
                self.assertEqual(len(macs), 256)
                self.assertEqual({op for op, _, _ in inputs}, {"PIM_Z_LOAD"})
                self.assertEqual(Counter(L.decode_address(a)[:4] for _, a, _ in inputs),
                    Counter(L.rank_bank_coords(rank, bank) for rank in range(16) for bank in range(16)))
                self.assertEqual({L.decode_address(a)[-1] for _, a, _ in inputs}, {360448+fragment*32})
                for channel in range(16):
                    offsets = [L.decode_address(a)[-1] for op, a, _ in macs
                               if L.decode_address(a)[0] == channel]
                    self.assertEqual(offsets, [327680+((tile*16+fragment)*16+slot)*32 for slot in range(16)])
                self.assertEqual({b for _, _, b in inputs+macs}, {fragment%2})

    def test_weight_scan_and_exports_preserve_original_tiled_Down(self):
        for case, (manifest, trace) in self.cases.items():
            for channel in range(16):
                offsets = [L.decode_address(a)[-1] for op, a, _ in trace
                           if op == "PIM_MAC_AB" and L.decode_address(a)[0] == channel]
                expected = ([] if case == "down_tiled_dram" else list(range(262144, 327680, 32)))
                if case != "gate_up_z_store":
                    expected += list(range(327680, 360448, 32))
                self.assertEqual(offsets, expected)
            if case != "gate_up_z_store":
                self.assertEqual(len(manifest["exports"]), 4)
                self.assertEqual(sum(record["bytes_fp32"] for record in manifest["exports"]), 1048576)
                self.assertEqual([(r["output_start"], r["output_count_per_rank"])
                                  for r in manifest["exports"]], [(t*256, 256) for t in range(4)])

    def test_no_bulk_gather_wait_and_all_phase_dependencies_have_barriers(self):
        for case, (manifest, trace) in self.cases.items():
            self.assertNotIn(("WAIT", 2048, None), trace)
            self.assertTrue(all(a in (4, 16) for op, a, _ in trace if op == "WAIT"))
            markers = [a for op, a, _ in trace if op == "BARRIER"]
            self.assertEqual(markers, list(range(len(markers))))
            for index, segment in enumerate(manifest["segments"]):
                self.assertEqual(segment["start"], 0 if not index else manifest["segments"][index-1]["end"])
                self.assertEqual(trace[segment["line_end"]-1], ("BARRIER", segment["end"], None))
                if segment["kind"] == "native_z_store":
                    body = trace[segment["line_start"]-1:segment["line_end"]-1]
                    self.assertEqual({op for op, _, _ in body}, {"PIM_Z_STORE"})
                    self.assertEqual(len(body), 4096)
        manifest, trace = self.cases["combined_timing_chain"]
        last_store = max(i for i, (op, _, _) in enumerate(trace) if op == "PIM_Z_STORE")
        first_load = next(i for i, (op, _, _) in enumerate(trace) if op == "PIM_Z_LOAD")
        self.assertEqual(trace[last_store+1:first_load], [("BARRIER", manifest["segments"][1]["end"], None)])

    def test_manifest_does_not_claim_nonlinear_or_bulk_SRAM(self):
        for case, (manifest, _) in self.cases.items():
            self.assertIsNone(manifest["nonlinear_latency_ck"])
            self.assertIsNone(manifest["gpu_partial_reduction_latency_ck"])
            self.assertFalse(manifest["numeric_compute"])
            self.assertFalse(manifest["integrated_chain"])
            self.assertEqual(manifest["timing_chain"], case == "combined_timing_chain")
            self.assertEqual(manifest["rank_local_z_sram_bytes"], 0)
            self.assertEqual(manifest["rank_bulk_GU_staging_bytes"], 0)
            self.assertEqual(manifest["analytical_gather_WAIT_ck"], 0)
            self.assertEqual(manifest["required_backend"], dict(pim_ffn_dram=True,
                pim_rank_rope=False, pim_input_broadcast=False, pim_mac_scope="rank", nCCDAB=16))


if __name__ == "__main__":
    unittest.main()

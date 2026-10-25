import unittest
import sys
import tempfile
from pathlib import Path

from analyze_run import category, partition, phase_kind, inventory

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import ffn_layout
import ffn_workload


class AttributionTests(unittest.TestCase):
    def test_inventory_distinguishes_horizon_from_attention_heads(self):
        operators = {row["operator"]: row for row in inventory()}
        self.assertEqual(operators["Q projection"]["input_shape"], [16,1024])
        self.assertEqual(operators["Q projection"]["output_shape"], [16,8,256])
        self.assertEqual(operators["QK^T"]["matrix_shape"], [8,256,984])
        self.assertEqual(operators["FFN Down"]["input_shape"], [16,4096])

    def test_three_resources_need_two_overlap_credits(self):
        p = partition(0, 12, dict(gpu_input=[(1, 8)], mac_array=[(3, 10)],
                                  local_write=[(5, 7)]))
        self.assertEqual(p["exclusive"]["gpu_input"], 2)
        self.assertEqual(p["exclusive"]["mac_array"], 2)
        self.assertEqual(p["exclusive"]["overlap"], 5)
        self.assertEqual(p["exclusive"]["other"], 3)
        self.assertEqual(p["overlap_credit"], 7)
        self.assertEqual(sum(p["exclusive"].values()), 12)

    def test_parallel_banks_are_union_not_summed_walltime(self):
        p = partition(0, 10, dict(local_write=[(0, 4)] * 512 + [(4, 8)] * 128))
        self.assertEqual(p["exclusive"]["local_write"], 8)
        self.assertEqual(p["exclusive"]["other"], 2)
        self.assertEqual(p["overlap_credit"], 0)

    def test_column_guard_is_not_arithmetic_and_can_overlap_input(self):
        p = partition(0, 8, dict(mac_array=[(0,2),(4,6)], array_column=[(2,4),(6,8)],
                                 gpu_input=[(3,5)]))
        self.assertEqual(p["inclusive"]["mac_array"], 4)
        self.assertEqual(p["inclusive"]["array_column"], 4)
        self.assertEqual(p["exclusive"]["mac_array"], 3)
        self.assertEqual(p["exclusive"]["array_column"], 3)
        self.assertEqual(p["exclusive"]["overlap"], 2)
        self.assertEqual(sum(p["exclusive"].values()), 8)

    def test_transfer_fences_are_distinct_from_pipeline_windows(self):
        self.assertEqual(phase_kind(dict(name="v_prefix_tail_read")), "dram_read")
        self.assertEqual(phase_kind(dict(name="k_local_writeback")), "local_write")
        self.assertEqual(phase_kind(dict(name="tile_0_result_export")), "gpu_output")
        self.assertEqual(phase_kind(dict(name="tile_0_input_and_compute")), "pipeline")
        self.assertEqual(phase_kind(dict(name="arbitrary", kind="gather")), "local_gather")

    def test_native_event_categories(self):
        self.assertEqual(category("PIM_RD_Q"), "q_dram_read")
        self.assertEqual(category("PIM_RELAY_Q"), "q_broadcast")
        self.assertEqual(category("MAC_K"), "mac_array")
        self.assertEqual(category("MAC_QK"), "mac_array")
        self.assertEqual(category("PIM_LOCAL_WR"), "local_write")
        self.assertEqual(category("PIM_LOCAL_GB"), "local_input")
        self.assertEqual(category("BG_REDUCE"), "bg_reduce")
        self.assertEqual(category("PIM_WR_BG"), "gpu_input")
        self.assertEqual(category("PIM_WR_Q"), "gpu_input")
        self.assertEqual(category("PIM_RD_BG"), "gpu_output")
        self.assertIsNone(category("ACTAB"))

    def test_dram_relay_is_not_gpu_input_and_closes(self):
        p = partition(0, 20, dict(q_dram_read=[(0,10)], q_broadcast=[(10,12)],
                                  mac_array=[(8,18)]))
        self.assertEqual(p["exclusive"]["gpu_input"], 0)
        self.assertEqual(p["exclusive"]["q_dram_read"], 8)
        self.assertEqual(p["exclusive"]["overlap"], 4)
        self.assertEqual(sum(p["exclusive"].values()), 20)

    def test_outside_phase_is_not_silently_discarded(self):
        with self.assertRaises(AssertionError):
            partition(0, 10, dict(mac_array=[(5, 11)]))


class H16FFNTests(unittest.TestCase):
    def test_layout_buffers_and_resident_regions(self):
        layout = ffn_layout.build_layout()
        self.assertEqual(layout["horizon"],16)
        self.assertEqual(layout["attention_reserved_bytes_per_bank"],42*1024)
        self.assertEqual(layout["layer_stride_bytes_per_bank"],90*1024)
        for op in ("gate_up","down_tiled"):
            region = layout["regions"][op]
            self.assertEqual(region["M"],16)
            self.assertEqual(region["input_fragment_bytes_per_rank"],512)
            self.assertEqual(region["input_double_buffer_bytes_per_rank"],1024)
            self.assertEqual(region["accumulator_bytes_per_bank"],1024)
        self.assertEqual(layout["regions"]["down_tiled"]["rank_local_z_sram_bytes"],4096)
        self.assertEqual(layout["regions"]["down_tiled"]["partial_result_fp32_bytes"],2*1024**2)

    def test_native_trace_request_coverage_scales_horizon_not_weight_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            for op, counts in (
                ("gate_up",dict(PIM_WR_GB=16384,PIM_MAC_AB=16384)),
                ("down_tiled",dict(PIM_LOCAL_GB=16384,PIM_MAC_AB=8192,PIM_RD_RES=65536))):
                path = Path(tmp) / (op+".trace")
                manifest = ffn_workload.generate(path,op)
                self.assertEqual(manifest["horizon"],16)
                self.assertEqual(manifest["mac_service_ck"],16)
                self.assertEqual(manifest["input_double_buffer_bytes_per_rank"],1024)
                for command,expected in counts.items():
                    self.assertEqual(manifest["counts"][command],expected)
                for fragment in manifest["input_fragments"]:
                    self.assertEqual(fragment["bursts_per_rank"],16)
                    self.assertEqual(fragment["bytes_per_rank"],512)
                if op == "gate_up":
                    gather = next(s for s in manifest["segments"] if s["name"] == "local_gather")
                    self.assertEqual(gather["bytes_per_rank"],16*1024)
                    self.assertEqual(gather["bytes_system"],512*1024)
                    self.assertEqual(gather["wait_cycles"],1024)
                else:
                    self.assertEqual(manifest["result_dq_bytes"],2*1024**2)
                    self.assertEqual(manifest["local_input_bytes"],512*1024)


if __name__ == "__main__":
    unittest.main()

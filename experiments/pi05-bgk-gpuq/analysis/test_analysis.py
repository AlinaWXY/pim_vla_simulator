import unittest

from analyze_run import category, partition, phase_kind


class AttributionTests(unittest.TestCase):
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
        self.assertEqual(category("MAC_K"), "mac_array")
        self.assertEqual(category("MAC_QK"), "mac_array")
        self.assertEqual(category("PIM_LOCAL_WR"), "local_write")
        self.assertEqual(category("PIM_LOCAL_GB"), "local_input")
        self.assertEqual(category("BG_REDUCE"), "bg_reduce")
        self.assertEqual(category("PIM_WR_BG"), "gpu_input")
        self.assertEqual(category("PIM_WR_Q"), "gpu_input")
        self.assertEqual(category("PIM_RD_BG"), "gpu_output")
        self.assertIsNone(category("ACTAB"))

    def test_outside_phase_is_not_silently_discarded(self):
        with self.assertRaises(AssertionError):
            partition(0, 10, dict(mac_array=[(5, 11)]))


if __name__ == "__main__":
    unittest.main()

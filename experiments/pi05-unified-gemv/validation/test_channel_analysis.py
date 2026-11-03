"""Independent edge cases for channel intervals and whole-phase accounting."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1]/"channel_analysis.py"
spec = importlib.util.spec_from_file_location("unified_channel_analysis", MODULE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class IntervalTests(unittest.TestCase):
    def test_command_edge_uses_frontend_clock_origin(self):
        self.assertEqual(C.issue_tick(1, 32), 1)
        self.assertEqual(C.issue_tick(3, 32), 65)

    def test_attacc_movement_and_ordinary_data_have_distinct_latency(self):
        rows = [{"clock": "2", "command": "RD"}, {"clock": "20", "command": "WR"},
                {"clock": "40", "command": "WRGB"}, {"clock": "42", "command": "RDRES"}]
        actual = C.dq_intervals(rows, {"nBL_min": 2, "nCL": 5, "nCWL": 3}, 4, "attacc")
        self.assertEqual([(e["begin"], e["end"]) for e in actual], [(25, 33), (89, 97), (157, 165), (165, 173)])

    def test_lpddr_pim_movement_retains_cl_cwl(self):
        rows = [{"clock": "2", "command": "WRGB"}, {"clock": "20", "command": "RDRES"}]
        actual = C.dq_intervals(rows, {"nBL_min": 2, "nCL": 5, "nCWL": 3}, 4, "lpddr")
        self.assertEqual([(e["begin"], e["end"]) for e in actual], [(17, 25), (97, 105)])

    def test_colliding_delayed_read_is_rejected_not_hidden_by_union(self):
        rows = [{"clock": "2", "command": "RD"}, {"clock": "7", "command": "WRGB"}]
        with self.assertRaisesRegex(ValueError, "DQ overlap"):
            C.dq_intervals(rows, {"nBL_min": 2, "nCL": 5, "nCWL": 3}, 4, "attacc")

    def test_half_open_clipping_and_idle_partition(self):
        occupied, idle = C.interval_accounting([{"begin": 1, "end": 3}, {"begin": 5, "end": 12}], 2, 10)
        self.assertEqual(occupied, 6)
        self.assertEqual(idle, [(3, 5)])
        self.assertEqual(C.interval_accounting([], 0, 10), (0, [(0, 10)]))

    def test_tail_clear_does_not_count_gap_before_delayed_burst(self):
        # The earlier command's data starts after this phase begins. Only the
        # suffix after its END is a complete contiguous candidate window.
        items = [{"issue": 1, "begin": 8, "end": 10}]
        self.assertEqual(C.tail_clear_window(items, 3, 15), (10, 15))
        self.assertEqual(C.tail_clear_window(items, 3, 9), (9, 9))
        self.assertEqual(C.tail_clear_window(items, 10, 15), (10, 15))


class CommandBusAccountingTests(unittest.TestCase):
    @staticmethod
    def command(clock, ready, *, scope="rank", rank=1):
        return dict(clock=clock, ready_at=ready, scope=scope, Rank=rank, command="test")

    def test_independent_rank_buses_allow_simultaneous_issue_but_same_rank_rejects_overlap(self):
        rows = [self.command(1, 3, rank=0), self.command(1, 3, rank=1)]
        result = C.command_bus_accounting(rows, [], 20, 2, .5)
        self.assertEqual(result["rank0"]["occupied_us"], 2)
        self.assertEqual(result["rank1"]["occupied_us"], 2)
        rows.append(self.command(2, 3, rank=1))
        with self.assertRaisesRegex(ValueError, "Overlapping rank1 CA"):
            C.command_bus_accounting(rows, [], 20, 2, .5)

    def test_global_CA_and_DQ_intersection_partition(self):
        # CK1..4 maps to frontend[1,7); DQ[3,5) overlaps exactly2 ticks.
        rows = [self.command(1, 4, scope="global")]
        result = C.command_bus_accounting(rows, [dict(begin=3, end=5)], 10, 2, 1)
        self.assertEqual(result["global"]["occupied_us"], 6)
        self.assertEqual(result["dq_idle_global_ca_busy_us"], 4)
        self.assertEqual(result["dq_idle_global_ca_free_us"], 4)
        self.assertEqual(result["dq_idle_global_ca_busy_us"]+result["dq_idle_global_ca_free_us"], 8)

    def test_half_open_endpoints_and_wall_clipping(self):
        rows = [self.command(1, 3, scope="global"), self.command(3, 7, scope="global")]
        # Adjacent[1,5) and[5,13) remain nonoverlapping. Clip to wall9.
        result = C.command_bus_accounting(rows, [dict(begin=5, end=20)], 9, 2, 1)
        self.assertEqual(result["global"]["occupied_us"], 8)
        self.assertEqual(result["global"]["idle_us"], 1)
        self.assertEqual(result["dq_idle_global_ca_busy_us"], 4)
        self.assertEqual(result["dq_idle_global_ca_free_us"], 1)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"simulation").mkdir()
        self.manifest = dict(expected_dq_bytes={"total": 128}, segments=[
            dict(name="export", kind="external_read", start=0, end=1, counts={"LD": 2, "BARRIER": 1}),
            dict(name="local_compute", kind="compute", start=1, end=2, counts={"PIM_MAC_AB": 16}),
            dict(name="input_with_large_gaps", kind="pipeline", start=2, end=3, counts={"PIM_WR_GB": 2}),
            dict(name="release", kind="gemv_release", start=3, end=4, counts={"PIM_GEMV_RELEASE": 2}),
            dict(name="wait_only", kind="compute", start=4, end=5, counts={"WAIT": 100, "BARRIER": 1}),
        ])
        self.result = dict(config={"frontend_ratio": 4, "channels": 2, "pim_data_path": "attacc"},
            resolved_timings_ck={"nBL_min": 2, "nCL": 5, "nCWL": 3},
            memory_cycles=30, latency_ns=30000, tCK_ns=1000, transaction_bytes=32,
            issued_commands={"RD": 2, "WRGB": 2, "MACAB": 2})
        self.dump("manifest.json", self.manifest)
        self.dump("simulation/result.json", self.result)
        self.csv("simulation/barriers.csv", ("marker", "frontend_tick"),
                 [dict(marker=i, frontend_tick=t) for i, t in enumerate((1, 9, 45, 65, 73, 81))])
        for channel in range(2):
            self.csv(f"simulation/timeline.csv.ch{channel}", ("clock", "command", "Channel"), [
                dict(clock=2, command="RD", Channel=channel),
                dict(clock=11, command="MACAB", Channel=channel),
                dict(clock=14+2*channel, command="WRGB", Channel=channel)])

    def dump(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding="utf-8")

    def csv(self, name, fields, rows):
        with (self.root/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_tail_corrected_whole_phase_and_two_partitions_close(self):
        report = C.analyze_stage(self.root, "fixture")
        self.assertEqual(report["stage_us"], 30)
        self.assertEqual(report["complete_zero_external_phase_us_all_channels"], 9)
        self.assertEqual(report["tail_clear_complete_zero_external_phase_us_all_channels"], 3)
        for row in report["channels"]:
            self.assertEqual(row["modelled_dq_occupied_us"], 4)
            self.assertEqual(row["modelled_dq_idle_us"], 26)
            self.assertEqual(row["modelled_dq_idle_outside_tail_clear_work_us"], 23)
            self.assertEqual(row["removed_boundary_data_tail_us"], 6)
            self.assertEqual(row["other_semantic_phase_us"], 11)
            self.assertEqual(row["boundary_overhead_us"], 10)
            self.assertEqual(sum(row[k] for k in ("complete_zero_external_phase_us", "other_semantic_phase_us", "boundary_overhead_us")), row["stage_us"])
        self.assertFalse(report["phases"][2]["complete_zero_external_phase_all_channels"])
        self.assertFalse(report["phases"][3]["meaningful_work"])
        self.assertFalse(report["phases"][4]["meaningful_work"])

    def test_count_corruption_is_rejected(self):
        self.result["issued_commands"]["WRGB"] = 3
        self.dump("simulation/result.json", self.result)
        with self.assertRaisesRegex(ValueError, "external counters"):
            C.analyze_stage(self.root, "fixture")

    def test_unclassified_command_is_rejected_at_exact_phase_end(self):
        self.csv("simulation/timeline.csv.ch1", ("clock", "command", "Channel"), [
            dict(clock=2, command="RD", Channel=1), dict(clock=21, command="WRGB", Channel=1)])
        with self.assertRaisesRegex(ValueError, "outside semantic phases"):
            C.analyze_stage(self.root, "fixture")

    def test_asymmetric_channels_are_not_forced_to_channel_zero(self):
        self.result["issued_commands"]["WRGB"] = 1
        self.manifest["expected_dq_bytes"]["total"] = 96
        self.dump("simulation/result.json", self.result)
        self.dump("manifest.json", self.manifest)
        self.csv("simulation/timeline.csv.ch1", ("clock", "command", "Channel"), [
            dict(clock=2, command="RD", Channel=1), dict(clock=11, command="MACAB", Channel=1)])
        report = C.analyze_stage(self.root, "fixture")
        self.assertEqual([r["modelled_dq_occupied_us"] for r in report["channels"]], [4, 2])
        self.assertEqual([r["complete_zero_external_phase_us"] for r in report["channels"]], [9, 14])
        self.assertEqual(report["complete_zero_external_phase_us_all_channels"], 9)

    def test_five_stage_report_does_not_add_combined_control(self):
        with tempfile.TemporaryDirectory() as aggregate_dir:
            root = Path(aggregate_dir)
            for case in C.PRIMARY:
                shutil.copytree(self.root, root/case)
            (root/"summary.json").write_text(json.dumps(dict(primary_cases=list(C.PRIMARY),
                measured_stage_sum_us_per_layer_step=150,
                runs={"combined_timing_chain": {"latency_us": 999}})), encoding="utf-8")
            report = C.write_report(root)
            self.assertEqual(report["stage_sum_us"], 150)
            self.assertEqual(report["complete_zero_external_phase_us_all_channels"], 45)
            self.assertIsNone(report["actual_GPU_usable_time_us"])
            self.assertFalse(report["validated_physical_DQ_waveform"])
            self.assertEqual(len(C.read_csv(root/"channel-partitions.csv")), 10)
            self.assertEqual(len(C.read_csv(root/"channel-phase-windows.csv")), 50)
            self.assertTrue((root/"CHANNEL-ANALYSIS.md").is_file())


if __name__ == "__main__":
    unittest.main()

"""Protect H=16 reference selection and exact same-version reproduction."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("h16_reproduce", Path(__file__).resolve().parents[1] / "reproduce.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.reference = self.root / "run", self.root / "reference"
        self.data = dict(schema_version=runner.SCHEMA_VERSION, horizon=16, attention_heads=8,
            primary_rows=[dict(operator=case, total_us=index+0.25)
                          for index, case in enumerate(runner.PRIMARY_CASES)],
            measured_pim_subset_us_per_layer_step=11.25,
            measured_pim_subset_ms_18layers_10steps=2.025,
            nested_channel_partition=[dict(cycles=12345)])
        for destination in (self.run, self.reference):
            self.write_breakdown(destination, self.data)
        for case, row in zip(runner.PRIMARY_CASES, self.data["primary_rows"]):
            directory = self.run / case if case in runner.PRIMARY_CASES[:3] else self.run / "ffn" / case
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps(dict(latency_us=row["total_us"])))

    def write_breakdown(self, root, data):
        directory = root / "all-linear-breakdown"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "breakdown.json").write_text(json.dumps(data))
        with (directory / "all-linear-breakdown.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["operator", "total_us"])
            writer.writeheader()
            writer.writerows(data["primary_rows"])

    def test_first_candidate_does_not_claim_exact_reproduction(self):
        result = runner.candidate_without_reference(self.run)
        self.assertEqual(result["comparison_mode"], "no_reference_first_H16_run")
        self.assertEqual(result["numeric_breakdown_fields_checked"], 0)
        self.assertIsNone(result["reference_run"])

    def test_same_h16_reference_compares_all_numeric_fields(self):
        result = runner.compare_reference(self.run, self.reference)
        self.assertEqual(result["comparison_mode"], "exact_same_schema")
        self.assertEqual(result["numeric_breakdown_fields_checked"], len(runner.numeric_leaves(self.data)))
        self.assertEqual(result["numeric_csv_fields_checked"], 5)

    def test_h20_reference_is_rejected_even_with_h16_schema_label(self):
        self.data["horizon"] = 20
        self.write_breakdown(self.reference, self.data)
        with self.assertRaisesRegex(AssertionError, "H=20"):
            runner.compare_reference(self.run, self.reference)

    def test_nested_cycle_mismatch_is_rejected(self):
        self.data["nested_channel_partition"][0]["cycles"] += 1
        self.write_breakdown(self.reference, self.data)
        with self.assertRaisesRegex(AssertionError, "numeric mismatch"):
            runner.compare_reference(self.run, self.reference)


if __name__ == "__main__":
    unittest.main()

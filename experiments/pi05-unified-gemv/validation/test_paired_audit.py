"""Faults that preserve byte counts but break the two-slot staging schedule."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from audit_native import trace_records
from paired_audit import audit_paired_staging, audit_paired_trace
import workload


def pair_fixture():
    staged = [dict(operation="SEAL", Channel=0, stage_generation=1, clock=1)]
    native = [dict(command="ACT2", Channel=0, Rank=1, clock=5, Row=256, Column=0),
              dict(command="ACTAB", Channel=0, Rank=1, clock=160, Row=160, Column=0)]
    gemv = []
    for fragment in range(2):
        for token in range(16):
            packet = fragment*16+token
            clock = 10+4*packet
            event = dict(Channel=0, Rank=1, Row=256, Column=packet, buffer=fragment,
                stage_generation=1, clock=clock, ready_at=clock+25)
            staged += [dict(event, operation="LOCALRD"),
                       dict(event, operation="BROADCAST", clock=clock+1)]
            native.append(dict(event, command="LOCALRD"))
        for column in range(4):
            clock = 181+16*(fragment*4+column)
            native.append(dict(command="MACAB", Channel=0, Rank=1, clock=clock, Row=160, Column=fragment*4+column))
            gemv.append(dict(operation="GEMV_MAC", Channel=0, clock=clock, ready_at=clock+16))
    return staged, native, gemv


class PairedAuditTests(unittest.TestCase):
    def check(self, fixture):
        return audit_paired_staging(*fixture, "down_gpu_input", complete_workload=False)

    def test_two_prefetched_fragments_use_exactly_existing_capacity(self):
        result = self.check(pair_fixture())
        self.assertEqual(result["input_batches_checked"], 1)
        self.assertEqual(result["maximum_prefetched_input_bytes_per_bank"], 1024)
        self.assertEqual(result["physical_input_capacity_bytes_per_bank"], 1024)
        self.assertEqual(result["row_command_counts"][0]["source_ACT2"], 1)
        self.assertEqual(result["row_command_counts"][0]["weight_ACTAB"], 1)

    def test_old_interleaving_fails_even_when_all_counts_are_unchanged(self):
        staged, native, gemv = pair_fixture()
        for event in native:
            if event["command"] == "MACAB" and event["Column"] < 4:
                event["clock"] -= 86
        for event in gemv[:4]:
            event["clock"] -= 86
            event["ready_at"] -= 86
        with self.assertRaisesRegex(AssertionError, "before both prefetched slots"):
            self.check((staged, native, gemv))

    def test_paired_input_cannot_alias_one_physical_slot(self):
        staged, native, gemv = pair_fixture()
        for event in staged:
            if event.get("buffer") == 1:
                event["buffer"] = 0
        with self.assertRaisesRegex(AssertionError, "aliases or exceeds"):
            self.check((staged, native, gemv))

    def test_weight_row_cannot_be_opened_between_the_two_input_fills(self):
        staged, native, gemv = pair_fixture()
        native.append(dict(command="ACTAB", Channel=0, Rank=1, clock=80, Row=160, Column=0))
        with self.assertRaisesRegex(AssertionError, "before both input slots"):
            self.check((staged, native, gemv))

    def test_descriptor_multiset_does_not_hide_old_trace_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"v.trace"
            workload.generate(path, "v")
            records = trace_records(path)
        audit_paired_trace(records, "v")
        second = next(e for e in records if e["operation"] == "PIM_STAGE_FRAGMENT" and
                      e["address"] >> 5 & 15 == 0 and e["buffer"] == 1)
        records.remove(second)
        last_mac = next(e for e in records if e["operation"] == "PIM_MAC_AB" and
                        e["address"] >> 5 & 15 == 0 and ((e["address"] >> 20)*2048+(e["address"] >> 9 & 63)*32) == 83968+7*32)
        records.insert(records.index(last_mac)+1, second)
        with self.assertRaisesRegex(AssertionError, "fill both slots"):
            audit_paired_trace(records, "v")


if __name__ == "__main__":
    unittest.main()

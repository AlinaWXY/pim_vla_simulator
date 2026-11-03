"""Faults invisible to byte counts or to a producer-supplied bank manifest."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from audit_native import audit_trace, trace_records
from paired_audit import audit_paired_staging
from staging_audit import audit_staging, audit_source_identity, audit_write_timing, BASE
from test_staging_audit import staged_fixture
import test_staging_audit
from test_paired_audit import pair_fixture
import workload


def two_bank_fixture():
    events, native, fills, timings = deepcopy(staged_fixture())
    timings.update(nCCDS_WR=2, nCCDL_WR=4)
    for e in events+native:
        if e.get("operation") in ("BEGIN", "SEAL"):
            continue
        packet = (e["Row"]*2048+e["Column"]*32-BASE)//32
        fragment, token = divmod(packet, 8)
        physical = BASE+((fragment//2)*8+token)*32
        e.update(BankGroup=fragment%2, Row=physical//2048, Column=physical%2048//32)
        if e.get("operation") == "WR":
            e["clock"] = 3+2*((fragment//2)*16+token*2+fragment%2)
            e["ready_at"] = e["clock"]+7
    return events, native, fills, timings


class TwoBankAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = two_bank_fixture()

    def check(self, events=None, native=None):
        e, n, fills, timings = self.fixture
        return audit_staging(events if events is not None else e,
            native if native is not None else n, fills, "chain", timings, staging_banks=2)

    def test_two_bank_storage_preserves_full_input_and_physical_SRF_capacity(self):
        result = self.check()
        self.assertEqual(result["source_banks"], [0, 4])
        self.assertEqual(result["external_input_bytes"], 16*16384)
        self.assertEqual(result["maximum_DRAM_staging_bytes_per_source_bank"], 8192)
        self.assertEqual(result["reserved_DRAM_staging_bytes_per_source_bank"], 65536)
        self.assertEqual(result["SRAM_bytes_per_bank"], 1024)
        self.assertEqual(result["writer"]["minimum_cross_BG_WR_gap_ck"], 2)

    def test_matching_wrong_native_and_staging_bank_still_fails_fixed_oracle(self):
        events, native, _, _ = deepcopy(self.fixture)
        for group in (events, native):
            for e in group:
                if e.get("BankGroup") == 1:
                    e.update(BankGroup=0, Bank=1)
        with self.assertRaisesRegex(AssertionError, "BG1/bank0"):
            self.check(events, native)

    def test_missing_second_bank_packet_cannot_alias_first_bank_address(self):
        events = deepcopy(self.fixture[0])
        target = next(e for e in events if e["operation"] == "WR" and e["BankGroup"] == 1)
        target["BankGroup"] = 0
        with self.assertRaisesRegex(AssertionError, "full-input source-address coverage"):
            self.check(events)

    def test_precharge_of_second_source_bank_checks_its_own_last_read(self):
        native = deepcopy(self.fixture[1])
        read = next(e for e in native if e["command"] == "LOCALRD" and e["BankGroup"] == 1)
        native.append(dict(read, command="PREpb", clock=read["clock"]+2))
        with self.assertRaisesRegex(AssertionError, "read-to-precharge timing"):
            self.check(native=native)

    def test_same_offset_in_other_bank_is_a_different_reduction_fragment(self):
        staged, native, fills = test_staging_audit.SourceIdentityTests().fixture()
        staged[0]["BankGroup"] = 1
        with self.assertRaisesRegex(AssertionError, "weight reduction fragment"):
            audit_source_identity(staged, native, fills, "chain", staging_banks=2)
        native[0]["Column"] = 32
        self.assertEqual(audit_source_identity(staged, native, fills, "chain", staging_banks=2), 1)

    def test_trace_write_permutation_fails_even_with_identical_address_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"down.trace"
            workload.generate(path, "down_gpu_input", staging_banks=2)
            records = trace_records(path)
        audit_trace(records, "down_gpu_input", staging_banks=2)
        indices = [i for i, e in enumerate(records) if e["operation"] == "ST" and e["address"] >> 5 & 15 == 0]
        records[indices[1]], records[indices[2]] = records[indices[2]], records[indices[1]]
        with self.assertRaisesRegex(AssertionError, "write order must interleave"):
            audit_trace(records, "down_gpu_input", staging_banks=2)

    def test_writer_short_gap_cannot_be_claimed_for_same_BG_or_serialized_WR(self):
        writes = [dict(Channel=0, stage_generation=1, BankGroup=bg, clock=clock)
                  for bg, clock in ((0, 1), (1, 3), (0, 5), (1, 7))]
        timings = dict(nCCDS_WR=2, nCCDL_WR=4)
        self.assertEqual(audit_write_timing(writes, timings, 2, True)["cross_BG_short_WR_pairs"], 3)
        writes[2]["clock"] = 4
        with self.assertRaisesRegex(AssertionError, "same-BG write spacing"):
            audit_write_timing(writes, timings, 2, True)
        for i, e in enumerate(writes):
            e["clock"] = i*4+1
        with self.assertRaisesRegex(AssertionError, "never issues cross-BG WR"):
            audit_write_timing(writes, timings, 2, True)

    def test_pair_uses_two_source_banks_at_same_row_before_weight_activation(self):
        staged, native, gemv = pair_fixture()
        for e in staged+native:
            if e.get("operation") in ("LOCALRD", "BROADCAST") or e.get("command") == "LOCALRD":
                e["BankGroup"], e["Column"] = divmod(e["Column"], 16)
                e["Bank"] = 0
        result = audit_paired_staging(staged, native, gemv, "down_gpu_input", False, staging_banks=2)
        self.assertEqual(result["first_two_batches_channel0"][0]["source_banks"], [0, 4])
        self.assertEqual(result["maximum_prefetched_input_bytes_per_bank"], 1024)
        native.append(dict(command="ACTAB", Channel=0, Rank=1, clock=80, Row=160, Column=0))
        with self.assertRaisesRegex(AssertionError, "before both input slots"):
            audit_paired_staging(staged, native, gemv, "down_gpu_input", False, staging_banks=2)


if __name__ == "__main__":
    unittest.main()

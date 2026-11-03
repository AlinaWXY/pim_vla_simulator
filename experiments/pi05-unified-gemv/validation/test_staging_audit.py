"""Fault injection against independent staging and command-bus reconstruction."""
from copy import deepcopy
import unittest

from staging_audit import audit_staging, audit_rank_ca, audit_source_identity, BASE


def staged_fixture():
    events, native, fills = [], [], []
    timings = dict(nCWL=2, nBL_min=2, nBL_max=2, nWR=3, nCL=3, nRTP=3)
    def event(ch, op, clock, ready, off=BASE, size=32, slot=-1):
        return dict(Channel=ch, Rank=1, BankGroup=0, Bank=0, Row=off//2048,
            Column=off%2048//32, clock=clock, ready_at=ready, operation=op,
            bytes=size, buffer=slot, stage_generation=1)
    for ch in range(16):
        events.append(event(ch, "BEGIN", 1, 1, size=16384))
        for packet in range(512):
            issued = 3+4*packet
            item = event(ch, "WR", issued, issued+7, BASE+packet*32)
            events.append(item)
            native.append(dict(item, command="WR"))
        seal = 3+4*511+7
        events.append(event(ch, "SEAL", seal, seal))
        for packet in range(512):
            issued, slot = seal+1+4*packet, packet//8%2
            item = event(ch, "LOCALRD", issued, issued+5, BASE+packet*32, slot=slot)
            events.append(item)
            native.append(dict(item, command="LOCALRD"))
            events.append(event(ch, "BROADCAST", issued+1, issued+5, BASE+packet*32, slot=slot))
            fills.append(dict(operation="GEMV_MULTICAST", Channel=ch, clock=issued+5, buffer=slot,
                token_mask=1 << (packet%8), lane_mask=65535, bank_mask=65535, generation=packet//16+1))
    return events, native, fills, timings


class StagingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = staged_fixture()

    def check(self, events=None, native=None, fills=None):
        original = self.fixture
        return audit_staging(events if events is not None else original[0],
            native if native is not None else original[1], fills if fills is not None else original[2],
            "chain", original[3])

    def test_valid_full_payload_once_in_one_bank(self):
        result = self.check()
        self.assertEqual(result["external_input_bytes"], 16*16384)
        self.assertEqual(result["internal_read_bytes"], 16*16384)
        self.assertEqual(result["multicast_destination_bytes"], 16*16*16384)

    def test_seal_cannot_precede_last_write_recovery(self):
        events = deepcopy(self.fixture[0])
        seal = next(e for e in events if e["operation"] == "SEAL")
        seal["clock"] -= 1
        with self.assertRaisesRegex(AssertionError, "before every DRAM write"):
            self.check(events=events)

    def test_read_cannot_start_before_full_input_is_sealed(self):
        events = deepcopy(self.fixture[0])
        read = next(e for e in events if e["operation"] == "LOCALRD")
        read["clock"] -= 2
        with self.assertRaisesRegex(AssertionError, "precedes full-input seal"):
            self.check(events=events)

    def test_full_address_coverage_detects_missing_source_read(self):
        events = deepcopy(self.fixture[0])
        del events[next(i for i, e in enumerate(events) if e["operation"] == "LOCALRD")]
        with self.assertRaisesRegex(AssertionError, "full-input source-address coverage"):
            self.check(events=events)

    def test_second_source_bank_is_rejected(self):
        events = deepcopy(self.fixture[0])
        next(e for e in events if e["operation"] == "WR")["Bank"] = 1
        with self.assertRaisesRegex(AssertionError, "exactly physical rank1/BG0/bank0"):
            self.check(events=events)

    def test_SRF_fill_must_match_source_token_and_completion(self):
        fills = deepcopy(self.fixture[2])
        fills[0]["token_mask"] = 2
        with self.assertRaisesRegex(AssertionError, "SRF multicast completions"):
            self.check(fills=fills)

    def test_internal_read_cannot_be_charged_as_external_RD(self):
        native = deepcopy(self.fixture[1])
        next(e for e in native if e["command"] == "LOCALRD")["command"] = "RD"
        with self.assertRaisesRegex(AssertionError, "actual native LOCALRD"):
            self.check(native=native)

    def test_write_cannot_omit_array_recovery_time(self):
        events = deepcopy(self.fixture[0])
        next(e for e in events if e["operation"] == "WR")["ready_at"] -= 1
        with self.assertRaisesRegex(AssertionError, "omits CWL/burst/write recovery"):
            self.check(events=events)

    def test_source_precharge_must_respect_read_recovery(self):
        native = deepcopy(self.fixture[1])
        first = next(e for e in native if e["command"] == "LOCALRD")
        native.append(dict(first, command="PREpb", clock=first["clock"]+2))
        with self.assertRaisesRegex(AssertionError, "read-to-precharge timing"):
            self.check(native=native)


class SourceIdentityTests(unittest.TestCase):
    def fixture(self):
        staging = [dict(operation="BROADCAST", Channel=0, ready_at=20, buffer=0,
                        stage_generation=1, Row=256, Column=0)]
        native = [dict(command="MACAB", Channel=0, clock=21, Row=0, Column=0)]
        fills = [dict(operation="GEMV_MULTICAST", Channel=0, clock=20, buffer=0, generation=1, token_mask=1),
                 dict(operation="GEMV_MAC", Channel=0, clock=21, buffer=0, generation=1)]
        return staging, native, fills

    def test_native_weight_fragment_must_match_actual_source(self):
        staged, native, fills = self.fixture()
        self.assertEqual(audit_source_identity(staged, native, fills, "chain"), 1)
        native[0]["Column"] = 32
        with self.assertRaisesRegex(AssertionError, "weight reduction fragment"):
            audit_source_identity(staged, native, fills, "chain")

    def test_all_tokens_in_a_slot_must_come_from_same_fragment(self):
        staged, native, fills = self.fixture()
        staged.append(dict(staged[0], Column=9, ready_at=22))
        fills.append(dict(fills[0], clock=22, token_mask=2))
        with self.assertRaisesRegex(AssertionError, "mixes different source fragments"):
            audit_source_identity(staged, native, fills, "chain")


def ca(clock, command, *, rank=1, duration=1, scope="rank"):
    return dict(clock=clock, ready_at=clock+duration, Channel=0, Rank=rank, command=command, scope=scope)


class CommandBusTests(unittest.TestCase):
    def fixture(self):
        native = [dict(clock=1, Channel=0, Rank=1, command="ACTAB"),
                  dict(clock=3, Channel=0, Rank=1, command="MACAB")]
        staging = [dict(clock=4, Channel=0, Rank=1, operation="BROADCAST")]
        events = [ca(0, "PIM_STAGE_BEGIN", scope="global"), ca(1, "ACTAB", duration=2),
                  ca(1, "ACTAB", duration=2, scope="global"), ca(3, "MACAB"), ca(4, "BCAST")]
        return events, native, staging

    def test_data_and_compute_share_one_rank_bus(self):
        result = audit_rank_ca(*self.fixture())
        self.assertTrue(result["transfer_broadcast_compute_share_rank_CA"])
        self.assertEqual(result["rank_command_coverage"], 3)

    def test_multicycle_command_competes_with_compute(self):
        events, native, staging = self.fixture()
        events[3].update(clock=2, ready_at=3)
        with self.assertRaisesRegex(AssertionError, "Overlapping command bus"):
            audit_rank_ca(events, native, staging)

    def test_global_dispatch_competes_with_external_commands(self):
        events, native, staging = self.fixture()
        events.append(ca(2, "PIM_STAGE_LOAD", scope="global"))
        with self.assertRaisesRegex(AssertionError, "Overlapping command bus"):
            audit_rank_ca(events, native, staging)

    def test_rank0_and_rank1_have_independent_local_buses(self):
        events, native, staging = self.fixture()
        events.append(ca(3, "RD", rank=0))
        native.append(dict(clock=3, Channel=0, Rank=0, command="RD"))
        audit_rank_ca(events, native, staging)

    def test_missing_rank_CA_evidence_is_rejected(self):
        events, native, staging = self.fixture()
        events.pop()
        with self.assertRaisesRegex(AssertionError, "every native command and broadcast"):
            audit_rank_ca(events, native, staging)

    def test_coarse_fragment_descriptor_cannot_expand_on_global_CA(self):
        events, native, staging = self.fixture()
        events.append(ca(5, "PIM_STAGE_FRAGMENT", scope="global"))
        records = [dict(address=0, operation="PIM_STAGE_BEGIN"), dict(address=0, operation="PIM_STAGE_FRAGMENT")]
        result = audit_rank_ca(events, native, staging, records=records)
        self.assertEqual(result["coarse_fragment_descriptors_checked"], 1)
        events.append(ca(6, "PIM_STAGE_LOAD", scope="global"))
        with self.assertRaisesRegex(AssertionError, "exactly the coarse staging descriptors"):
            audit_rank_ca(events, native, staging, records=records)


if __name__ == "__main__":
    unittest.main()

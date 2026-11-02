"""Fault injection for the independent native-event checker and address oracle."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import workload
from audit_native import (audit_slot_events, audit_trace, audit_native_commands, audit_mac_order,
                          expected_addresses, trace_key, trace_records, decode, GEOMETRY, NATIVE)
from collections import Counter


def event(op, clock, *, duration=2, slot=0, banks=65535, tokens=1, lanes=65535,
          generation=1, columns=0):
    return dict(operation=op, clock=clock, ready_at=clock+duration, service_cycles=duration,
        Channel=0, Rank=1, buffer=slot, bank_mask=banks, token_mask=tokens,
        lane_mask=lanes, generation=generation, mac_columns=columns, bytes=32)


def ordinary_fragment():
    fills = [event("GEMV_MULTICAST", 1+2*t, tokens=1 << t) for t in range(16)]
    macs = [event("GEMV_MAC", 33+16*s, duration=16, tokens=65535, columns=s+1) for s in range(8)]
    return fills+macs+[event("GEMV_RELEASE", 161, duration=0, tokens=65535, columns=8)]


class NativeAuditFaultTests(unittest.TestCase):
    def test_valid_multicast_and_consumption(self):
        result = audit_slot_events(ordinary_fragment(), "v")
        self.assertEqual(result["physical_bank_releases"], 16)

    def test_missing_token_is_detected_before_consume(self):
        events = ordinary_fragment()
        del events[15]
        with self.assertRaisesRegex(AssertionError, "missing bank/token/lane"):
            audit_slot_events(events, "v")

    def test_fill_completion_must_precede_mac(self):
        events = ordinary_fragment()
        events[15].update(ready_at=35, service_cycles=4)
        with self.assertRaisesRegex(AssertionError, "before fill"):
            audit_slot_events(events, "v")

    def test_active_consumer_cannot_be_overwritten(self):
        events = ordinary_fragment()
        events.insert(17, event("GEMV_MULTICAST", 34, generation=2))
        with self.assertRaisesRegex(AssertionError, "overwrote consuming"):
            audit_slot_events(events, "v")

    def test_partial_weight_consumption_cannot_be_released(self):
        events = ordinary_fragment()
        del events[-2]
        with self.assertRaisesRegex(AssertionError, "incomplete input/weight"):
            audit_slot_events(events, "v")

    def test_distribution_port_conflict_is_detected(self):
        events = ordinary_fragment()
        events[1].update(clock=2, ready_at=4)
        with self.assertRaisesRegex(AssertionError, "distribution port"):
            audit_slot_events(events, "v")

    def test_down_scatter_needs_all_feature_lanes(self):
        events = [event("GEMV_Z_SCATTER", 1+2*b, tokens=65535, lanes=1 << b) for b in range(15)]
        events.append(event("GEMV_MAC", 33, duration=16, tokens=65535, columns=1))
        with self.assertRaisesRegex(AssertionError, "missing bank/token/lane"):
            audit_slot_events(events, "down_tiled_dram", require_final_release=False)

    def test_Q_ownership_blocks_ordinary_write_to_same_slot(self):
        events = [event("GEMV_Q_LOCAL", 1, banks=1, tokens=1),
                  event("GEMV_MULTICAST", 3, tokens=2)]
        with self.assertRaisesRegex(AssertionError, "Different input kinds"):
            audit_slot_events(events, "chain", require_final_release=False)

    def test_complete_stage_must_release_slots(self):
        with self.assertRaisesRegex(AssertionError, "Live slot ownership"):
            audit_slot_events(ordinary_fragment()[:-1], "v")


class IndependentAddressTests(unittest.TestCase):
    def test_all_cases_match_independent_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            for case in workload.CASES:
                with self.subTest(case=case):
                    trace = Path(root) / f"{case}.trace"
                    manifest = workload.generate(trace, case)
                    records = trace_records(trace)
                    self.assertFalse(any(e["operation"] == "WAIT" for e in records))
                    actual = Counter(trace_key(e) for e in records if e["operation"] != "BARRIER")
                    self.assertEqual(actual, expected_addresses(case))
                    self.assertEqual(manifest["buffers"]["gemv_total_bytes_per_bank"], 1024)
                    self.assertEqual(manifest["buffers"]["separate_rank_MAC_input_bytes"], 0)

    def test_correlated_trace_and_native_address_error_fails_independent_oracle(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "v.trace"
            workload.generate(trace, "v")
            records = trace_records(trace)
            target = next(e for e in records if e["operation"] == "PIM_MAC_AB")
            target["address"] ^= 1 << 20
            native = [dict(command=NATIVE[e["operation"]], **dict(zip(GEOMETRY, decode(e["address"]))))
                      for e in records if e["operation"] in NATIVE]
            # Producer and consumer agree on the same wrong address, so a
            # trace-vs-native comparison alone is insufficient.
            audit_native_commands(records, native)
            with self.assertRaisesRegex(AssertionError, "independent matrix/address"):
                audit_trace(records, "v")

    def test_same_address_multiset_with_wrong_reduction_order_is_rejected(self):
        native = []
        for ch in range(16):
            for i in range(512):
                offset = 83968+i*32
                native.append(dict(Channel=ch, clock=i+1, command="MACAB",
                                   Row=offset//2048, Column=offset%2048//32))
        native[0]["Column"], native[1]["Column"] = native[1]["Column"], native[0]["Column"]
        with self.assertRaisesRegex(AssertionError, "independent operator/reduction order"):
            audit_mac_order(native, "v")

    def test_serial_control_has_one_monotonic_barrier_namespace(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "serial.trace"
            manifest = workload.generate(trace, "serial_layer_control")
            records = trace_records(trace)
            barriers = [e["address"] for e in records if e["operation"] == "BARRIER"]
            self.assertEqual(barriers, list(range(len(barriers))))
            self.assertEqual(manifest["expected_dq_bytes"]["total"], 3872*1024)
            self.assertFalse(manifest["required_backend"]["ffn_z_preinstalled"])
            for a, b in zip(manifest["parts"], manifest["parts"][1:]):
                self.assertEqual(a["end_marker"], b["start_marker"])
                self.assertEqual(a["last_line"]+1, b["first_line"])

    def test_non_H16_geometry_is_rejected(self):
        import horizon_attention, horizon_value, horizon_ffn
        for call in (horizon_attention.layout_for, horizon_value.layout, horizon_ffn.layout):
            with self.assertRaises(ValueError):
                call(20)


if __name__ == "__main__":
    unittest.main()

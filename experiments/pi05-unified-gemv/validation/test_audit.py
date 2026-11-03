"""Fault injection for the independent native-event checker and address oracle."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import workload
from audit_native import (audit_slot_events, audit_trace, audit_native_commands, audit_mac_order, audit_external_fills,
                          expected_addresses, expected_native_commands, trace_key, trace_records, decode, GEOMETRY, NATIVE)
from collections import Counter


def event(op, clock, *, duration=2, slot=0, banks=65535, tokens=1, lanes=65535,
          generation=1, columns=0):
    return dict(operation=op, clock=clock, ready_at=clock+duration, service_cycles=duration,
        Channel=0, Rank=1, buffer=slot, bank_mask=banks, token_mask=tokens,
        lane_mask=lanes, generation=generation, mac_columns=columns, bytes=32)


def ordinary_fragment():
    fills = [event("GEMV_MULTICAST", 1+2*t, duration=0, tokens=1 << t) for t in range(16)]
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
        events = [event("GEMV_Q_LOCAL", 1+2*(b*16+t), banks=1 << b, tokens=1 << t)
                  for b in range(16) for t in range(16)]
        events.append(event("GEMV_QK", 512, duration=16, tokens=65535, columns=1))
        with self.assertRaisesRegex(AssertionError, "before fill"):
            audit_slot_events(events, "chain", require_final_release=False)

    def test_active_consumer_cannot_be_overwritten(self):
        events = ordinary_fragment()
        events.insert(17, event("GEMV_MULTICAST", 34, duration=0, generation=2))
        with self.assertRaisesRegex(AssertionError, "overwrote consuming"):
            audit_slot_events(events, "v")

    def test_partial_weight_consumption_cannot_be_released(self):
        events = ordinary_fragment()
        del events[-2]
        with self.assertRaisesRegex(AssertionError, "incomplete input/weight"):
            audit_slot_events(events, "v")

    def test_Q_distribution_port_conflict_is_detected(self):
        events = [event("GEMV_Q_LOCAL", 1, banks=1, tokens=1),
                  event("GEMV_Q_REMOTE", 2, banks=2, tokens=2)]
        with self.assertRaisesRegex(AssertionError, "distribution port"):
            audit_slot_events(events, "chain", require_final_release=False)

    def test_external_multicast_cannot_hide_added_latency(self):
        events = ordinary_fragment()
        events[0].update(ready_at=3, service_cycles=2)
        with self.assertRaisesRegex(AssertionError, "zero added internal latency"):
            audit_slot_events(events, "v")

    def test_external_fill_cannot_precede_DQ_completion(self):
        native = [dict(command="WRGB", Channel=0, clock=10)]
        fills = [event("GEMV_MULTICAST", 11, duration=0)]
        with self.assertRaisesRegex(AssertionError, "DQ completion"):
            audit_external_fills(native, fills)
        fills[0].update(clock=12, ready_at=12)
        self.assertEqual(audit_external_fills(native, fills), 1)

    def test_down_GPU_input_needs_every_token(self):
        events = [event("GEMV_MULTICAST", 1+2*t, duration=0, tokens=1 << t) for t in range(15)]
        events.append(event("GEMV_MAC", 33, duration=16, tokens=65535, columns=1))
        with self.assertRaisesRegex(AssertionError, "missing bank/token/lane"):
            audit_slot_events(events, "down_gpu_input", require_final_release=False)

    def test_down_consumes_exactly_four_weight_columns_per_fragment(self):
        fills = ordinary_fragment()[:16]
        macs = [event("GEMV_MAC", 33+16*c, duration=16, tokens=65535, columns=c+1) for c in range(4)]
        released = event("GEMV_RELEASE", 97, duration=0, tokens=65535, columns=4)
        audit_slot_events(fills+macs+[released], "down_gpu_input")
        with self.assertRaisesRegex(AssertionError, "Too many weight columns"):
            audit_slot_events(fills+macs+[event("GEMV_MAC", 97, duration=16, tokens=65535, columns=5)],
                              "down_gpu_input", require_final_release=False)

    def test_Q_ownership_blocks_ordinary_write_to_same_slot(self):
        events = [event("GEMV_Q_LOCAL", 1, banks=1, tokens=1),
                  event("GEMV_MULTICAST", 3, duration=0, tokens=2)]
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
                    expected = expected_addresses(case)
                    self.assertTrue(actual == expected, f"{case}: missing {sum((expected-actual).values())}, extra {sum((actual-expected).values())} address records")
                    self.assertEqual(manifest["buffers"]["gemv_total_bytes_per_bank"], 1024)
                    self.assertEqual(manifest["buffers"]["separate_rank_MAC_input_bytes"], 0)

    def test_correlated_trace_and_native_address_error_fails_independent_oracle(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "v.trace"
            workload.generate(trace, "v")
            records = trace_records(trace)
            target = next(e for e in records if e["operation"] == "PIM_MAC_AB")
            target["address"] ^= 1 << 20
            native = [dict(command=key[0], **dict(zip(GEOMETRY, key[1:])))
                      for key, count in expected_native_commands(records).items() for _ in range(count)]
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
            self.assertEqual(manifest["expected_dq_bytes"]["total"], 5472*1024)
            self.assertFalse(manifest["required_backend"]["pim_ffn_dram"])
            for a, b in zip(manifest["parts"], manifest["parts"][1:]):
                self.assertEqual(a["end_marker"], b["start_marker"])
                self.assertEqual(a["last_line"]+1, b["first_line"])

    def test_non_H16_geometry_is_rejected(self):
        import horizon_attention, horizon_value, horizon_ffn
        for call in (horizon_attention.layout_for, horizon_value.layout, horizon_ffn.layout):
            with self.assertRaises(ValueError):
                call(20)


class FFNLayoutTests(unittest.TestCase):
    def test_down_output_shards_cover_complete_outputs_and_bank_weight_space(self):
        import horizon_ffn as F
        ownership = Counter()
        for output in range(1024):
            first = F.decode_address(F.down_weight_location(0, output))
            last = F.decode_address(F.down_weight_location(4095, output))
            self.assertEqual(first[:3], last[:3])
            channel, rank, bank, _ = first
            self.assertEqual(rank, 1)
            self.assertEqual(channel, output//64)
            ownership[channel, bank] += 1
        self.assertEqual(set(ownership.values()), {4})
        self.assertEqual(len(ownership), 256)
        # Every sampled bank contains its four full 4096-feature dot products,
        # with no aliased cells, padding holes or dependence on another rank.
        for channel, bank in ((0, 0), (0, 15), (7, 9), (15, 15)):
            offsets = set()
            for output in range(channel*64+bank*4, channel*64+bank*4+4):
                for reduction in range(4096):
                    addr = F.down_weight_location(reduction, output)
                    owner = F.decode_address(addr)
                    self.assertEqual(owner[:3], (channel, 1, bank))
                    self.assertEqual(F.decode_down_weight(addr), (reduction, output))
                    offsets.add(owner[3])
            self.assertEqual(offsets, set(range(327680, 360448, 2)))

    def test_gate_up_pair_ownership_matches_for_GPU_elementwise_product(self):
        import horizon_ffn as F
        for feature in range(4096):
            for reduction in (0, 15, 16, 1023):
                gate = F.gu_weight_location("gate", reduction, feature)
                up = F.gu_weight_location("up", reduction, feature)
                self.assertNotEqual(gate, up)
                g, u = F.decode_address(gate), F.decode_address(up)
                self.assertEqual(g[:3], u[:3])
                self.assertEqual(g[0], feature//256)
                self.assertEqual(F.decode_gu_weight(gate), ("gate", reduction, feature))
                self.assertEqual(F.decode_gu_weight(up), ("up", reduction, feature))

    def test_every_rank_receives_the_full_GPU_Z_once_without_transpose(self):
        import horizon_ffn as F
        expected = set(range(0, 16*4096*2, 32))
        total = 0
        for channel in range(16):
            sources = set()
            for fragment in range(256):
                for token in range(16):
                    packet = F.gpu_input_packet("down", channel, fragment, token)
                    self.assertEqual(packet["source_tensor"], "Z_gpu")
                    self.assertEqual(packet["bytes"], 32)
                    self.assertEqual(packet["target_bank_mask"], 65535)
                    self.assertEqual(F.decode_address(packet["destination"]),
                                     (channel, 1, 0, 524288+(fragment*16+token)*32))
                    self.assertEqual(packet["storage_before_srf"], "bank0 DRAM")
                    self.assertEqual(F.decode_address(packet["gemv_destination"]), (channel, 1, 0, token*32))
                    self.assertEqual(packet["source_byte_offset"]//8192, token)
                    self.assertEqual(packet["source_byte_offset"]%8192//2, packet["feature_first"])
                    sources.add(packet["source_byte_offset"])
                    total += packet["bytes"]
            self.assertEqual(sources, expected)
        self.assertEqual(total, 2*1024**2)


if __name__ == "__main__":
    unittest.main()

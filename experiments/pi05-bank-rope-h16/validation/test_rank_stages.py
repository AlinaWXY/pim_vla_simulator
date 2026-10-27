"""Adversarial native-timing examples for the independent rank-stage audit."""
from collections import Counter
import copy
import unittest

import audit_rank_stages as A


def event(command='MACAB', clock=1, **changes):
    item = dict(command=command, clock=clock, Channel=0, Rank=1, BankGroup=-1,
                Bank=-1, Row=0, Column=0, pim_buffer=0, pim_rank_mask=2)
    item.update(changes)
    return item


def fragments(count=3, slots=2):
    writes, macs, clock = [], [], 1
    for fragment in range(count):
        for query in range(16):
            writes.append(event('WRGB', clock, Column=query, pim_buffer=fragment % 2))
            clock += 4
        for slot in range(slots):
            macs.append(event('MACAB', clock, Column=fragment*slots+slot, pim_buffer=fragment % 2))
            clock += 16
    return writes, macs


def check_fragments(writes, macs):
    return A.audit_fragment_dependencies(writes, macs, base=0, fragments=3,
        slots=2, local=False, local_ck=2, burst_ck=4)


class IndependentRankStages(unittest.TestCase):
    def test_independent_address_oracle_matches_matrix_work(self):
        for case, expected in A.EXPECTED_COUNTS.items():
            addresses = A.expected_addresses(case)
            self.assertEqual(Counter({op: sum(rows.values()) for op, rows in addresses.items()}), Counter(expected))
            self.assertTrue(all(key[1] == 1 for rows in addresses.values() for key in rows))
            self.assertEqual({key[0] for rows in addresses.values() for key in rows}, set(range(16)))
        # Local256-term reduction, not the old128-term reduction.
        self.assertEqual([p[1] for p in A.pipeline_shapes('down_tiled')], [16]*4)

    def test_native_dq_cannot_hide_an_extra_read(self):
        native = [dict(command=command) for command, count in
                  (('WRGB', 15872), ('MACAB', 7936), ('RDRES', 4096)) for _ in range(count)]
        self.assertEqual(A.audit_external_payload(native, 0, 'pv')['total'], 624*1024)
        with self.assertRaisesRegex(AssertionError, 'Native work counts'):
            A.audit_external_payload(native+[dict(command='RD')], 0, 'pv')

    def test_correct_rank_address_does_not_allow_rank_broadcast(self):
        A.audit_rank_scope([event()], [])
        for corrupt in (dict(Rank=0), dict(pim_rank_mask=3), dict(BankGroup=0)):
            with self.subTest(corrupt=corrupt), self.assertRaises(AssertionError):
                A.audit_rank_scope([event(**corrupt)], [])

    def test_native_row_must_be_open_in_every_bank(self):
        activate = event('ACTAB', 1, Row=10)
        mac = event('MACAB', 100, Row=10)
        self.assertEqual(A.audit_rows([activate, mac]), 16)
        wrong = dict(activate, Row=11)
        with self.assertRaisesRegex(AssertionError, 'closed/wrong row'):
            A.audit_rows([wrong, mac])
        precharge = event('PREpb', 50, BankGroup=1, Bank=2)
        with self.assertRaisesRegex(AssertionError, 'closed/wrong row'):
            A.audit_rows([activate, precharge, mac])

    def test_fragment_requires_all_sixteen_queries_ready(self):
        writes, macs = fragments()
        self.assertEqual(check_fragments(writes, macs), 3)
        early = copy.deepcopy(macs)
        early[0]['clock'] -= 1
        with self.assertRaisesRegex(AssertionError, 'before complete input'):
            check_fragments(writes, early)
        duplicate = copy.deepcopy(writes)
        duplicate[15]['Column'] = 14
        with self.assertRaisesRegex(AssertionError, 'sixteen-query'):
            check_fragments(duplicate, macs)

    def test_fragment_cannot_reuse_live_pingpong_buffer(self):
        writes, macs = fragments()
        writes[32]['clock'] = macs[1]['clock']+15
        with self.assertRaisesRegex(AssertionError, 'reused before previous consumer'):
            check_fragments(writes, macs)

    def test_fragment_cannot_swap_weight_columns(self):
        writes, macs = fragments()
        macs[0]['Column'], macs[1]['Column'] = macs[1]['Column'], macs[0]['Column']
        with self.assertRaisesRegex(AssertionError, 'output-column order'):
            check_fragments(writes, macs)

    def test_local_readiness_checks_actual_mac_not_only_recorded_field(self):
        native = [event('MACAB', 10)]
        local = [dict(Channel=0, Rank=1, pim_buffer=0, clock=26, ready_at=28,
                      bytes=32, port_cycles=2, buffer_in_use_until=26, port_ready_before=0)]
        A.audit_local_events(local, native, 2)
        forged = [dict(local[0], buffer_in_use_until=0)]
        with self.assertRaisesRegex(AssertionError, 'actual native MAC'):
            A.audit_local_events(forged, native, 2)
        early = [dict(local[0], clock=25, ready_at=27)]
        with self.assertRaisesRegex(AssertionError, 'before MAC completion'):
            A.audit_local_events(early, native, 2)

    def test_local_port_cannot_overlap_itself(self):
        first = dict(Channel=0, Rank=1, pim_buffer=0, clock=10, ready_at=12,
                     bytes=32, port_cycles=2, buffer_in_use_until=0, port_ready_before=0)
        second = dict(first, pim_buffer=1, clock=12, ready_at=14, port_ready_before=12)
        A.audit_local_events([first, second], [], 2)
        second.update(clock=11, ready_at=13)
        with self.assertRaisesRegex(AssertionError, 'Overlapping transfers'):
            A.audit_local_events([first, second], [], 2)

    def test_local_fragment_mac_waits_for_completion(self):
        writes, macs = fragments()
        locals_ = [dict(w, offset_bytes=w['Column']*32, ready_at=w['clock']+2) for w in writes]
        self.assertEqual(A.audit_fragment_dependencies(locals_, macs, base=0, fragments=3,
            slots=2, local=True, local_ck=2, burst_ck=4), 3)
        locals_[15]['ready_at'] = macs[0]['clock']+1
        with self.assertRaisesRegex(AssertionError, 'before complete input'):
            A.audit_fragment_dependencies(locals_, macs, base=0, fragments=3,
                slots=2, local=True, local_ck=2, burst_ck=4)

    def test_result_export_requires_compute_and_movement_tails(self):
        macs, exports = [event('MACAB', 100)], [event('RDRES', 116)]
        start, stop = A.tick(116, 32), A.tick(120, 32)
        A.audit_export_dependency(macs, exports, start, stop, 32, 4)
        with self.assertRaisesRegex(AssertionError, 'producer MAC'):
            A.audit_export_dependency(macs, [dict(exports[0], clock=115)], start, stop, 32, 4)
        with self.assertRaisesRegex(AssertionError, 'movement tail'):
            A.audit_export_dependency(macs, exports, start, stop-1, 32, 4)

    def test_native_two_cycle_burst_keeps_legacy_four_cycle_tail(self):
        macs, exports = [event('MACAB', 100)], [event('RDRES', 116)]
        start = A.tick(116, 32)
        A.audit_export_dependency(macs, exports, start, A.tick(120, 32), 32, 2)
        with self.assertRaisesRegex(AssertionError, 'movement tail'):
            A.audit_export_dependency(macs, exports, start, A.tick(118, 32), 32, 2)
        # Inputs become usable after nBL_min, without adding the output WAIT4.
        writes, macs = fragments()
        macs[0]['clock'] = writes[15]['clock']+2
        self.assertEqual(A.audit_fragment_dependencies(writes, macs, base=0,
            fragments=3, slots=2, local=False, local_ck=2, burst_ck=2), 3)


if __name__ == '__main__':
    unittest.main()

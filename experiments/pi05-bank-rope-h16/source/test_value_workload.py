"""Ownership, complete-cache coverage and GPU boundary checks for V/PV."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import value_layout as L
from value_workload import generate


def decode(address):
    # Independent RoBaRaCoCh decode: catches rank/channel bits as well as offsets.
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3,
            (address >> 20) * 2048 + (address >> 9 & 63) * 32 + (address & 31))


def read_trace(path):
    return [(f[0], int(f[1]), int(f[2]) if len(f) == 3 else None)
            for f in (line.split() for line in Path(path).read_text().splitlines())]


class ValueLayoutTests(unittest.TestCase):
    def test_every_feature_has_one_rank1_owner_with_eight_slots(self):
        for head in range(8):
            owners = defaultdict(list)
            for feature in range(256):
                owner, slot = L.feature_owner(head, feature)
                self.assertEqual(owner[1], 1)
                self.assertIn(owner[0], (2 * head, 2 * head + 1))
                self.assertEqual(slot * 32 + (owner[0] % 2) * 16 +
                                 owner[2] * 4 + owner[3], feature)
                owners[owner].append(slot)
                self.assertEqual(L.feature_from_slot(owner[0] % 2, owner[2], owner[3], slot), feature)
            self.assertEqual(len(owners), 32)
            self.assertTrue(all(slots == list(range(8)) for slots in owners.values()))

    def test_weight_cache_ranges_do_not_overlap_qk_or_ffn(self):
        self.assertEqual(L.ATTENTION_END, 82 * 1024)
        self.assertEqual(L.REGIONS['w_v']['base'], 82 * 1024)
        self.assertEqual(L.REGIONS['v_cache']['base'], 98 * 1024)
        self.assertEqual(L.manifest()['end_before_ffn'], 114 * 1024)
        self.assertLessEqual(L.manifest()['end_before_ffn'], 256 * 1024)
        for name, count, location in [('w_v', 1024, L.weight_location),
                                      ('v_cache', 992, L.value_location)]:
            by_bank = defaultdict(set)
            # One entire head suffices to enumerate every within-bank position.
            for feature in range(256):
                for reduction in range(count):
                    address = location(2, reduction, feature)
                    self.assertEqual(decode(L.physical_address(*address)), address)
                    by_bank[address[:4]].add(address[-1])
            base = L.REGIONS[name]['base']
            expected = set(range(base, base + count * 8 * 2, 2))
            self.assertEqual(len(by_bank), 32)
            self.assertTrue(all(offsets == expected for offsets in by_bank.values()))


class ValueTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.traces, cls.manifests = {}, {}
        for case in ('v', 'pv'):
            path = Path(cls.temp.name) / f'{case}.trace'
            cls.manifests[case] = generate(path, case=case)
            cls.traces[case] = read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rank1_only_and_complete_inputs_per_fragment(self):
        for case, fragments in [('v', 64), ('pv', 62)]:
            trace, manifest = self.traces[case], self.manifests[case]
            for op, address, _ in trace:
                if op not in ('BARRIER', 'WAIT'):
                    self.assertEqual(decode(address)[1], 1)
            self.assertEqual(len(manifest['input_fragments']), fragments)
            for fragment, record in enumerate(manifest['input_fragments']):
                commands = trace[record['line_start'] - 1:record['line_end']]
                got = [(op, decode(addr), buffer) for op, addr, buffer in commands]
                expected = [('PIM_WR_GB', (channel, 1, 0, 0, query * 32), fragment % 2)
                            for query in range(16) for channel in range(16)]
                self.assertEqual(got, expected)

    def test_one_full_weight_or_value_scan_reused_by_all_queries(self):
        # In particular PV must cover 992 tokens, not Q/K's 496-token rank shard.
        for case, base, columns in [('v', 82 * 1024, 512), ('pv', 98 * 1024, 496)]:
            scans = defaultdict(list)
            for op, addr, _ in self.traces[case]:
                if op == 'PIM_MAC_AB':
                    channel, rank, bg, bank, offset = decode(addr)
                    self.assertEqual((rank, bg, bank), (1, 0, 0))
                    scans[channel].append(offset)
            self.assertEqual(set(scans), set(range(16)))
            self.assertTrue(all(offsets == list(range(base, base + columns * 32, 32))
                                for offsets in scans.values()))
            self.assertEqual(self.manifests[case]['matrix']['mac_service_ck'], 16)
            # All 16 physical banks read a unique 32B payload per MAC request.
            expected_bytes = (4 * 1024**2 if case == 'v' else 8 * 992 * 256 * 2)
            self.assertEqual(sum(map(len, scans.values())) * 16 * 32, expected_bytes)

    def test_gpu_rmw_preserves_prefix_and_writes_both_tail_blocks(self):
        trace = self.traces['v']
        reads = [decode(addr) for op, addr, _ in trace if op == 'LD']
        writes = [decode(addr) for op, addr, _ in trace if op == 'ST']
        self.assertEqual(len(reads), len(set(reads)))
        self.assertEqual(len(writes), len(set(writes)))
        expected_reads, expected_writes = set(), set()
        base = 98 * 1024
        for channel in range(16):
            for bg in range(4):
                for bank in range(4):
                    for slot in range(8):
                        first = (channel, 1, bg, bank, base + (60 * 8 + slot) * 32)
                        expected_reads.add(first)
                        expected_writes.add(first)
                        expected_writes.add((channel, 1, bg, bank, base + (61 * 8 + slot) * 32))
        self.assertEqual(set(reads), expected_reads)
        self.assertEqual(set(writes), expected_writes)
        self.assertFalse(any(op in ('LD', 'ST') for op, _, _ in self.traces['pv']))

    def test_dq_bytes_and_fp32_exports_from_actual_trace(self):
        for case, expected in [('v', (512, 128, 64, 128, 832)),
                               ('pv', (496, 128, 0, 0, 624))]:
            counts = Counter(op for op, _, _ in self.traces[case])
            commands = ('PIM_WR_GB', 'PIM_RD_RES', 'LD', 'ST')
            amounts = [counts[op] * 32 for op in commands]
            self.assertEqual(tuple(amounts + [sum(amounts)]), tuple(n * 1024 for n in expected))
            per_channel = defaultdict(list)
            for op, addr, _ in self.traces[case]:
                if op == 'PIM_RD_RES':
                    ch, rank, bg, bank, offset = decode(addr)
                    self.assertEqual((rank, bg, bank), (1, 0, 0))
                    per_channel[ch].append(offset)
            self.assertEqual(set(per_channel), set(range(16)))
            self.assertTrue(all(offsets == list(range(0, 8192, 32))
                                for offsets in per_channel.values()))
            self.assertEqual(self.manifests[case]['expected_dq_bytes']['total'], sum(amounts))

    def test_legacy_completion_tail_precedes_export(self):
        for case in ('v', 'pv'):
            trace = self.traces[case]
            first_export = next(i for i, (op, _, _) in enumerate(trace) if op == 'PIM_RD_RES')
            self.assertEqual([op for op, _, _ in trace[first_export - 3:first_export]],
                             ['BARRIER', 'WAIT', 'BARRIER'])
            self.assertEqual(trace[first_export - 2][1], 16)
            last_export = max(i for i, (op, _, _) in enumerate(trace) if op == 'PIM_RD_RES')
            self.assertEqual([op for op, _, _ in trace[last_export + 1:last_export + 4]],
                             ['BARRIER', 'WAIT', 'BARRIER'])
            self.assertEqual(trace[last_export + 2][1], 4)
            backend = self.manifests[case]['required_backend']
            self.assertFalse(backend['pim_rank_rope'])
            self.assertEqual(backend['pim_mac_scope'], 'rank')
            self.assertEqual(backend['nCCDAB'], 16)


if __name__ == '__main__':
    unittest.main()

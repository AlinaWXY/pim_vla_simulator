"""Whole-H token count, variable work-row placement and KV tail conservation."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import horizon_value as V


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 16 & 3,
            address >> 18 & 3, (address >> 20)*2048+(address >> 9 & 63)*32+(address & 31))


class HorizonValueTests(unittest.TestCase):
    def test_dynamic_attention_end_keeps_weight_and_cache_regions_disjoint(self):
        for h, padded, vbase, cache in ((16, 992, 83968, 100352), (20, 992, 83968, 100352),
                                       (30, 1024, 83968, 100352), (40, 1024, 86016, 102400)):
            with self.subTest(h=h):
                layout = V.layout(h)
                self.assertEqual(layout['padded_sequence'], padded)
                self.assertEqual(layout['regions']['w_v']['base'], vbase)
                self.assertEqual(layout['regions']['v_cache']['base'], cache)
                self.assertEqual(layout['regions']['v_cache']['reserved_bytes_per_bank'], 16384)
                self.assertLessEqual(layout['end_before_ffn'], 262144)
                self.assertEqual(layout['input_double_buffer_bytes_per_rank'], h*64)
                self.assertEqual(layout['accumulator_bytes_per_bank'], h*32)

    def test_same_feature_ownership_for_every_horizon(self):
        for head in range(8):
            owners = defaultdict(set)
            for feature in range(256):
                owner, slot = V.feature_owner(head, feature)
                self.assertEqual(owner[1], 1)
                self.assertEqual(feature, slot*32+(owner[0] % 2)*16+owner[2]*4+owner[3])
                owners[owner].add(slot)
            self.assertEqual(len(owners), 32)
            self.assertTrue(all(slots == set(range(8)) for slots in owners.values()))

    def test_whole_horizon_inputs_outputs_and_native_dq_budget(self):
        expected = {16: (832, 624), 20: (992, 780), 30: (1520, 1200), 40: (1920, 1600)}
        with tempfile.TemporaryDirectory() as tmp:
            for h in (16, 20, 30, 40):
                for case, dq_kib in zip(('v', 'pv'), expected[h]):
                    with self.subTest(h=h, case=case):
                        path = Path(tmp)/f'{case}-{h}.trace'
                        manifest = V.generate(path, h, case)
                        trace = [(f[0], int(f[1]), int(f[2]) if len(f) == 3 else -1)
                                 for f in (line.split() for line in path.read_text().splitlines())]
                        counts = Counter(op for op, _, _ in trace)
                        self.assertEqual(sum(counts[op]*32 for op in ('PIM_WR_GB','PIM_RD_RES','LD','ST')), dq_kib*1024)
                        self.assertEqual(manifest['expected_dq_bytes']['total'], dq_kib*1024)
                        self.assertEqual(manifest['matrix']['M'], h)
                        self.assertEqual(manifest['required_backend']['nCCDAB'], h)
                        self.assertEqual(manifest['query_padding'], 0)
                        self.assertEqual([value for op, value, _ in trace if op == 'WAIT'], [h, 4])
                        fragments = 64 if case == 'v' else (62 if h <= 20 else 64)
                        self.assertEqual(counts['PIM_MAC_AB'], fragments*8*16)
                        self.assertEqual(counts['PIM_RD_RES']*32, h*8*1024)
                        for record in manifest['input_fragments']:
                            rows = trace[record['line_start']-1:record['line_end']]
                            self.assertEqual([(op, decode(addr), buf) for op, addr, buf in rows],
                                [('PIM_WR_GB', (ch, 1, 0, 0, query*32), record['fragment'] % 2)
                                 for query in range(h) for ch in range(16)])
                        self.assertTrue(all(decode(addr)[1] == 1 for op, addr, _ in trace if op not in ('BARRIER','WAIT')))

    def test_one_prefix_rmw_then_complete_suffix_and_masked_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            for h in (20, 30, 40):
                path = Path(tmp)/str(h)
                manifest = V.generate(path, h, 'v')
                cache = 102400 if h == 40 else 100352
                blocks = (60, 61) if h == 20 else (60, 61, 62, 63)
                reads, writes = [], []
                for line in path.read_text().splitlines():
                    op, value, *_ = line.split()
                    if op == 'LD': reads.append(decode(int(value)))
                    if op == 'ST': writes.append(decode(int(value)))
                expected_read, expected_write = set(), set()
                for ch in range(16):
                    for bg in range(4):
                        for bank in range(4):
                            for slot in range(8):
                                expected_read.add((ch,1,bg,bank,cache+(60*8+slot)*32))
                                for block in blocks:
                                    expected_write.add((ch,1,bg,bank,cache+(block*8+slot)*32))
                self.assertEqual(Counter(reads), Counter(expected_read))
                self.assertEqual(Counter(writes), Counter(expected_write))
                self.assertEqual(manifest['layout']['suffix_tokens'], [968, 968+h])
                self.assertEqual(manifest['layout']['padding_tokens'], [968+h, 992 if h == 20 else 1024])


if __name__ == '__main__':
    unittest.main()

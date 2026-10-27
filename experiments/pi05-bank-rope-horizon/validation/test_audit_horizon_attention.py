"""H30 score-burst and H40 row/coverage boundary tests, independent of generator."""
from contextlib import contextmanager
import copy
import unittest

import audit_horizon_attention as A


@contextmanager
def horizon(value):
    old = A.QUERIES, A.LOCAL_SUFFIX, A.LOCAL_PADDED
    A.QUERIES, A.LOCAL_SUFFIX = value, value//2
    A.LOCAL_PADDED = {16:496, 20:496, 30:512, 40:512}[value]
    try:
        yield
    finally:
        A.QUERIES, A.LOCAL_SUFFIX, A.LOCAL_PADDED = old


def event(operation, byte, clock=100, duration=40, channel=0, bank=0, **extra):
    row, within = divmod(byte, 2048)
    item = dict(operation=operation, Channel=channel, Rank=1, BankGroup=bank//4,
        Bank=bank%4, Row=row, Column=within//32, clock=clock, ready_at=clock+duration,
        service_cycles=duration, target_banks=16, buffer=0)
    item.update(extra)
    return item


def key_byte(token, value):
    return 65536+token*32 if token<484 else 81920+(value+token-484)*32


def projection_events(value):
    local = value//2
    events = []
    for channel in range(16):
        for fragment in range(64):
            start = 100+fragment*2000
            for token in range(local):
                events.append(event('PIM_WR_GB',token*32,start+token*2,2,channel,
                    buffer=fragment%2,source_line=len(events)+1))
            for slot in range(32):
                events.append(event('PIM_MAC_AB',(fragment*32+slot)*32,
                    start+local*2+10+slot*local,local,channel,
                    buffer=fragment%2,source_line=len(events)+1))
    return events


class WholeHorizonAttentionBoundaries(unittest.TestCase):
    def test_H30_score_burst_requires_both_overlapping_K_tokens(self):
        with horizon(30):
            reductions = [event('PIM_RANK_REDUCE',key_byte(0,30),60,40),
                          event('PIM_RANK_REDUCE',key_byte(1,30),90,40)]
            # Scores0..119 belong to K0. Burst96..127 includes K0 and K1.
            A.audit_score_readiness([event('PIM_RD_BG',96,130,2)],reductions)
            with self.assertRaisesRegex(AssertionError,'precedes its K-token'):
                A.audit_score_readiness([event('PIM_RD_BG',96,129,2)],reductions)
            with self.assertRaisesRegex(AssertionError,'no corresponding'):
                A.audit_score_readiness([event('PIM_RD_BG',96,130,2)],reductions[:1])

    def test_H30_last_score_burst_and_out_of_range(self):
        with horizon(30):
            last = event('PIM_RANK_REDUCE',key_byte(511,30),60,40)
            A.audit_score_readiness([event('PIM_RD_BG',512*120-32,100,2)],[last])
            with self.assertRaisesRegex(AssertionError,'outside'):
                A.audit_score_readiness([event('PIM_RD_BG',512*120,100,2)],[last])

    def test_H40_K_padding_continues_into_next_row(self):
        with horizon(40):
            macs = [event('PIM_MAC_QK',key_byte(token,40),100+token*40,40,channel)
                    for channel in range(16) for token in range(512)]
            self.assertEqual(A.audit_k_scan(macs),4*1024**2)
            beyond = [e for e in macs if e['Row']==41]
            self.assertEqual(len(beyond),16*4)
            self.assertEqual({e['Column'] for e in beyond},{0,1,2,3})
            with self.assertRaisesRegex(AssertionError,'K scan missing'):
                A.audit_k_scan([e for e in macs if e['Row']!=41])
            duplicate = list(macs)
            duplicate[-1] = duplicate[-2]
            with self.assertRaisesRegex(AssertionError,'K scan missing'):
                A.audit_k_scan(duplicate)
            last = event('PIM_RANK_REDUCE',key_byte(511,40),60,40)
            self.assertEqual((last['Row'],last['Column']),(41,3))
            A.audit_score_readiness([event('PIM_RD_BG',512*160-32,100,2)],[last])

    def test_projection_counts_and_service_follow_half_horizon(self):
        for value in (20,30,40):
            with horizon(value):
                events = projection_events(value)
                self.assertEqual(set(A.audit_projection(events)),set(range(16)))
                missing = copy.copy(events)
                missing.pop(next(i for i,e in enumerate(events) if e['operation']=='PIM_WR_GB'
                                 and e['Column']==value//2-1))
                with self.assertRaisesRegex(AssertionError,'input fragment count'):
                    A.audit_projection(missing)
                wrong = copy.deepcopy(events)
                mac = next(e for e in wrong if e['operation']=='PIM_MAC_AB')
                mac['service_cycles'] = 8
                with self.assertRaisesRegex(AssertionError,'local tokens'):
                    A.audit_projection(wrong)

    def test_H40_per_bank_local_and_peer_cover_queries_above_bit31(self):
        with horizon(40):
            exports, remotes = [], []
            for channel in range(16):
                for bank in range(16):
                    for token in range(20):
                        byte = 81920+(20+token)*32
                        exports.append(event('PIM_Q_EXPORT',byte,100,25,channel,bank))
                        remotes.append(event('PIM_Q_REMOTE',byte,125,2,channel,bank))
            A.expected_geometry(exports,'PIM_Q_EXPORT',range(20,40))
            A.expected_geometry(remotes,'PIM_Q_REMOTE',range(20,40))
            A.audit_peer_dependencies(exports,remotes)
            logical_queries = {2*token+parity for token in range(20) for parity in range(2)}
            self.assertEqual(logical_queries,set(range(40)))
            self.assertEqual(sum(1<<q for q in logical_queries),(1<<40)-1)
            truncated = [e for e in exports if e['Column']<36]
            with self.assertRaisesRegex(AssertionError,'exactly once'):
                A.expected_geometry(truncated,'PIM_Q_EXPORT',range(20,40))
            bad_remote = copy.deepcopy(remotes)
            bad_remote[-1]['clock'] = 124
            with self.assertRaisesRegex(AssertionError,'precedes peer export'):
                A.audit_peer_dependencies(exports,bad_remote)


if __name__ == '__main__':
    unittest.main()

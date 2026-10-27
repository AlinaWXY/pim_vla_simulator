"""Fault examples ensure whole-H auditing cannot silently accept H16 work."""
from collections import Counter
import copy
import unittest

import audit_horizon_value as A


def event(command, clock, byte, buffer):
    return dict(command=command, clock=clock, Channel=0, Rank=1, BankGroup=-1,
                Bank=-1, Row=byte//2048, Column=byte % 2048//32,
                pim_buffer=buffer, pim_rank_mask=2)


def fragment_events(h, fragments=3):
    writes, macs, clock = [], [], 1
    for fragment in range(fragments):
        for query in range(h):
            writes.append(event('WRGB', clock, query*32, fragment % 2))
            clock += 4
        for slot in range(8):
            macs.append(event('MACAB', clock, 100352+(fragment*8+slot)*32, fragment % 2))
            clock += h
    return writes, macs


class WholeHorizonValueAudit(unittest.TestCase):
    def test_counts_are_derived_from_whole_horizon_and_kv_padding(self):
        expected_kib = {16:(832,624),20:(992,780),30:(1520,1200),40:(1920,1600)}
        for h in expected_kib:
            for case, kib in zip(('v','pv'),expected_kib[h]):
                oracle=A.expected_addresses(h,case)
                counts=Counter({op:sum(addresses.values()) for op,addresses in oracle.items()})
                self.assertEqual(counts,A.shape(h,case)['counts'])
                self.assertEqual(sum(counts[op]*32 for op in ('PIM_WR_GB','PIM_RD_RES','LD','ST')),kib*1024)
        self.assertNotEqual(A.shape(30,'v')['base'],A.shape(40,'v')['base'])

    def test_tail_query_rows_cannot_be_missing_or_duplicated(self):
        for h in (20,30,40):
            writes,macs=fragment_events(h)
            A.audit_fragments(writes,macs,horizon=h,base=100352,fragments=3,burst_ck=2)
            with self.assertRaisesRegex(AssertionError,'whole-H matrix'):
                A.audit_fragments(writes[:-1],macs,horizon=h,base=100352,fragments=3,burst_ck=2)
            bad=copy.deepcopy(writes)
            bad[h-1]['Column']=h-2
            with self.assertRaisesRegex(AssertionError,'H query row'):
                A.audit_fragments(bad,macs,horizon=h,base=100352,fragments=3,burst_ck=2)

    def test_mac_must_wait_for_last_query_input_and_full_h_service(self):
        writes,macs=fragment_events(30)
        bad=copy.deepcopy(macs)
        bad[0]['clock']=writes[29]['clock']+1
        with self.assertRaisesRegex(AssertionError,'all H input rows'):
            A.audit_fragments(writes,bad,horizon=30,base=100352,fragments=3,burst_ck=2)
        bad=copy.deepcopy(macs)
        bad[1]['clock']=bad[0]['clock']+16
        with self.assertRaisesRegex(AssertionError,'all H query vectors'):
            A.audit_fragments(writes,bad,horizon=30,base=100352,fragments=3,burst_ck=2)

    def test_buffer_lifetime_uses_h_not_legacy16(self):
        writes,macs=fragment_events(40)
        writes[80]['clock']=macs[7]['clock']+16
        with self.assertRaisesRegex(AssertionError,'before prior MAC'):
            A.audit_fragments(writes,macs,horizon=40,base=100352,fragments=3,burst_ck=2)


if __name__=='__main__':
    unittest.main()

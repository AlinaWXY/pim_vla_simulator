"""Full-Wo residency proof, separate from the frozen active native workload.

The full logical matrix is a product of eight head blocks. Exhaustively prove
one block's cell mapping, then prove every rank/head/bank's translated block
is disjoint, covers the reservation, and decodes to the intended global head.
"""
from collections import Counter
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"source"))
import o_projection_layout as O


def frozen_active_location(channel,feature,output):
    """Independent formula of the addresses used by the native experiment."""
    tile=output//512
    bank=(output%512)//32
    column=output%32
    pair=feature%128
    fragment=pair//8
    lane=2*(pair%8)+feature//128
    offset=786432+tile*16384+fragment*1024+column*32+lane*2
    return channel,1,bank//4,bank%4,offset


class FullOWeightResidencyTests(unittest.TestCase):
    def test_one_complete_head_block_is_a_bijection(self):
        # Head and rank only translate/rotate this common inner layout. Check
        # all 262144 cells, including every lane, fragment, column and bank.
        channel,head=5,6
        positions=set()
        counts=Counter()
        for feature in range(256):
            for output in range(1024):
                loc=O.resident_weight_location(channel,head*256+feature,output)
                self.assertNotIn(loc,positions)
                positions.add(loc)
                counts[loc[2]*4+loc[3]]+=1
                self.assertEqual(O.decode_resident_weight(loc),(channel,head*256+feature,output))
                active=frozen_active_location(channel,feature,output)
                self.assertEqual(loc,(*active[:-1],active[-1]+4*32768))
        self.assertEqual(len(positions),256*1024)
        self.assertEqual(counts,Counter({bank:16384 for bank in range(16)}))

    def test_every_rank_head_and_bank_has_disjoint_complete_intervals(self):
        for channel in range(16):
            for bank in range(16):
                intervals=[]
                for head in range(8):
                    slot=(head-channel//2)%8
                    first=O.resident_weight_location(channel,head*256,bank*32)
                    last=O.resident_weight_location(channel,head*256+255,512+bank*32+31)
                    begin,end=786432+slot*32768,786432+(slot+1)*32768
                    self.assertEqual(first,(channel,1,bank//4,bank%4,begin))
                    self.assertEqual(last,(channel,1,bank//4,bank%4,end-2))
                    self.assertEqual(O.decode_resident_weight(first),(channel,head*256,bank*32))
                    self.assertEqual(O.decode_resident_weight(last),(channel,head*256+255,512+bank*32+31))
                    self.assertGreaterEqual(begin,786432)
                    self.assertLessEqual(end,1048576)
                    intervals.append((begin,end))
                self.assertEqual(sorted(intervals),[(786432+s*32768,786432+(s+1)*32768) for s in range(8)])
        # All existing non-Wo allocations end below this reservation. The
        # calibrated 8-Gbit x16 organization has 32768 rows of 2048 B/bank.
        self.assertGreaterEqual(786432,655360+16384)
        self.assertLessEqual(1048576,32768*2048)

    def test_active_native_addresses_remain_exact_for_every_rank(self):
        # Every local feature, rank, bank and output tile, with both edge and
        # interior output columns. The exhaustive block test covers all cols.
        outputs=[tile*512+bank*32+column for tile in range(2)
            for bank in range(16) for column in (0,1,30,31)]
        for channel in range(16):
            head,parity=divmod(channel,2)
            for feature in range(256):
                for output in outputs:
                    expected=frozen_active_location(channel,feature,output)
                    self.assertEqual(O.weight_location(head,parity,feature,output),expected)
                    self.assertEqual(O.resident_weight_location(channel,head*256+feature,output),expected)
                    self.assertEqual(O.decode_weight(expected),(head,parity,feature,output))

    def test_capacity_contract_distinguishes_resident_and_accessed_weights(self):
        contract=O.contract()
        weights=contract["weights"]
        self.assertEqual(contract["weight_shape_per_rank"],[2048,1024])
        self.assertEqual(contract["active_weight_shape_per_rank"],[256,1024])
        self.assertEqual(weights["bytes_per_rank"],2048*1024*2)
        self.assertEqual(weights["total_bytes"],16*2048*1024*2)
        self.assertEqual(weights["active_bytes_per_rank"],256*1024*2)
        self.assertEqual(weights["active_total_bytes"],16*256*1024*2)
        self.assertEqual(weights["inactive_total_bytes"],56*1024*1024)
        self.assertEqual(weights["bytes_per_bank"]*16,weights["bytes_per_rank"])
        self.assertEqual(weights["end_bytes_exclusive"]-weights["base_bytes"],weights["bytes_per_bank"])
        self.assertEqual(weights["active_end_bytes_exclusive"],819200)
        self.assertEqual(weights["copies_per_head"],16)
        self.assertEqual(weights["active_copies_per_head"],2)
        self.assertEqual(weights["initialization_total_bytes"],weights["total_bytes"])
        self.assertTrue(weights["initialization_excluded"])
        self.assertIsNone(weights["initialization_latency_us"])
        self.assertEqual(weights["inactive_weight_reads_per_step"],0)

    def test_invalid_global_cells_and_physical_addresses_are_rejected(self):
        for args in ((-1,0,0),(16,0,0),(0,-1,0),(0,2048,0),(0,0,-1),(0,0,1024)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                O.resident_weight_location(*args)
        for loc in ((0,0,0,0,786432),(16,1,0,0,786432),(0,1,4,0,786432),
                    (0,1,0,4,786432),(0,1,0,0,786430),(0,1,0,0,786433),(0,1,0,0,1048576)):
            with self.subTest(location=loc),self.assertRaises(ValueError):
                O.decode_resident_weight(loc)


if __name__=="__main__":
    unittest.main()

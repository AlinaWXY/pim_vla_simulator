"""Negative checks for mask evidence, physical rows and split-ACT ownership."""
from copy import deepcopy
import unittest
from act2_audit import audit_act2
from audit_native import audit_array_rows
from masked_audit import annotate_masked_commands


def event(command, clock, bank=0, row=512, mask=None):
    e=dict(command=command, clock=clock, ready_at=clock+(2 if command=="ACTMASK" else 1), Channel=0, Rank=1,
        BankGroup=bank//4 if bank>=0 else -1, Bank=bank%4 if bank>=0 else -1,
        Row=row, Column=0, type=27)
    if mask is not None:e['bank_mask']=mask
    return e


class MaskedCommandTests(unittest.TestCase):
    def test_exact_mask_binding_rejects_missing_duplicate_and_data_bank(self):
        native=[event('MACMASK',8,-1)]
        masks=[dict(native[0],bank_mask=65534)]
        bound=annotate_masked_commands(native,masks,1)
        self.assertEqual(bound[0]['bank_mask'],65534)
        self.assertNotIn('bank_mask',native[0])
        for wrong,pattern in (([], 'unique mask'),(masks*2,'unique mask'),
                              ([dict(masks[0],bank_mask=65535)],'dedicated data bank')):
            with self.subTest(pattern=pattern),self.assertRaisesRegex(AssertionError,pattern):
                annotate_masked_commands(native,wrong,1)

    def test_tail_column_excludes_exactly_padding_banks(self):
        # 256 features over15 compute banks: column17 has exactly one feature.
        native=[dict(event('MACMASK',8,-1),Column=17)]
        annotate_masked_commands(native,[dict(native[0],bank_mask=2)],1)
        with self.assertRaisesRegex(AssertionError,'computes padding'):
            annotate_masked_commands(native,[dict(native[0],bank_mask=6)],1)

    def test_pending_data_activation_allows_disjoint_MAC_but_not_data_access(self):
        good=[event('ACT1',1,0,256),event('MACMASK',2,-1,512,65534),event('ACT2',3,0,256)]
        self.assertEqual(audit_act2(good,dict(nAAD=8))['status'],'passed')
        bad=deepcopy(good);bad[1]['bank_mask']=65535
        with self.assertRaisesRegex(AssertionError,'target_bank_access_before_ACT2'):
            audit_act2(bad,dict(nAAD=8))
        bad=deepcopy(good);bad[1].pop('bank_mask')
        with self.assertRaisesRegex(AssertionError,'bound physical bank masks'):
            audit_act2(bad,dict(nAAD=8))

    def test_masked_precharge_preserves_other_open_rows_and_pending_owner(self):
        native=[event('ACT1',1,0,256),event('ACTMASK',2,-1,512,65534),
                event('ACT2',4,0,256),event('PREMASK',5,-1,512,65534),
                event('LOCALRD',6,0,256)]
        audit_act2(native,dict(nAAD=8));audit_array_rows(native)
        bad=native+[event('MACMASK',7,-1,512,2)]
        with self.assertRaisesRegex(AssertionError,'wrong/closed row'):
            audit_array_rows(bad)
        pending=[event('ACT1',1,0,256),event('PREMASK',2,-1,512,1)]
        with self.assertRaisesRegex(AssertionError,'pending_ACT1_cancelled'):
            audit_act2(pending,dict(nAAD=8))


if __name__=='__main__':unittest.main()

"""Small adversarial cases for the independent auditor, without Ramulator."""
import copy
import unittest

import audit_run as A


def event(channel=0, column=8, clock=10, ready=35, **extra):
    value = dict(Channel=channel, Rank=1, BankGroup=0, Bank=0,
                 Row=40, Column=column, clock=clock, ready_at=ready)
    value.update(extra)
    return value


class AuditCounterexamples(unittest.TestCase):
    def test_peer_export_must_complete_not_merely_issue(self):
        source = [event()]
        remote = [event(channel=1, clock=35, ready=37)]
        A.audit_peer_dependencies(source, remote)
        early = copy.deepcopy(remote)
        early[0].update(clock=34, ready_at=36)
        with self.assertRaisesRegex(AssertionError, 'precedes peer export completion'):
            A.audit_peer_dependencies(source, early)
        wrong_bank = copy.deepcopy(remote)
        wrong_bank[0]['Bank'] = 1
        with self.assertRaisesRegex(AssertionError, 'no peer export'):
            A.audit_peer_dependencies(source, wrong_bank)

    def test_k_duplicate_replacing_missing_token_preserves_count(self):
        macs = []
        for c in range(16):
            for token in range(496):
                byte = 65536 + token * 32 if token < 484 else 81920 + (token - 484 + 16) * 32
                macs.append(event(channel=c, column=byte % 2048 // 32, Row=byte // 2048,
                                  clock=100, ready=116, service_cycles=16, target_banks=16))
        self.assertEqual(A.audit_k_scan(macs), 8 * 992 * 256 * 2)
        corrupt = copy.deepcopy(macs)
        corrupt[1]['Column'] = corrupt[0]['Column']
        with self.assertRaisesRegex(AssertionError, 'K scan missing/duplicates'):
            A.audit_k_scan(corrupt)

    def test_native_dq_counts_cannot_hide_an_extra_read(self):
        exports, remotes, scores = [None] * 2048, [None] * 2048, [None] * 15872
        native = ([dict(command='WRGB')] * 10240 + [dict(command='RD')] * 2048
                  + [dict(command='RDRES')] * 15872 + [dict(command='LOCALRD')] * 20)
        self.assertEqual(A.audit_external_payload(native, exports, remotes, scores, 8192)['total'], 880 * 1024)
        with self.assertRaisesRegex(AssertionError, 'unaccounted external DQ'):
            A.audit_external_payload(native + [dict(command='RD')], exports, remotes, scores, 8192)

    def test_valid_address_is_insufficient_when_wrong_row_open(self):
        activate = event(clock=1, command='ACT2')
        read = event(clock=40, command='LOCALRD')
        self.assertEqual(A.audit_rows([activate, read]), 1)
        bad_activate = dict(activate, Row=39)
        with self.assertRaisesRegex(AssertionError, 'closed/wrong row'):
            A.audit_rows([bad_activate, read])

    def test_rope_output_fifo_is_not_unbounded(self):
        events = [event(column=8, clock=10, operation='PIM_ROPE_EXEC'),
                  event(column=9, clock=11, operation='PIM_ROPE_EXEC'),
                  event(column=10, clock=22, operation='PIM_ROPE_EXEC'),
                  event(column=8, clock=20, operation='PIM_ROPE_STORE'),
                  event(column=9, clock=24, operation='PIM_ROPE_STORE'),
                  event(column=10, clock=28, operation='PIM_ROPE_STORE')]
        A.audit_output_fifo(events, 2)
        events[2]['clock'] = 12
        with self.assertRaisesRegex(AssertionError, 'exceeds two entries'):
            A.audit_output_fifo(events, 2)

    def test_tiled_scores_wait_for_their_own_reduction(self):
        reductions = [event(Row=32, column=0, ready=20),
                      event(Row=32, column=16, ready=200),
                      event(Row=40, column=16, ready=400)]
        scores = []
        for token, clock in ((0, 30), (16, 210), (484, 410)):
            for half in (0, 1):
                byte = token * 64 + half * 32
                scores.append(event(Row=byte // 2048, column=byte % 2048 // 32,
                                    clock=clock + half * 2))
        # Earlier tiles may leave before later tiles finish; suffix K lives in
        # a noncontiguous work row but still maps to its token-major scores.
        A.audit_score_readiness(scores, reductions)
        for index, too_early in ((0, 19), (2, 199), (4, 399)):
            corrupted = copy.deepcopy(scores)
            corrupted[index]['clock'] = too_early
            with self.assertRaisesRegex(AssertionError, 'precedes its K-token rank reduction'):
                A.audit_score_readiness(corrupted, reductions)


if __name__ == '__main__':
    unittest.main()

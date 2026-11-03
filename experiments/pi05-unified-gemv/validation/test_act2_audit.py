"""Protocol faults that open-row and aggregate-command-count checks miss."""
import unittest

from act2_audit import audit_act2, audit_staging_act2


def event(clock, command, *, channel=0, rank=1, bg=0, bank=0, row=256, type_id=27):
    return dict(clock=clock, command=command, Channel=channel, Rank=rank,
                BankGroup=bg, Bank=bank, Row=row, Column=0, type=type_id)


class SplitActivationTests(unittest.TestCase):
    def audit(self, events, **kwargs):
        return audit_act2(events, {"nAAD": 8}, **kwargs)

    def test_scoped_pass_does_not_hide_other_internal_failure(self):
        events = [event(1, "ACT1", type_id=14), event(20, "ACT2", type_id=14),
                  event(30, "ACT1"), event(31, "ACT2")]
        result = audit_staging_act2(events, {"nAAD": 8})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["global_audit"]["status"], "failed")
        self.assertEqual(result["known_other_internal_violations"]["late_ACT2"], 1)
        with self.assertRaisesRegex(AssertionError, "late_ACT2"):
            self.audit(events)

    def test_scoped_audit_catches_other_path_interrupting_staging(self):
        for command in ("PREab", "REFab", "ACT1"):
            events = [event(1, "ACT1"), event(2, command, type_id=14), event(3, "ACT2")]
            with self.subTest(command=command), self.assertRaisesRegex(AssertionError, "Staging ACT1/ACT2"):
                audit_staging_act2(events, {"nAAD": 8})

    def test_new_staging_ACT1_cannot_ignore_other_pending_owner(self):
        events = [event(1, "ACT1", type_id=14), event(2, "ACT1", bank=1),
                  event(3, "ACT2", type_id=14), event(4, "ACT2", bank=1)]
        with self.assertRaisesRegex(AssertionError, "ACT1_while_channel_ACT2_pending"):
            audit_staging_act2(events, {"nAAD": 8})

    def test_deadline_is_inclusive_and_gap_is_measured_per_physical_bank(self):
        result = self.audit([event(10, "ACT1"), event(18, "ACT2"),
                             event(20, "ACT1", rank=0, row=4, type_id=0),
                             event(21, "ACT2", rank=0, row=4, type_id=0)])
        self.assertEqual(result["gap_histogram_ck"], {1: 1, 8: 1})
        self.assertEqual(result["classifications"]["staging"]["paired_ACT2"], 1)
        self.assertEqual(result["classifications"]["ordinary"]["paired_ACT2"], 1)

    def test_late_ACT2_is_rejected_even_when_row_is_correct(self):
        with self.assertRaisesRegex(AssertionError, "late_ACT2"):
            self.audit([event(10, "ACT1"), event(19, "ACT2")])

    def test_zero_cycle_ACT2_is_not_a_valid_pair(self):
        with self.assertRaisesRegex(AssertionError, "before_ACT1_command_finishes"):
            self.audit([event(10, "ACT1"), event(10, "ACT2")])

    def test_PRE_cannot_cancel_pending_target(self):
        for command in ("PREpb", "PREab", "PREHB"):
            with self.subTest(command=command), self.assertRaisesRegex(AssertionError, "pending_ACT1_cancelled"):
                self.audit([event(10, "ACT1", bg=1), event(11, command, bg=1), event(12, "ACT2", bg=1)])

    def test_refresh_cannot_cancel_pending_target(self):
        for command in ("REFpb", "REFab"):
            with self.subTest(command=command), self.assertRaisesRegex(AssertionError, "pending_ACT1_cancelled"):
                self.audit([event(10, "ACT1"), event(11, command), event(12, "ACT2")])

    def test_orphan_and_wrong_row_ACT2_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "orphan_ACT2"):
            self.audit([event(11, "ACT2")])
        with self.assertRaisesRegex(AssertionError, "ACT2_row_mismatch"):
            self.audit([event(10, "ACT1"), event(11, "ACT2", row=257)])

    def test_pending_activation_must_not_survive_end_of_log(self):
        with self.assertRaisesRegex(AssertionError, "pending_ACT2_at_end"):
            self.audit([event(10, "ACT1")])

    def test_read_and_precharge_other_banks_are_legal(self):
        result = self.audit([event(10, "ACT1"), event(11, "RD", bank=1),
            event(12, "PREpb", bank=1), event(13, "PREHB", bg=2),
            event(14, "PREab", rank=0), event(15, "ACT2")])
        self.assertEqual(result["paired_ACT2"], 1)

    def test_read_of_activating_bank_is_not_a_substitute_for_ACT2(self):
        with self.assertRaisesRegex(AssertionError, "target_bank_access_before_ACT2"):
            self.audit([event(10, "ACT1"), event(11, "LOCALRD"), event(12, "ACT2")])

    def test_compute_access_covers_all_target_banks_before_ACT2(self):
        for command in ("MACAB", "MACK", "MACQK", "MACHB"):
            # The command's encoded bank is not the complete physical target.
            events = [event(10, "ACT1", bg=1, bank=3, type_id=14),
                      event(11, command, bg=0, bank=0, type_id=4),
                      event(12, "ACT2", bg=1, bank=3, type_id=14)]
            with self.subTest(command=command), self.assertRaisesRegex(AssertionError, "target_bank_access_before_ACT2"):
                self.audit(events)

    def test_compute_in_other_half_or_rank_does_not_touch_pending_bank(self):
        report = self.audit([event(10, "ACT1", bg=1, type_id=18),
                             event(11, "MACHB", bg=2),
                             event(12, "MACAB", rank=0),
                             event(13, "ACT2", bg=1, type_id=18)])
        self.assertEqual(report["status"], "passed")

    def test_activation_owner_kind_survives_stage_window_boundary(self):
        stage = [dict(Channel=0, stage_generation=1, operation="SEAL", clock=5),
                 dict(Channel=0, stage_generation=1, operation="BROADCAST", ready_at=12)]
        events = [event(10, "ACT1", type_id=-1), event(20, "ACT2", type_id=-1)]
        report = audit_staging_act2(events, {"nAAD": 8}, staging_events=stage, strict=False)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["staging"]["late_ACT2"], 1)
        self.assertEqual(report["global_audit"]["violations_by_kind"]["staging"]["late_ACT2"], 1)

    def test_internal_owner_and_staging_conflict_are_both_attributed(self):
        events = [event(10, "ACT1", type_id=18), event(11, "MACAB"),
                  event(12, "ACT2", type_id=18)]
        report = self.audit(events, strict=False)
        for category in ("other_internal", "staging"):
            self.assertEqual(report["violations_by_kind"][category]["target_bank_access_before_ACT2"], 1)
        self.assertEqual(report["classifications"]["other_internal"]["paired_ACT2"], 1)
        with self.assertRaisesRegex(AssertionError, "target_bank_access_before_ACT2"):
            audit_staging_act2(events, {"nAAD": 8})

    def test_channel_protocol_is_shared_by_both_ranks(self):
        for command in ("ACT1", "REFpb", "REFab"):
            with self.subTest(command=command), self.assertRaisesRegex(AssertionError, "channel_ACT2_pending"):
                self.audit([event(10, "ACT1"), event(11, command, rank=0), event(12, "ACT2")])

    def test_independent_channels_can_activate_and_refresh_concurrently(self):
        result = self.audit([event(10, "ACT1"), event(11, "ACT1", channel=1),
            event(12, "REFab", channel=2), event(12, "ACT2", channel=1), event(13, "ACT2")])
        self.assertEqual(result["paired_ACT2"], 2)

    def test_other_rank_ACT2_cannot_claim_this_rank_owner(self):
        with self.assertRaisesRegex(AssertionError, "orphan_ACT2"):
            self.audit([event(10, "ACT1"), event(11, "ACT2", rank=0), event(12, "ACT2")])

    def test_half_precharge_selects_two_bank_groups(self):
        with self.assertRaisesRegex(AssertionError, "pending_ACT1_cancelled"):
            self.audit([event(10, "ACT1", bg=1), event(11, "PREHB", bg=0), event(12, "ACT2", bg=1)])

    def test_wildcard_rank_precharge_does_not_hide_cancellation(self):
        with self.assertRaisesRegex(AssertionError, "pending_ACT1_cancelled"):
            self.audit([event(10, "ACT1"), event(11, "PREab", rank=-1, bg=-1, bank=-1)])

    def test_report_only_retains_failure_and_bounds_examples(self):
        events = [event(10*i, "ACT2") for i in range(12)]
        report = self.audit(events, strict=False, max_examples=3)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["violations"], {"orphan_ACT2": 12})
        self.assertEqual(len(report["examples"]), 3)
        with self.assertRaises(AssertionError):
            self.audit(events)

    def test_source_region_write_is_ordinary_and_unknown_source_is_not_assumed_internal(self):
        events = [event(1, "ACT1", type_id=1), event(2, "ACT2", type_id=1),
                  event(20, "ACT1", type_id=-1), event(21, "ACT2", type_id=-1)]
        report = self.audit(events)
        self.assertEqual(report["classifications"]["ordinary"]["paired_ACT2"], 1)
        self.assertEqual(report["classifications"]["staging_region_unclassified"]["paired_ACT2"], 1)


if __name__ == "__main__":
    unittest.main()

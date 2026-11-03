"""Fault injection for queue bounds, per-channel progress and coarse children."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from audit_native import audit_trace, trace_records, expected_native_commands, decode
from dispatch_audit import audit_ingress, audit_rank_descriptors, expand_rank_records, TYPE_IDS
from native_guards import address
import workload


def ingress_fixture():
    records = [dict(line=i+1, operation=op, address=addr, buffer=-1) for i, (op, addr) in enumerate((
        ("BARRIER", 0), ("LD", address(2048)), ("LD", address(4096)), ("LD", address(2048, ch=1)),
        ("BARRIER", 1), ("PIM_Q_EXPORT", address(82176, ch=1)),
        ("PIM_Q_REMOTE", address(82176)), ("BARRIER", 2)))]
    front, ingress = [], []
    schedule = {1: (1, 5, 0, 1), 2: (2, 8, 1, 2), 3: (1, 3, 0, 1),
                5: (9, 15, 8, 9), 6: (15, 18, 14, 15)}
    for index, (accept, complete, enqueue, dispatch) in schedule.items():
        r = records[index]
        identity = dict(channel=r["address"] >> 5 & 15, address=r["address"], type_id=TYPE_IDS[r["operation"]], rank_batch=0)
        front += [dict(identity, operation="ACCEPT", trace_index=index, frontend_tick=accept),
                  dict(identity, operation="COMPLETE", trace_index=index, frontend_tick=complete)]
        ingress += [dict(identity, operation="ENQUEUE", request_id=index, clock=enqueue, queue_depth=1, capacity=1),
                    dict(identity, operation="DISPATCH", request_id=index, clock=dispatch, queue_depth=0, capacity=1)]
    front.append(dict(front[-2], operation="PEER_WAIT", frontend_tick=9))
    front.sort(key=lambda e: e["frontend_tick"])
    ingress.sort(key=lambda e: (e["clock"], e["operation"] == "ENQUEUE"))
    barriers = [dict(marker=0, frontend_tick=1, requests_completed=0),
                dict(marker=1, frontend_tick=9, requests_completed=3),
                dict(marker=2, frontend_tick=19, requests_completed=5)]
    return records, front, ingress, barriers


def descriptor_fixture():
    records = [dict(line=1, operation="PIM_RANK_ROPE_COEF", address=address(81920), buffer=-1)]
    parent = dict(zip(("Channel", "Rank", "BankGroup", "Bank", "Row", "Column"), decode(records[0]["address"])))
    parent.update(descriptor_id=1, trace_index=0, operation="PIM_RANK_ROPE_COEF", buffer=-1, children=16)
    events = [dict(parent, event="DISPATCH", clock=1, ready_at=1)]
    fine = []
    for bank in range(16):
        start, ready = 2+4*bank, 5+4*bank
        child = dict(parent, BankGroup=bank//4, Bank=bank%4, operation="PIM_ROPE_COEF")
        events.append(dict(child, event="CHILD_COMPLETE", clock=ready+1, ready_at=ready))
        fine.append(dict(child, clock=start, ready_at=ready))
    events.append(dict(parent, event="COMPLETE", clock=66, ready_at=65))
    ca = [dict(Channel=0, clock=1, command="PIM_RANK_ROPE_COEF", scope="global")]
    return records, events, fine, ca


class IndependentIngressTests(unittest.TestCase):
    def check(self, fixture):
        return audit_ingress(*fixture, capacity=1)

    def test_finite_queue_and_late_peer_keep_all_requests_accounted(self):
        result = self.check(ingress_fixture())
        self.assertEqual(result["queue_high_water_by_channel"], {0: 1, 1: 1})
        self.assertEqual(result["peer_export_dependencies_checked"], 1)
        self.assertEqual(result["peer_wait_events"], 1)

    def test_queue_depth_cannot_be_reported_as_unbounded_or_zero(self):
        fixture = ingress_fixture()
        fixture[2][0]["queue_depth"] = 2
        with self.assertRaisesRegex(AssertionError, "depth is inaccurate or exceeds"):
            self.check(fixture)

    def test_channel_FIFO_cannot_dispatch_second_request_first(self):
        fixture = ingress_fixture()
        first = next(e for e in fixture[2] if e["operation"] == "DISPATCH" and e["request_id"] == 1)
        first.update(request_id=2, address=fixture[0][2]["address"])
        with self.assertRaisesRegex(AssertionError, "FIFO order"):
            self.check(fixture)

    def test_barrier_must_include_every_channel_completion(self):
        fixture = ingress_fixture()
        fixture[3][1]["frontend_tick"] = 7
        with self.assertRaisesRegex(AssertionError, "before pending callback"):
            self.check(fixture)

    def test_peer_must_wait_for_export_completion_not_acceptance(self):
        fixture = ingress_fixture()
        next(e for e in fixture[1] if e["operation"] == "ACCEPT" and e["trace_index"] == 6)["frontend_tick"] = 14
        with self.assertRaisesRegex(AssertionError, "latest peer export completes"):
            self.check(fixture)

    def test_repeated_peer_address_requires_new_export_instance(self):
        records, front, ingress, barriers = ingress_fixture()
        records.insert(-1, dict(line=8, operation="PIM_Q_EXPORT", address=address(82176, ch=1), buffer=-1))
        records.insert(-1, dict(line=9, operation="PIM_Q_REMOTE", address=address(82176), buffer=-1))
        records[-1]["line"] = 10
        for index, accept, complete in ((7, 19, 25), (8, 20, 27)):
            r = records[index]
            identity = dict(channel=r["address"] >> 5 & 15, address=r["address"], type_id=TYPE_IDS[r["operation"]], rank_batch=0)
            front += [dict(identity, operation="ACCEPT", trace_index=index, frontend_tick=accept),
                      dict(identity, operation="COMPLETE", trace_index=index, frontend_tick=complete)]
            ingress += [dict(identity, operation="ENQUEUE", request_id=index, clock=accept-1, queue_depth=1, capacity=1),
                        dict(identity, operation="DISPATCH", request_id=index, clock=accept, queue_depth=0, capacity=1)]
        front.sort(key=lambda e: e["frontend_tick"])
        ingress.sort(key=lambda e: (e["clock"], e["operation"] == "ENQUEUE"))
        barriers[-1].update(frontend_tick=28, requests_completed=7)
        with self.assertRaisesRegex(AssertionError, "latest peer export completes"):
            self.check((records, front, ingress, barriers))

    def test_adverse_peer_guard_consumes_coefficients_before_every_reload(self):
        from native_guards import trace_for
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"guard.trace"
            for name in ("Dispatch_late_peer", "Dispatch_repeated_peer_source"):
                trace_for(path, name)
                state = {}
                for record in expand_rank_records(trace_records(path)):
                    if record["operation"] not in ("PIM_ROPE_COEF", "PIM_ROPE_EXEC"):
                        continue
                    channel, rank, bg, bank, row, column = decode(record["address"])
                    key = channel, bg, bank
                    token, uses = state.get(key, (-1, 3))
                    if record["operation"] == "PIM_ROPE_COEF":
                        self.assertEqual(uses, 3, (name, record["line"], "unconsumed coefficients"))
                        state[key] = column, 0
                    else:
                        self.assertEqual(token, column%8)
                        use = 1 if column < 16 else 2
                        self.assertFalse(uses & use, (name, "duplicate coefficient use"))
                        state[key] = token, uses | use
                self.assertEqual({uses for _, uses in state.values()}, {3})


class RankDescriptorTests(unittest.TestCase):
    def test_rank_parent_joins_sixteen_fine_bank_completions(self):
        result = audit_rank_descriptors(*descriptor_fixture())
        self.assertEqual(result["parent_descriptors_checked"], 1)
        self.assertEqual(result["fine_children_checked"], 16)

    def test_missing_child_cannot_be_hidden_by_declared_children_count(self):
        fixture = descriptor_fixture()
        fixture[1].pop(5)
        with self.assertRaisesRegex(AssertionError, "exactly sixteen"):
            audit_rank_descriptors(*fixture)

    def test_aliased_bank_child_cannot_preserve_count_only(self):
        fixture = descriptor_fixture()
        fixture[1][2].update(Bank=0)
        with self.assertRaisesRegex(AssertionError, "fine bank/address coverage"):
            audit_rank_descriptors(*fixture)

    def test_parent_cannot_complete_before_slowest_child(self):
        fixture = descriptor_fixture()
        fixture[1][-1]["clock"] -= 1
        with self.assertRaisesRegex(AssertionError, "join all sixteen"):
            audit_rank_descriptors(*fixture)

    def test_child_callback_requires_real_fine_completion(self):
        fixture = descriptor_fixture()
        fixture[1][1]["ready_at"] -= 1
        with self.assertRaisesRegex(AssertionError, "before its fine service"):
            audit_rank_descriptors(*fixture)

    def test_global_bus_cannot_count_sixteen_coarse_admissions(self):
        fixture = descriptor_fixture()
        fixture[3].append(dict(fixture[3][0], clock=2))
        with self.assertRaisesRegex(AssertionError, "one admission"):
            audit_rank_descriptors(*fixture)

    def test_Q_export_child_keeps_RD_but_must_not_repay_global_CA(self):
        fixture = descriptor_fixture()
        for group in fixture[:3]:
            for e in group:
                e["operation"] = e["operation"].replace("ROPE_COEF", "Q_EXPORT")
        fixture[3][0]["command"] = "PIM_RANK_Q_EXPORT"
        self.assertEqual(audit_rank_descriptors(*fixture)["Q_EXPORT_internal_RD_global_CA_exclusion_checks"], 16)
        fixture[3].append(dict(Channel=0, clock=2, command="RD", scope="global"))
        with self.assertRaisesRegex(AssertionError, "child RD incorrectly occupies global CA"):
            audit_rank_descriptors(*fixture)

    def test_noncanonical_parent_is_rejected_by_independent_expansion(self):
        record = descriptor_fixture()[0][0]
        record["address"] |= 1 << 18
        with self.assertRaisesRegex(AssertionError, "canonical rank1"):
            expand_rank_records([record])

    def test_all_rank_descriptor_cases_preserve_fixed_tensor_and_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in workload.CASES:
                path = Path(directory)/f"{case}.trace"
                workload.generate(path, case, staging_banks=2, rank_descriptors=False)
                fine = expected_native_commands(trace_records(path))
                workload.generate(path, case, staging_banks=2, rank_descriptors=True)
                records = trace_records(path)
                audit_trace(records, case, staging_banks=2)
                self.assertEqual(expected_native_commands(records), fine, case)


if __name__ == "__main__":
    unittest.main()

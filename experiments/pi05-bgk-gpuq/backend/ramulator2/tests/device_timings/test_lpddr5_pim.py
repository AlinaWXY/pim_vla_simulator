import pytest
import ramulator

from tests.device_timings.harness import DeviceUnderTest


def device(**overrides):
    return DeviceUnderTest(ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", **overrides
    ))


def test_broadcast_reads_check_every_bank_not_just_first():
    d = device()
    ab = d.addr_vec(BankGroup=-1, Bank=-1, Row=1)
    # First bank is closed; last bank conflicts. PREab must win over ACTAB.
    last = d.addr_vec(BankGroup=3, Bank=3, Row=0)
    d.issue("ACT1", last, 0)
    d.issue("ACT2", last, 1)
    assert d.probe("MACAB", ab, 100).preq == "PREab"
    d.issue("PREab", ab, 100)
    assert d.probe("MACAB", ab, 101).preq == "ACTAB"
    d.assert_earliest_ready_at("ACTAB", ab, 116)
    d.issue("ACTAB", ab, 116)
    d.assert_earliest_ready_at("MACAB", ab, 132)
    for bg in range(4):
        for bank in range(4):
            assert d.probe("RD", d.addr_vec(BankGroup=bg, Bank=bank, Row=1), 132).row_hit


def test_mac_interval_and_array_recovery_boundaries():
    d = device()
    ab = d.addr_vec(BankGroup=-1, Bank=-1)
    d.issue("ACTAB", ab, 0)
    d.assert_earliest_ready_at("MACAB", ab, 16)
    d.issue("MACAB", ab, 16)
    d.assert_earliest_ready_at("MACAB", ab, 20)
    d.assert_earliest_ready_at("PREab", ab, 35)  # ACT reception + nRAS.


@pytest.mark.parametrize("refresh,recovery", [("REFab", 168), ("REFpb", 96)])
def test_pim_activation_obeys_refresh_recovery(refresh, recovery):
    d = device()
    ab = d.addr_vec(BankGroup=-1, Bank=-1)
    target = ab if refresh == "REFab" else d.addr_vec(BankGroup=3, Bank=3)
    d.issue(refresh, target, 0)
    d.assert_earliest_ready_at("ACTAB", ab, recovery - 1)


def test_other_rank_stays_closed():
    d = device(rank=2)
    ab = d.addr_vec(BankGroup=-1, Bank=-1)
    d.issue("ACTAB", ab, 0)
    assert not d.probe("RD", d.addr_vec(Rank=1), 100).row_open


def test_mac_does_not_inherit_external_read_rank_switch_or_burst():
    d = device(rank=2, nCS=100, nBL_min=8)
    a = d.addr_vec(Rank=0, BankGroup=-1, Bank=-1)
    b = d.addr_vec(Rank=1, BankGroup=-1, Bank=-1)
    d.issue("ACTAB", a, 0)
    d.issue("ACTAB", b, 2)
    d.issue("MACAB", a, 100)
    # No DQ delay between independent ranks. The controller separately
    # enforces one issue per clock; this device harness only checks DRAM.
    assert d.probe("MACAB", b, 101).ready
    d.issue("MACAB", b, 101)
    d.assert_earliest_ready_at("MACAB", a, 104)


@pytest.mark.parametrize("preceding,following", [("RD", "MACAB"), ("MACAB", "RD")])
def test_external_reads_do_not_share_mac_dq_timing_across_ranks(preceding, following):
    d = device(rank=2, nCS=100, nBL_min=8)
    a = d.addr_vec(Rank=0, BankGroup=-1, Bank=-1)
    b = d.addr_vec(Rank=1, BankGroup=-1, Bank=-1)
    d.issue("ACTAB", a, 0)
    d.issue("ACTAB", b, 2)
    source = d.addr_vec(Rank=0) if preceding == "RD" else a
    target = d.addr_vec(Rank=1) if following == "RD" else b
    d.issue(preceding, source, 100)
    assert d.probe(following, target, 101).ready


def test_local_array_column_floor_and_recovery_still_apply():
    d = device(nCCDAB=1)
    ab = d.addr_vec(BankGroup=-1, Bank=-1)
    d.issue("ACTAB", ab, 0)
    d.issue("MACAB", ab, 100)
    d.assert_earliest_ready_at("MACAB", ab, 104)
    d.assert_earliest_ready_at("PREab", ab, 106)
    # Retain the same array recovery when a legacy half-bank phase follows.
    d.assert_earliest_ready_at("PREHB", d.addr_vec(BankGroup=0, Bank=-1), 106)


def test_channel_scope_checks_conflict_and_activates_every_rank_bank():
    d = device(rank=2)
    all_banks = d.addr_vec(Rank=-1, BankGroup=-1, Bank=-1, Row=1)
    last = d.addr_vec(Rank=1, BankGroup=3, Bank=3, Row=0)
    d.issue("ACT1", last, 0)
    d.issue("ACT2", last, 1)
    assert d.probe("MACAB", all_banks, 100).preq == "PREab"
    d.issue("PREab", all_banks, 100)
    d.assert_earliest_ready_at("ACTAB", all_banks, 116)
    d.issue("ACTAB", all_banks, 116)
    d.assert_earliest_ready_at("MACAB", all_banks, 132)
    d.issue("MACAB", all_banks, 132)
    for rank in range(2):
        for bg in range(4):
            for bank in range(4):
                addr = d.addr_vec(Rank=rank, BankGroup=bg, Bank=bank, Row=1)
                assert d.probe("RD", addr, 200).row_hit
        rank_banks = d.addr_vec(Rank=rank, BankGroup=-1, Bank=-1, Row=1)
        d.assert_earliest_ready_at("MACAB", rank_banks, 136)


def test_channel_activation_waits_for_refresh_in_nonzero_rank():
    d = device(rank=2)
    rank_one = d.addr_vec(Rank=1, BankGroup=-1, Bank=-1)
    all_banks = d.addr_vec(Rank=-1, BankGroup=-1, Bank=-1)
    d.issue("REFab", rank_one, 0)
    d.assert_earliest_ready_at("ACTAB", all_banks, 167)

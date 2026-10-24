import pytest
import ramulator

from tests.device_timings.harness import DeviceUnderTest


def device():
    return DeviceUnderTest(ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", rank=2,
    ))


@pytest.mark.parametrize("selected_bg", range(4))
def test_half_bank_state_and_timing_cover_both_groups(selected_bg):
    d = device()
    half = d.addr_vec(BankGroup=selected_bg, Bank=-1, Row=1)
    d.issue("ACTHB", half, 0)
    d.assert_earliest_ready_at("MACHB", half, 16)
    d.issue("MACHB", half, 16)
    d.assert_earliest_ready_at("MACHB", half, 20)
    d.assert_earliest_ready_at("PREHB", half, 35)
    for rank in range(2):
        for bg in range(4):
            for bank in range(4):
                addr = d.addr_vec(Rank=rank, BankGroup=bg, Bank=bank, Row=1)
                selected = rank == 0 and bg // 2 == selected_bg // 2
                assert d.probe("RD", addr, 20).row_open == selected
                if selected:
                    d.assert_earliest_ready_at("RD", addr, 20)
    d.issue("PREHB", half, 35)
    d.assert_earliest_ready_at("ACTHB", half, 51)


def test_conflict_in_second_group_wins_over_closed_first_group():
    d = device()
    last = d.addr_vec(BankGroup=1, Bank=3)
    other = d.addr_vec(BankGroup=3, Bank=3)
    for addr, clk in ((last, 0), (other, 10)):
        d.issue("ACT1", addr, clk)
        d.issue("ACT2", addr, clk + 1)
    half = d.addr_vec(BankGroup=0, Bank=-1, Row=1)
    assert d.probe("MACHB", half, 100).preq == "PREHB"
    d.issue("PREHB", half, 100)
    assert d.probe("MACHB", half, 101).preq == "ACTHB"
    assert d.probe("RD", other, 101).row_hit


def test_half_compute_does_not_occupy_read_dq():
    d = device()
    other = d.addr_vec(BankGroup=2)
    half = d.addr_vec(BankGroup=0, Bank=-1)
    d.issue("ACT1", other, 0)
    d.issue("ACT2", other, 1)
    d.issue("ACTHB", half, 10)
    d.issue("MACHB", half, 26)
    assert d.probe("RD", other, 27).ready
    d.issue("RD", other, 27)
    d.assert_earliest_ready_at("MACHB", half, 30)


@pytest.mark.parametrize("refresh,bg,ready", [
    ("REFab", -1, 167), ("REFpb", 1, 95), ("REFpb", 3, 5),
])
def test_half_activation_obeys_refresh_scope(refresh, bg, ready):
    d = device()
    d.issue(refresh, d.addr_vec(BankGroup=bg, Bank=-1 if bg == -1 else 3), 0)
    d.assert_earliest_ready_at("ACTHB", d.addr_vec(BankGroup=0, Bank=-1), ready)

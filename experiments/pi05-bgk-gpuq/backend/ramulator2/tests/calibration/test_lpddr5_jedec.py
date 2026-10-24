"""Independent expected values and command boundaries from JESD209-5C."""

import math

import pytest

import ramulator
from tests.controller_scheduling.harness import ControllerUnderTest
from tests.device_timings.harness import DeviceUnderTest


def dram(**overrides):
    return ramulator.dram.LPDDR5(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", **overrides
    )


def test_6400_table_values():
    _, t = dram().resolve()
    # Tables 225,229,230,339,381; CKR4, x16 BG BL16, no ECC/DBI/DVFS.
    expected = dict(
        rate=6400,
        tCK_ps=1250,
        nBL_min=2,
        nBL_max=4,
        nCL=17,
        nCWL=9,
        nRCD=15,
        nRP=15,
        nRPab=17,
        nRAS=34,
        nRC=49,
        nWR=28,
        nRTP=6,
        nPPD=2,
        nCCDS=2,
        nCCDL=4,
        nCCDS_WR=2,
        nCCDL_WR=4,
        nRRDS=4,
        nRRDL=4,
        nWTRS=5,
        nWTRL=10,
        nFAW=16,
        nWCKPST=1,
        nCAS=1,
        nAAD=8,
        nCS=2,
    )
    assert {key: t[key] for key in expected} == expected
    assert DeviceUnderTest(dram()).timings == t


@pytest.mark.parametrize("density,rfc,rfcpb", [(8, 210, 120), (16, 280, 140)])
def test_density_refresh_table_240(density, rfc, rfcpb):
    m = ramulator.dram.LPDDR5(org_preset=f"LPDDR5_{density}Gb_x16", timing_preset="LPDDR5_6400")
    _, t = m.resolve()
    assert t["nRFC"] == math.ceil(rfc / 1.25)
    assert t["nRFCpb"] == math.ceil(rfcpb / 1.25)
    assert t["nPBR2PBR"] == 72 and t["nPBR2ACT"] == 6
    assert t["nREFI"] == 3124 and t["nREFIpb"] == 390


def activate(d, a, c):
    d.issue("ACT1", a, c)
    d.issue("ACT2", a, c + 1)


@pytest.mark.parametrize(
    "first,second,gap", [("RD", "RD", 4), ("RD", "WR", 12), ("WR", "RD", 1), ("WR", "WR", 3)]
)
def test_rank_switch_table_377(first, second, gap):
    d = DeviceUnderTest(dram(rank=2))
    a, b = d.addr_vec(), d.addr_vec(Rank=1)
    activate(d, a, 0)
    activate(d, b, 30)
    d.issue(first, a, 100)
    d.assert_earliest_ready_at(second, b, 100 + gap)
    lat1 = 17 if first == "RD" else 9
    lat2 = 17 if second == "RD" else 9
    assert 100 + gap + lat2 >= 100 + lat1 + 2


@pytest.mark.parametrize("cmd,param", [("RD", "nRCD"), ("WR", "nRCD"), ("PREpb", "nRAS")])
def test_activate_bounds(cmd, param):
    d = DeviceUnderTest(dram())
    a = d.addr_vec()
    activate(d, a, 0)
    d.assert_earliest_ready_at(cmd, a, 1 + d.timings[param])


@pytest.mark.parametrize("refresh,param", [("REFab", "nRFC"), ("REFpb", "nRFCpb")])
def test_refresh_recovery(refresh, param):
    d = DeviceUnderTest(dram())
    a = d.addr_vec()
    d.issue(refresh, a, 0)
    ready = d.timings[param]
    d.issue("ACT1", a, ready - 1)
    d.assert_earliest_ready_at("ACT2", a, ready)


def controller():
    return ControllerUnderTest(
        ramulator.controller.LPDDR5(
            dram=dram(),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        )
    )


@pytest.mark.parametrize("extra,sync", [(0, False), (1, True)])
def test_wck_valid_through_inclusive_maximum(extra, sync):
    d = controller()
    a = d.addr_vec()
    d.send_request("Read", a)
    now = 0
    while True:
        now += 1
        events = d.tick()
        read = next((e for e in events if e.command == "RD"), None)
        if read:
            break
        assert now < 100
    # Table 212: RL + BL/n_max + floor(WCKPST/CK), inclusive.
    deadline = read.clk + 17 + 4 + 1 + extra
    while now < deadline - 1:
        d.tick()
        now += 1
    d.send_request("Read", a)
    events = d.tick()
    assert [e.command for e in events] == (["CAS_RD"] if sync else ["RD"])
    if sync:
        assert [e.command for e in d.tick()] == ["RD"]


def test_multirank_requires_presynchronized_wck():
    kwargs = dict(
        dram=dram(rank=2),
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    with pytest.raises(RuntimeError, match="pre-synchronized"):
        ControllerUnderTest(ramulator.controller.LPDDR5(**kwargs))
    ControllerUnderTest(ramulator.controller.LPDDR5(**kwargs, wck_sync_mode="always_on"))

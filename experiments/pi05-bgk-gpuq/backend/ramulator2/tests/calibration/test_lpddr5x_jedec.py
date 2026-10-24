"""LPDDR5X-8533 standard-derived values and read/write timing separation."""

import math

import pytest

import ramulator
from tests.controller_scheduling.harness import ControllerUnderTest
from tests.device_timings.harness import DeviceUnderTest


def dram(**kwargs):
    return ramulator.dram.LPDDR5X(
        org_preset="LPDDR5X_8Gb_x16", timing_preset="LPDDR5X_8533", **kwargs
    )


def test_8533_tables_and_ck_quantization():
    org, t = dram().resolve()
    assert org["row"] * org["column"] * 16 * 16 == 8 * 1024**3
    assert 7500 < 8_000_000 / t["tCK_ps"] <= 8533
    expected = dict(
        tCK_ps=938,
        rate=8533,
        nCL=23,
        nCWL=12,
        nWR=37,
        nRTP=8,
        nRCD=20,
        nRCDWR=9,
        nRP=20,
        nRPab=23,
        nRAS=45,
        nRC=65,
        nPPD=2,
        nCCDS=2,
        nCCDL=4,
        nCCDS_WR=2,
        nCCDL_WR=4,
        nWTRS=7,
        nWTRL=13,
        nRRDS=4,
        nRRDL=4,
        nFAW=16,
        nRFC=224,
        nRFCpb=128,
        nPBR2PBR=96,
        nPBR2ACT=8,
        nREFI=4164,
        nREFIpb=520,
        nWCKPST=1,
        nCAS=1,
        nAAD=8,
        nCS=2,
    )
    assert {k: t[k] for k in expected} == expected
    assert DeviceUnderTest(dram()).timings == t


@pytest.mark.parametrize("cmd,cycles", [("RD", 20), ("RDA", 20), ("WR", 9), ("WRA", 9)])
def test_read_write_activation_boundaries(cmd, cycles):
    d = DeviceUnderTest(dram())
    a = d.addr_vec()
    d.issue("ACT1", a, 0)
    d.issue("ACT2", a, 1)
    d.assert_earliest_ready_at(cmd, a, 1 + cycles)


@pytest.mark.parametrize(
    "first,second,gap", [("RD", "RD", 4), ("RD", "WR", 15), ("WR", "RD", 1), ("WR", "WR", 3)]
)
def test_rank_switch_bounds(first, second, gap):
    d = DeviceUnderTest(dram(rank=2))
    a = d.addr_vec()
    b = d.addr_vec(Rank=1)
    d.issue("ACT1", a, 0)
    d.issue("ACT2", a, 1)
    d.issue("ACT1", b, 30)
    d.issue("ACT2", b, 31)
    d.issue(first, a, 100)
    d.assert_earliest_ready_at(second, b, 100 + gap)


@pytest.mark.parametrize("density,rfc,rfcpb", [(8, 210, 120), (16, 280, 140)])
def test_8533_refresh_density(density, rfc, rfcpb):
    m = ramulator.dram.LPDDR5X(
        org_preset=f"LPDDR5X_{density}Gb_x16", timing_preset="LPDDR5X_8533", rank=2
    )
    _, t = m.resolve()
    assert t["nRFC"] == math.ceil(rfc * 1000 / 938)
    assert t["nRFCpb"] == math.ceil(rfcpb * 1000 / 938)


@pytest.mark.parametrize(
    "override",
    [{"rate": 9600}, {"tCK_ps": 1250}, {"column": 64}, {"bankgroup": 1}, {"row": 1}, {"rank": 3}],
)
def test_invalid_config_rejected(override):
    with pytest.raises(ValueError):
        dram(**override).to_config()


def test_controller_uses_short_write_rcd():
    d = ControllerUnderTest(
        ramulator.controller.LPDDR5(
            dram=dram(),
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        )
    )
    d.send_request("Write", d.addr_vec())
    history = d.run_until_idle(256)
    clocks = {e.command: e.clk for e in history}
    assert clocks["WR"] - clocks["ACT2"] == 9
    assert clocks["WR"] - clocks["CAS_WR"] == 1

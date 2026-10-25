"""AttAcc's compute/movement paths, including shared transfer bandwidth."""

import pytest

from simulator import MemoryConfig
from test_input_broadcast import run


def test_result_and_input_share_movement_path_but_do_not_stall_mac(tmp_path):
    result, events = run(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 0 0", "BARRIER 0",
        # The result belongs to the preceding window, not this next MAC.
        "PIM_MAC_AB 32 0",
        *[f"PIM_RD_RES {rank * 2048 + i * 32}"
          for i in range(50) for rank in range(2)],
        "PIM_WR_GB 0 1", "PIM_MAC_AB 64 1", "BARRIER 1",
        "PIM_RD_RES 0",
    ], pim_mac_scope="channel", pim_data_path="attacc")
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    reads = [e["clock"] for e in events if e["command"] == "RDRES"]
    assert reads[0] < macs[1] < reads[-2]
    assert macs[1] - macs[0] == 50
    # No channel-wide RDRES -> MAC = CL+BL lock.
    before_mac = max(t for t in reads if t < macs[1])
    assert macs[1] - before_mac < 25
    transfers = [e for e in events if e["command"] in ("WRGB", "RDRES")]
    assert all(b["clock"] - a["clock"] >= 2 for a, b in zip(transfers, transfers[1:]))
    for rank in ("0", "1"):
        local = [e["clock"] for e in transfers if e["Rank"] == rank]
        assert all(b - a >= 4 for a, b in zip(local, local[1:]))
    assert result["resolved_timings_ck"]["nPIMWrLatency"] == 0
    assert result["resolved_timings_ck"]["nPIMRdLatency"] == 0
    assert result["completed_requests"] == result["requests"]["total"]


def test_attacc_has_no_cwl_input_ready_tail(tmp_path):
    result, events = run(tmp_path, [
        *[f"PIM_WR_GB {i * 32} 0" for i in range(60)],
        "PIM_MAC_AB 0 0", "BARRIER 0", "PIM_RD_RES 0",
    ], pim_mac_scope="channel", pim_data_path="attacc")
    last_write = max(e["clock"] for e in events if e["command"] == "WRGB")
    mac = next(e["clock"] for e in events if e["command"] == "MACAB")
    assert result["resolved_timings_ck"]["nBL_min"] <= mac - last_write < 14
    read = next(e["clock"] for e in events if e["command"] == "RDRES")
    assert read < mac + 50
    assert result["pim_model"]["data_path"] == "attacc"


def test_attacc_still_protects_active_fragment_slot(tmp_path):
    _, events = run(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 0 0",
        "PIM_WR_GB 32 0", "PIM_MAC_AB 32 0",
    ], pim_mac_scope="channel", pim_data_path="attacc")
    mac = next(e["clock"] for e in events if e["command"] == "MACAB")
    writes = [e["clock"] for e in events if e["command"] == "WRGB"]
    assert writes[1] >= mac + 50


def test_attacc_constraints_do_not_modify_legacy_instances():
    import ramulator
    options = dict(org_preset="LPDDR5X_8Gb_x16", timing_preset="LPDDR5X_8533")
    legacy = ramulator.dram.LPDDR5XPIM(**options)
    before = legacy.to_config()
    attacc = ramulator.dram.LPDDR5XPIM(**options, pim_data_path="attacc")
    assert attacc.to_config() != before
    assert legacy.to_config() == before
    assert ramulator.dram.LPDDR5XPIM(**options).to_config() == before


@pytest.mark.parametrize("mode", ["", None, 1, "both"])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="lpddr or attacc"):
        MemoryConfig(pim=True, pim_data_path=mode)


def test_attacc_requires_pim():
    with pytest.raises(ValueError, match="requires pim"):
        MemoryConfig(pim_data_path="attacc")


def test_attacc_rejects_half_bank(tmp_path):
    with pytest.raises(ValueError, match="all-bank"):
        run(tmp_path, ["PIM_MAC_HB 0"], pim_input_broadcast=False,
            pim_mac_scope="rank", pim_data_path="attacc")


@pytest.mark.parametrize("timing", ["nPIMWrLatency", "nPIMRdLatency"])
def test_attacc_rejects_conflicting_completion_overrides(tmp_path, timing):
    with pytest.raises(ValueError, match="completion latency is fixed"):
        run(tmp_path, ["PIM_WR_GB 0 0"], pim_data_path="attacc",
            timing_overrides={timing: 25})

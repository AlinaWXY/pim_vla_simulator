"""One MAC command covers every rank; input and output dependencies remain real."""

import pytest

from simulator import MemoryConfig
from test_input_broadcast import run


@pytest.mark.parametrize("model", ["lpddr5-6400", "lpddr5x-8533"])
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("broadcast", [False, True])
def test_channel_mac_accounts_for_all_banks_and_waits_for_all_inputs(tmp_path, model, ranks, broadcast):
    writes = [f"PIM_WR_GB {rank * 2048} 0"
              for rank in ((0,) if broadcast else range(ranks))]
    result, events = run(tmp_path, [
        *writes, "PIM_MAC_AB 0 0", "PIM_MAC_AB 32 0",
        *[f"PIM_RD_RES {rank * 2048}" for rank in range(ranks)],
    ], model=model, ranks=ranks, pim_input_broadcast=broadcast, pim_mac_scope="channel")
    macs = [e for e in events if e["command"] == "MACAB"]
    assert len(macs) == 2
    assert {e["Rank"] for e in macs} == {"-1"}
    assert {e["mask"] for e in macs} == {(1 << ranks) - 1}
    assert macs[1]["clock"] - macs[0]["clock"] == 50
    last_write = max(e["clock"] for e in events if e["command"] == "WRGB")
    assert macs[0]["clock"] >= last_write + result["resolved_timings_ck"]["nPIMWrLatency"]
    assert result["traffic_bytes"]["pim_internal"] == 2 * 32 * 16 * ranks
    assert result["pim_model"]["banks_per_command"] == 16 * ranks
    assert result["issued_commands"]["ACTAB"] == 1
    for rank in range(ranks):
        read = next(e for e in events if e["command"] == "RDRES" and int(e["Rank"]) == rank)
        assert read["clock"] >= macs[-1]["clock"] + 50


def test_channel_mac_requires_nonzero_rank_input_too(tmp_path):
    with pytest.raises(RuntimeError, match="never filled"):
        run(tmp_path, ["PIM_WR_GB 0 0", "PIM_MAC_AB 0 0"],
            pim_input_broadcast=False, pim_mac_scope="channel")


def test_channel_mac_does_not_reach_another_channel(tmp_path):
    with pytest.raises(RuntimeError, match="never filled"):
        run(tmp_path, ["PIM_WR_GB 0 0", "PIM_MAC_AB 32 0"],
            channels=2, pim_mac_scope="channel")


def test_channel_mac_owns_nonzero_rank_slot_and_allows_other_slot_prefetch(tmp_path):
    _, events = run(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_WR_GB 2048 0", "PIM_MAC_AB 0 0",
        *[f"PIM_WR_GB {rank * 2048 + i * 32} 1" for i in range(60) for rank in range(2)],
        "PIM_MAC_AB 32 1",
        "PIM_WR_GB 2112 0", "PIM_RD_RES 0", "PIM_RD_RES 2048",
    ], pim_input_broadcast=False, pim_mac_scope="channel")
    first = next(e["clock"] for e in events if e["command"] == "MACAB")
    slot_one = [e["clock"] for e in events if e["command"] == "WRGB" and e["pim_buffer"] == "1"]
    overwrite = next(e["clock"] for e in events
                     if e["command"] == "WRGB" and e["Rank"] == "1"
                     and e["Column"] == "2" and e["pim_buffer"] == "0")
    assert any(first < tick < first + 50 for tick in slot_one)
    assert overwrite >= first + 50


@pytest.mark.parametrize("lines,message", [
    (["PIM_MAC_AB 2048 0"], "rank-0"),
    (["PIM_MAC_AB 0"], "explicit BUFFER"),
    (["PIM_WR_GB 0"], "explicit BUFFER"),
    (["PIM_MAC_HB 0"], "all-bank"),
])
def test_channel_mac_rejects_ambiguous_traces(tmp_path, lines, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        run(tmp_path, lines, pim_input_broadcast=False, pim_mac_scope="channel")


@pytest.mark.parametrize("scope", ["all", "", None, 1])
def test_invalid_scope_is_rejected(scope):
    with pytest.raises(ValueError, match="rank or channel"):
        MemoryConfig(pim=True, pim_mac_scope=scope)


def test_channel_scope_requires_pim():
    with pytest.raises(ValueError, match="requires pim"):
        MemoryConfig(pim_mac_scope="channel")

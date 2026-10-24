"""Channel input fanout: one DQ transfer, independent rank ownership."""

import csv
from dataclasses import replace

import pytest

from simulator import MemoryConfig, run_trace


def run(tmp_path, lines, **overrides):
    trace = tmp_path / "input.trace"
    trace.write_text("\n".join(lines) + "\n")
    cfg = replace(MemoryConfig(model="lpddr5x-8533", pim=True, ranks=2,
                               pim_input_broadcast=True, refresh=False,
                               timing_overrides={"nCCDAB": 50}), **overrides)
    result = run_trace(cfg, trace, tmp_path / "run")
    events = []
    for ch in range(cfg.channels):
        with (tmp_path / "run" / f"timeline.csv.ch{ch}").open() as stream:
            events += [dict(e, clock=int(e["clock"]), mask=int(e["pim_rank_mask"]))
                       for e in csv.DictReader(stream)]
    return result, events


@pytest.mark.parametrize("model", ["lpddr5-6400", "lpddr5x-8533"])
@pytest.mark.parametrize("ranks", [1, 2])
def test_one_burst_delivers_to_every_rank_without_duplicate_dq(tmp_path, model, ranks):
    result, events = run(tmp_path, [
        *[f"PIM_WR_GB {i * 32} 0" for i in range(4)],
        *[f"PIM_MAC_AB {rank * 2048} 0" for rank in range(ranks)],
        *[f"PIM_RD_RES {rank * 2048}" for rank in range(ranks)],
    ], model=model, ranks=ranks)
    writes = [e for e in events if e["command"] == "WRGB"]
    # Early two-cycle ACTAB can now use a write gap and delay the next CA
    # issue; the DQ burst itself still has a two-cycle minimum interval.
    assert writes[0]["clock"] == 1
    assert all(b["clock"] - a["clock"] >= 2 for a, b in zip(writes, writes[1:]))
    assert {e["mask"] for e in writes} == {(1 << ranks) - 1}
    assert result["traffic_bytes"]["pim_input_write"] == 128
    assert result["traffic_bytes"]["pim_input_delivered"] == 128 * ranks
    assert result["completed_requests"] == 4 + ranks * 2
    ready = writes[-1]["clock"] + result["resolved_timings_ck"]["nPIMWrLatency"]
    for rank in range(ranks):
        mac = next(e for e in events if e["command"] == "MACAB" and int(e["Rank"]) == rank)
        read = next(e for e in events if e["command"] == "RDRES" and int(e["Rank"]) == rank)
        assert mac["clock"] >= ready
        assert mac["mask"] == read["mask"] == 1 << rank
        assert read["clock"] >= mac["clock"] + 50


def test_disabled_mode_does_not_fill_another_rank(tmp_path):
    assert MemoryConfig(pim=True).pim_input_broadcast is False
    with pytest.raises(RuntimeError, match="never filled"):
        run(tmp_path, ["PIM_WR_GB 0 0", "PIM_MAC_AB 2048 0"], pim_input_broadcast=False)


def test_broadcast_cannot_cross_channels(tmp_path):
    with pytest.raises(RuntimeError, match="never filled"):
        run(tmp_path, ["PIM_WR_GB 0 0", "PIM_MAC_AB 32 0"], channels=2)


def test_shared_write_waits_for_nonzero_rank_and_preserves_pending_mac(tmp_path):
    # Rank 0 is idle. A write addressed to it still targets rank 1 and cannot
    # bypass rank 1's queued second MAC or overwrite its active SRAM slot.
    _, events = run(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 2048 0", "PIM_MAC_AB 2080 0",
        "PIM_WR_GB 32 0", "PIM_MAC_AB 2112 0", "PIM_RD_RES 2048",
    ])
    writes = [e["clock"] for e in events if e["command"] == "WRGB"]
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    assert macs[1] - macs[0] == 50
    assert writes[1] >= macs[1] + 50
    assert macs[2] >= writes[1] + 14


def test_other_slot_prefetch_overlaps_nonzero_rank_compute(tmp_path):
    _, events = run(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 2048 0", "PIM_MAC_AB 2080 0",
        *[f"PIM_WR_GB {i * 32} 1" for i in range(60)],
        "PIM_MAC_AB 2112 1", "PIM_RD_RES 2048",
    ])
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    prefetch = [e["clock"] for e in events if e["command"] == "WRGB" and e["pim_buffer"] == "1"]
    assert macs[1] - macs[0] == 50
    assert any(macs[0] < tick < macs[1] for tick in prefetch)
    assert macs[2] >= max(prefetch) + 14


@pytest.mark.parametrize("lines,message", [
    (["PIM_WR_GB 2048 0"], "rank-0"),
    (["PIM_WR_GB 0"], "explicit BUFFER"),
    (["PIM_MAC_AB 0"], "explicit BUFFER"),
    (["PIM_MAC_HB 0"], "all-bank"),
])
def test_ambiguous_or_unsupported_broadcast_traces_fail(tmp_path, lines, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        run(tmp_path, lines)


@pytest.mark.parametrize("value", [1, "true", None])
def test_broadcast_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        MemoryConfig(pim=True, pim_input_broadcast=value)


def test_broadcast_requires_pim():
    with pytest.raises(ValueError, match="requires pim"):
        MemoryConfig(pim_input_broadcast=True)


def test_broadcast_write_only_stream_retains_two_cycle_dq_spacing(tmp_path):
    _, events = run(tmp_path, [f"PIM_WR_GB {i * 32} 0" for i in range(4)])
    assert [e["clock"] for e in events if e["command"] == "WRGB"] == [1, 3, 5, 7]

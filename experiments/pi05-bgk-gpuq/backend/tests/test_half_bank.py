"""Half-bank scope, completion, and real compute/read overlap through Ramulator."""

import csv
from dataclasses import replace

import pytest

from simulator import MemoryConfig, run_trace


def run(path, lines, **overrides):
    path.mkdir()
    trace = path / "input.trace"
    trace.write_text("\n".join(lines) + "\n")
    result = run_trace(
        replace(MemoryConfig(pim=True, refresh=False), **overrides), trace, path / "run"
    )
    events = []
    for channel in range(result["config"]["channels"]):
        with (path / "run" / f"timeline.csv.ch{channel}").open() as stream:
            events.extend(dict(row, clock=int(row["clock"])) for row in csv.DictReader(stream))
    return result, events


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("model,first_mac,latency", [
    ("lpddr5-6400", 17, 0), ("lpddr5x-8533", 22, 0),
])
def test_half_bank_compute_matches_unscaled_timing(tmp_path, half, model, first_mac, latency):
    result, events = run(tmp_path / "half", [f"PIM_MAC_HB {half * 4096}"], model=model)
    assert [(e["command"], e["clock"]) for e in events] == [("ACTHB", 1), ("MACHB", first_mac)]
    assert {int(e["BankGroup"]) for e in events} == {half * 2}
    assert result["memory_cycles"] == first_mac + latency
    assert result["traffic_bytes"]["pim_internal"] == 8 * 32
    assert result["requests"]["pim_mac_hb"] == result["completed_requests"] == 1
    assert result["pim_model"]["half_bank"]["transfer_latency_scale"] == 1.0


@pytest.mark.parametrize("half", [0, 1])
def test_only_selected_eight_banks_open(tmp_path, half):
    result, _ = run(tmp_path / "scope", [
        f"PIM_MAC_HB {half * 4096}", "BARRIER 0",
        *[f"LD {bank * 2048}" for bank in range(16)],
    ])
    assert result["issued_commands"]["ACTHB"] == 1
    assert result["issued_commands"]["ACT1"] == result["issued_commands"]["ACT2"] == 8
    assert result["issued_commands"]["RD"] == 16


@pytest.mark.parametrize("reads_first", [False, True])
@pytest.mark.parametrize("wck_mode", ["always_on", "need_sync"])
def test_disjoint_reads_overlap_and_beat_same_work_with_barrier(tmp_path, reads_first, wck_mode):
    compute = [f"PIM_MAC_HB {i * 32}" for i in range(16)]
    reads = [f"LD {4096 + i * 32}" for i in range(24)]
    first, second = (reads, compute) if reads_first else (compute, reads)
    concurrent, events = run(tmp_path / "concurrent", first + second, wck_mode=wck_mode)
    serial, _ = run(tmp_path / "serial", first + ["BARRIER 0"] + second, wck_mode=wck_mode)
    macs = [e["clock"] for e in events if e["command"] == "MACHB"]
    rds = [e["clock"] for e in events if e["command"] == "RD"]
    assert min(rds) < max(macs)
    assert min(macs) < max(rds) + 19
    assert concurrent["memory_cycles"] < serial["memory_cycles"]
    assert concurrent["traffic_bytes"] == serial["traffic_bytes"]
    assert concurrent["completed_requests"] == serial["completed_requests"] == 40
    assert concurrent["memory_cycles"] == max(max(macs), max(rds) + 19)
    assert min(b - a for a, b in zip(macs, macs[1:])) >= 4


@pytest.mark.parametrize("reads_first", [False, True])
def test_same_half_is_serialized(tmp_path, reads_first):
    lines = ["PIM_MAC_HB 0", "LD 2048"]
    if reads_first:
        lines.reverse()
    result, events = run(tmp_path / "conflict", lines)
    clocks = {e["command"]: e["clock"] for e in events}
    if reads_first:
        assert clocks["MACHB"] >= clocks["RD"] + 19
    else:
        assert clocks["RD"] >= clocks["MACHB"] + 4
    assert result["completed_requests"] == 2


def test_row_conflict_precharges_only_compute_half(tmp_path):
    result, events = run(tmp_path / "rows", [
        "LD 4096", "BARRIER 0", "PIM_MAC_HB 0", "PIM_MAC_HB 32768",
        "BARRIER 1", "LD 4096",
    ])
    assert result["issued_commands"]["PREHB"] == 1
    assert result["issued_commands"]["PREab"] == 0
    assert result["issued_commands"]["ACT1"] == 1
    macs = [e["clock"] for e in events if e["command"] == "MACHB"]
    pre = next(e["clock"] for e in events if e["command"] == "PREHB")
    assert pre >= macs[0] + 6  # Read-to-precharge array recovery.
    assert macs[1] >= pre + 17 + 15


@pytest.mark.parametrize("channels,ranks", [(1, 2), (2, 1), (4, 2)])
def test_address_selects_channel_rank_and_half(tmp_path, channels, ranks):
    lines = [
        f"PIM_MAC_HB {(half * 2 * ranks + rank) * 2048 * channels + channel * 32}"
        for channel in range(channels) for rank in range(ranks) for half in range(2)
    ]
    result, events = run(tmp_path / "topology", lines, channels=channels, ranks=ranks)
    assert {(int(e["Channel"]), int(e["Rank"]), int(e["BankGroup"]))
            for e in events if e["command"] == "ACTHB"} == {
        (ch, rank, half * 2)
        for ch in range(channels) for rank in range(ranks) for half in range(2)
    }
    assert result["traffic_bytes"]["pim_internal"] == len(lines) * 256


def test_refresh_backpressure_and_other_pim_phases_complete(tmp_path):
    lines = ["PIM_WR_GB 0"]
    for i in range(1000):
        lines += [f"PIM_MAC_HB {i % 64 * 32}", f"LD {4096 + i % 64 * 32}"]
    lines += ["PIM_RD_RES 0", "PIM_MAC_AB 0", "ST 4096", "PIM_MAC_HB 0", "BARRIER 0"]
    result, events = run(tmp_path / "long", lines, refresh=True)
    assert result["completed_requests"] == 2005
    assert result["issued_commands"]["REFab"] > 0
    assert result["issued_commands"]["MACHB"] == 1001
    assert result["issued_commands"]["RD"] == 1000
    for index, event in enumerate(events):
        if event["command"] == "REFab":
            prior = [e["clock"] for e in events[:index] if e["command"] == "MACHB"]
            assert event["clock"] >= max(prior) + 6 + 17  # RTP + RPab.


def test_half_bank_requires_pim_enable(tmp_path):
    with pytest.raises(ValueError, match="pim: true"):
        run(tmp_path / "disabled", ["PIM_MAC_HB 0"], pim=False)

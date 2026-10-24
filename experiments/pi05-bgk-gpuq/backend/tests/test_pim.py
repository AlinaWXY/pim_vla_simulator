"""End-to-end timing oracles, independent of backend constraint generation."""

import csv
from dataclasses import replace

import pytest
import ramulator

from simulator import MemoryConfig, run_trace
from simulator.trace import generate_trace


def run(tmp_path, lines, **overrides):
    trace = tmp_path / "input.trace"
    trace.write_text("\n".join(lines) + "\n")
    result = run_trace(
        replace(MemoryConfig(pim=True, refresh=False), **overrides), trace, tmp_path / "run"
    )
    events = []
    for channel in range(result["config"]["channels"]):
        with (tmp_path / "run" / f"timeline.csv.ch{channel}").open() as stream:
            events.extend(dict(row, clock=int(row["clock"])) for row in csv.DictReader(stream))
    return result, events


@pytest.mark.parametrize("model,first_mac,tail,ck_ns", [
    ("lpddr5-6400", 17, 0, 1.25),
    ("lpddr5x-8533", 22, 0, 0.938),
])
@pytest.mark.parametrize("pim_request,activate,mac,banks", [
    ("PIM_MAC_AB", "ACTAB", "MACAB", 16),
    ("PIM_MAC_HB", "ACTHB", "MACHB", 8),
])
@pytest.mark.parametrize("overrides,interval", [({}, 4), ({"nCCDAB": 6}, 6)])
def test_eight_pim_row_hits_match_hand_calculation(
    tmp_path, model, first_mac, tail, ck_ns, pim_request, activate, mac, banks, overrides, interval
):
    result, events = run(
        tmp_path, [f"{pim_request} {i * 32}" for i in range(8)],
        model=model, timing_overrides=overrides,
    )
    # Activation starts at CK1 and completes reception at CK2. Read RCD
    # gives the first MAC; same-BG column reads permit one MAC every 4 CK.
    # MAC callbacks retire at issue: no inferred PE completion tail.
    # An explicit 6-CK override changes only the issue spacing.
    assert [(e["command"], e["clock"]) for e in events] == [
        (activate, 1),
        *[(mac, first_mac + interval * i) for i in range(8)],
    ]
    assert result["memory_cycles"] == first_mac + 7 * interval + tail
    assert result["latency_ns"] == pytest.approx((first_mac + 7 * interval + tail) * ck_ns)
    assert result["completed_requests"] == 8
    assert result["issued_commands"][activate] == 1
    assert result["traffic_bytes"]["pim_internal"] == 8 * banks * 32
    assert result["traffic_bytes"]["dram_read"] == 0
    assert result["energy_pj"] is None


def test_row_conflict_preserves_array_recovery_without_compute_tail(tmp_path):
    result, events = run(tmp_path, ["PIM_MAC_AB 0", "PIM_MAC_AB 32768"])
    assert [(e["command"], e["clock"]) for e in events] == [
        ("ACTAB", 1),
        ("MACAB", 17),
        ("PREab", 36),
        ("ACTAB", 52),
        ("MACAB", 68),
    ]
    assert result["memory_cycles"] == 68
    assert result["latency_ns"] == 85.0
    assert all(e["Bank"] == e["BankGroup"] == "-1" for e in events)


def test_mixed_modes_are_completion_fences_and_rows_persist(tmp_path):
    result, events = run(tmp_path, ["ST 0", "PIM_MAC_AB 32768", "LD 32768"])
    by_cmd = {
        name: [e["clock"] for e in events if e["command"] == name]
        for name in ("WR", "ACTAB", "PREab", "MACAB", "RD")
    }
    assert by_cmd["PREab"][0] >= by_cmd["WR"][0] + 9 + 4 + 28
    assert by_cmd["RD"][0] >= by_cmd["MACAB"][0] + 4
    assert result["issued_commands"]["ACT1"] == 1  # only the initial ST
    assert result["issued_commands"]["ACTAB"] == 1
    assert result["completed_requests"] == 3
    assert result["memory_cycles"] >= by_cmd["RD"][0] + 19


@pytest.mark.parametrize("channels,ranks", [(1, 2), (2, 1), (4, 2)])
def test_broadcast_is_scoped_to_addressed_channel_and_rank(tmp_path, channels, ranks):
    # Channel bits follow the 32B transaction; rank bit follows 64 columns.
    rank_stride = 2048 * channels
    lines = [
        f"PIM_MAC_AB {rank * rank_stride + channel * 32}"
        for channel in range(channels)
        for rank in range(ranks)
    ]
    result, events = run(tmp_path, lines, channels=channels, ranks=ranks)
    acts = [e for e in events if e["command"] == "ACTAB"]
    assert {(int(e["Channel"]), int(e["Rank"])) for e in acts} == {
        (c, r) for c in range(channels) for r in range(ranks)
    }
    assert len(acts) == channels * ranks
    assert result["completed_requests"] == channels * ranks
    assert result["traffic_bytes"]["pim_internal"] == channels * ranks * 512


def test_refresh_interrupts_pim_and_recovers(tmp_path):
    # At 4 CK per MAC, 1000 requests still cross the 3124-CK refresh period.
    result, events = run(tmp_path, ["PIM_MAC_AB 0"] * 1000, refresh=True)
    refreshes = [e for e in events if e["command"] == "REFab"]
    assert refreshes
    assert result["completed_requests"] == 1000
    assert result["issued_commands"]["MACAB"] == 1000
    assert result["issued_commands"]["ACTAB"] >= 2
    for refresh in refreshes:
        later_act = next(
            e for e in events if e["command"] == "ACTAB" and e["clock"] > refresh["clock"]
        )
        # tRFC is measured to the second CA cycle of ACTAB.
        assert later_act["clock"] + 1 >= refresh["clock"] + 168


def test_pim_derivation_follows_final_normal_overrides():
    model = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", nCCDL=5, nCL=20
    )
    _, t = model.resolve()
    assert t["nCCDAB"] == 5
    assert "nPIMLatency" not in t
    explicit = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400",
        nCCDAB=10,
    )
    assert explicit.resolve()[1]["nCCDAB"] == 10


def test_pim_requires_explicit_enable(tmp_path):
    trace = tmp_path / "trace"
    trace.write_text("PIM_MAC_AB 0\n")
    with pytest.raises(ValueError, match="pim: true"):
        run_trace(MemoryConfig(), trace, tmp_path / "run")


@pytest.mark.parametrize("count", [1, 65])
@pytest.mark.parametrize("pim_request", ["PIM_MAC_AB", "PIM_MAC_HB"])
def test_mac_completion_retires_at_issue(tmp_path, count, pim_request):
    result, events = run(tmp_path, [f"{pim_request} 0"] * count)
    assert result["completed_requests"] == count
    assert result["memory_cycles"] == events[-1]["clock"]


@pytest.mark.parametrize("tail", [0, 38, -1, True, 1.5])
def test_removed_compute_tail_is_rejected(tail):
    with pytest.raises(ValueError):
        MemoryConfig(pim=True, timing_overrides={"nPIMLatency": tail})
    with pytest.raises(ValueError, match="Unknown .* override"):
        ramulator.dram.LPDDR5PIM(
            org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", nPIMLatency=tail,
        ).resolve()


def test_all_banks_are_open_for_subsequent_normal_reads(tmp_path):
    lines = ["PIM_MAC_AB 0", *[f"LD {bank * 2048}" for bank in range(16)]]
    result, _ = run(tmp_path, lines)
    assert result["issued_commands"]["ACTAB"] == 1
    assert result["issued_commands"]["ACT1"] == result["issued_commands"]["ACT2"] == 0
    assert result["issued_commands"]["RD"] == 16


def test_pim_preserves_trace_order_across_rows(tmp_path):
    result, events = run(tmp_path, ["PIM_MAC_AB 0", "PIM_MAC_AB 32768", "PIM_MAC_AB 32"])
    assert [e["Row"] for e in events if e["command"] == "MACAB"] == ["0", "1", "0"]
    assert result["issued_commands"]["ACTAB"] == 3


@pytest.mark.parametrize("channels", [1, 4])
def test_enabling_pim_preserves_normal_only_trace_latency(tmp_path, channels):
    trace = generate_trace(tmp_path / "normal.trace", requests=128)
    ordinary = run_trace(MemoryConfig(channels=channels), trace, tmp_path / "ordinary")
    enabled = run_trace(MemoryConfig(channels=channels, pim=True), trace, tmp_path / "enabled")
    assert enabled["memory_cycles"] == ordinary["memory_cycles"]
    assert all(
        enabled["issued_commands"][cmd] == count
        for cmd, count in ordinary["issued_commands"].items()
    )

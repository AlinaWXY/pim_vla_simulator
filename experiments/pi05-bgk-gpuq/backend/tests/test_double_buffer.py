"""Behavioral checks for fragment input readiness, ownership, and overlap."""

import csv

import pytest

from simulator import MemoryConfig, run_trace
from simulator.trace import inspect_trace


def simulate(tmp_path, lines, interval=50):
    trace = tmp_path / "input.trace"
    trace.write_text("\n".join(lines) + "\n")
    result = run_trace(
        MemoryConfig(model="lpddr5x-8533", pim=True, refresh=False,
                     timing_overrides={"nCCDAB": interval}),
        trace, tmp_path / "run",
    )
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        events = [dict(e, clock=int(e["clock"]), pim_buffer=int(e["pim_buffer"]))
                  for e in csv.DictReader(stream)]
    return result, events


def test_next_buffer_fills_while_active_buffer_keeps_computing(tmp_path):
    # A blocked second MAC(A) must not prevent prefetching B. Conversely,
    # writes to B must not keep postponing MAC(A)'s input readiness.
    result, events = simulate(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 0 0", "PIM_MAC_AB 32 0",
        *[f"PIM_WR_GB {i * 32} 1" for i in range(60)],
        "PIM_MAC_AB 64 1", "PIM_RD_RES 0",
    ])
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    prefetch = [e["clock"] for e in events if e["command"] == "WRGB" and e["pim_buffer"] == 1]
    assert macs[1] - macs[0] == 50
    assert any(macs[0] < tick < macs[1] for tick in prefetch)
    assert macs[1] < max(prefetch)
    assert macs[2] >= max(prefetch) + 14
    reads = [e["clock"] for e in events if e["command"] == "RDRES"]
    assert min(reads) >= macs[-1] + 50
    assert "nPIMLatency" not in result["resolved_timings_ck"]
    assert result["completed_requests"] == 65


def test_active_buffer_cannot_be_overwritten(tmp_path):
    _, events = simulate(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 0 0",
        "PIM_WR_GB 32 0", "PIM_MAC_AB 32 0", "PIM_RD_RES 0",
    ])
    writes = [e["clock"] for e in events if e["command"] == "WRGB"]
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    assert writes[1] >= macs[0] + 50
    assert macs[1] >= writes[1] + 14


def test_annotated_mac_requires_input(tmp_path):
    with pytest.raises(RuntimeError, match="never filled"):
        simulate(tmp_path, ["PIM_MAC_AB 0 1"])


def test_activation_can_prepare_during_same_buffer_input_fill(tmp_path):
    result, events = simulate(tmp_path, [
        *[f"PIM_WR_GB {i * 32} 0" for i in range(60)],
        "PIM_MAC_AB 0 0", "PIM_RD_RES 0",
    ])
    last_write = max(e["clock"] for e in events if e["command"] == "WRGB")
    activate = next(e["clock"] for e in events if e["command"] == "ACTAB")
    mac = next(e["clock"] for e in events if e["command"] == "MACAB")
    assert activate < last_write
    assert mac >= last_write + result["resolved_timings_ck"]["nPIMWrLatency"]
    assert mac >= activate + result["resolved_timings_ck"]["nRCD"] + 1


def test_precharge_can_prepare_next_row_during_input_fill_without_skipping_mac(tmp_path):
    result, events = simulate(tmp_path, [
        "PIM_WR_GB 0 0", "PIM_MAC_AB 0 0",
        *[f"PIM_WR_GB {i * 32} 1" for i in range(60)],
        "PIM_MAC_AB 32768 1", "PIM_RD_RES 0",
    ])
    macs = [e["clock"] for e in events if e["command"] == "MACAB"]
    precharge = next(e["clock"] for e in events if e["command"] == "PREab")
    last_write = max(e["clock"] for e in events if e["command"] == "WRGB")
    activate = next(e["clock"] for e in events if e["command"] == "ACTAB" and e["Row"] == "1")
    assert precharge >= macs[0] + result["resolved_timings_ck"]["nRTP"]
    assert precharge < activate < last_write
    assert macs[1] >= last_write + result["resolved_timings_ck"]["nPIMWrLatency"]


@pytest.mark.parametrize("line", [
    "PIM_WR_GB 0 2", "PIM_MAC_AB 0 -1", "PIM_RD_RES 0 0", "LD 0 1",
    "BARRIER 0 0", "PIM_WR_GB 0 0 1",
])
def test_trace_rejects_invalid_buffer_annotations(tmp_path, line):
    trace = tmp_path / "bad.trace"
    trace.write_text(line + "\n")
    with pytest.raises(ValueError):
        inspect_trace(trace, transaction_bytes=32, capacity_bytes=1024**3, pim=True)


def test_result_drain_covers_both_ranks(tmp_path):
    trace = tmp_path / "ranks.trace"
    trace.write_text("PIM_WR_GB 0 0\nPIM_WR_GB 2048 0\n"
                     "PIM_MAC_AB 0 0\nPIM_MAC_AB 2048 0\n"
                     "PIM_RD_RES 0\nPIM_RD_RES 2048\n")
    run_trace(MemoryConfig(model="lpddr5x-8533", pim=True, ranks=2, refresh=False,
                          timing_overrides={"nCCDAB": 50}), trace, tmp_path / "run")
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        events = list(csv.DictReader(stream))
    for rank in ("0", "1"):
        mac = next(int(e["clock"]) for e in events if e["command"] == "MACAB" and e["Rank"] == rank)
        read = next(int(e["clock"]) for e in events if e["command"] == "RDRES" and e["Rank"] == rank)
        assert read >= mac + 50


def test_prefetch_preserves_requested_cross_rank_write_order(tmp_path):
    trace = tmp_path / "ordered.trace"
    lines, expected = [], []
    for fragment in range(4):
        for row in range(25):
            for rank in range(2):
                lines.append(f"PIM_WR_GB {rank * 2048 + row * 32} {fragment % 2}")
                expected.append((rank, row, fragment % 2))
        for rank in range(2):
            lines.append(f"PIM_MAC_AB {rank * 2048 + fragment * 32} {fragment % 2}")
    lines += ["PIM_RD_RES 0", "PIM_RD_RES 2048"]
    trace.write_text("\n".join(lines) + "\n")
    run_trace(MemoryConfig(model="lpddr5x-8533", pim=True, ranks=2, refresh=False,
                          timing_overrides={"nCCDAB": 25}), trace, tmp_path / "run")
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        actual = [(int(e["Rank"]), int(e["Column"]), int(e["pim_buffer"]))
                  for e in csv.DictReader(stream) if e["command"] == "WRGB"]
    assert actual == expected

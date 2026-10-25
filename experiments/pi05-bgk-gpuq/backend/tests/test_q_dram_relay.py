"""Native source reads, finite staging, multicast, and SRAM dependencies."""

import csv
from dataclasses import replace

import pytest

from simulator import MemoryConfig, run_trace


def address(rank=0, bg=0, bank=0, row=12, col=0, channel=0):
    # Fixed 16-channel, two-rank RoBaRaCoCh test geometry.
    return (row << 20) | (bank << 18) | (bg << 16) | (rank << 15) | (col << 9) | (channel << 5)


def op(name, slot=None, **where):
    return f"{name} {address(**where)}" + ("" if slot is None else f" {slot}")


def run(tmp_path, lines, **overrides):
    trace = tmp_path / "input.trace"
    trace.write_text("\n".join(lines) + "\n")
    cfg = replace(MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, pim=True,
        refresh=False, frontend_ratio=16, pim_input_broadcast=True,
        pim_mac_scope="channel", pim_data_path="attacc", timing_overrides={"nCCDAB": 20},
    ), **overrides)
    result = run_trace(cfg, trace, tmp_path / "run")
    with (tmp_path / "run" / "bgk_events.csv.ch0").open() as stream:
        events = [{k: v if k == "operation" else int(v) for k, v in row.items()}
                  for row in csv.DictReader(stream)]
    with (tmp_path / "run" / "timeline.csv.ch0").open() as stream:
        commands = [{**row, "clock": int(row["clock"])} for row in csv.DictReader(stream)]
    return result, events, commands


def test_q_reads_return_before_charged_relay_and_mac(tmp_path):
    result, events, commands = run(tmp_path, [
        *[op("PIM_RD_Q", 0, bank=b) for b in range(4)],
        op("PIM_MAC_QK", 0, row=8), "BARRIER 1",
    ])
    reads = [e for e in events if e["operation"] == "PIM_RD_Q"]
    relays = [e for e in events if e["operation"] == "PIM_RELAY_Q"]
    mac = next(e for e in events if e["operation"] == "PIM_MAC_QK")
    t = result["resolved_timings_ck"]
    assert len(reads) == len(relays) == result["issued_commands"]["RD"] == 4
    assert result["issued_commands"]["WRGB"] == 4
    for read, relay in zip(reads, relays):
        assert read["target_banks"] == 1 and relay["target_banks"] == 8
        assert read["ready_at"] - read["clock"] == t["nCL"] + t["nBL_min"]
        assert relay["clock"] >= read["ready_at"]
        assert relay["ready_at"] - relay["clock"] == t["nBL_min"]
        for key in ("Rank", "BankGroup", "Bank", "Row", "Column", "buffer"):
            assert read[key] == relay[key]
    assert mac["clock"] >= max(e["ready_at"] for e in relays)
    assert result["memory_cycles"] >= mac["ready_at"]
    assert result["traffic_bytes"]["pim_input_write"] == 0
    assert result["traffic_bytes"]["q_dram_read"] == result["traffic_bytes"]["q_internal_relay"] == 128
    assert result["traffic_bytes"]["bgk_input_delivered_to_bank_buffers"] == 128 * 8
    assert result["completed_requests"] == 5
    assert len([c for c in commands if c["command"] == "ACT1"]) == 4
    assert len([c for c in commands if c["command"] == "ACT2"]) == 4


def test_q_source_row_hit_and_conflict_use_native_precharge(tmp_path):
    result, events, commands = run(tmp_path, [
        op("PIM_RD_Q", 0, row=12, col=0),
        op("PIM_RD_Q", 0, row=12, col=1),
        op("PIM_RD_Q", 0, row=13, col=0), "BARRIER 1",
    ])
    assert result["issued_commands"]["RD"] == 3
    assert result["issued_commands"]["ACT1"] == result["issued_commands"]["ACT2"] == 2
    assert result["issued_commands"]["PREpb"] == 1
    # These are ordinary reads, so their normal row-hit statistics are real.
    controller = result["raw_stats"]["memory_system"]["controller"][0]
    assert controller["read_row_hits"] >= 1
    assert {e["Row"] for e in events} == {12, 13}


def test_q_staging_is_bounded_to_one_32_byte_burst(tmp_path):
    result, events, _ = run(tmp_path, [
        *[op("PIM_RD_Q", 0, bank=b, col=q) for b in range(4) for q in range(20)],
        "BARRIER 1",
    ])
    reads = [e for e in events if e["operation"] == "PIM_RD_Q"]
    relays = [e for e in events if e["operation"] == "PIM_RELAY_Q"]
    assert len(reads) == len(relays) == 80
    assert all(read["clock"] >= relay["ready_at"] for read, relay in zip(reads[1:], relays))
    assert result["memory_cycles"] >= relays[-1]["ready_at"]


@pytest.mark.parametrize("next_slot", [0, 1])
def test_q_relay_preserves_active_slot_but_can_fill_inactive_slot(tmp_path, next_slot):
    _, events, _ = run(tmp_path, [
        *[op("PIM_WR_Q", 0, bank=b, row=8) for b in range(4)],
        op("PIM_MAC_QK", 0, row=8),
        op("PIM_RD_Q", next_slot, row=12), "BARRIER 1",
    ], timing_overrides={"nCCDAB": 20, "nCCDQK": 200})
    mac = next(e for e in events if e["operation"] == "PIM_MAC_QK")
    relay = next(e for e in events if e["operation"] == "PIM_RELAY_Q")
    if next_slot == 0:
        assert relay["clock"] >= mac["ready_at"]
    else:
        assert mac["clock"] < relay["clock"] < mac["ready_at"]


@pytest.mark.parametrize("where", [{"rank": 1}, {"bg": 1}])
def test_q_source_must_be_explicit_rank0_bg0(tmp_path, where):
    with pytest.raises(RuntimeError, match="rank0/BG0"):
        run(tmp_path, [op("PIM_RD_Q", 0, **where)])


def test_q_refill_is_not_broadcast_to_other_bank_indices(tmp_path):
    with pytest.raises(RuntimeError, match="unfilled bank-local"):
        run(tmp_path, [op("PIM_RD_Q", 0, bank=0), op("PIM_MAC_QK", 0, row=8)])


def test_gpu_and_internal_q_traffic_are_separate(tmp_path):
    result, _, _ = run(tmp_path, [
        op("PIM_WR_Q", 0, bank=0), op("PIM_RD_Q", 1, bank=1), "BARRIER 1",
    ])
    assert result["traffic_bytes"]["pim_input_write"] == 32
    assert result["traffic_bytes"]["q_internal_relay"] == 32
    assert result["traffic_bytes"]["pim_movement_write_total"] == 64


def test_legacy_attacc_input_waits_for_last_complete_burst(tmp_path):
    result, _, commands = run(tmp_path, [
        *[op("PIM_WR_GB", 0, row=0, col=i) for i in range(40)],
        op("PIM_MAC_AB", 0, row=0), "BARRIER 1",
    ])
    last_write = max(c["clock"] for c in commands if c["command"] == "WRGB")
    mac = next(c["clock"] for c in commands if c["command"] == "MACAB")
    assert mac >= last_write + result["resolved_timings_ck"]["nBL_min"]
    assert result["pim_model"]["input_buffer_write_latency_ck"] == result["resolved_timings_ck"]["nBL_min"]

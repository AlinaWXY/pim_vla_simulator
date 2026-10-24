"""Independent layout/work conservation, addressing, and completion checks."""

import csv
import json
from collections import Counter
from dataclasses import replace

import pytest
import ramulator

from simulator import MemoryConfig, run_trace
from simulator.ffn import (
    FFNConfig,
    build_layout,
    decode_address,
    encode_address,
    generate_ffn,
    run_ffn,
    weight_location,
)


@pytest.mark.parametrize("ranks,outputs", [(1, (16, 16, 4)), (2, (8, 8, 2))])
def test_layout_balances_outputs_and_keeps_k_in_one_bank(ranks, outputs):
    cfg = FFNConfig(ranks=ranks)
    layout = build_layout(cfg)
    assert tuple(m["outputs_per_bank"] for m in layout["matrices"]) == outputs
    assert len({m["weights_per_bank"] for m in layout["matrices"]}) == 1
    occupied = set()
    for matrix in layout["matrices"]:
        bank_weights = Counter()
        for n in range(matrix["N"]):
            first = weight_location(cfg, matrix, n)
            last = weight_location(cfg, matrix, n, matrix["K"] - 1)
            assert first["global_bank"] == last["global_bank"]
            assert last["bank_byte_offset"] - first["bank_byte_offset"] == 2 * (matrix["K"] - 1)
            bank_weights[first["global_bank"]] += matrix["K"]
            # Every vector occupies a distinct interval in its bank.
            key = (first["global_bank"], first["bank_byte_offset"])
            assert key not in occupied
            occupied.add(key)
            if matrix["name"] == "down":
                assert last["row"] - first["row"] == 3
        assert len(bank_weights) == cfg.total_banks
        assert set(bank_weights.values()) == {16384 // ranks}


def test_horizon_does_not_change_any_weight_address(tmp_path):
    cfg = FFNConfig(packages=1, channels_per_package=1, hidden_size=16,
                    intermediate_size=32, horizon=1)
    generate_ffn(cfg, tmp_path / "h1")
    generate_ffn(replace(cfg, horizon=3), tmp_path / "h3")
    assert (tmp_path / "h1/layout.json").read_bytes() == (tmp_path / "h3/layout.json").read_bytes()
    assert (tmp_path / "h1/weights.csv").read_bytes() == (tmp_path / "h3/weights.csv").read_bytes()
    manifests = [json.loads((tmp_path / h / "manifest.json").read_text()) for h in ("h1", "h3")]
    for op in ("PIM_MAC_AB", "PIM_WR_GB", "PIM_RD_RES"):
        assert manifests[1]["command_counts"][op] == 3 * manifests[0]["command_counts"][op]


@pytest.mark.parametrize("packages,channels,ranks", [(1, 1, 1), (2, 2, 2), (8, 2, 2)])
def test_address_mapping_round_trip(packages, channels, ranks):
    cfg = FFNConfig(packages=packages, channels_per_package=channels, ranks=ranks)
    for c in range(cfg.channels):
        for r in range(ranks):
            for bg in range(4):
                for bank in range(4):
                    for offset in (0, 32, 2046, 2048, 8190, 32768):
                        address = encode_address(cfg, c, r, bg, bank, offset)
                        assert decode_address(cfg, address) == dict(
                            channel=c, rank=r, bankgroup=bg, bank=bank,
                            row=offset // 2048, column_bytes=offset % 2048,
                        )


@pytest.mark.parametrize("kwargs", [
    {"packages": 3}, {"horizon": 0}, {"ranks": 3}, {"hidden_size": 1000},
    {"intermediate_size": 128}, {"input_buffer_bytes": 2048},
    {"nonlinear_cycles_per_token": -1}, {"nonlinear_cycles_per_token": True},
])
def test_reject_unsupported_or_unbalanced_layout(kwargs):
    with pytest.raises(ValueError):
        FFNConfig(**kwargs)


@pytest.mark.parametrize("model,cls,ck,latency", [
    ("lpddr5-6400", "LPDDR5PIM", 1250, 0),
    ("lpddr5x-8533", "LPDDR5XPIM", 938, 0),
])
def test_pim_derives_from_selected_calibrated_model(model, cls, ck, latency):
    base = "LPDDR5" if model == "lpddr5-6400" else "LPDDR5X"
    org, timings = getattr(ramulator.dram, cls)(
        org_preset=f"{base}_16Gb_x16", timing_preset=f"{base}_{model.rsplit('-', 1)[1]}"
    ).resolve()
    assert timings["tCK_ps"] == ck
    assert timings["nCCDAB"] == 4
    assert "nPIMLatency" not in timings
    if base == "LPDDR5X":
        assert timings["nRRDS"] == 4
        assert timings["nFAW"] == 16
        assert timings["nRCDWR"] == 9
    assert org["bankgroup"] * org["bank"] == 16


def test_8533_single_mac_hand_timeline(tmp_path):
    trace = tmp_path / "one.trace"
    trace.write_text("PIM_MAC_AB 0\n")
    result = run_trace(MemoryConfig(model="lpddr5x-8533", pim=True, refresh=False),
                       trace, tmp_path / "run")
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        events = list(csv.DictReader(stream))
    # ACTAB CK1/2; read RCD=20 -> MAC and callback at CK22; no compute tail.
    assert [(e["command"], int(e["clock"])) for e in events] == [("ACTAB", 1), ("MACAB", 22)]
    assert result["memory_cycles"] == 22


def test_ffn_completion_buffers_and_nonlinear_wait(tmp_path):
    cfg = FFNConfig(packages=1, channels_per_package=1, horizon=2,
                    hidden_size=16, intermediate_size=32, refresh=False)
    omitted = run_ffn(cfg, tmp_path / "omitted")
    explicit = run_ffn(replace(cfg, nonlinear_cycles_per_token=100), tmp_path / "explicit")
    assert not omitted["nonlinear_compute_included"]
    assert explicit["nonlinear_compute_included"]
    assert explicit["memory_cycles"] - omitted["memory_cycles"] == 200
    assert explicit["phase_latency_ns"]["nonlinear"] == pytest.approx(200 * 0.938)
    assert omitted["issued_commands"]["MACAB"] * 256 == 2 * 3 * 16 * 32
    assert omitted["issued_commands"]["WRGB"] == 2 * (16 + 32) // 16
    assert omitted["traffic_bytes"]["pim_result_read"] == 2 * (32 + 32 + 16) * 4
    assert omitted["traffic_bytes"]["dram_read"] == 0
    assert omitted["traffic_bytes"]["dram_write"] == 0


def test_generated_addresses_reach_real_backend_banks(tmp_path):
    cfg = FFNConfig(packages=1, channels_per_package=2, ranks=2, horizon=1,
                    hidden_size=64, intermediate_size=128, refresh=False)
    trace = tmp_path / "address.trace"
    targets = [(1, 1, 3, 2, 4 * 2048 + 64), (0, 0, 1, 3, 2048 + 32)]
    trace.write_text("".join(f"LD {encode_address(cfg, *t)}\n" for t in targets))
    result = run_trace(replace(cfg.memory_config(), pim=False), trace, tmp_path / "run")
    seen = set()
    for ch in range(cfg.channels):
        with (tmp_path / "run" / f"timeline.csv.ch{ch}").open() as stream:
            for e in csv.DictReader(stream):
                if e["command"] == "RD":
                    keys = ("Channel", "Rank", "BankGroup", "Bank", "Row")
                    seen.add(tuple(int(e[k]) for k in keys))
    assert seen == {(c, r, bg, b, off // 2048) for c, r, bg, b, off in targets}
    assert result["completed_requests"] == 2


def test_barriers_cannot_finish_on_issue_and_buffer_commands_do_not_activate(tmp_path):
    trace = tmp_path / "buffers.trace"
    trace.write_text("PIM_WR_GB 0\nBARRIER 0\nPIM_RD_RES 0\nBARRIER 1\nWAIT 10\nBARRIER 2\n")
    result = run_trace(MemoryConfig(model="lpddr5x-8533", pim=True, refresh=False),
                       trace, tmp_path / "run")
    assert result["issued_commands"]["ACTAB"] == 0
    assert result["issued_commands"]["ACT1"] == 0
    assert result["completed_requests"] == 2
    assert result["raw_stats"]["frontend"]["barriers_completed"] == 3
    assert result["memory_cycles"] >= 1 + 14 + 25 + 10


@pytest.mark.parametrize("op,command,gap", [("PIM_WR_GB", "WRGB", 3),
                                          ("PIM_RD_RES", "RDRES", 4)])
def test_buffer_transfers_respect_cross_rank_dq_turnaround(tmp_path, op, command, gap):
    trace = tmp_path / "rank.trace"
    trace.write_text(f"{op} 0\n{op} 2048\n")
    run_trace(MemoryConfig(model="lpddr5x-8533", ranks=2, pim=True, refresh=False),
              trace, tmp_path / "run")
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        events = [e for e in csv.DictReader(stream) if e["command"] == command]
    assert [int(e["Rank"]) for e in events] == [0, 1]
    assert int(events[1]["clock"]) - int(events[0]["clock"]) == gap


def test_buffer_transfers_reject_unmodeled_wck_startup(tmp_path):
    trace = tmp_path / "buffer.trace"
    trace.write_text("PIM_WR_GB 0\n")
    with pytest.raises(ValueError, match="always_on"):
        run_trace(MemoryConfig(pim=True, wck_mode="need_sync"), trace, tmp_path / "run")


@pytest.mark.parametrize("pim_request,command", [("PIM_MAC_AB", "MACAB"), ("PIM_MAC_HB", "MACHB")])
def test_unannotated_mac_has_no_extra_result_delay(tmp_path, pim_request, command):
    trace = tmp_path / "mac-result.trace"
    trace.write_text(f"{pim_request} 0\nPIM_RD_RES 0\n")
    result = run_trace(MemoryConfig(model="lpddr5x-8533", pim=True, refresh=False),
                       trace, tmp_path / "run")
    with (tmp_path / "run/timeline.csv.ch0").open() as stream:
        events = {e["command"]: int(e["clock"]) for e in csv.DictReader(stream)}
    # HB admission waits for its callback before accepting a buffer phase;
    # AB can queue RDRES earlier and let the device enforce the dependency.
    assert events["RDRES"] - events[command] == 1
    assert result["memory_cycles"] == events["RDRES"] + 25


@pytest.mark.parametrize("reuse,expected_input,expected_acc", [(1, 8192, 4),
                                                            (5, 40960, 20),
                                                            (20, 81920, 40)])
def test_row_reuse_buffer_budgets(reuse, expected_input, expected_acc):
    cfg = FFNConfig(horizon=10, reuse_tokens=reuse)
    assert cfg.input_buffer_bytes == expected_input
    assert cfg.accumulator_buffer_bytes_per_bank == expected_acc


@pytest.mark.parametrize("kwargs", [
    {"reuse_tokens": 0}, {"reuse_tokens": True},
    {"reuse_tokens": 2, "input_buffer_bytes": 8192},
    {"reuse_tokens": 2, "accumulator_buffer_bytes_per_bank": 4},
    {"accumulator_buffer_bytes_per_bank": True},
])
def test_row_reuse_rejects_invalid_or_insufficient_buffers(kwargs):
    with pytest.raises(ValueError):
        FFNConfig(**kwargs)


def test_reuse_one_preserves_legacy_trace_byte_for_byte(tmp_path):
    cfg = FFNConfig(packages=1, channels_per_package=2, ranks=2, horizon=2,
                    hidden_size=64, intermediate_size=128)
    generate_ffn(cfg, tmp_path / "actual")
    layout = build_layout(cfg)
    lines = []
    marker = 0

    def emit(op, value):
        lines.append(f"{op} 0x{value:x}\n")

    def fence():
        nonlocal marker
        emit("BARRIER", marker)
        marker += 1

    def command(op, offset):
        for rank in range(cfg.ranks):
            for channel in range(cfg.channels):
                emit(op, encode_address(cfg, channel, rank, 0, 0, offset))

    def load(k):
        fence()
        for byte in range(0, k * 2, 32):
            command("PIM_WR_GB", byte)
        fence()

    def project(matrix):
        fence()
        for output in range(matrix["outputs_per_bank"]):
            offset = matrix["base_row"] * 2048 + output * matrix["vector_bytes"]
            for byte in range(0, matrix["vector_bytes"], 32):
                command("PIM_MAC_AB", offset + byte)
            fence()
            for byte in (0, 32):
                command("PIM_RD_RES", byte)
            fence()
        fence()

    for _ in range(cfg.horizon):
        load(cfg.hidden_size)
        project(layout["matrices"][0])
        project(layout["matrices"][1])
        fence()
        emit("WAIT", 0)
        fence()
        load(cfg.intermediate_size)
        project(layout["matrices"][2])
    assert (tmp_path / "actual/ffn.trace").read_text() == "".join(lines)


def test_reuse_preserves_layout_and_all_column_mac_work(tmp_path):
    manifests = []
    mac_addresses = []
    for reuse in (1, 2, 4):
        cfg = FFNConfig(packages=1, channels_per_package=2, ranks=2, horizon=5,
                        hidden_size=64, intermediate_size=128, reuse_tokens=reuse)
        target = tmp_path / str(reuse)
        manifests.append(generate_ffn(cfg, target))
        trace_lines = (target / "ffn.trace").read_text().splitlines()
        mac_addresses.append(Counter(line for line in trace_lines
                                     if line.startswith("PIM_MAC_AB ")))
        for name in ("layout.json", "weights.csv"):
            assert (target / name).read_bytes() == (tmp_path / "1" / name).read_bytes()
    for manifest in manifests[1:]:
        for op in ("PIM_MAC_AB", "PIM_WR_GB", "PIM_RD_RES"):
            assert manifest["command_counts"][op] == manifests[0]["command_counts"][op]
        assert manifest["traffic_bytes"] == manifests[0]["traffic_bytes"]
    assert mac_addresses[0] == mac_addresses[1] == mac_addresses[2]


def test_row_reuse_context_covers_every_dot_product_without_padding(tmp_path):
    # Down vectors are 2080 B: straddle rows at nonuniform offsets. Tail tile is 1 token.
    cfg = FFNConfig(packages=1, channels_per_package=1, horizon=3,
                    hidden_size=32, intermediate_size=1040, reuse_tokens=2)
    manifest = generate_ffn(cfg, tmp_path / "trace")
    matrices = {m["name"]: m for m in build_layout(cfg)["matrices"]}
    trace = (tmp_path / "trace/ffn.trace").read_text().splitlines()
    coverage = Counter()
    tile_sizes = set()
    with (tmp_path / "trace/compute_schedule.csv").open() as stream:
        for raw in csv.DictReader(stream):
            s = {key: (value if key == "phase" else int(value)) for key, value in raw.items()}
            matrix = matrices[s["phase"]]
            tile_sizes.add(s["tile_size"])
            assert s["token"] == s["tile_start"] + s["input_slot"]
            assert s["accumulator_slot"] == s["input_slot"] < s["tile_size"]
            assert s["token"] < cfg.horizon
            assert s["input_byte_offset"] == (
                s["input_slot"] * matrix["K"] * 2 + s["reduction_start"] * 2
            )
            assert s["input_byte_offset"] + 2 * s["elements"] <= cfg.input_buffer_bytes
            assert s["bank_byte_offset"] // 2048 == s["bank_row"]
            assert (s["bank_byte_offset"] + 2 * s["elements"] - 1) // 2048 == s["bank_row"]
            commands = trace[s["first_trace_line"] - 1:s["last_trace_line"]]
            assert len(commands) == s["elements"] // 16
            for i, command in enumerate(commands):
                op, address = command.split()
                assert op == "PIM_MAC_AB"
                decoded = decode_address(cfg, int(address, 0))
                assert decoded["row"] * 2048 + decoded["column_bytes"] == (
                    s["bank_byte_offset"] + i * 32
                )
            for k in range(s["reduction_start"], s["reduction_start"] + s["elements"]):
                coverage[s["phase"], s["token"], s["local_output"], k] += 1
    expected = {(m["name"], h, n, k): 1 for m in matrices.values()
                for h in range(cfg.horizon) for n in range(m["outputs_per_bank"])
                for k in range(m["K"])}
    assert coverage == expected
    assert tile_sizes == {1, 2}
    assert manifest["buffers"]["additional_weight_sram_bytes_per_bank"] == 0


def test_row_reuse_backend_reduces_act_not_mac_or_data_movement(tmp_path):
    results = []
    for reuse in (1, 2):
        cfg = FFNConfig(packages=1, channels_per_package=1, horizon=3,
                        hidden_size=32, intermediate_size=1040, reuse_tokens=reuse,
                        refresh=False)
        result = run_ffn(cfg, tmp_path / str(reuse))
        assert result["issued_commands"]["ACTAB"] == result["row_reuse"]["ideal_actab_no_refresh"]
        assert result["issued_commands"]["PREab"] == result["issued_commands"]["ACTAB"] - 1
        assert result["pim_issue_interval_ck"] == 4
        results.append(result)
    assert results[1]["issued_commands"]["ACTAB"] < results[0]["issued_commands"]["ACTAB"]
    for op in ("MACAB", "WRGB", "RDRES"):
        assert results[0]["issued_commands"][op] == results[1]["issued_commands"][op]
    assert results[0]["traffic_bytes"] == results[1]["traffic_bytes"]
    assert results[1]["latency_ns"] < results[0]["latency_ns"]


def test_tiled_nonlinear_wait_covers_tail_tokens(tmp_path):
    cfg = FFNConfig(packages=1, channels_per_package=1, horizon=5, reuse_tokens=2,
                    hidden_size=16, intermediate_size=32, refresh=False)
    omitted = run_ffn(cfg, tmp_path / "omitted")
    explicit = run_ffn(replace(cfg, nonlinear_cycles_per_token=100), tmp_path / "explicit")
    assert explicit["memory_cycles"] - omitted["memory_cycles"] == 500
    assert explicit["phase_latency_ns"]["nonlinear"] == pytest.approx(500 * 0.938)


@pytest.mark.parametrize("kwargs", [
    {"macab_tokens_per_command": 0},
    {"reuse_tokens": 3, "macab_tokens_per_command": 2},
    {"macab_issue_interval_ck": 0},
    {"macab_issue_interval_ck": True},
])
def test_fused_macab_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FFNConfig(**kwargs)


def test_fused_macab_preserves_work_and_reduces_real_commands(tmp_path):
    base = FFNConfig(packages=1, channels_per_package=1, horizon=5,
                     hidden_size=16, intermediate_size=32, refresh=False)
    legacy = generate_ffn(base, tmp_path / "legacy")
    fused_cfg = replace(
        base, reuse_tokens=3, macab_tokens_per_command=3,
        input_buffer_bytes=None, accumulator_buffer_bytes_per_bank=None,
    )
    fused = generate_ffn(fused_cfg, tmp_path / "fused")
    assert legacy["logical_mac_operations"] == fused["logical_mac_operations"] == 5 * 1536
    assert legacy["command_counts"]["PIM_MAC_AB"] == 30
    assert fused["command_counts"]["PIM_MAC_AB"] == 12
    assert fused["enhanced_macab"]["enabled"]
    assert fused["row_reuse"]["weight_scans"] == 2
    assert fused["traffic_bytes"]["weight_internal"] * 5 == (
        legacy["traffic_bytes"]["weight_internal"] * 2
    )


def test_fused_macab_backend_uses_overridden_issue_interval(tmp_path):
    cfg = FFNConfig(
        packages=1, channels_per_package=1, horizon=5,
        hidden_size=16, intermediate_size=32, reuse_tokens=3,
        macab_tokens_per_command=3, macab_issue_interval_ck=8, refresh=False,
    )
    result = run_ffn(cfg, tmp_path / "run")
    assert result["pim_issue_interval_ck"] == 8
    assert result["issued_commands"]["MACAB"] == 12
    assert result["logical_mac_operations"] == 5 * 1536
    assert result["enhanced_macab"]["tokens_per_full_command"] == 3

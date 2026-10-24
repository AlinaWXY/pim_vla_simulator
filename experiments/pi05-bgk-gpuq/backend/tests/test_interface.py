import json
from dataclasses import replace

import pytest

from simulator import MemoryConfig, run_trace
from simulator.trace import generate_trace


def test_yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: lpddr5-6400\nchannels: 4\n")
    assert MemoryConfig.from_yaml(path).channels == 4
    path.write_text("channels: 4\nchannles: 2\n")
    with pytest.raises(ValueError, match="Unknown"):
        MemoryConfig.from_yaml(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": 3},
        {"channels": True},
        {"refresh": "false"},
        {"ranks": 2, "wck_mode": "need_sync"},
        {"density_gbit": 4},
        {"model": "pim"},
        {"model": []},
        {"timing_overrides": {"tCK_ps": 1}},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MemoryConfig(**kwargs)


@pytest.mark.parametrize("channels", [1, 4])
def test_trace_completes_and_counts_real_commands(tmp_path, channels):
    config = replace(MemoryConfig(), channels=channels)
    trace = generate_trace(tmp_path / "mixed.trace", requests=128)
    result = run_trace(config, trace, tmp_path / "result")
    assert result["completed_requests"] == 128
    assert result["requests"]["reads"] == 102
    assert result["requests"]["writes"] == 26
    assert result["issued_commands"]["RD"] == 102
    assert result["issued_commands"]["WR"] == 26
    assert result["traffic_bytes"]["dram_read"] == 102 * 32
    assert result["traffic_bytes"]["dram_write"] == 26 * 32
    assert result["energy_pj"] is None
    assert result["memory_cycles"] > 0
    assert result["latency_ns"] == result["memory_cycles"] * 1.25
    assert json.loads((tmp_path / "result/result.json").read_text()) == result
    assert len(list((tmp_path / "result").glob("commands.ch*.csv"))) == channels
    again = run_trace(config, trace, tmp_path / "again")
    assert again["memory_cycles"] == result["memory_cycles"]
    assert again["issued_commands"] == result["issued_commands"]
    with pytest.raises(FileExistsError):
        run_trace(config, trace, tmp_path / "result")


def test_write_recovery_not_only_enqueue(tmp_path):
    trace = generate_trace(tmp_path / "write.trace", requests=1, pattern="write")
    result = run_trace(MemoryConfig(), trace, tmp_path / "result")
    assert result["completed_requests"] == 1
    assert result["memory_cycles"] >= 15 + 9 + 4 + 28


@pytest.mark.parametrize(
    "text,match",
    [
        ("MACAB 0", "PIM"),
        ("LD 1", "aligned"),
        ("LD -32", "capacity"),
        ("LD 1073741824", "capacity"),
        ("", "Empty"),
        ("LD 0\n\n", "line 2"),
    ],
)
def test_invalid_trace_is_rejected(tmp_path, text, match):
    path = tmp_path / "bad.trace"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        run_trace(MemoryConfig(), path, tmp_path / "result")
    assert not (tmp_path / "result/result.json").exists()


def test_runaway_limit_is_enforced(tmp_path):
    trace = generate_trace(tmp_path / "trace", requests=64)
    with pytest.raises(RuntimeError, match="limit exceeded"):
        run_trace(MemoryConfig(max_cycles=1), trace, tmp_path / "result")


def test_branch_default_model(tmp_path):
    from pathlib import Path

    config = MemoryConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    trace = generate_trace(tmp_path / "trace", requests=16)
    result = run_trace(config, trace, tmp_path / "result")
    assert result["completed_requests"] == 16
    assert result["tCK_ns"] == (1.25 if config.model == "lpddr5-6400" else 0.938)

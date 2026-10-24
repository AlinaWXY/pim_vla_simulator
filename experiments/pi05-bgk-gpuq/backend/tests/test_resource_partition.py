"""Check actual cross-channel overlap, partition fairness and timing denominators."""

import csv
from dataclasses import replace

import pytest

from simulator.resource_partition import CASES, Experiment, run_case


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("partitions")
    cfg = Experiment(reads=8192, pim_rounds=4096, pattern="row-hit")
    return {case: run_case(root / case, case, cfg) for case in CASES}


def test_fair_work_and_physical_interface_limit(cases):
    a, b, c = (cases[case] for case in CASES)
    assert {r["gpu_read_bytes"] for r in cases.values()} == {8192 * 32}
    assert len({r["workload"]["logical_read_sha256"] for r in cases.values()}) == 1
    assert b["workload"]["pim_banks"] == c["workload"]["pim_banks"] == 32
    assert b["pim_internal_bytes"] == c["pim_internal_bytes"]
    assert [r["active_gpu_channels"] for r in (a, b, c)] == [4, 2, 4]
    assert a["full_system_peak_GBps"] == pytest.approx(51.2)
    assert b["aggregate_bandwidth_GBps"] / a["aggregate_bandwidth_GBps"] == pytest.approx(
        0.5, abs=0.03)
    assert c["aggregate_bandwidth_GBps"] / a["aggregate_bandwidth_GBps"] > 0.95
    assert b["per_active_channel_bandwidth_GBps"] == pytest.approx(
        a["per_active_channel_bandwidth_GBps"], rel=0.04)
    for r in cases.values():
        assert r["aggregate_bandwidth_GBps"] <= r["active_channel_peak_GBps"]
        assert 0 < r["dq_utilization_active_interfaces"] <= 1


def test_actual_overlap_and_complementary_banks(cases):
    b, c = cases["half-lpddr"], cases["half-bank"]
    for channel in b["per_channel"][:2]:
        assert channel["rd_bursts"] == 0
        assert channel["commands"]["MACAB"] == 4096
        assert channel["pim_first_mac_ck"] < b["last_gpu_rd_complete_ck"]
        assert channel["pim_last_complete_ck"] > b["first_gpu_rd_ck"]
    for channel in b["per_channel"][2:]:
        assert channel["rd_bursts"] == 4096
        assert channel["pim_last_complete_ck"] is None
    for channel in c["per_channel"]:
        assert set(channel["rd_bankgroup_counts"]) == {"2", "3"}
        assert channel["commands"]["MACHB"] == 4096
        assert channel["pim_covers_gpu_window"]
        assert channel["rd_during_local_pim"] == channel["rd_bursts"]
        assert 0 < channel["read_queue_gpu_window_avg"] <= 128


def test_pim_tail_does_not_dilute_gpu_bandwidth(tmp_path):
    cfg = Experiment(reads=2048, pim_rounds=1024, pattern="row-hit")
    short = run_case(tmp_path / "short", "half-bank", cfg)
    long = run_case(tmp_path / "long", "half-bank", replace(cfg, pim_rounds=2048))
    assert long["simulation_cycles"] > short["simulation_cycles"]
    assert long["aggregate_bandwidth_GBps"] == short["aggregate_bandwidth_GBps"]
    assert long["gpu_active_cycles"] == short["gpu_active_cycles"]
    assert long["last_pim_complete_ck"] > long["last_gpu_rd_complete_ck"]
    assert long["per_channel"][0]["read_queue_gpu_window_avg"] == (
        short["per_channel"][0]["read_queue_gpu_window_avg"])
    assert long["per_channel"][0]["read_queue_whole_sim_avg"] < (
        short["per_channel"][0]["read_queue_whole_sim_avg"])


def test_stream_mapping_matches_issued_addresses(tmp_path):
    cfg = Experiment(reads=256, pim_rounds=128)
    run_case(tmp_path / "run", "half-bank", cfg)
    with (tmp_path / "run" / "gpu_mapping.csv").open() as stream:
        expected = {(int(e["channel"]), int(e["bankgroup"]), int(e["bank"]),
                     int(e["row"]), int(e["column_tx"])) for e in csv.DictReader(stream)}
    actual = set()
    for channel in range(cfg.channels):
        with (tmp_path / "run" / f"timeline.csv.ch{channel}").open() as stream:
            for e in csv.DictReader(stream):
                if e["command"] == "RD":
                    # RoBaRaCoCh represents Column in transaction indices.
                    actual.add(tuple(int(e[key]) for key in
                                     ("Channel", "BankGroup", "Bank", "Row", "Column")))
    assert actual == expected

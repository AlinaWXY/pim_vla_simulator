"""Work conservation and explicit GPU delivery accounting for the coarse sweep."""

from simulator.hybrid import estimate, make_plan


def test_grid_conserves_full_ffn_work_and_uses_all_pim_banks():
    plan = make_plan()
    assert len(plan["cases"]) == 230
    assert len({case["id"] for case in plan["cases"]}) == 230
    for case in plan["cases"]:
        cfg, layout = case["config"], case["layout"]
        width = cfg["intermediate_size"]
        pim_weights = sum(m["K"] * m["N"] for m in layout["matrices"])
        assert pim_weights + 3 * 1024 * (4096 - width) == 3 * 1024 * 4096
        for matrix in layout["matrices"]:
            assert matrix["outputs_per_bank"] * layout["total_banks"] == matrix["N"]
        assert layout["matrices"][2]["K"] == width
        assert layout["matrices"][2]["N"] == 1024


def test_half_weights_half_bandwidth_preserves_gpu_bound():
    full = estimate(50, 0, 0, 0)
    half = estimate(50, 4, 2048, 100)
    assert full["gpu_weight_bound_ns"] == half["gpu_weight_bound_ns"]
    assert half["pim_output_bytes"] == 204800
    assert half["delivery_write_ns"] == 204800 / 136.5
    assert half["add_traffic_ns"] == 3 * 204800 / 136.5
    assert full["delivery_write_ns"] == full["add_traffic_ns"] == 0


def test_pim_only_delivers_without_partial_sum_add():
    row = estimate(10, 8, 4096, 240000)
    assert row["gpu_weight_bytes"] == 0
    assert row["add_traffic_ns"] == 0
    assert row["scenario_ns"] == 240000 + 40960 / 273


def test_splitting_width_does_not_shrink_output():
    small = estimate(20, 2, 256, 100)
    large = estimate(20, 2, 3840, 100)
    assert small["pim_output_bytes"] == large["pim_output_bytes"] == 81920
    assert small["add_traffic_ns"] == large["add_traffic_ns"]

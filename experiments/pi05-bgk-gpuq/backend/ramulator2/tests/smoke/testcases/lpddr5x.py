import ramulator

CONFIG = dict(
    dram_class="LPDDR5X",
    org_preset="LPDDR5X_8Gb_x16",
    timing_preset="LPDDR5X_8533",
    dram_kwargs={},
    controller_class="LPDDR5",
    fast_ctrl_extra_kwargs=dict(refresh_manager=ramulator.refresh_manager.NoRefresh()),
    frontend_clock_ratio=4,
    stream_cls=8,
)

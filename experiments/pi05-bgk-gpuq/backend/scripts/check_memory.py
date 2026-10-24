"""Resolved timing export and steady-state traffic checks for the selected model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ramulator2"))
import ramulator  # noqa: E402
from tests.controller_scheduling.harness import ControllerUnderTest  # noqa: E402
from tests.latency_throughput.utils.runner import run_single_config_point  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standard", choices=["LPDDR5", "LPDDR5X"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not hasattr(ramulator.dram, args.standard):
        p.error("Selected model is absent from this branch")
    preset = "LPDDR5_6400" if args.standard == "LPDDR5" else "LPDDR5X_8533"
    cls = getattr(ramulator.dram, args.standard)
    model = cls(org_preset=f"{args.standard}_8Gb_x16", timing_preset=preset)
    org, t = model.resolve()
    ceiling = 32 / (t["nBL_min"] * t["tCK_ps"] / 1000)
    cfg = dict(
        name=args.standard,
        dram_class=args.standard,
        org_preset=f"{args.standard}_8Gb_x16",
        timing_preset=preset,
        controller_class="LPDDR5",
        scheduler_class="FRFCFSRowHit",
        controller_kwargs={"wck_sync_mode": "always_on"},
        stagger_stream_rows=True,
        stream_cls=64,
    )
    results = []
    for refresh in (False, True):
        for ratio in (0, 50, 100):
            stats = run_single_config_point(
                cfg,
                nop_counter=1,
                read_ratio=ratio,
                num_probe_requests=0,
                num_streaming_requests=30000,
                streaming_only=True,
                refresh_enabled=refresh,
                frontend_clock_ratio=8,
                warmup_cycles=1000,
            )
            bw = stats["memory_system"]["controller"]["total_throughput_MBps"] / 1000
            assert 0 < bw <= ceiling * 1.001
            if ratio == 100 and not refresh:
                assert bw >= ceiling * 0.95
            results.append(dict(read_percent=ratio, refresh=refresh, GBps=bw, stats=stats))
            print(f"{preset}: read={ratio}%, refresh={refresh}, {bw:.6f} GB/s")
    # A separate completion-aware microtrace, not an admission-only bandwidth window.
    d = ControllerUnderTest(
        ramulator.controller.LPDDR5(
            dram=model,
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
            wck_sync_mode="always_on",
        )
    )
    for i in range(24):
        d.send_request(
            "Read" if i % 3 else "Write",
            d.addr_vec(BankGroup=i % 4, Bank=(i // 4) % 4, Row=i // 16, Column=(i % 4) * 16),
        )
    history = d.run_until_idle(max_ticks=20000)
    assert sum(e.command == "RD" for e in history) == 16
    assert sum(e.command == "WR" for e in history) == 8
    # The inherited controller acknowledges WR on issue; include explicit recovery.
    for _ in range(t["nCWL"] + t["nBL_max"] + t["nWR"]):
        d.tick()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                model=preset,
                organization=org,
                timings=t,
                nominal_GBps=t["rate"] * 16 / 8000,
                quantized_GBps=ceiling,
                scope="x16 BG BL16 CKR4, RL0/WLA, ECC/DBI/DVFS/DFE/ODT off, pre-synchronized WCK",
                streaming_window="admission-limited steady state, not complete kernel latency",
                results=results,
                completed_microtrace_requests=24,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Completion-aware mixed microtrace: 24 requests passed")


if __name__ == "__main__":
    main()

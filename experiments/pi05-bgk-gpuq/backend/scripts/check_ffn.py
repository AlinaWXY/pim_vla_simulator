"""Bounded end-to-end FFN trace regression; run on allocated CPU resources."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from simulator.ffn import FFNConfig, run_ffn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = FFNConfig()
    cases = [
        ("8533-8pkg-2ch-1rank-h1", replace(baseline, horizon=1)),
        ("8533-8pkg-2ch-1rank-h2", replace(baseline, horizon=2)),
        ("8533-8pkg-2ch-1rank-h50", baseline),
        ("8533-8pkg-2ch-2rank-h50", replace(baseline, ranks=2)),
        ("6400-8pkg-2ch-1rank-h1", replace(baseline, model="lpddr5-6400", horizon=1)),
    ]
    summary = []
    for name, cfg in cases:
        result = run_ffn(cfg, args.output / name)
        assert result["logical_mac_operations"] == cfg.horizon * 3 * 1024 * 4096
        assert result["issued_commands"]["MACAB"] == cfg.horizon * 3 * 16384
        assert result["issued_commands"]["WRGB"] == cfg.horizon * cfg.channels * cfg.ranks * 320
        assert result["issued_commands"]["RDRES"] == cfg.horizon * 1152
        assert result["energy_pj"] is None
        if cfg.horizon == 50:
            assert result["issued_commands"]["REFab"] > 0
        summary.append(dict(case=name, **result))
        print(f"{name}: {result['memory_cycles']} CK, {result['latency_ns'] / 1000:.3f} us",
              flush=True)
    assert summary[0]["layout_sha256"] == summary[1]["layout_sha256"] == summary[2]["layout_sha256"]
    assert summary[0]["issued_commands"]["MACAB"] * 50 == summary[2]["issued_commands"]["MACAB"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

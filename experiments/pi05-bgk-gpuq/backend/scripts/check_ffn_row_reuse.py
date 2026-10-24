"""Five-case H=10 row-open reuse regression. Run on approved CPU resources."""

import argparse
import csv
import json
from pathlib import Path

from simulator.ffn import FFNConfig, run_ffn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for reuse in (1, 2, 4, 5, 10):
        # Keep the historical 6-CK spacing to check the original trace layout.
        # The removed proxy completion tail means old latencies no longer apply.
        cfg = FFNConfig(horizon=10, reuse_tokens=reuse, macab_issue_interval_ck=6)
        result = run_ffn(cfg, args.output / f"h10-r{reuse}")
        assert result["pim_issue_interval_ck"] == 6
        assert result["issued_commands"]["MACAB"] == 491520
        assert result["issued_commands"]["WRGB"] == 51200
        assert result["issued_commands"]["RDRES"] == 11520
        assert result["traffic_bytes"]["pim_internal"] == 240 * 1024**2
        assert result["issued_commands"]["REFab"] > 0
        if summary:
            assert result["layout_sha256"] == summary[0]["layout_sha256"]
            assert result["traffic_bytes"] == summary[0]["traffic_bytes"]
        else:
            # Independent legacy run 1543611, before row-reuse support.
            raw = json.loads((args.output / "h10-r1/simulation/result.json").read_text())
            assert raw["requests"]["sha256"] == (
                "9459287fb2d1774fdba7904f47d72298294c7d1cd4eda4efb4d779ec597aad94"
            )
        summary.append(result)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"R={reuse}: {result['latency_ns'] / 1000:.3f} us; "
              f"ACTAB={result['issued_commands']['ACTAB']}; "
              f"PREab={result['issued_commands']['PREab']}; "
              f"MACAB={result['issued_commands']['MACAB']}", flush=True)

    fields = ["reuse_tokens", "weight_scans", "latency_us", "speedup", "ACTAB", "PREab",
              "MACAB", "REFab", "input_kib_per_channel_rank", "accumulator_bytes_per_bank"]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary:
            writer.writerow(dict(
                reuse_tokens=result["config"]["reuse_tokens"],
                weight_scans=result["row_reuse"]["weight_scans"],
                latency_us=result["latency_ns"] / 1000,
                speedup=summary[0]["latency_ns"] / result["latency_ns"],
                **{name: result["issued_commands"][name] for name in fields[4:8]},
                input_kib_per_channel_rank=result["buffers"]["input_bytes_per_channel_rank"] / 1024,
                accumulator_bytes_per_bank=result["buffers"]["accumulator_bytes_per_bank"],
            ))


if __name__ == "__main__":
    main()

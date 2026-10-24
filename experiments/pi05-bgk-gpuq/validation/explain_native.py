"""Small independent diagnostics explaining array cadence and K writeback."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(root):
    directory = root / "k"
    manifest = load(directory / "manifest.json")
    result = load(directory / "simulation/result.json")
    ratio = result["config"]["frontend_ratio"]
    tck = result["tCK_ns"]
    barriers = {int(r["marker"]): int(r["frontend_tick"])
                for r in read(directory / "simulation/barriers.csv")}
    wb = next(s for s in manifest["segments"] if s["kind"] == "local_write")
    lo, hi = barriers[wb["start"]], barriers[wb["end"]]
    per_channel = []
    for channel in range(16):
        native = read(directory / "simulation" / f"timeline.csv.ch{channel}")
        events = read(directory / "simulation" / f"bgk_events.csv.ch{channel}")
        selected = [r for r in native if lo <= (int(r["clock"])-1)*ratio+1 < hi]
        macs = [r for r in events if r["operation"] == "PIM_MAC_K"]
        within_row = Counter(int(b["clock"])-int(a["clock"]) for a, b in zip(macs, macs[1:])
                             if a["Row"] == b["Row"])
        bank = defaultdict(list)
        for r in events:
            if r["operation"] == "PIM_LOCAL_WR":
                bank[(r["Rank"], r["BankGroup"], r["Bank"])].append(r)
        assert all(len({r["Row"] for r in rows}) == 1 for rows in bank.values())
        commands = Counter(r["command"] for r in selected)
        assert not any(commands[c] for c in ("WR", "WRA", "RD", "RDA", "WRGB", "RDRES"))
        per_channel.append(dict(channel=channel, K_MACs=len(macs),
            same_row_MAC_interval_CK_histogram=dict(sorted(within_row.items())),
            writeback_CA_commands=dict(commands), active_write_banks=len(bank),
            bank_write_bytes_histogram=dict(Counter(len(rows)*32 for rows in bank.values())),
            maximum_bank_data_service_ns=max(len(rows) for rows in bank.values())
                * result["resolved_timings_ck"]["nLOCALWR"] * tck,
            local_written_bytes=sum(len(rows)*32 for rows in bank.values())))
    summaries = {p.name: load(p / "summary.json") for p in root.iterdir()
                 if p.is_dir() and (p / "summary.json").exists()}
    output = dict(K_native_writeback_phase_us=(hi-lo)*tck/(ratio*1000),
        K_write_payload_budget_ns=max(r["maximum_bank_data_service_ns"] for r in per_channel),
        channels=per_channel,
        controls=dict(QK_double_us=summaries["qk"]["latency_us"],
            QK_single_us=summaries["qk_serial"]["latency_us"],
            K_double_us=summaries["k"]["latency_us"],
            K_single_us=summaries["k_serial"]["latency_us"],
            K_local_port_4CK_us=summaries["k"]["latency_us"],
            K_local_port_8CK_us=summaries["k_local8"]["latency_us"],
            K_QK_chain_us=summaries["chain_k_qk"]["latency_us"],
            K_QK_separate_sum_us=summaries["k"]["latency_us"]+summaries["qk"]["latency_us"]),
        explanation="Column cadence is a real native array guard, distinct from the 2CK K PE service. Local write phase includes native row setup and recovery; payload lower bound is not added to it.")
    path = root / "native-diagnostics.json"
    path.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k != "channels"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)

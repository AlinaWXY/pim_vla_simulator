"""Completion-aware GPU bandwidth experiments on static GPU/PIM partitions."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CASES = ("full-read", "half-lpddr", "half-bank")


@dataclass(frozen=True)
class Experiment:
    channels: int = 4
    reads: int = 65536
    pim_rounds: int = 16384
    queue_depth: int = 128
    pattern: str = "streaming"
    refresh: bool = False
    max_cycles: int = 2000000

    def __post_init__(self):
        if self.channels < 2 or self.channels & (self.channels - 1):
            raise ValueError("channels must be a power of two >= 2")
        if self.reads <= 0 or self.reads % (self.channels * 16):
            raise ValueError("reads must be positive and divisible by channels * 16")
        if min(self.queue_depth, self.pim_rounds, self.max_cycles) <= 0:
            raise ValueError("queue_depth, pim_rounds and max_cycles must be positive")
        if self.pattern not in ("streaming", "row-hit", "row-conflict"):
            raise ValueError("unknown pattern")


def resolve_model():
    import ramulator

    model = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", rank=1,
    )
    org, timing = model.resolve()
    return model, org, timing


def physical_address(channel, bg, bank, row, column_tx, channels, org, prefetch):
    """Inverse of CacheLineInterleave + RoBaRaCoCh, with one rank."""
    columns = org["column"] // prefetch
    tx_bytes = org["channel_width"] * prefetch // 8
    if not (0 <= row < org["row"] and 0 <= column_tx < columns):
        raise ValueError("workload exceeds DRAM row/column capacity")
    intra = ((row * org["bank"] + bank) * org["bankgroup"] + bg) * columns + column_tx
    return (intra * channels + channel) * tx_bytes


def gpu_coordinates(index, case, cfg, columns, single_bg=False):
    """Map the same logical request IDs onto each scheme's available GPU banks.

    All layouts stripe channel, then BG, then bank, then column, then row.
    Compression into fewer banks preserves the ordered logical byte stream;
    physical addresses necessarily differ between resource partitions.
    """
    active = cfg.channels // 2 if case == "half-lpddr" else cfg.channels
    local, channel = divmod(index, active)
    if case == "half-lpddr":
        channel += cfg.channels // 2
    groups = (2,) if single_bg else (2, 3) if case == "half-bank" else (0, 1, 2, 3)
    local, group_index = divmod(local, len(groups))
    local, bank = divmod(local, 4)
    row, column = divmod(local, columns)
    if cfg.pattern == "row-hit":
        row = 0
    elif cfg.pattern == "row-conflict":
        row, column = local, 0
    return channel, groups[group_index], bank, row, column


def write_inputs(output, case, cfg, org, prefetch, *, no_pim=False, single_bg=False):
    columns = org["column"] // prefetch
    logical_hash = hashlib.sha256()
    with ((output / "input.trace").open("w") as trace,
          (output / "gpu_mapping.csv").open("w") as maps):
        writer = csv.writer(maps)
        writer.writerow(["logical_request", "logical_byte_address", "physical_byte_address",
                         "channel", "bankgroup", "bank", "row", "column_tx"])
        for index in range(cfg.reads):
            coords = gpu_coordinates(index, case, cfg, columns, single_bg)
            address = physical_address(*coords, cfg.channels, org, prefetch)
            logical_address = index * prefetch * org["channel_width"] // 8
            logical_hash.update(f"LD {logical_address}\n".encode())
            trace.write(f"{coords[0]} LD {address}\n")
            writer.writerow([index, logical_address, address, *coords])
        pim_channels = [] if case == "full-read" or no_pim else list(range(
            cfg.channels // 2 if case == "half-lpddr" else cfg.channels
        ))
        op = "PIM_MAC_AB" if case == "half-lpddr" else "PIM_MAC_HB"
        for channel in pim_channels:
            for index in range(cfg.pim_rounds):
                # Identical row-hit computation per participating bank in B/C.
                address = physical_address(channel, 0, 0, 0, index % columns,
                                           cfg.channels, org, prefetch)
                trace.write(f"{cfg.channels + channel} {op} {address}\n")
    banks_per_command = 16 if case == "half-lpddr" else 8
    return {
        "logical_read_sha256": logical_hash.hexdigest(),
        "pim_channels": pim_channels,
        "pim_banks": len(pim_channels) * banks_per_command,
        "pim_commands": len(pim_channels) * cfg.pim_rounds,
        "pim_bank_operations": len(pim_channels) * banks_per_command * cfg.pim_rounds,
    }


def summarize(output, cfg, org, timing, tx_bytes, stats):
    controllers = stats["memory_system"]["controller"]
    if isinstance(controllers, dict):
        controllers = [controllers]
    completions = [[] for _ in controllers]
    pim_completions = [[] for _ in controllers]
    with (output / "completions.csv").open() as stream:
        for item in csv.DictReader(stream):
            event = {key: int(value) for key, value in item.items()}
            dest = completions if event["type"] == 0 else pim_completions
            dest[event["channel"]].append(event)
    channels = []
    tck_ns = timing["tCK_ps"] / 1000
    peak = tx_bytes / (timing["nBL_min"] * tck_ns)
    for channel, ctrl in enumerate(controllers):
        with (output / f"timeline.csv.ch{channel}").open() as stream:
            events = list(csv.DictReader(stream))
        rds = [e for e in events if e["command"] in ("RD", "RDA")]
        macs = [e for e in events if e["command"] in ("MACAB", "MACHB")]
        count = len(rds)
        done = completions[channel]
        if count != len(done) or count != ctrl["num_read_reqs_served"]:
            raise RuntimeError("RD issue/completion/accounting mismatch")
        first = int(rds[0]["clock"]) if count else None
        last = max(e["depart"] for e in done) if count else None
        cycles = last - first if count else 0
        if count:
            if first != ctrl["first_read_issue_ck"] or last != ctrl["last_read_complete_ck"]:
                raise RuntimeError("Controller GPU window disagrees with timeline/callbacks")
            if last != int(rds[-1]["clock"]) + timing["nCL"] + timing["nBL_min"]:
                raise RuntimeError("Unexpected model read completion latency")
            if any(int(b["clock"]) - int(a["clock"]) < timing["nBL_min"]
                   for a, b in zip(rds, rds[1:])):
                raise RuntimeError("DQ bursts overlap")
        dq_cycles = count * timing["nBL_min"]
        pim_first = int(macs[0]["clock"]) if macs else None
        pim_last = max((e["depart"] for e in pim_completions[channel]), default=None)
        channels.append({
            "channel": channel, "rd_bursts": count, "gpu_read_bytes": count * tx_bytes,
            "first_gpu_rd_ck": first, "last_gpu_rd_complete_ck": last,
            "gpu_active_cycles": cycles,
            "bandwidth_GBps": count * tx_bytes / (cycles * tck_ns) if cycles else 0,
            "peak_GBps": peak, "dq_busy_cycles": dq_cycles,
            "dq_utilization": dq_cycles / cycles if cycles else 0,
            "avg_read_latency_ck": sum(e["depart"] - e["arrive"] for e in done) / count
            if count else None,
            "read_queue_gpu_window_avg": ctrl["read_queue_gpu_window_avg"],
            "read_queue_whole_sim_avg": ctrl["read_queue_len_avg"],
            "read_row_hits": ctrl["read_row_hits"],
            "read_row_misses": ctrl["read_row_misses"],
            "read_row_conflicts": ctrl["read_row_conflicts"],
            "rd_bankgroup_counts": dict(Counter(e["BankGroup"] for e in rds)),
            "commands": dict(Counter(e["command"] for e in events)),
            "pim_first_mac_ck": pim_first, "pim_last_complete_ck": pim_last,
            "pim_covers_gpu_window": bool(
                count and macs and pim_first <= first and pim_last >= last),
            "rd_during_local_pim": sum(pim_first <= int(e["clock"]) < pim_last for e in rds)
            if macs else 0,
        })
    active = [c for c in channels if c["rd_bursts"]]
    first = min(c["first_gpu_rd_ck"] for c in active)
    last = max(c["last_gpu_rd_complete_ck"] for c in active)
    cycles = last - first
    byte_count = sum(c["gpu_read_bytes"] for c in active)
    bandwidth = byte_count / (cycles * tck_ns)
    return {
        "per_channel": channels, "active_gpu_channels": len(active),
        "gpu_read_bytes": byte_count, "first_gpu_rd_ck": first,
        "last_gpu_rd_complete_ck": last, "gpu_active_cycles": cycles,
        "aggregate_bandwidth_GBps": bandwidth,
        "per_active_channel_bandwidth_GBps": bandwidth / len(active),
        "active_channel_peak_GBps": peak * len(active),
        "full_system_peak_GBps": peak * cfg.channels,
        "dq_utilization_active_interfaces": sum(c["dq_busy_cycles"] for c in active)
        / (cycles * len(active)),
        "dq_utilization_all_interfaces": sum(c["dq_busy_cycles"] for c in active)
        / (cycles * cfg.channels),
        "avg_read_latency_ck": sum(c["avg_read_latency_ck"] * c["rd_bursts"] for c in active)
        / sum(c["rd_bursts"] for c in active),
        "simulation_cycles": max(c["cycles"] for c in controllers),
        "last_pim_complete_ck": max((c["pim_last_complete_ck"] or 0 for c in channels)),
    }


def run_case(output, case, cfg, *, no_pim=False, single_bg=False):
    import ramulator

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model, org, timing = resolve_model()
    if (org["bankgroup"], org["bank"], org["rank"]) != (4, 4, 1):
        raise ValueError("partition layouts require one rank with 4 BG x 4 banks")
    tx_bytes = model.internal_prefetch_size * org["channel_width"] // 8
    workload = write_inputs(output, case, cfg, org, model.internal_prefetch_size,
                            no_pim=no_pim, single_bg=single_bg)
    controllers = [ramulator.controller.LPDDR5PIM(
        dram=resolve_model()[0], scheduler=ramulator.scheduler.FRFCFSRowHit(),
        refresh_manager=ramulator.refresh_manager.AllBank() if cfg.refresh
        else ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(), addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        wck_sync_mode="always_on", read_buffer_size=cfg.queue_depth,
        controller_plugins=[ramulator.controller_plugin.CmdTraceRecorder(
            path=str(output / "timeline.csv"))],
    ) for _ in range(cfg.channels)]
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=controllers,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    # Raw config also makes the new frontend usable without importing its generated wrapper.
    frontend = {"impl": "PartitionTrace", "clock_ratio": 1,
                "path": str(output / "input.trace"),
                "completions_path": str(output / "completions.csv"),
                "max_ticks": cfg.max_cycles}
    config = {"frontend": frontend, "memory_system": memory.to_config()}
    (output / "ramulator_config.json").write_text(json.dumps(config, indent=2) + "\n")
    sim = ramulator.Simulation(frontend, memory)
    sim.run()
    sim.finalize()
    stats = sim.stats
    expected = cfg.reads + workload["pim_commands"]
    if stats["frontend"]["requests_completed"] != expected:
        raise RuntimeError("Not all requests completed")
    result = {
        "case": case, "config": asdict(cfg), "no_pim": no_pim, "single_bg": single_bg,
        "organization": org, "timings": timing, "transaction_bytes": tx_bytes,
        "internal_prefetch": model.internal_prefetch_size,
        "nominal_channel_peak_GBps": timing["rate"] * org["channel_width"] / 8000,
        "workload": workload, "raw_stats": stats,
        **summarize(output, cfg, org, timing, tx_bytes, stats),
    }
    if result["gpu_read_bytes"] != cfg.reads * tx_bytes:
        raise RuntimeError("GPU byte count changed")
    actual_mac = sum(c["commands"].get("MACAB", 0) + c["commands"].get("MACHB", 0)
                     for c in result["per_channel"])
    if actual_mac != workload["pim_commands"]:
        raise RuntimeError("PIM command count mismatch")
    result["pim_internal_bytes"] = workload["pim_bank_operations"] * tx_bytes
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{output.name}: {result['aggregate_bandwidth_GBps']:.4f} GB/s aggregate, "
          f"{result['per_active_channel_bandwidth_GBps']:.4f} GB/s / active channel", flush=True)
    return result


def write_report(output, results):
    baseline = results[0]["aggregate_bandwidth_GBps"]
    rows, channel_rows = [], []
    for result in results:
        row = {key: value for key, value in result.items()
               if not isinstance(value, (dict, list))}
        row["normalized_bandwidth"] = result["aggregate_bandwidth_GBps"] / baseline
        rows.append(row)
        channel_rows.extend({"case": result["label"], **c} for c in result["per_channel"])
    for name, data in (("summary.csv", rows), ("per_channel.csv", channel_rows)):
        with (output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "schema_version": 1,
        "definitions": {
            "bandwidth": "bytes / ((last RD completion - first RD issue) * tCK); decimal GB/s",
            "aggregate": "global min(first RD issue) to global max(last RD completion)",
            "dq_utilization": "RD bursts * nBL_min / active cycles; no PIM bytes included",
            "read_latency": "controller admission to callback depart; excludes source backpressure",
            "queue": "read_buffer only; right-endpoint tick samples on (first RD, last completion]",
            "injection": "one independent ordered GPU stream/channel and PIM stream/channel; "
            "one attempt/stream/CK; source-local backpressure; all begin at CK0",
            "address_fairness": "identical ordered logical byte addresses and striping rule; "
            "physical addresses remap to available banks; see gpu_mapping.csv",
            "pim": "same rounds per PIM bank; row-hit internal MAC only, no WRGB/RDRES",
        },
        "summary": rows, "results": results,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = rows[:3]
    labels = ["Full-read", "Half-LPDDR", "Half-bank"]
    colors = ["#416D9C", "#CC7750", "#3E937D"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in zip(axes, ("aggregate_bandwidth_GBps", "normalized_bandwidth"),
                             ("Aggregate GPU bandwidth (GB/s)", "Normalized to Full-read")):
        bars = ax.bar(labels, [row[key] for row in main], color=colors, width=0.62)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_ylim(0, max(row[key] for row in main) * 1.22)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(rows[0]["full_system_peak_GBps"], ls="--", color="gray", lw=1)
    axes[1].axhline(1, ls="--", color="gray", lw=1)
    cfg = results[0]["config"]
    fig.suptitle(f"LPDDR5-6400 | {cfg['channels']} x16 channels | {cfg['pattern']} reads | "
                 f"refresh {'on' if cfg['refresh'] else 'off'}")
    fig.savefig(output / "bandwidth.png", dpi=180)
    fig.savefig(output / "bandwidth.svg")
    plt.close(fig)


def snapshot_sources(output):
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    (output / "working-tree.patch").write_bytes(diff)
    paths = subprocess.check_output(
        ["git", "ls-files", "--modified", "--others", "--exclude-standard"], cwd=repo, text=True,
    ).splitlines()
    hashes = {}
    for path in sorted(set(paths)):
        source = repo / path
        if source.is_file():
            target = output / "source_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "provenance.json").write_text(json.dumps({
        "commit": revision, "dirty": bool(paths), "source_sha256": hashes,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--reads", type=int, default=65536)
    parser.add_argument("--pim-rounds", type=int, default=16384)
    parser.add_argument("--queue-depth", type=int, default=128)
    parser.add_argument("--pattern", choices=("streaming", "row-hit", "row-conflict"),
                        default="streaming")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    cfg = Experiment(channels=args.channels, reads=args.reads, pim_rounds=args.pim_rounds,
                     queue_depth=args.queue_depth, pattern=args.pattern, refresh=args.refresh)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []

    def run(label, case, config=cfg, **kwargs):
        result = run_case(args.output / label, case, config, **kwargs)
        result["label"] = label
        results.append(result)

    for case in CASES:
        run(case, case)
    b, c = results[1:3]
    if b["workload"]["pim_bank_operations"] != c["workload"]["pim_bank_operations"]:
        raise RuntimeError("Unequal PIM work")
    if len({r["workload"]["logical_read_sha256"] for r in results}) != 1:
        raise RuntimeError("Logical GPU stream changed")
    if args.diagnostics:
        run("half-bank-no-pim", "half-bank", no_pim=True)
        run("half-bank-q8", "half-bank", replace(cfg, queue_depth=8))
        run("half-bank-q512", "half-bank", replace(cfg, queue_depth=512))
        run("half-bank-one-bg", "half-bank", single_bg=True)
        run("full-read-row-hit", "full-read", replace(cfg, pattern="row-hit"))
        run("half-bank-row-hit", "half-bank", replace(cfg, pattern="row-hit"))
    write_report(args.output, results)
    snapshot_sources(args.output)


if __name__ == "__main__":
    main()

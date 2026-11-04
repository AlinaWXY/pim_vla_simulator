"""Isolated native guards for bounded QKT score tiles and final DRAM export."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "source")]

CASES = {
    "Score_complete": None,
    "Score_disabled": "PIM_SCORE_SPILL requires pim_score_dram_spill",
    "Score_missing_tile_spill": "Score tile overwritten before complete local spill",
    "Score_spill_before_compute": "Score spill tile order or complete MAC coverage mismatch",
    "Score_read_before_complete_store": "Score export before complete DRAM store",
    "Score_duplicate_read": "Duplicate score DRAM read",
    "Score_missing_final_read": "Incomplete QKT score spill/export at end of trace",
    "Score_unaligned_descriptor": "Score spill requires canonical rank1 tile descriptor",
}


def address(offset, channel=0, bank=0):
    row, within = divmod(offset, 2048)
    return row << 20 | (bank % 4) << 18 | (bank // 4) << 16 | 1 << 15 | (within // 32) << 9 | channel << 5


def is_channel(line, opcode, channel=0):
    parts = line.split()
    return len(parts) > 1 and parts[0] == opcode and (int(parts[1]) >> 5 & 15) == channel


def make_trace(path, name):
    import workload
    manifest = workload.generate(path, "chain", rank_staging=True, staging_banks=2,
        rank_descriptors=True, channel_independent_dispatch=True,
        v_token_sharding=True, pv_token_sharding=True, qkv_fusion=True,
        score_dram_spill=True)
    (path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = path.read_text(encoding="ascii").splitlines()
    spill = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_SCORE_SPILL"))
    if name == "Score_missing_tile_spill":
        del lines[spill]
    elif name == "Score_spill_before_compute":
        moved = lines.pop(spill)
        index = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_MAC_QK"))
        lines.insert(index, moved)
    elif name == "Score_read_before_complete_store":
        moved = lines.pop(next(i for i, line in enumerate(lines) if is_channel(line, "LD")))
        # Insert after the first tile's completion barrier, before another MAC.
        index = next(i for i in range(spill, len(lines)) if lines[i].startswith("BARRIER "))
        lines.insert(index + 1, moved)
    elif name == "Score_duplicate_read":
        index = next(i for i, line in enumerate(lines) if is_channel(line, "LD"))
        lines.insert(index + 1, lines[index])
    elif name == "Score_missing_final_read":
        del lines[max(i for i, line in enumerate(lines) if is_channel(line, "LD"))]
    elif name == "Score_unaligned_descriptor":
        lines[spill] = f"PIM_SCORE_SPILL {address(655392)}"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{k: int(v) if v.lstrip("-").isdigit() else v for k, v in row.items()}
            for row in csv.DictReader(stream)]


def positive_audit(simulation):
    """Reconstruct tile lifetimes and match stores to real native commands."""
    for channel in range(16):
        events = read_csv(simulation / f"score_spill.csv.ch{channel}")
        native = read_csv(simulation / f"timeline.csv.ch{channel}")
        stores = [e for e in events if e["operation"] == "LOCALWR"]
        expected = Counter((tile, packet, packet % 2, 0,
            (655360 + tile * 512 + packet // 2 * 32) // 2048,
            (655360 + tile * 512 + packet // 2 * 32) % 2048 // 32)
            for tile in range(31) for packet in range(32))
        actual = Counter((e["tile"], e["packet"], e["BankGroup"], e["Bank"], e["Row"], e["Column"]) for e in stores)
        assert actual == expected, "Native score address coverage mismatch"
        native_stores = Counter((e["clock"], e["BankGroup"], e["Bank"], e["Row"], e["Column"])
            for e in native if e["command"] == "LOCALWR")
        recorded_stores = Counter((e["clock"], e["BankGroup"], e["Bank"], e["Row"], e["Column"]) for e in stores)
        assert not recorded_stores - native_stores, "Score store has no actual LOCALWR"
        previous_done = -1
        for tile in range(31):
            macs = [e for e in events if e["operation"] == "QK_MAC" and e["tile"] == tile]
            starts = [e for e in events if e["operation"] == "SPILL_BEGIN" and e["tile"] == tile]
            ends = [e for e in events if e["operation"] == "TILE_DONE" and e["tile"] == tile]
            writes = [e for e in stores if e["tile"] == tile]
            transfers = [e for e in events if e["operation"] == "TRANSFER" and e["tile"] == tile]
            assert len(macs) == 16 and len(starts) == len(ends) == 1 and len(transfers) == 32
            assert min(e["clock"] for e in macs) >= previous_done
            assert starts[0]["clock"] >= max(e["ready_at"] for e in macs)
            assert ends[0]["clock"] >= max(e["ready_at"] for e in writes)
            by_packet = {e["packet"]: e for e in transfers}
            assert all(e["clock"] >= by_packet[e["packet"]]["ready_at"] for e in writes)
            previous_done = ends[0]["ready_at"]
        accepted = [e for e in events if e["operation"] == "READ_ACCEPT"]
        completed = [e for e in events if e["operation"] == "READ_COMPLETE"]
        assert len(accepted) == len(completed) == 992
        assert min(e["clock"] for e in accepted) >= previous_done
        first_mac = min(e["clock"] for e in events if e["operation"] == "QK_MAC")
        external = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}
        assert not any(e["command"] in external and first_mac <= e["clock"] < previous_done for e in native)
    return dict(channels=16, tiles_per_channel=31, native_local_writes=15872,
        internal_phase_external_commands=0, score_read_bytes=507904)


def worker(backend, output, name):
    from reproduce import setup, config_for, dump
    setup(backend)
    from simulator.backend import run_trace
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "guard.trace"
    make_trace(trace, name)
    config = config_for("chain", refresh=CASES[name] is None, max_cycles=1_000_000,
        rank_staging=True, staging_banks=2, rank_descriptors=True,
        channel_independent_dispatch=True, v_token_sharding=True,
        pv_token_sharding=True, qkv_fusion=True, score_dram_spill=name != "Score_disabled")
    try:
        result = run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = CASES[name] is not None and isinstance(error, RuntimeError) and CASES[name] in str(error)
        evidence = dict(case=name, passed=passed, expected_message=CASES[name],
            exception_type=type(error).__name__, message=str(error))
    else:
        evidence = dict(case=name, passed=CASES[name] is None, memory_cycles=result["memory_cycles"],
            message="Native lifetime checks completed" if CASES[name] is None else "Invalid trace accepted")
        if evidence["passed"]:
            try:
                evidence["native_audit"] = positive_audit(output / "simulation")
            except Exception as error:
                evidence.update(passed=False, exception_type=type(error).__name__, message=str(error))
    dump(output / "case-result.json", evidence)
    print(json.dumps(evidence), flush=True)
    return 0 if evidence["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if args.case:
        return worker(backend, output, args.case)
    from reproduce import setup, dump
    env = setup(backend)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in CASES:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--backend", str(backend),
            "--output", str(output / name), "--case", name], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        (output / (name + ".log")).write_text(run.stdout, encoding="utf-8")
        path = output / name / "case-result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(
            case=name, passed=False, message="Native worker did not produce evidence")
        result["worker_returncode"] = run.returncode
        result["passed"] = result["passed"] and run.returncode == 0
        results.append(result)
    summary = dict(passed=all(r["passed"] for r in results), case_count=len(results),
        subprocess_isolation=True, native_cases=results)
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

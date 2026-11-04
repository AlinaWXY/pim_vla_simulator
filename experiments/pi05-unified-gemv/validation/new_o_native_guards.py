"""Native-process guards for BF16 PV exports and the local PV-to-Wo chain."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "source")]

CASES = {
    "PV_BF16_complete": None,
    "O_local_complete": None,
    "PV_spill_missing_producer": "PV spill requires complete PV accumulators",
    "PV_spill_missing_packet": "staging seal requires every declared input write",
    "PV_spill_duplicate_packet": "Duplicate PV local input packet",
    "PV_spill_wrong_destination": "staging requires rank1 BG0/BG1 bank0",
    "O_input_without_seal": "staging load requires sealed input",
    "O_rewind_before_export": "Output owner change before complete result drain",
    "O_duplicate_rewind": "O rewind requires complete first tile",
    "O_missing_rewind": "duplicate staging load",
    "O_missing_final_export": "Incomplete local PV to O projection chain",
    "O_wrong_weight_column": "Unified GEMV MAC column order or operator switch before release",
    "PV_BF16_excess_export": "Output export address/count mismatch",
}


def address(offset=0, channel=0, bank=0):
    row, within = divmod(offset, 2048)
    return row << 20 | (bank % 4) << 18 | (bank // 4) << 16 | 1 << 15 | (within // 32) << 9 | channel << 5


def is_channel(line, opcode, channel=0):
    parts = line.split()
    return len(parts) > 1 and parts[0] == opcode and (int(parts[1]) >> 5 & 15) == channel


def offset_of(line):
    value = int(line.split()[1])
    return (value >> 20) * 2048 + (value >> 9 & 63) * 32


def make_trace(path, name):
    import workload
    use_o = not name.startswith("PV_BF16")
    if name == "PV_spill_missing_producer":
        lines = [f"PIM_STAGE_BEGIN {address(524288)} 8192 16", f"PIM_PV_SPILL {address(524288)}"]
    else:
        workload.generate(path, "pv", rank_staging=True, staging_banks=2,
            rank_descriptors=True, channel_independent_dispatch=True,
            v_token_sharding=True, pv_token_sharding=True,
            o_projection=use_o, pv_bf16_output=True)
        lines = path.read_text(encoding="ascii").splitlines()
        if name == "PV_spill_missing_packet":
            del lines[next(i for i, line in enumerate(lines) if is_channel(line, "PIM_PV_SPILL"))]
        elif name == "PV_spill_duplicate_packet":
            index = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_PV_SPILL"))
            lines.insert(index + 1, lines[index])
        elif name == "PV_spill_wrong_destination":
            index = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_PV_SPILL"))
            lines[index] = f"PIM_PV_SPILL {address(524288, bank=1)}"
        elif name == "O_input_without_seal":
            first = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_PV_SPILL"))
            index = next(i for i in range(first, len(lines)) if is_channel(lines[i], "PIM_STAGE_SEAL"))
            del lines[index]
        elif name == "O_rewind_before_export":
            index = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_RD_RES"))
            lines.insert(index, f"PIM_O_REWIND {address(524288)}")
        elif name == "O_duplicate_rewind":
            index = next(i for i, line in enumerate(lines) if is_channel(line, "PIM_O_REWIND"))
            lines.insert(index + 1, lines[index])
        elif name == "O_missing_rewind":
            del lines[next(i for i, line in enumerate(lines) if is_channel(line, "PIM_O_REWIND"))]
        elif name == "O_missing_final_export":
            index = max(i for i, line in enumerate(lines) if is_channel(line, "PIM_RD_RES"))
            del lines[index]
        elif name == "O_wrong_weight_column":
            index = next(i for i, line in enumerate(lines)
                if is_channel(line, "PIM_MAC_AB") and offset_of(line) >= 786432)
            lines[index] = f"PIM_MAC_AB {address(786464)} 0"
        elif name == "PV_BF16_excess_export":
            index = max(i for i, line in enumerate(lines) if is_channel(line, "PIM_RD_RES"))
            lines.insert(index + 1, f"PIM_RD_RES {address(8192)}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return use_o


def worker(backend, output, name):
    from reproduce import setup, config_for, dump
    setup(backend)
    from simulator.backend import run_trace
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "guard.trace"
    use_o = make_trace(trace, name)
    config = config_for("pv", refresh=CASES[name] is None, max_cycles=1_000_000,
        rank_staging=True, staging_banks=2, rank_descriptors=True,
        channel_independent_dispatch=True, v_token_sharding=True,
        pv_token_sharding=True, o_projection=use_o, pv_bf16_output=True)
    try:
        result = run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = CASES[name] is not None and isinstance(error, RuntimeError) and CASES[name] in str(error)
        evidence = dict(case=name, passed=passed, expected_message=CASES[name],
            exception_type=type(error).__name__, message=str(error))
    else:
        evidence = dict(case=name, passed=CASES[name] is None,
            memory_cycles=result["memory_cycles"],
            message="Native lifetime/capacity checks completed" if CASES[name] is None else "Invalid trace accepted")
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

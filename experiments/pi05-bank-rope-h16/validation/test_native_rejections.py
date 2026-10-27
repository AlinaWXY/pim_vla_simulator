"""Exercise invalid dependency/ownership traces against the compiled backend.

Each case runs in a fresh Python process because the native simulator owns
process-wide component registrations. No manifest is used as an oracle.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def physical_address(channel, rank, bg, bank, row, column):
    return (row << 20) | (bank << 18) | (bg << 16) | (rank << 15) | (column << 9) | (channel << 5)


WORK_Q = physical_address(0, 1, 0, 0, 40, 8)
CASES = {
    "rank0_pim": (
        "PIM_WR_GB 0x0 0\n",
        "rank0 is reserved for GPU",
    ),
    "remote_without_peer_export": (
        f"PIM_Q_REMOTE 0x{WORK_Q:x} 0\n",
        "Q_REMOTE has no preceding peer Q_EXPORT",
    ),
    "rope_execute_without_input": (
        f"PIM_ROPE_EXEC 0x{WORK_Q:x} 0\n",
        "RoPE execute needs matching loaded 32B input",
    ),
    "export_without_rope_store": (
        f"PIM_Q_EXPORT 0x{WORK_Q:x}\n",
        "Q export precedes RoPE store",
    ),
}


def worker(backend, output, name):
    sys.path[:0] = [str(backend / "ramulator2/python"), str(backend)]
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig

    output.mkdir(parents=True, exist_ok=False)
    trace = output / "invalid.trace"
    trace.write_text(CASES[name][0], encoding="ascii")
    config = MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, pim=True, refresh=False,
        pim_input_broadcast=False, pim_mac_scope="rank", pim_data_path="attacc",
        pim_rank_rope=True, frontend_ratio=32, max_cycles=2_000,
        rope_ii_cycles=1, rope_latency_cycles=4, pim_cast_cycles=1, rank_reduce_cycles=4,
        timing_overrides={"nCCDAB": 8, "nCCDK": 2, "nCCDQK": 16, "nLOCALWR": 4},
    )
    try:
        run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = isinstance(error, RuntimeError) and CASES[name][1] in str(error)
        result = {
            "case": name, "passed": passed, "compiled_backend_invoked": True,
            "exception_type": type(error).__name__, "message": str(error),
            "expected_exception_type": "RuntimeError", "expected_message": CASES[name][1],
        }
    else:
        result = {"case": name, "passed": False, "compiled_backend_invoked": True,
                  "message": "The invalid trace completed without rejection"}
    (output / "case-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend, output = args.backend.resolve(strict=True), args.output.resolve()
    if args.case:
        return worker(backend, output, args.case)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(backend / "ramulator2/python"), str(backend), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(backend / "ramulator2"), env.get("LD_LIBRARY_PATH", "")])
    env["PYTHONNOUSERSITE"] = "1"
    results = []
    for name in CASES:
        command = [sys.executable, str(Path(__file__).resolve()), "--backend", str(backend),
                   "--output", str(output / name), "--case", name]
        run = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60)
        (output / f"{name}.log").write_text(run.stdout, encoding="utf-8")
        result_path = output / name / "case-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "case": name, "passed": False, "message": "Worker did not produce native rejection evidence"}
        result["worker_returncode"] = run.returncode
        result["passed"] = bool(result.get("passed")) and run.returncode == 0
        results.append(result)
    summary = {"passed": all(case["passed"] for case in results), "cases": results,
               "case_count": len(results), "subprocess_isolation": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

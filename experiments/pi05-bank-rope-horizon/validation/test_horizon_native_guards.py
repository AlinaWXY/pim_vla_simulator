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


CASES = {"missing_query39": "Rank QK requires all",
         "missing_z_token_block": "Down MAC requires all16 source banks"}


def make_trace(path, name):
    if name == "missing_z_token_block":
        lines=[]
        for block in range(3):
            for bank in range(16):
                if (block,bank)==(2,15):
                    continue
                addr=physical_address(0,1,bank//4,bank%4,176,block)
                lines.append(f"PIM_Z_LOAD {addr} 0")
        lines.append(f"PIM_MAC_AB {physical_address(0,1,0,0,160,0)} 0")
        path.write_text("\n".join(lines)+"\n",encoding="ascii")
        return
    # One real whole-H chain, with exactly the high query bit39 missing in
    # bank0 of channel0. The paired channel supplies odd queries to channel0.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
    import horizon_attention as attention
    attention.generate(path,horizon=40)
    missing=physical_address(0,1,0,0,40,39)
    lines=path.read_text(encoding="ascii").splitlines()
    target=f"PIM_Q_REMOTE {missing} 0"
    if lines.count(target)!=1:
        raise AssertionError("Guard must remove exactly one query39 remote transfer")
    path.write_text("\n".join(line for line in lines if line!=target)+"\n",encoding="ascii")


def worker(backend, output, name):
    sys.path[:0] = [str(backend / "ramulator2/python"), str(backend)]
    from simulator.backend import run_trace
    from simulator.config import MemoryConfig

    output.mkdir(parents=True, exist_ok=False)
    trace = output / "invalid.trace"
    make_trace(trace,name)
    config = MemoryConfig(
        model="lpddr5x-8533", channels=16, ranks=2, pim=True, refresh=False,
        pim_input_broadcast=False, pim_mac_scope="rank", pim_data_path="attacc",
        pim_rank_rope=name=="missing_query39", pim_horizon=40,
        pim_ffn_dram=name=="missing_z_token_block", ffn_z_preinstalled=name=="missing_z_token_block",
        frontend_ratio=32, max_cycles=200_000 if name=="missing_query39" else 5_000,
        rope_ii_cycles=1, rope_latency_cycles=4, pim_cast_cycles=1, rank_reduce_cycles=4,
        timing_overrides={"nCCDAB": 20 if name=="missing_query39" else 40, "nCCDK": 2, "nCCDQK": 40, "nLOCALWR": 4},
    )
    try:
        run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = isinstance(error, RuntimeError) and CASES[name] in str(error)
        result = {
            "case": name, "passed": passed, "compiled_backend_invoked": True,
            "exception_type": type(error).__name__, "message": str(error),
            "expected_exception_type": "RuntimeError", "expected_message": CASES[name],
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
                             stderr=subprocess.STDOUT, timeout=180)
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

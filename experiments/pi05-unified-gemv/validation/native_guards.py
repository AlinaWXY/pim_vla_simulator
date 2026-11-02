"""Exercise shared-slot hazards in isolated compiled-backend processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "source"), str(HERE / "validation")]

CASES = {
    "missing_token": "requires every token and lane",
    "duplicate_token": "overwrite before full fragment consumption",
    "overwrite_incomplete_fragment": "overwrite before full fragment consumption",
    "release_incomplete_fragment": "release of unconsumed input fragment",
    "operator_switch_without_release": "Unified GEMV MAC column order or operator switch before release",
    "Down_missing_GPU_token": "requires every token and lane",
    "Down_wrong_weight_column": "column order or operator switch",
    "Down_release_incomplete_fragment": "release of unconsumed input fragment",
    "Down_overwrite_incomplete_fragment": "overwrite before full fragment consumption",
    "Q_held_ordinary_write": "Q slot held until PIM_GEMV_RELEASE",
    "Q_fill_without_projection_release": "Q fill requires explicit release",
    "QKT_missing_query": "Unified QKT missing query/lane coverage",
    "Q_release_before_scan": "Q release requires completed 496-token K scan",
    "QKT_then_V_reuses_same_slot": None,
    "Down_GPU_input_complete": None,
}


def address(offset=0, ch=0, bank=0):
    row, column = divmod(offset, 2048)
    return row << 20 | (bank % 4) << 18 | (bank // 4) << 16 | 1 << 15 | (column // 32) << 9 | ch << 5


def command(op, offset=0, slot=0, ch=0, bank=0):
    return f"{op} {address(offset, ch, bank)} {slot}"


def trace_for(path, name):
    if name.startswith("Q"):
        import workload
        manifest = workload.generate(path, "chain")
        lines = path.read_text(encoding="ascii").splitlines()
        if name in ("Q_held_ordinary_write", "Q_release_before_scan"):
            segment = next(s for s in manifest["segments"] if s["name"] == "q_remote_buffer_fill")
            lines = lines[:segment["line_end"]]
            lines.append(command("PIM_WR_GB" if name == "Q_held_ordinary_write" else "PIM_GEMV_RELEASE"))
        elif name == "Q_fill_without_projection_release":
            segment = next(s for s in manifest["segments"] if s["name"] == "projection_gemv_release")
            lines = [line for n, line in enumerate(lines, 1)
                     if not (segment["line_start"] <= n <= segment["line_end"] and line.startswith("PIM_GEMV_RELEASE"))]
        elif name == "QKT_missing_query":
            target = command("PIM_Q_REMOTE", 81920+8*32)
            if lines.count(target) != 1:
                raise AssertionError("Query deletion must identify exactly one bank/query")
            lines.remove(target)
        elif name == "QKT_then_V_reuses_same_slot":
            lines += [command("PIM_WR_GB", t*32) for t in range(16)]
            lines += [command("PIM_MAC_AB", 83968+s*32) for s in range(8)]
            lines.append(command("PIM_GEMV_RELEASE"))
        else:
            raise ValueError(name)
    elif name.startswith("Down_"):
        tokens = 15 if name == "Down_missing_GPU_token" else 16
        lines = [command("PIM_WR_GB", t*32) for t in range(tokens)]
        if name == "Down_wrong_weight_column":
            lines.append(command("PIM_MAC_AB", 327680+32))
        else:
            columns = 4 if name == "Down_GPU_input_complete" else 1
            lines += [command("PIM_MAC_AB", 327680+s*32) for s in range(columns)]
            if name in ("Down_GPU_input_complete", "Down_release_incomplete_fragment"):
                lines.append(command("PIM_GEMV_RELEASE"))
            elif name == "Down_overwrite_incomplete_fragment":
                lines.append(command("PIM_WR_GB"))
    else:
        tokens = 15 if name == "missing_token" else 16
        lines = [command("PIM_WR_GB", t*32) for t in range(tokens)]
        if name == "duplicate_token":
            lines.append(command("PIM_WR_GB"))
        else:
            lines.append(command("PIM_MAC_AB", 83968))
            if name == "overwrite_incomplete_fragment":
                lines.append(command("PIM_WR_GB"))
            elif name == "release_incomplete_fragment":
                lines.append(command("PIM_GEMV_RELEASE"))
            elif name == "operator_switch_without_release":
                lines.append(command("PIM_MAC_AB", 100352+32))
    path.write_text("\n".join(lines)+"\n", encoding="ascii")


def positive_audit(directory, name):
    from audit_native import read_csv, read_json, audit_slot_events
    events = [e for ch in range(16)
              for e in read_csv(directory / "simulation" / f"unified_gemv.csv.ch{ch}")]
    if name == "Down_GPU_input_complete":
        result = audit_slot_events(events, "down_gpu_input")
        if result["operation_counts"].get("GEMV_MULTICAST") != 16 or result["operation_counts"].get("GEMV_MAC") != 4:
            raise AssertionError("Positive Down control did not consume sixteen GPU rows and four weight columns")
        from audit_native import audit_external_fills
        native = [e for ch in range(16)
                  for e in read_csv(directory / "simulation" / f"timeline.csv.ch{ch}")]
        result["external_fill_checks"] = audit_external_fills(native, events)
        return result
    # The full QK trace releases all banks; channel0 then writes one V fragment
    # into the exact same physical slot. Split only at its native Q release.
    q_releases, last_qk = {}, {}
    for event in events:
        if event["operation"] == "GEMV_QK":
            last_qk[event["Channel"]] = max(last_qk.get(event["Channel"], 0), event["ready_at"])
    for event in events:
        ch = event["Channel"]
        if event["operation"] == "GEMV_RELEASE" and event["buffer"] == 0 and event["clock"] >= last_qk.get(ch, float("inf")):
            q_releases[ch] = min(q_releases.get(ch, event["clock"]), event["clock"])
    if len(q_releases) != 16:
        raise AssertionError("Cross-operator control lacks Q ownership-release evidence")
    old = [e for e in events if e["clock"] <= q_releases[e["Channel"]]]
    new = [e for e in events if e["clock"] > q_releases[e["Channel"]]]
    before = audit_slot_events(old, "chain")
    after = audit_slot_events(new, "v")
    if set(e["buffer"] for e in new) != {0} or len([e for e in new if e["operation"] == "GEMV_MAC"]) != 8:
        raise AssertionError("V did not reuse Q's physical slot0")
    q_generation = {e["Channel"]: e["generation"] for e in old if e["operation"] == "GEMV_QK"}
    if any(e["generation"] != q_generation[e["Channel"]]+1 for e in new):
        raise AssertionError("V slot state was reset instead of continuing the shared Q generation")
    native = [e for ch in range(16)
              for e in read_csv(directory / "simulation" / f"timeline.csv.ch{ch}")]
    refresh = [e for e in native if e["command"].startswith("REF")]
    config = read_json(directory / "simulation/result.json")["config"]
    if not config["refresh"] or {e["Channel"] for e in refresh} != set(range(16)):
        raise AssertionError("QKT/V same-simulation control lacks enabled and observed native refresh")
    return dict(QKT=before, V=after, same_bank_slot_reused=True,
        generation_continues_without_reset=True, single_native_simulation=True,
        refresh_enabled=True, native_refresh_commands=len(refresh))


def worker(backend, output, name):
    from reproduce import setup, config_for, dump
    setup(backend)
    from simulator.backend import run_trace
    output.mkdir(parents=True)
    trace = output / "guard.trace"
    trace_for(trace, name)
    is_q = name.startswith("Q")
    is_down = name.startswith("Down_")
    config = config_for("chain" if is_q else "down_gpu_input" if is_down else "v",
                        refresh=CASES[name] is None, max_cycles=500_000)
    try:
        native = run_trace(config, trace, output / "simulation")
    except Exception as error:
        passed = CASES[name] is not None and isinstance(error, RuntimeError) and CASES[name] in str(error)
        result = dict(case=name, passed=passed, compiled_backend_invoked=True,
            expected_message=CASES[name], exception_type=type(error).__name__, message=str(error))
    else:
        if CASES[name] is not None:
            result = dict(case=name, passed=False, compiled_backend_invoked=True,
                          message="Invalid dependency trace was accepted")
        else:
            try:
                audit = positive_audit(output, name)
                result = dict(case=name, passed=True, compiled_backend_invoked=True,
                              memory_cycles=native["memory_cycles"], native_audit=audit)
            except Exception as error:
                result = dict(case=name, passed=False, compiled_backend_invoked=True,
                              exception_type=type(error).__name__, message=str(error))
    dump(output / "case-result.json", result)
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
    from reproduce import setup, dump
    env = setup(backend)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in CASES:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--backend", str(backend),
            "--output", str(output / name), "--case", name], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        (output / f"{name}.log").write_text(run.stdout, encoding="utf-8")
        evidence = output / name / "case-result.json"
        result = json.loads(evidence.read_text(encoding="utf-8")) if evidence.is_file() else dict(
            case=name, passed=False, message="Worker did not produce native evidence")
        result["worker_returncode"] = run.returncode
        result["passed"] = bool(result["passed"]) and run.returncode == 0
        results.append(result)
    summary = dict(passed=all(r["passed"] for r in results), case_count=len(results),
        subprocess_isolation=True, native_cases=results)
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

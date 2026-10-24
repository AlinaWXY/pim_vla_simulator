"""Coarse GPU/PIM FFN sweep; GPU costs are bandwidth bounds, not measurements."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .ffn import FFNConfig, build_layout, run_ffn


def make_plan():
    cases = []
    for h in (10, 20, 30, 40, 50):
        for packages in (1, 2, 4, 8):
            widths = (4096,) if packages == 8 else range(256, 4096, 256)
            for width in widths:
                cfg = FFNConfig(packages=packages, horizon=h, intermediate_size=width)
                cases.append({
                    "id": f"h{h}-p{packages}-f{width}",
                    "config": asdict(cfg),
                    "layout": build_layout(cfg),
                })
    return {
        "schema_version": 1,
        "total_packages": 8,
        "full_intermediate_size": 4096,
        "gpu_total_bandwidth_gbps": 273.0,
        "cases": cases,
        "scope": "coarse candidates only; PIM package counts 3/5/6/7 unsupported",
        "assumptions": [
            "ordinary MACAB, reuse_tokens=1, refresh enabled, one rank/channel",
            "PIM shared-logic GELU-product compute is excluded",
            "GPU reads its three BF16 weight shards once per FFN, no cache credit",
            "GPU bandwidth scales linearly with its package count",
            "GPU arithmetic, activation traffic and launch overhead are excluded",
            "PIM RDRES already included; do not count that export twice",
            "delivery/add modeled after both branches finish, avoiding GPU DQ overlap",
            "shared logic writes FP32 PIM output to GPU-side LPDDR",
            "GPU add reads two FP32 partials and writes one FP32 final output",
            "delivery and add use GPU-side peak bandwidth; sync/compute cost excluded",
            "PIM input WRGB is included; upstream input distribution is excluded",
            "synthetic independent channels, not a measured Thor partition",
        ],
    }


def estimate(h, pim_packages, pim_width, pim_ns, bandwidth_gbps=273.0):
    """Return an explicit idealized cost scenario, never an end-to-end measurement."""
    if not (0 <= pim_packages <= 8 and 0 <= pim_width <= 4096):
        raise ValueError("Invalid split")
    if (pim_packages == 0) != (pim_width == 0):
        raise ValueError("GPU-only requires zero PIM width/packages")
    if (pim_packages == 8) != (pim_width == 4096):
        raise ValueError("PIM-only requires full PIM width/packages")
    if h <= 0 or bandwidth_gbps <= 0 or pim_ns < 0:
        raise ValueError("Invalid horizon, bandwidth, or latency")
    gpu_packages = 8 - pim_packages
    gpu_width = 4096 - pim_width
    weight_bytes = 3 * 1024 * gpu_width * 2
    result_bytes = h * 1024 * 4
    # Decimal GB/s equals bytes/ns. PIM-only hands the result to a subsequent
    # GPU phase with all packages available after PIM has completed.
    available_gbps = bandwidth_gbps * (gpu_packages / 8 if gpu_packages else 1)
    gpu_ns = weight_bytes / available_gbps
    delivery_bytes = result_bytes if pim_packages else 0
    add_bytes = 3 * result_bytes if pim_packages and gpu_packages else 0
    delivery_ns = delivery_bytes / available_gbps
    add_ns = add_bytes / available_gbps
    return {
        "horizon": h, "pim_packages": pim_packages, "gpu_packages": gpu_packages,
        "pim_width": pim_width, "gpu_width": gpu_width,
        "pim_fraction": pim_width / 4096,
        "gpu_weight_bytes": weight_bytes, "pim_output_bytes": delivery_bytes,
        "gpu_bandwidth_gbps": available_gbps,
        "pim_ns": pim_ns, "gpu_weight_bound_ns": gpu_ns,
        "branch_max_ns": max(pim_ns, gpu_ns),
        "delivery_write_ns": delivery_ns, "add_traffic_ns": add_ns,
        "scenario_ns": max(pim_ns, gpu_ns) + delivery_ns + add_ns,
        "bottleneck": "PIM" if pim_ns >= gpu_ns else "GPU weight bandwidth",
    }


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def summarize(root):
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    missing = []
    for case in plan["cases"]:
        path = root / "cases" / case["id"] / "result.json"
        if not path.exists():
            missing.append(case["id"])
            continue
        result = json.loads(path.read_text())
        if result["config"] != case["config"]:
            raise ValueError(f"Configuration mismatch: {path}")
        cfg = case["config"]
        row = estimate(cfg["horizon"], cfg["packages"], cfg["intermediate_size"],
                       result["latency_ns"], plan["gpu_total_bandwidth_gbps"])
        row["id"] = case["id"]
        row["macab_commands"] = result["issued_commands"]["MACAB"]
        rows.append(row)
    for h in sorted({c["config"]["horizon"] for c in plan["cases"]}):
        row = estimate(h, 0, 0, 0, plan["gpu_total_bandwidth_gbps"])
        row.update(id=f"h{h}-gpu-only", macab_commands=0)
        rows.append(row)
    best = []
    best_hybrid = []
    for h in sorted({r["horizon"] for r in rows}):
        group = [r for r in rows if r["horizon"] == h]
        best.append(min(group, key=lambda r: r["scenario_ns"]))
        hybrids = [r for r in group if 0 < r["pim_packages"] < 8]
        if hybrids:
            best_hybrid.append(min(hybrids, key=lambda r: r["scenario_ns"]))
    report = {
        "complete": not missing, "missing_cases": missing,
        "scope": plan["scope"], "assumptions": plan["assumptions"],
        "best_available_candidates": best, "rows": rows,
        "best_available_hybrids": best_hybrid,
        "warning": "Idealized scenario; neither measured Thor performance nor global optimum",
    }
    write_json(root / "summary.json", report)
    with (root / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "plan":
        root.mkdir(parents=True, exist_ok=False)
        plan = make_plan()
        write_json(root / "plan.json", plan)
        print(json.dumps({"cases": len(plan["cases"]), "root": str(root)}))
    elif args.action == "run":
        plan = json.loads((root / "plan.json").read_text())
        if args.index is None or not 0 <= args.index < len(plan["cases"]):
            parser.error("run requires a valid --index")
        case = plan["cases"][args.index]
        result = run_ffn(FFNConfig(**case["config"]), root / "cases" / case["id"])
        print(json.dumps({"id": case["id"], "latency_ns": result["latency_ns"]}))
    else:
        report = summarize(root)
        print(json.dumps({"complete": report["complete"],
                          "missing_count": len(report["missing_cases"]),
                          "best": report["best_available_candidates"]}, indent=2))


if __name__ == "__main__":
    main()

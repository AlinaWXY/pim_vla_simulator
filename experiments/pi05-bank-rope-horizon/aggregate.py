"""Summarize four fresh native runs; never substitutes scaled H16 timings."""
import argparse
import csv
import json
from pathlib import Path


def dump(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def aggregate(root):
    cases = ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram")
    rows, detail, provenance = [], {}, []
    for h in (16, 20, 30, 40):
        directory = root / f"H{h}"
        summary = json.loads((directory / "summary.json").read_text())
        channel = json.loads((directory / "channel-window-summary.json").read_text())
        assert summary["status"] == "passed" and summary["horizon"] == h
        assert all(summary["runs"][c]["audit_status"] == "passed" for c in (*cases, "combined_timing_chain"))
        assert abs(channel["zero_external_command_phase_us"] - summary["zero_external_command_phase_us"]) < 1e-9
        provenance.append(json.loads((directory / "provenance.json").read_text()))
        row = dict(H=h, padded_KV=((968+h+31)//32)*32,
            measured_stage_sum_us=summary["measured_stage_sum_us_per_layer_step"],
            measured_stage_sum_ms_18x10=summary["measured_stage_sum_ms_18x10"],
            zero_external_command_phase_us=summary["zero_external_command_phase_us"],
            zero_external_command_phase_fraction=summary["zero_external_command_phase_fraction"],
            external_DQ_KiB=summary["external_DQ_bytes_per_layer_step"]/1024,
            Q_bank_buffer_bytes=32*h, V_FFN_rank_input_double_bytes=64*h,
            QK_projection_accumulator_bytes_per_bank=64*h,
            V_PV_accumulator_bytes_per_bank=32*h,
            GU_accumulator_bytes_per_bank=128*h, Down_accumulator_bytes_per_bank=64*h,
            Z_DRAM_bytes_per_bank=512*((h+15)//16), Z_DRAM_reserved_per_bank=2048,
            attention_work_rows=summary["layouts"]["attention"]["regions"]["work"]["reserved_bytes_per_bank"]//2048)
        for c in cases:
            row[f"{c}_us"] = summary["runs"][c]["latency_us"]
            row[f"{c}_quiet_phase_us"] = channel["stages"][c]["zero_external_command_phase_us"]
        rows.append(row)
        detail[str(h)] = summary
    assert len({p["native_library_sha256"] for p in provenance}) == 1, "Mixed native builds"
    assert abs(rows[0]["measured_stage_sum_us"] - 124.474476) < 1e-9
    assert abs(rows[0]["zero_external_command_phase_us"] - 37.9815253125) < 1e-9
    summary = dict(status="passed", policy="Whole horizon; buffers scale; one batch; heads remain8",
        baseline_commit="41199cee64c40c55cafb330ed838d7e5c13fc0b8",
        native_library_sha256=provenance[0]["native_library_sha256"], rows=rows, runs=detail,
        actual_GPU_usable_time_us=None, complete_AE_end_to_end_latency_ms=None,
        quiet_phase_definition="Only whole semantic phases containing zero external transfer commands on every channel; exclude all gaps inside input/output phases. Boundary guards/CA/rank-switch costs not deducted.")
    dump(root / "summary.json", summary)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Whole-horizon AE sweep: H=16,20,30,40", "",
        "All entries are fresh native Ramulator runs with the same compiled backend. H is action-token count; the number of attention heads remains8. Token buffers and accumulators expand with H, and all H tokens execute as one batch.", "",
        "H16 reproduces the frozen xinyao_1027_h16 baseline exactly. Timings preserve the earlier sum of separately measured stages; nonlinear/GPU arithmetic and VLM concurrency remain unmeasured.", "",
        "| H | Padded KV | Stage sum/us | x18x10/ms | Zero-external-command phases/us | Fraction | External DQ/KiB |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['H']} | {r['padded_KV']} | {r['measured_stage_sum_us']:.6f} | {r['measured_stage_sum_ms_18x10']:.6f} | {r['zero_external_command_phase_us']:.6f} | {100*r['zero_external_command_phase_fraction']:.2f}% | {r['external_DQ_KiB']:g} |")
    lines += ["", "## Timing by stage", "", "| H | Q/K chain | V | PV | Gate/Up + Z write | Down |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| " + str(r["H"]) + " | " + " | ".join(f"{r[c+'_us']:.6f}" for c in cases) + " |")
    lines += ["", "All stage times are microseconds per layer and denoising step.", "",
        "## Complete phases without external transfer commands", "",
        "| H | Q/K local spill, RoPE and QKT compute | Z local store | Down local input/compute | Total/us |", "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['H']} | {r['chain_quiet_phase_us']:.6f} | {r['gate_up_z_store_quiet_phase_us']:.6f} | {r['down_tiled_dram_quiet_phase_us']:.6f} | {r['zero_external_command_phase_us']:.6f} |")
    lines += ["", "The phase segmentation is unchanged. All short and long gaps inside input/output phases are excluded. This is a candidate-window statistic before data tails, read/write turnaround, rank-switch, CA and GPU row constraints; it is not measured usable GPU bandwidth. No rank0 background workload is supplied, and the AttAcc PIM transfer model retains its prior limitations.", "",
        "## Layout and buffer changes", "",
        "| H | Q buffer/B per bank | V/FFN input double buffer/B per rank | GU accumulator/B per bank | Down accumulator/B per bank | Work rows/bank | Z DRAM/B per bank |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['H']} | {r['Q_bank_buffer_bytes']} | {r['V_FFN_rank_input_double_bytes']} | {r['GU_accumulator_bytes_per_bank']} | {r['Down_accumulator_bytes_per_bank']} | {r['attention_work_rows']} | {r['Z_DRAM_bytes_per_bank']} |")
    lines += ["", "- Weight ownership and copies do not change: Q/K retain two token-owner copies per head; V/FFN weights remain single-copy and feature-sharded.",
        "- Q/K production assigns H/2 tokens to each peer rank, keeps RoPE pairs in their original banks, and exchanges only the missing half of Q. Each enlarged bank query buffer holds all H queries, so QKT still scans each local K column once.",
        "- The work area stores H/2 coefficient, H/2 Q, H/2 suffix-K columns, followed by K padding. H40 needs a second reserved row only for the final four padding-K columns; coefficient/Q/K RoPE data still fit in the first row. V regions shift accordingly; FFN weight bases stay fixed.",
        "- Z uses [feature_slot16][ceil(H/16) token blocks][token_lane16] BF16. The last column has padded lanes; padded lanes are discarded when filling the H-token Down buffer. It remains within the existing2KiB DRAM reservation per bank.",
        "- The32B bank staging and streamed128B+32B nonlinear staging remain bounded; no full-Z rank SRAM is introduced. Nonlinear arithmetic is still excluded.",
        "- Projection/QK MAC service scales to H/2 and H cycles respectively; V/PV/FFN MAC service is H cycles. RoPE pair II/latency, local-port throughput, bank topology, frequency and refresh policy stay fixed.", "",
        "## Reproduction and evidence", "",
        "Each H directory contains workload traces, per-channel native timelines, completion events, independent layout/dataflow audits, source provenance and the channel-window analysis. Combined FFN cases check write-before-read dependencies and are not added to the five-stage sum. See README.md for build/run commands."]
    (root / "REPORT.md").write_bytes(("\n".join(lines) + "\n").encode())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = aggregate(args.root.resolve(strict=True))
    print(json.dumps(result["rows"], indent=2))

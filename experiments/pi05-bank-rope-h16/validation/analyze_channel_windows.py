"""Read-only analysis of PIM traffic; does not infer GPU schedulable bandwidth.

Payload-equivalent spare time is not a reconstructed physical DQ waveform.
Zero-transfer phases are candidates before boundary, CA and GPU row constraints.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

EXTERNAL = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}


def analyze(root):
    ae = json.loads((root / "summary.json").read_text())
    phases, stages, hashes = [], {}, {}
    for case in ae["primary_cases"]:
        directory = root / case
        paths = [directory / "manifest.json", directory / "simulation/result.json",
                 directory / "simulation/barriers.csv"]
        manifest = json.loads(paths[0].read_text())
        result = json.loads(paths[1].read_text())
        with paths[2].open(newline="") as stream:
            markers = {int(r["marker"]): int(r["frontend_tick"]) for r in csv.DictReader(stream)}
        ratio = result["config"]["frontend_ratio"]
        ck_us = result["tCK_ns"] / 1000
        assert result["config"]["model"] == "lpddr5x-8533"
        per_channel = []
        timelines = []
        total_counts = Counter()
        for channel in range(result["config"]["channels"]):
            path = directory / f"simulation/timeline.csv.ch{channel}"
            paths.append(path)
            with path.open(newline="") as stream:
                external = [r for r in csv.DictReader(stream) if r["command"] in EXTERNAL]
            timelines.append(external)
            counts = Counter(r["command"] for r in external)
            total_counts.update(counts)
            gaps = Counter(int(b["clock"]) - int(a["clock"]) for a, b in zip(external, external[1:]))
            per_channel.append(dict(channel=channel, commands=dict(counts),
                first_external_issue_CK=int(external[0]["clock"]),
                last_external_issue_CK=int(external[-1]["clock"]),
                inter_command_interval_histogram_CK=dict(sorted(gaps.items())),
                payload_equivalent_us=len(external) * 2 * ck_us))
        assert dict(total_counts) == {c: n for c, n in result["issued_commands"].items() if c in EXTERNAL and n}
        assert sum(total_counts.values()) * 32 == manifest["expected_dq_bytes"]["total"]
        # This run is balanced; do not assume that for arbitrary workloads.
        assert all(c["commands"] == per_channel[0]["commands"] for c in per_channel)
        no_transfer_us = 0
        classified = [0] * len(timelines)
        for segment in manifest["segments"]:
            begin, end = markers[segment["start"]], markers[segment["end"]]
            counts = [sum(begin <= (int(r["clock"]) - 1) * ratio + 1 < end for r in timeline)
                      for timeline in timelines]
            classified = [a + b for a, b in zip(classified, counts)]
            duration = (end - begin) / ratio * ck_us
            no_external = not any(counts)
            no_transfer_us += duration if no_external else 0
            phases.append(dict(case=case, phase=segment["name"],
                begin_us=begin / ratio * ck_us, end_us=end / ratio * ck_us,
                duration_us=duration, external_command_counts_per_channel=counts,
                zero_external_command_phase=no_external))
        assert classified == [len(t) for t in timelines]
        payload = per_channel[0]["payload_equivalent_us"]
        stages[case] = dict(stage_us=result["latency_ns"] / 1000,
            payload_equivalent_us_per_channel=payload,
            payload_equivalent_spare_us_per_channel=result["latency_ns"] / 1000 - payload,
            zero_external_command_phase_us=no_transfer_us, channels=per_channel)
        hashes.update({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    total_us = sum(s["stage_us"] for s in stages.values())
    payload_us = sum(s["payload_equivalent_us_per_channel"] for s in stages.values())
    return dict(source=str(root), units="per channel, per layer, per step",
        stage_sum_us=total_us, payload_equivalent_us=payload_us,
        payload_equivalent_spare_us=total_us-payload_us,
        payload_equivalent_spare_fraction=1-payload_us/total_us,
        zero_external_command_phase_us=sum(s["zero_external_command_phase_us"] for s in stages.values()),
        actual_GPU_usable_time_us=None,
        limitations=["AttAcc WRGB/RDRES do not retain full LPDDR direction/rank turnaround timing",
                     "No concurrent rank0 GPU workload; full-bank PIM/ordinary admission is serialized",
                     "Zero-command phases need boundary data tails, rank switching, CA and GPU row timing checked",
                     "Five independently timed stages; no continuous full-AE overlap claim"],
        transfer_assumption="32B per command at nBL_min=2 CK, tCK=0.938ns; not a physical DQ interval trace",
        stages=stages, phases=phases, source_sha256=hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run.resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output / "phases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case", "phase", "begin_us", "end_us", "duration_us", "zero_external_command_phase"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["phases"])
    print(json.dumps({k: v for k, v in result.items() if k not in ("stages", "phases", "source_sha256")}, indent=2))

"""Compare token-sharded V projection with the 23e6614 feature-sharded run.

Reads complete native results, independently rebuilds all DQ/CA partitions,
and checks the intentional V traffic/geometry changes separately from the
unchanged workloads. Only the five primary stages contribute to the total.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import compare_act2_deadline as common
from compare_localwr_act2 import audit_protocol, load_module

CHANGED = {"v", "serial_layer_control"}
INPUT_SAVING_BYTES = 16 * 16 * 1024


def compare(before, after, analyzer, analyzer_path):
    require, pair = common.require, common.pair
    root = Path(__file__).resolve().parent
    auditor_path = root / "experiments/pi05-unified-gemv/validation/act2_audit.py"
    auditor = load_module("v_token_act_auditor", auditor_path)
    protocol = {mode: audit_protocol(data, analyzer, auditor, common.CASES, strict=True)
                for mode, data in (("before", before), ("after", after))}
    result = dict(status="passed", title="V projection token sharding; V cache and PV retain feature sharding",
        baseline_revision="23e661478f136f55f11855d23529643cb060b378",
        units="microseconds per layer/denoising step; DQ/CA time per channel; bytes",
        primary_cases=list(common.PRIMARY), control_cases=list(common.CONTROLS),
        aggregation="Five independent primary stages only; controls are excluded from the sum",
        sources={mode: dict(directory=str(data["root"]), summary_sha256=common.digest(data["root"] / "summary.json"),
            trace_sha256=data["traces"], accounting_source_sha256={case: stage["source_sha256"] for case, stage in data["stages"].items()})
            for mode, data in (("before", before), ("after", after))},
        analyzer=dict(path=str(analyzer_path), sha256=common.digest(analyzer_path)),
        protocol_auditor=dict(path=str(auditor_path), sha256=common.digest(auditor_path)),
        act_protocol=protocol, checks=dict(all_16_channel_partitions_close=True, full_ACT_protocol_passed=True),
        limitations=["GPU cast, packing and reordering arithmetic remain unmeasured; existing transfer phases are timed.",
            "DQ/CA idle is a modeled resource window, not measured concurrent Thor service.",
            "Per-phase differences include row-state, refresh and scheduling changes."], cases={})
    require(after["summary"].get("v_token_sharding") is True, "After run must use token-sharded V")
    require(not before["summary"].get("v_token_sharding", False), "Before run must use feature-sharded V")
    for case in common.CASES:
        a, b = (data["summary"]["runs"][case] for data in (before, after))
        an, bn = (data["natives"][case] for data in (before, after))
        old_manifest = common.read_json(before["root"] / case / "manifest.json")
        manifest = common.read_json(after["root"] / case / "manifest.json")
        require(old_manifest["buffers"]["gemv_slot_bytes"] == manifest["buffers"]["gemv_slot_bytes"] == 512, f"{case}: SRF size")
        require(old_manifest["buffers"]["gemv_slots_per_bank"] == manifest["buffers"]["gemv_slots_per_bank"] == 2, f"{case}: SRF slots")
        timings_a, timings_b = dict(an["resolved_timings_ck"]), dict(bn["resolved_timings_ck"])
        allowed_timing = {}
        if case == "v":
            require(timings_a.pop("nCCDAB") == 16 and timings_b.pop("nCCDAB") == 8,
                    "V MAC interval must match 16-to-8 local token change")
            allowed_timing = dict(nCCDAB=pair(16, 8))
        require(timings_a == timings_b and an["tCK_ns"] == bn["tCK_ns"], f"{case}: unrelated DRAM timing changed")
        config_a, config_b = deepcopy(an["config"]), deepcopy(bn["config"])
        if case == "v":
            require(config_a["timing_overrides"].pop("nCCDAB") == 16
                    and config_b["timing_overrides"].pop("nCCDAB") == 8, "V configured MAC interval")
        require(config_a == config_b, f"{case}: unrelated native configuration changed")
        expected_saving = INPUT_SAVING_BYTES if case in CHANGED else 0
        require(a["external_DQ_bytes"] - b["external_DQ_bytes"] == expected_saving,
                f"{case}: expected only 256KiB V input DQ saving")
        if case in CHANGED:
            for key, delta in (("gpu_input_staging_write", -INPUT_SAVING_BYTES),
                               ("input_staging_array_read", -INPUT_SAVING_BYTES),
                               ("pim_internal", 4*1024*1024)):
                require(b["traffic_bytes"][key]-a["traffic_bytes"][key] == delta,
                        f"{case}: unexpected {key} change")
            command_deltas = {"MACAB": 8192, "LOCALRD": -8192, "WR": -8192,
                              "RD": 0, "RDRES": 0, "WRGB": 0, "LOCALWR": 0}
            for command, delta in command_deltas.items():
                require(bn["issued_commands"].get(command, 0)-an["issued_commands"].get(command, 0) == delta,
                        f"{case}: unexpected {command} count change")
            prefix = "v/" if case == "serial_layer_control" else ""
            old_phases = {p["phase"]: p for p in before["stages"][case]["phases"]}
            new_phases = {p["phase"]: p for p in after["stages"][case]["phases"]}
            for name in ("v_result_export", "v_prefix_tail_read", "v_suffix_write"):
                require([c["external_commands"] for c in old_phases[prefix+name]["channels"]]
                        == [c["external_commands"] for c in new_phases[prefix+name]["channels"]],
                        f"{case}/{name}: output/cache command counts changed")
        if case not in CHANGED:
            require(before["traces"][case] == after["traces"][case], f"{case}: unrelated trace changed")
            require(a["memory_cycles"] == b["memory_cycles"] and a["traffic_bytes"] == b["traffic_bytes"],
                    f"{case}: unrelated native behavior changed")
            old_audit = common.read_json(before["root"] / case / "native-audit.json")
            new_audit = common.read_json(after["root"] / case / "native-audit.json")
            require(old_audit["native_sha256"] == new_audit["native_sha256"], f"{case}: unrelated raw command logs changed")
        require(set(a["phase_us"]) == set(b["phase_us"]), f"{case}: phase contract changed")
        phases = {name: pair(a["phase_us"][name], b["phase_us"][name]) for name in a["phase_us"]}
        phases["phase_boundary_overhead"] = pair(a["phase_boundary_us"], b["phase_boundary_us"])
        common.close(sum(v["after_minus_before"] for v in phases.values()), b["latency_us"]-a["latency_us"], f"{case}: phase closure")
        entry = dict(scope="primary_stage" if case in common.PRIMARY else "control_only",
            latency_us=pair(a["latency_us"], b["latency_us"]),
            memory_cycles=pair(a["memory_cycles"], b["memory_cycles"]),
            external_DQ_bytes=pair(a["external_DQ_bytes"], b["external_DQ_bytes"]),
            saved_us=a["latency_us"]-b["latency_us"],
            saved_percent=100*(a["latency_us"]-b["latency_us"])/a["latency_us"],
            changed_compute_timing=allowed_timing, phases=phases, channels=[], phase_kinds=[],
            native_command_counts={command: pair(an["issued_commands"].get(command, 0), bn["issued_commands"].get(command, 0))
                for command in sorted(set(an["issued_commands"]) | set(bn["issued_commands"]))})
        for channel in range(16):
            left, right = (data["stages"][case]["channels"][channel] for data in (before, after))
            require(left["external_DQ_bytes"]-right["external_DQ_bytes"] == expected_saving//16, f"{case}/ch{channel}: input bytes")
            lm, rm = common.channel_metrics(left), common.channel_metrics(right)
            require(set(lm) == set(rm), f"{case}: resource metrics")
            entry["channels"].append(dict(channel=channel, metrics={key: pair(lm[key], rm[key]) for key in lm}))
            lk, rk = left["phase_kind_breakdown"], right["phase_kind_breakdown"]
            require(set(lk) == set(rk), f"{case}: phase kinds")
            for kind in lk:
                entry["phase_kinds"].append(dict(channel=channel, kind=kind,
                    metrics={key: pair(lk[kind][key], rk[kind][key]) for key in common.KIND_FIELDS}))
        result["cases"][case] = entry
    total = dict(scope="primary_sum", included_cases=list(common.PRIMARY), channels=[])
    for metric in ("latency_us", "external_DQ_bytes"):
        total[metric] = pair(*(sum(result["cases"][case][metric][mode] for case in common.PRIMARY) for mode in ("before", "after")))
    total["saved_us"] = -total["latency_us"]["after_minus_before"]
    total["saved_percent"] = total["saved_us"]*100/total["latency_us"]["before"]
    for channel in range(16):
        fields = result["cases"]["chain"]["channels"][channel]["metrics"]
        metrics = {key: pair(*(sum(result["cases"][case]["channels"][channel]["metrics"][key][mode]
                    for case in common.PRIMARY) for mode in ("before", "after"))) for key in fields}
        for mode in ("before", "after"):
            common.validate_partition({key: value[mode] for key, value in metrics.items()}, f"total/ch{channel}/{mode}")
        total["channels"].append(dict(channel=channel, metrics=metrics))
    result["primary_five_stage_sum"] = total
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyzer, analyzer_path = common.load_analyzer()
    result = compare(common.load_run(args.before, analyzer), common.load_run(args.after, analyzer), analyzer, analyzer_path)
    output = args.output or args.after / "v-token-comparison"
    common.write_report(result, output)
    (output / "act2-deadline-comparison.json").replace(output / "v-token-comparison.json")
    print(json.dumps(dict(status="passed", primary_latency_us=result["primary_five_stage_sum"]["latency_us"],
        output=str(output.resolve())), indent=2))


if __name__ == "__main__":
    main()

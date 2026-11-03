"""Compare local V cache writes and token-sharded PV against 810e01e.

Rebuild every DQ/CA partition from complete native timelines, audit strict
ACT1/ACT2 ownership/deadlines, and isolate intended traffic/config changes.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import compare_act2_deadline as common
from compare_localwr_act2 import audit_protocol, load_module

BASELINE = "810e01e8422dcf811fcfaa15c75f39bfd82e30f7"
FUSED_PRIMARY = ("chain", "pv", "gate_up_gpu", "down_gpu_input")
CHANGED = {"chain", "v", "pv", "serial_layer_control"}
SAVINGS = dict(chain=0, v=327680, pv=122880, serial_layer_control=712704)
COMMAND_DELTAS = {
    "chain": dict(MACAB=16384, LOCALRD=4096, LOCALWR=4096, WR=0, RD=0, RDRES=0, WRGB=0),
    "v": dict(MACAB=0, LOCALRD=4096, LOCALWR=4096, WR=-4096, RD=-2048, RDRES=-4096, WRGB=0),
    "pv": dict(MACAB=0, LOCALRD=-7936, LOCALWR=0, WR=-7936, RD=0, RDRES=4096, WRGB=0),
    "serial_layer_control": dict(MACAB=0, LOCALRD=-12032, LOCALWR=4096, WR=-20224, RD=-2048, RDRES=0, WRGB=0),
}


def compare(before, after, analyzer, analyzer_path):
    require, pair = common.require, common.pair
    auditor_path = Path(__file__).resolve().parent / "experiments/pi05-unified-gemv/validation/act2_audit.py"
    auditor = load_module("pv_token_act_auditor", auditor_path)
    protocol = {mode: audit_protocol(data, analyzer, auditor, common.CASES, strict=True)
                for mode, data in (("before", before), ("after", after))}
    require(before["summary"]["v_token_sharding"] and after["summary"]["v_token_sharding"], "V projection token sharding retained")
    require(not before["summary"].get("pv_token_sharding", False) and after["summary"].get("pv_token_sharding"), "PV comparison modes")
    require(not before["summary"].get("qkv_fusion", False) and after["summary"].get("qkv_fusion"), "Fused QKV comparison modes")
    result = dict(status="passed", title="Fused QKV, local V writeback and KV-token-sharded PV with GPU partial-sum reduction",
        baseline_revision=BASELINE, units="microseconds per layer/step; DQ/CA time per channel; bytes",
        primary_cases=dict(before=list(common.PRIMARY), after=list(FUSED_PRIMARY)), control_cases=list(common.CONTROLS),
        standalone_v_after_is_control_only=True,
        aggregation="Before: five independent stages. After: four stages, with V included in the fused chain. Standalone V after is diagnostic only.",
        sources={mode: dict(directory=str(data["root"]), summary_sha256=common.digest(data["root"] / "summary.json"),
            trace_sha256=data["traces"], accounting_source_sha256={case: stage["source_sha256"] for case, stage in data["stages"].items()})
            for mode, data in (("before", before), ("after", after))},
        analyzer=dict(path=str(analyzer_path), sha256=common.digest(analyzer_path)),
        protocol_auditor=dict(path=str(auditor_path), sha256=common.digest(auditor_path)),
        act_protocol=protocol, checks=dict(all_16_channel_partitions_close=True, full_ACT_protocol_passed=True),
        limitations=["GPU summation of PV partial results is unmeasured; both result transfers are timed.",
            "V local cast, gather/RMW and store follow explicitly modeled costs; no measured silicon or tensor inference.",
            "DQ/CA idle windows do not measure concurrent Thor service.",
            "Prefix cache is preinstalled in the selected layout; initial cache conversion is outside each step."], cases={})
    for case in common.CASES:
        a, b = (data["summary"]["runs"][case] for data in (before, after))
        an, bn = (data["natives"][case] for data in (before, after))
        ma, mb = (common.read_json(data["root"] / case / "manifest.json") for data in (before, after))
        require(ma["buffers"]["gemv_total_bytes_per_bank"] == mb["buffers"]["gemv_total_bytes_per_bank"] == 1024,
                f"{case}: physical SRF capacity")
        ta, tb = dict(an["resolved_timings_ck"]), dict(bn["resolved_timings_ck"])
        allowed = {}
        require(ta == tb and an["tCK_ns"] == bn["tCK_ns"], f"{case}: unrelated DRAM timings")
        ac, bc = deepcopy(an["config"]), deepcopy(bn["config"])
        for key, expected in (("pim_token_pv", case in CHANGED), ("v_cache_preinstalled", case == "pv"),
                              ("pim_whole_srf", True), ("pim_qkv_fusion", case in ("chain", "serial_layer_control"))):
            require(not ac.pop(key, False) and bc.pop(key, False) == expected, f"{case}: {key} contract")
        if case in ("v", "pv"):
            # The old path omitted its unused cast setting from result.json.
            # Local V writeback now exposes the actual one-cycle caster.
            require("pim_cast_cycles" not in ac and bc.pop("pim_cast_cycles") == 1,
                    f"{case}: new local-cast parameter must be explicitly recorded")
        require(ac == bc, f"{case}: unrelated configuration changed")
        saving = SAVINGS.get(case, 0)
        require(a["external_DQ_bytes"]-b["external_DQ_bytes"] == saving, f"{case}: DQ byte accounting")
        if case in CHANGED:
            for command, delta in COMMAND_DELTAS[case].items():
                require(bn["issued_commands"].get(command, 0)-an["issued_commands"].get(command, 0) == delta,
                        f"{case}: {command} count")
        else:
            require(before["traces"][case] == after["traces"][case], f"{case}: unaffected trace")
            require(a["traffic_bytes"] == b["traffic_bytes"], f"{case}: unchanged tensor/data traffic")
            for command in ("MACAB", "MACQK", "LOCALRD", "LOCALWR", "WR", "WRA", "RD", "RDA", "RDRES", "WRGB"):
                require(an["issued_commands"].get(command, 0) == bn["issued_commands"].get(command, 0),
                        f"{case}: unchanged data/MAC command count for {command}")
            aa, ab = (common.read_json(data["root"] / case / "native-audit.json") for data in (before, after))
            # Preserve tensor work while auditing whole-SRF data ownership.
            # Legal PRE/ACT may overlap the previous batch's PE tail; only
            # the actual next input read waits for complete batch consumption.
            # Any scheduling/refresh effects must come from measured timelines.
            require(ab["input_batching"]["full_batch_consumption_before_any_view_overwrite"],
                    f"{case}: whole-SRF ownership proof missing")
        phases = {name: pair(a["phase_us"].get(name, 0), b["phase_us"].get(name, 0))
                  for name in sorted(set(a["phase_us"]) | set(b["phase_us"]))}
        phases["phase_boundary_overhead"] = pair(a["phase_boundary_us"], b["phase_boundary_us"])
        for mode, run in (("before", a), ("after", b)):
            common.close(sum(p[mode] for p in phases.values()), run["latency_us"], f"{case}/{mode}: phase closure")
        entry = dict(scope="before_primary_after_diagnostic" if case == "v" else "primary_stage" if case in FUSED_PRIMARY else "control_only",
            latency_us=pair(a["latency_us"], b["latency_us"]), memory_cycles=pair(a["memory_cycles"], b["memory_cycles"]),
            external_DQ_bytes=pair(a["external_DQ_bytes"], b["external_DQ_bytes"]), saved_us=a["latency_us"]-b["latency_us"],
            saved_percent=100*(a["latency_us"]-b["latency_us"])/a["latency_us"], changed_compute_timing=allowed,
            phases=phases, channels=[], phase_kinds=[],
            native_command_counts={key: pair(an["issued_commands"].get(key, 0), bn["issued_commands"].get(key, 0))
                for key in sorted(set(an["issued_commands"]) | set(bn["issued_commands"]))})
        for channel in range(16):
            left, right = (data["stages"][case]["channels"][channel] for data in (before, after))
            require(left["external_DQ_bytes"]-right["external_DQ_bytes"] == saving//16, f"{case}/ch{channel}: DQ bytes")
            lm, rm = common.channel_metrics(left), common.channel_metrics(right)
            require(set(lm) == set(rm), f"{case}: resource fields")
            entry["channels"].append(dict(channel=channel, metrics={key: pair(lm[key], rm[key]) for key in lm}))
            lk, rk = left["phase_kind_breakdown"], right["phase_kind_breakdown"]
            for kind in sorted(set(lk) | set(rk)):
                entry["phase_kinds"].append(dict(channel=channel, kind=kind, metrics={
                    key: pair(lk.get(kind, {}).get(key, 0), rk.get(kind, {}).get(key, 0)) for key in common.KIND_FIELDS}))
        result["cases"][case] = entry
    total = dict(scope="primary_sum", included_cases=dict(before=list(common.PRIMARY), after=list(FUSED_PRIMARY)), channels=[])
    for metric in ("latency_us", "external_DQ_bytes"):
        total[metric] = pair(*(sum(result["cases"][case][metric][mode] for case in selected)
            for mode, selected in (("before", common.PRIMARY), ("after", FUSED_PRIMARY))))
    total["saved_us"] = -total["latency_us"]["after_minus_before"]
    total["saved_percent"] = total["saved_us"]*100/total["latency_us"]["before"]
    for channel in range(16):
        fields = result["cases"]["chain"]["channels"][channel]["metrics"]
        metrics = {key: pair(*(sum(result["cases"][case]["channels"][channel]["metrics"][key][mode]
            for case in selected) for mode, selected in (("before", common.PRIMARY), ("after", FUSED_PRIMARY)))) for key in fields}
        for mode in ("before", "after"):
            common.validate_partition({key: value[mode] for key, value in metrics.items()}, f"total/ch{channel}/{mode}")
        total["channels"].append(dict(channel=channel, metrics=metrics))
    require(total["external_DQ_bytes"]["after_minus_before"] == -712704, "Expected 696KiB primary DQ reduction")
    result["primary_stage_sum"] = total
    result["sum_name"] = "primary_stage_sum"
    # A comparable attention block absorbs the previous independent V stage.
    attention = dict(before_cases=["chain", "v"], after_cases=["chain"], channels=[])
    for metric in ("latency_us", "external_DQ_bytes"):
        attention[metric] = pair(result["cases"]["chain"][metric]["before"]+result["cases"]["v"][metric]["before"],
                                 result["cases"]["chain"][metric]["after"])
    for channel in range(16):
        fields = result["cases"]["chain"]["channels"][channel]["metrics"]
        attention["channels"].append(dict(channel=channel, metrics={key: pair(
            result["cases"]["chain"]["channels"][channel]["metrics"][key]["before"]+
            result["cases"]["v"]["channels"][channel]["metrics"][key]["before"], fields[key]["after"]) for key in fields}))
    result["attention_block"] = attention
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyzer, path = common.load_analyzer()
    result = compare(common.load_run(args.before, analyzer), common.load_run(args.after, analyzer, primary=FUSED_PRIMARY), analyzer, path)
    output = args.output or args.after / "pv-token-comparison"
    common.write_report(result, output)
    (output / "act2-deadline-comparison.json").replace(output / "pv-token-comparison.json")
    print(json.dumps(dict(status="passed", primary_latency_us=result["primary_stage_sum"]["latency_us"], output=str(output)), indent=2))


if __name__ == "__main__":
    main()

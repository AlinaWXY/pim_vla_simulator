"""Rebuild FFN bank-partition and double-buffer comparisons from native logs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BASELINE = "7d2730dadc0b5610c6635e800d5d306bfb4164a6"
PRIMARY = ("gate_up_gpu", "down_gpu_input")
FFN = (*PRIMARY, "combined_timing_chain")
MODES = ("before", "legacy", "db1", "db2", "no-overlap1", "no-overlap2", "serial1", "serial2")
METRICS = ("stage_us", "modelled_dq_occupied_us", "modelled_dq_idle_us",
    "tail_clear_complete_zero_external_phase_us", "modelled_dq_idle_outside_tail_clear_work_us",
    "external_DQ_bytes")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()

def require(condition, message):
    if not condition: raise AssertionError(message)

def load_analyzer():
    path = Path(__file__).resolve().parent/"experiments/pi05-unified-gemv/channel_analysis.py"
    spec = importlib.util.spec_from_file_location("ffn_channel_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path

def flatten_channel(row):
    metrics = {key: row[key] for key in METRICS}
    bus = row["command_bus"]
    for name in ("dq_idle_global_ca_busy_us", "dq_idle_global_ca_free_us"):
        metrics[name] = bus[name]
    for name in ("global", "rank0", "rank1"):
        metrics[name+"_ca_busy_us"] = bus[name]["occupied_us"]
    return metrics

def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def compare(root, output):
    analyzer, analyzer_path = load_analyzer()
    sys.path[:0]=[str(analyzer_path.parent/"validation"),str(analyzer_path.parent/"source")]
    from act2_audit import audit_act2
    from audit_native import read_csv
    from masked_audit import annotate_masked_commands
    summaries, reports, audits, native = {}, {}, {}, {}
    protocols={}
    checked = {}
    for mode in MODES:
        directory=root/mode
        summary=read(directory/"summary.json")
        require(summary["status"]=="passed", f"{mode}: run failed")
        summaries[mode]=summary; reports[mode]={}; audits[mode]={}; native[mode]={}; protocols[mode]={}
        expected = ("serial_layer_control",) if mode.startswith("serial") else tuple(summary["runs"])
        require(set(expected)==set(summary["runs"]), f"{mode}: cases")
        if mode not in ("legacy", "serial1", "serial2"):
            require(set(expected)==set(FFN), f"{mode}: FFN cases")
        for case in expected:
            case_dir=directory/case
            print(f"Rebuilding {mode}/{case} from complete raw timelines", flush=True)
            audit=read(case_dir/"native-audit.json"); audits[mode][case]=audit
            require(audit["status"]=="passed" and audit["split_activation"]["status"]=="passed",
                f"{mode}/{case}: native/ACT audit")
            require(digest(case_dir/"workload.trace")==summary["runs"][case]["trace_sha256"]==audit["trace_sha256"],
                f"{mode}/{case}: trace binding")
            for name, sha in audit["native_sha256"].items():
                path=case_dir/"simulation"/name
                checked[str(path)]=digest(path)
                require(checked[str(path)]==sha, f"{mode}/{case}: raw {name}")
            native[mode][case]=read(case_dir/"simulation/result.json")
            events=lambda stem: [event for ch in range(16)
                if (case_dir/"simulation"/f"{stem}.csv.ch{ch}").is_file()
                for event in read_csv(case_dir/"simulation"/f"{stem}.csv.ch{ch}")]
            native_rows=annotate_masked_commands(events("timeline"),events("masked_events"),
                native[mode][case]["config"].get("ffn_data_banks") or None)
            protocols[mode][case]=audit_act2(native_rows,native[mode][case]["resolved_timings_ck"],
                staging_events=events("staging_events"))
            require(protocols[mode][case]["status"]=="passed", f"{mode}/{case}: rebuilt strict ACT protocol")
            reports[mode][case]=analyzer.analyze_stage(case_dir, case)
            require(len(reports[mode][case]["channels"])==16, f"{mode}/{case}: channel count")
            run=summary["runs"][case]; measured=native[mode][case]; report=reports[mode][case]
            require(run["memory_cycles"]==measured["memory_cycles"], f"{mode}/{case}: cycles binding")
            require(abs(run["latency_us"]-report["stage_us"])<1e-9, f"{mode}/{case}: latency binding")
            require(run["external_DQ_bytes"]==sum(r["external_DQ_bytes"] for r in report["channels"]),
                f"{mode}/{case}: DQ binding")
            require(run["traffic_bytes"]==measured["traffic_bytes"], f"{mode}/{case}: native traffic binding")
            phase_times={}
            for part in report["phases"]:
                phase_times[part["phase"]]=phase_times.get(part["phase"],0)+part["duration_us"]
            require(set(phase_times)==set(run["phase_us"]) and all(abs(value-run["phase_us"][name])<1e-9
                for name,value in phase_times.items()), f"{mode}/{case}: phase binding")
            require(abs(sum(phase_times.values())+run["phase_boundary_us"]-run["latency_us"])<1e-9,
                f"{mode}/{case}: phase closure")
            if mode not in ("before","legacy"):
                contract=audit["ffn_split"]
                banks=int(mode[-1]); enabled=not mode.startswith("no-overlap")
                scalar=134217728 if case=="gate_up_gpu" else 67108864 if case=="down_gpu_input" else 201326592
                require(contract["status"]=="passed" and contract["useful_scalar_MACs"]==scalar and
                    contract["padding_MACs"]==0, f"{mode}/{case}: scalar work/padding")
                require(contract["data_banks"]==[0,4][:banks] and contract["data_compute_bank_isolation"] and
                    contract["source_SRF_peak_inflight_bytes"]<=1024, f"{mode}/{case}: physical bank/SRF resources")
                require(contract["ffn_overlap"] is enabled and all(
                    (part["LOCALRD_MAC_overlap_ck"]>0 and part["LOCALRD_issues_during_MAC"]>0 and part["BCAST_issues_during_MAC"]>0)
                    if enabled else (part["LOCALRD_MAC_overlap_ck"]==part["LOCALRD_issues_during_MAC"]==part["BCAST_issues_during_MAC"]==0)
                    for part in contract["overlap"]),f"{mode}/{case}: measured overlap mode")
    for case in FFN:
        a,b=(summaries[mode]["runs"][case] for mode in ("before","legacy"))
        for key in ("latency_us","memory_cycles","trace_sha256","external_DQ_bytes","phase_us","traffic_bytes"):
            require(a[key]==b[key], f"{case}: legacy {key}")
        require(audits["before"][case]["native_sha256"]==audits["legacy"][case]["native_sha256"], f"{case}: legacy raw logs")
    for banks in (1,2):
        overlap, serial=f"db{banks}",f"no-overlap{banks}"
        for case in FFN:
            old=summaries["before"]["runs"][case]
            a,b=(summaries[mode]["runs"][case] for mode in (overlap,serial))
            require(a["trace_sha256"]==b["trace_sha256"], f"{overlap}/{case}: ablation must use identical trace")
            require(a["external_DQ_bytes"]==b["external_DQ_bytes"]==old["external_DQ_bytes"], f"{overlap}/{case}: DQ conservation")
            require(native[overlap][case]["resolved_timings_ck"]==native[serial][case]["resolved_timings_ck"],
                f"{overlap}/{case}: ablation timings")
            configs=[dict(native[mode][case]["config"]) for mode in (overlap,serial)]
            for config,enabled in zip(configs,(True,False)):
                require(config.pop("ffn_data_banks")==banks and config.pop("ffn_overlap") is enabled,
                    f"{overlap}/{case}: dedicated-bank/overlap flags")
            require(configs[0]==configs[1], f"{overlap}/{case}: unrelated ablation configuration")
            require(read(root/overlap/"provenance.json")["native_library_sha256"]==
                read(root/serial/"provenance.json")["native_library_sha256"], f"{overlap}/{case}: ablation library")
            require(native[overlap][case]["tCK_ns"]==native["before"][case]["tCK_ns"], f"{overlap}/{case}: DRAM clock")
            for key,value in native["before"][case]["resolved_timings_ck"].items():
                require(native[overlap][case]["resolved_timings_ck"][key]==value, f"{overlap}/{case}: timing {key}")
    latencies, channels, phases, kinds, commands = [], [], [], [], []
    for mode in MODES:
        for case, report in reports[mode].items():
            run=summaries[mode]["runs"][case]
            latencies.append(dict(mode=mode,case=case,latency_us=run["latency_us"],memory_cycles=run["memory_cycles"],external_DQ_bytes=run["external_DQ_bytes"]))
            for row in report["channels"]:
                channels.append(dict(mode=mode,case=case,channel=row["channel"],**flatten_channel(row)))
                for kind,part in row["phase_kind_breakdown"].items():
                    kinds.append(dict(mode=mode,case=case,channel=row["channel"],kind=kind,**part))
            for phase,value in {**run["phase_us"],"phase_boundary_overhead":run["phase_boundary_us"]}.items():
                phases.append(dict(mode=mode,case=case,phase=phase,latency_us=value))
            for command,count in native[mode][case]["issued_commands"].items():
                commands.append(dict(mode=mode,case=case,command=command,count=count))
    totals={}
    for mode in ("before","legacy","db1","db2","no-overlap1","no-overlap2"):
        total=dict(latency_us=sum(summaries[mode]["runs"][c]["latency_us"] for c in PRIMARY),
            external_DQ_bytes=sum(summaries[mode]["runs"][c]["external_DQ_bytes"] for c in PRIMARY),channels=[])
        require(total["external_DQ_bytes"]==3211264, f"{mode}: complete FFN DQ total")
        for ch in range(16):
            parts=[flatten_channel(reports[mode][c]["channels"][ch]) for c in PRIMARY]
            total["channels"].append(dict(channel=ch,**{key:sum(p[key] for p in parts) for key in parts[0]}))
        totals[mode]=total
    result=dict(status="passed",baseline_commit=BASELINE,primary_cases=list(PRIMARY),
        units="microseconds per layer/step; DQ/CA times per channel", independent_stage_totals=totals,
        runs={mode:summaries[mode]["runs"] for mode in MODES},channels=channels,
        raw_files_checked=len(checked),analyzer_sha256=digest(analyzer_path),
        rebuilt_ACT_protocol=protocols,
        sources={mode:dict(summary_sha256=digest(root/mode/"summary.json"),
            native_library_sha256=read(root/mode/"provenance.json")["native_library_sha256"],
            accounting_source_sha256={case:report["source_sha256"] for case,report in reports[mode].items()}) for mode in MODES},
        audit_contracts={mode:{case:{key:value for key,value in audit.items() if key.startswith("ffn")}
            for case,audit in audits[mode].items()} for mode in MODES},
        checks=dict(legacy_FFN_exact=True,ablation_identical_traces=True,all_DQ_bytes_conserved=True,
            all_native_ACT_audits_passed=True,all_channel_partitions_recomputed=True),
        limitations=["GPU GeLU/multiply/cast and producer delay remain unmeasured.",
            "DQ/global CA idle windows do not measure concurrent GPU traffic service.",
            "MAC target masks and local broadcast costs are architectural model assumptions.",
            "Timing and tensor-address coverage are simulated; no numerical inference is executed."])
    output.mkdir(parents=True,exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name,rows in (("latency",latencies),("channels",channels),("phases",phases),("phase-kinds",kinds),("commands",commands)):
        write_csv(output/(name+".csv"),rows)
    print(json.dumps(dict(status="passed",totals_us={m:t["latency_us"] for m,t in totals.items()}),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compare(args.root,args.output)

if __name__=="__main__": main()

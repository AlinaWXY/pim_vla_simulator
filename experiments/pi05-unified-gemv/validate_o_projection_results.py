"""Bind four-way results, negative guards and unchanged legacy timing evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--legacy",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--validation-evidence",type=Path,
        help="Reuse completed tests from an identical native library and source snapshot")
    a=p.parse_args()
    measured=read(a.experiment/"provenance.json")
    legacy_provenance=read(a.legacy/"provenance.json")
    if legacy_provenance["native_library_sha256"]!=measured["native_library_sha256"]:
        raise AssertionError("Legacy control did not use the measured native library")
    legacy_sources={"experiments/"+path.replace("\\", "/").split("/experiments/",1)[1]:sha
        for path,sha in legacy_provenance["source_sha256"].items()}
    if legacy_sources!=measured["source_sha256"]:
        raise AssertionError("Legacy control source snapshot differs from four-way measurement")
    evidence={}
    for case in ("chain","pv","gate_up_gpu","down_gpu_input","serial_layer_control"):
        old,new=read(a.reference/case/"summary.json"),read(a.legacy/case/"summary.json")
        for field in ("memory_cycles","latency_us","external_DQ_bytes","trace_sha256","traffic_bytes"):
            if new[field]!=old[field]:
                raise AssertionError((case,field,"legacy regression differs"))
        # Native event hashes are independent of paths and runner metadata.
        old_audit,new_audit=read(a.reference/case/"native-audit.json"),read(a.legacy/case/"native-audit.json")
        for filename,sha in old_audit["native_sha256"].items():
            if new_audit["native_sha256"].get(filename)!=sha:
                raise AssertionError((case,filename,"legacy native events differ"))
        evidence[case]=dict(exact_timing=True,exact_trace=True,exact_native_events=True,
            memory_cycles=new["memory_cycles"],latency_us=new["latency_us"])
    summary=read(a.experiment/"summary.json")
    if summary["status"]!="passed":
        raise AssertionError("Four-way simulation failed")
    tests=a.validation_evidence or a.experiment
    if a.validation_evidence:
        tested,measured=read(tests/"provenance.json"),read(a.experiment/"provenance.json")
        for field in ("native_library_sha256","source_sha256"):
            if tested[field]!=measured[field]:
                raise AssertionError((field,"Test evidence does not match measured artifact"))
        if not tested["tests_executed"]:
            raise AssertionError("Supplied evidence was not produced by a test run")
    elif not summary["tests_executed"]:
        raise AssertionError("Full four-way tests were not executed")
    unit_log=(tests/"runner-logs/unit-tests.log").read_text(encoding="utf-8")
    unit_count=re.search(r"Ran (\d+) tests in",unit_log)
    if not unit_count or not unit_log.rstrip().endswith("OK"):
        raise AssertionError("Unit test evidence is incomplete or failed")
    checks={}
    for label in ("legacy-native-guards","score-native-guards","o-native-guards"):
        path=tests/label/"summary.json"
        if not path.is_file():
            path=tests/label/"native-guards.json"
        item=read(path)
        if not item.get("passed",item.get("status")=="passed"):
            raise AssertionError((label,"guards failed"))
        checks[label]=item
    variants=summary["variants"]
    modes={"neither":(False,False),"score_only":(True,False),"oproj_only":(False,True),"both":(True,True)}
    if set(variants)!=set(modes):
        raise AssertionError("All four ablation modes are required")
    for name,item in variants.items():
        if set(item["runs"])!={"chain","pv","gate_up_gpu","down_gpu_input","serial_layer_control"} or item["measured_stage_sum_us_per_layer_step"] is None:
            raise AssertionError((name,"All primary stages and serial control are required"))
        score,proj=modes[name]
        if (item["score_dram_spill"],item["o_projection"])!=(score,proj):
            raise AssertionError((name,"Incorrect ablation switch settings"))
        for case,run in item["runs"].items():
            if (run["score_dram_spill"],run["o_projection"],run["pv_bf16_output"])!=(
                score and case in ("chain","serial_layer_control"),proj and case in ("pv","serial_layer_control"),
                case in ("pv","serial_layer_control")):
                raise AssertionError((name,case,"Native stage switches differ from intended mode"))
        if not item["pv_bf16_output"] or any(r["audit_status"]!="passed" for r in item["runs"].values()):
            raise AssertionError((name,"precision/audit mismatch"))
    for left,right,stage in (("neither","oproj_only","chain"),("score_only","both","chain"),
                             ("neither","score_only","pv"),("oproj_only","both","pv")):
        for field in ("trace_sha256","memory_cycles","external_DQ_bytes"):
            if variants[left]["runs"][stage][field]!=variants[right]["runs"][stage][field]:
                raise AssertionError((left,right,stage,field,"switches are not independent"))
    files=[p for p in a.experiment.rglob("*") if p.is_file() and p.name in
        ("summary.json","native-audit.json","comparison.json","channel-analysis.json","provenance.json")]
    result=dict(status="passed",legacy_regression=evidence,native_guards=checks,
        unit_tests=int(unit_count.group(1)),tests_run_separately=bool(a.validation_evidence),
        validation_evidence=str(tests),same_tested_native_library_and_sources=True,
        same_legacy_control_native_library_and_sources=True,
        four_way_switch_independence=True,common_PV_BF16_precision=True,
        result_hashes={str(p.relative_to(a.experiment)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        numerical_validation=False,complete_end_to_end_latency_us=None)
    correction=tests/"assertion-correction.json"
    if correction.is_file():
        result["test_assertion_correction"]=read(correction)
    (a.experiment/"validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","legacy_cases":len(evidence),"four_way_variants":len(variants)}))


if __name__=="__main__":
    main()

"""Read-only continuous DQ/shared-global-CA accounting for historical runs.

Primary windows retain complete meaningful phases with no issues on the named
resource. Physical idle gaps and CA-sliced DQ-work envelopes are diagnostics;
neither their sums nor tiny fragments are presented as complete work windows.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from channel_analysis import dq_intervals, issue_tick, read_csv, read_json, require
from continuous_dq import complete_work_windows, maximal_idle_windows, semantic_scopes, union_intervals, window_record, write_csv

RESOURCES=("dq","global_ca","dq_global_ca")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_disjoint(items,label):
    items.sort(key=lambda e:(e["begin"],e["end"]))
    require(all(e["begin"]<e["end"] for e in items),f"Empty {label} reservation")
    require(all(a["end"]<=b["begin"] for a,b in zip(items,items[1:])),f"Overlapping {label} reservations")
    return items


def ca_intervals(rows,ratio,channel):
    """Decode recorded resources, without double-counting mirrored rank rows."""
    resources={"global_ca":[],"rank0_ca":[],"rank1_ca":[]}
    for row in rows:
        require(int(row["Channel"])==channel,"CA channel label mismatch")
        rank,scope=int(row["Rank"]),row["scope"]
        require(rank in (-1,0,1) and scope in ("global","rank"),"Unknown CA resource")
        start,end=issue_tick(row["clock"],ratio),issue_tick(row["ready_at"],ratio)
        item=dict(begin=start,end=end,issue=start,command=row["command"])
        names=["global_ca"] if scope=="global" else [f"rank{r}_ca" for r in ((0,1) if rank==-1 else (rank,))]
        for name in names:
            resources[name].append(dict(item))
    return {name:check_disjoint(items,name) for name,items in resources.items()}


def verified_legacy_contract(path,directory,manifest,result,channel_count):
    """Validate an explicit source-bound reconstruction contract before use.

    The contract is authored from the historical source, not guessed from a
    modern command name. Raw hashes bind it to the selected historical run.
    """
    path,directory=Path(path).resolve(),Path(directory).resolve()
    contract=read_json(path)
    require(contract.get("model")=="legacy_flat_shared_ca","Unsupported legacy CA model")
    require(not result["config"].get("pim_rank_controller",False),"Flat CA contract cannot reconstruct a hierarchical controller")
    source_root=(path.parent/contract["source_root"]).resolve()
    source_hashes=contract["source_sha256"]
    require(source_hashes and contract.get("source_evidence"),"Legacy CA contract lacks source evidence")
    for name,expected in source_hashes.items():
        candidate=(source_root/name).resolve()
        require(candidate.is_file() and sha256(candidate)==expected,f"Legacy CA source hash mismatch: {name}")
    case=contract["cases"][manifest["case"]]
    for filename,key in (("manifest.json","manifest_sha256"),("simulation/result.json","result_sha256"),
                         ("simulation/barriers.csv","barriers_sha256")):
        require(sha256(directory/filename)==case[key],f"Legacy CA case hash mismatch: {filename}")
    expected=case["timeline_sha256"]
    require(set(expected)=={f"timeline.csv.ch{i}" for i in range(channel_count)},"Legacy contract must bind every channel")
    for name,digest in expected.items():
        require(sha256(directory/"simulation"/name)==digest,f"Legacy CA timeline hash mismatch: {name}")
    cycles,internal=contract["command_cycles_ck"],set(contract["no_ca_commands"])
    require(cycles and all(isinstance(n,int) and n>0 for n in cycles.values()),"Invalid legacy CA command width")
    require(not set(cycles)&internal,"Legacy CA/internal command sets overlap")
    provenance=dict(status="reconstructed_verified",model=contract["model"],contract_sha256=sha256(path),
        source_sha256=source_hashes,source_evidence=contract["source_evidence"],command_cycles_ck=cycles,
        no_ca_commands=sorted(internal),
        limitations="Reconstruction of this historical simulator's flat shared CA reservations; no rank-private bus model or electrical validation is added.")
    return cycles,internal,provenance


def reconstruct_legacy_ca(rows,ratio,cycles,internal):
    items=[]
    for row in rows:
        command=row["command"]
        if command in internal:
            continue
        require(command in cycles,f"Legacy CA command has no verified width: {command}")
        begin=issue_tick(row["clock"],ratio)
        items.append(dict(begin=begin,end=begin+cycles[command]*ratio,issue=begin,command=command))
    return check_disjoint(items,"reconstructed shared CA")


def diagnostic_joint_windows(dq_work,ca,phases):
    """CA-sliced DQ-complete envelopes, explicitly allowed to be fragmentary."""
    pieces=[]
    for parent_index,work in enumerate(dq_work):
        _,idle=maximal_idle_windows(ca,work["begin"],work["end"])
        for begin,end in idle:
            overlaps=[p for p in phases if p["meaningful"] and p["begin"]<end and p["end"]>begin]
            if not overlaps:
                continue
            # A CA pulse in a control bridge breaks that bridge; don't retain
            # an isolated control-only prefix/suffix as useful work.
            begin,end=max(begin,overlaps[0]["begin"]),min(end,overlaps[-1]["end"])
            pieces.append(dict(begin=begin,end=end,parent_dq_work_window_index=parent_index,
                phase_names=[p["name"] for p in phases if p["begin"]<end and p["end"]>begin]))
    return pieces


def unavailable_row(identity,begin,end,tick_us,reason):
    return dict(identity,available=False,unavailable_reason=reason,scope_begin_tick=begin,scope_end_tick=end,
        scope_duration_us=(end-begin)*tick_us,resource_busy_us=None,idle_total_us=None,
        window_count=None,longest_duration_us=None,complete_work_window_count=None,
        complete_work_idle_sum_us=None,longest_complete_work_window_us=None,
        diagnostic_joint_window_count=None,diagnostic_joint_longest_us=None)


def analyze_case(directory,flow,legacy_ca_contract=None):
    directory=Path(directory).resolve()
    paths=[directory/"manifest.json",directory/"simulation/result.json",directory/"simulation/barriers.csv"]
    manifest,result=read_json(paths[0]),read_json(paths[1])
    cfg=result["config"]
    ratio,count=int(cfg["frontend_ratio"]),int(cfg["channels"])
    require(ratio>0 and count==16,"Historical comparison requires all 16 channels")
    require(result["transaction_bytes"]==32,"Expected 32-byte external transaction")
    wall=int(result["memory_cycles"])*ratio
    tick_us=result["tCK_ns"]/(ratio*1000)
    require(abs(wall*tick_us-result["latency_ns"]/1000)<1e-8,"Native wall-time mismatch")
    barriers=read_csv(paths[2])
    markers={int(r["marker"]):int(r["frontend_tick"]) for r in barriers}
    require(len(markers)==len(barriers),"Duplicate phase barrier")
    ca_paths=[directory/f"simulation/rank_ca_events.csv.ch{i}" for i in range(count)]
    ca_present=[p.is_file() for p in ca_paths]
    require(all(ca_present) or not any(ca_present),"Partial CA logs cannot establish all-channel availability")
    fallback=None
    if all(ca_present):
        require(cfg.get("pim_rank_controller",False),"Recorded rank CA logs require an enabled hierarchical controller")
        provenance=dict(status="recorded",model="hierarchical_global_and_rank_ca",
            definition="Only scope=global occupies shared global CA. scope=rank occupies the named rank resource, including internal fine commands. A command may be logged once globally and separately on its target rank; those are distinct resources.")
    elif legacy_ca_contract is not None:
        cycles,internal,provenance=verified_legacy_contract(legacy_ca_contract,directory,manifest,result,count)
        fallback=(cycles,internal)
    else:
        provenance=dict(status="unavailable",model=None,
            reason="No complete native rank_ca_events logs and no verified legacy flat-CA reconstruction contract.")
    dq,ca,rank_stats={}, {}, []
    external_count=0
    for channel in range(count):
        path=directory/f"simulation/timeline.csv.ch{channel}"
        paths.append(path)
        rows=read_csv(path)
        require(all(int(r["Channel"])==channel for r in rows),"Native channel label mismatch")
        require(all(int(a["clock"])<=int(b["clock"]) for a,b in zip(rows,rows[1:])),"Nonmonotonic command trace")
        dq[channel]=dq_intervals(rows,result["resolved_timings_ck"],ratio,cfg["pim_data_path"])
        external_count+=len(dq[channel])
        if all(ca_present):
            paths.append(ca_paths[channel])
            resources=ca_intervals(read_csv(ca_paths[channel]),ratio,channel)
            ca[channel]=resources["global_ca"]
            rank_stats.append(dict(channel=channel,**{f"{name}_busy_us":maximal_idle_windows(items,0,wall)[0]*tick_us
                for name,items in resources.items()}))
        elif fallback is not None:
            ca[channel]=reconstruct_legacy_ca(rows,ratio,*fallback)
    require(external_count*32==manifest["expected_dq_bytes"]["total"],"Native DQ byte contract mismatch")
    # All-channel idle means simultaneous freedom, so use the union, while
    # retaining the original issues for resource-specific phase qualification.
    dq["all_channels"]=[e for i in range(count) for e in dq[i]]
    if ca:
        ca["all_channels"]=[e for i in range(count) for e in ca[i]]
    records,physical,complete,diagnostic=[],[],[],[]
    for scope in semantic_scopes(manifest,markers,wall):
        begin,end,phases=scope["begin"],scope["end"],scope["phases"]
        for channel,external in dq.items():
            base=dict(flow=flow,variant=flow,case=manifest["case"],stage=scope["stage"],scope=scope["scope"],channel=channel)
            resource_items={"dq":external}
            if ca:
                resource_items.update(global_ca=ca[channel],dq_global_ca=external+ca[channel])
            dq_work=complete_work_windows(external,begin,end,phases)
            pieces=diagnostic_joint_windows(dq_work,ca[channel],phases) if ca else []
            for resource in RESOURCES:
                identity=dict(base,resource=resource)
                if resource not in resource_items:
                    records.append(unavailable_row(identity,begin,end,tick_us,provenance["reason"]))
                    continue
                items=resource_items[resource]
                row,windows,work=window_record(union_intervals(items),begin,end,variant=flow,case=manifest["case"],
                    stage=scope["stage"],scope=scope["scope"],channel=channel,tick_us=tick_us,
                    phases=phases,metadata=scope.get("metadata"),issue_intervals=items)
                row.update(flow=flow,resource=resource,available=True,ca_status=provenance["status"] if resource!="dq" else "not_required")
                row["resource_busy_us"]=row.pop("dq_busy_us")
                row["idle_total_us"]=row.pop("dq_idle_total_us")
                if resource=="dq_global_ca":
                    row.update(diagnostic_joint_window_count=len(pieces),
                        diagnostic_joint_longest_us=max((p["end"]-p["begin"] for p in pieces),default=0)*tick_us,
                        diagnostic_joint_sum_us=sum(p["end"]-p["begin"] for p in pieces)*tick_us)
                records.append(row)
                for window in windows:
                    window.update(flow=flow,resource=resource)
                for window in work:
                    window.update(flow=flow,resource=resource)
                physical.extend(windows);complete.extend(work)
            for index,piece in enumerate(pieces):
                diagnostic.append(dict(base,resource="dq_global_ca",window_index=index,
                    begin_tick=piece["begin"],end_tick=piece["end"],begin_us=piece["begin"]*tick_us,
                    end_us=piece["end"]*tick_us,duration_us=(piece["end"]-piece["begin"])*tick_us,
                    parent_dq_work_window_index=piece["parent_dq_work_window_index"],phase_names=piece["phase_names"],
                    may_be_fragmentary=True))
    return dict(records=records,complete_work_windows=complete,physical_windows=physical,
        diagnostic_joint_windows=diagnostic,ca_provenance=provenance,rank_ca_accounting=rank_stats,
        source_sha256={str(p.relative_to(directory)):sha256(p) for p in paths},
        input_root=str(directory),flow=flow,case=manifest["case"])


class GzipRows:
    """Append independently analyzed cases without retaining all physical gaps."""
    def __init__(self,path):
        self.stream=gzip.open(path,"wt",encoding="utf-8",newline="")
        self.writer=None

    def write(self,rows):
        for row in rows:
            if self.writer is None:
                self.writer=csv.DictWriter(self.stream,fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

    def close(self):
        self.stream.close()


def write_report(cases,output):
    """Analyze already-existing cases; no simulator invocation or raw writes."""
    output=Path(output).resolve()
    for case in cases:
        directory=Path(case["directory"]).resolve(strict=True)
        require(output!=directory and directory not in output.parents,"Output must be outside every immutable raw case")
    output.mkdir(parents=True,exist_ok=True)
    physical=GzipRows(output/"dq-ca-physical-windows.csv.gz")
    diagnostic=GzipRows(output/"dq-ca-diagnostic-windows.csv.gz")
    records,complete,evidence=[],[],[]
    try:
        for case in cases:
            report=analyze_case(case["directory"],case["flow"],case.get("legacy_ca_contract"))
            records.extend(report.pop("records"));complete.extend(report.pop("complete_work_windows"))
            physical.write(report.pop("physical_windows"));diagnostic.write(report.pop("diagnostic_joint_windows"))
            evidence.append(report)
    finally:
        physical.close();diagnostic.close()
    result=dict(schema_version=1,records=records,complete_work_windows=complete,evidence=evidence,
        definitions=dict(primary="For each resource, retain complete meaningful semantic phases with zero issues on that resource; remove preceding resource tails, merge exact-adjacent eligible phases, and permit fully idle control-only bridges only between work phases. Any resource-issuing phase is excluded entirely, including idle holes. No duration threshold is applied.",
            resource_dq="External data bursts including ordinary read/write CL/CWL delays.",
            resource_global_ca="Shared channel/global command resource only; rank-private command resources are separate and not added to global CA.",
            resource_dq_global_ca="Both external DQ and shared global CA must be idle. The union of their busy reservations and all their issue times qualify primary phases.",
            physical="Maximal idle gaps of the resource inside its scope, including possibly tiny transfer-phase gaps; diagnostics only.",
            diagnostic_joint="DQ-complete work envelopes intersected with shared-global-CA idle. CA pulses split these windows, which may be fragmentary and are not the strict complete-work primary metric.",
            sums="complete_work_idle_sum_us is a sum of separate complete windows; only longest_complete_work_window_us is one continuous interval. idle_total_us and diagnostic_joint_sum_us are also sums, never continuous time.",
            clocks="Half-open integer frontend ticks; first memory edge=(CK-1)*frontend_ratio+1. Scope boundaries clip windows. Independent native runs are never concatenated.",
            simultaneous="channel=all_channels uses the union of all 16 channels' occupied resources and retains every original issue for phase qualification.",
            missing="Unavailable CA is null, not zero. Legacy flat CA is reconstructed only with a source- and raw-hash-verified explicit contract.",
            limitations="Resource idle is not measured GPU service time or an electrical waveform. Strict complete-work results depend on semantic phase granularity; zero means no complete phase qualifies, not that the bus is always occupied."),
        files=dict(summary="dq-ca-summary.csv",complete_work_windows="dq-ca-complete-work-windows.csv",
            physical_windows="dq-ca-physical-windows.csv.gz",diagnostic_windows="dq-ca-diagnostic-windows.csv.gz"))
    (output/"dq-ca-windows.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    write_csv(output/"dq-ca-summary.csv",records)
    write_csv(output/"dq-ca-complete-work-windows.csv",complete)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True,help="JSON list of {flow,directory,legacy_ca_contract?}; paths relative to this JSON")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases=read_json(args.cases)
    for case in cases:
        for key in ("directory","legacy_ca_contract"):
            if key in case:
                case[key]=str((args.cases.resolve().parent/case[key]).resolve())
    report=write_report(cases,args.output)
    print(json.dumps(dict(records=len(report["records"]),cases=len(report["evidence"]),output=str(args.output))))


if __name__=="__main__":
    main()

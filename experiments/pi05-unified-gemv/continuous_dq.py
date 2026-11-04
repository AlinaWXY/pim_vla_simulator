"""Exact maximal DQ-idle windows from immutable native command traces.

All arithmetic uses integer frontend ticks and half-open intervals. Work-phase
boundaries do not split an idle window. Independent runs are never concatenated;
the serial control has both clipped stage scopes and its actual full-run scope.
"""
import argparse
from bisect import bisect_left
import csv
import gzip
import hashlib
import json
from pathlib import Path

from channel_analysis import _has_work, dq_intervals, interval_accounting, read_csv, read_json, require, tail_clear_window

VARIANTS=("neither","score_only","oproj_only","both")
CASES=("chain","pv","gate_up_gpu","down_gpu_input","serial_layer_control")


def union_intervals(intervals):
    """Union simultaneous-channel busy intervals; also merge touching bursts."""
    merged=[]
    for item in sorted(intervals,key=lambda e:(e["begin"],e["end"])):
        begin,end=int(item["begin"]),int(item["end"])
        require(begin<=end,"Reversed DQ interval")
        if begin==end:
            continue
        if merged and begin<=merged[-1]["end"]:
            merged[-1]["end"]=max(merged[-1]["end"],end)
        else:
            merged.append(dict(begin=begin,end=end))
    return merged


def maximal_idle_windows(intervals,begin,end):
    """Return the exact complement within one scope, including burst tails."""
    require(isinstance(begin,int) and isinstance(end,int),"Scope ticks must be integers")
    busy,idle=interval_accounting(union_intervals(intervals),begin,end)
    require(busy+sum(b-a for a,b in idle)==end-begin,"Continuous DQ accounting does not close")
    require(all(a<b for a,b in idle),"Empty idle window")
    return busy,idle


def percentile(values,fraction):
    if not values:
        return None
    ordered=sorted(values)
    position=(len(ordered)-1)*fraction
    lower=int(position)
    upper=min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def complete_work_windows(intervals,begin,end,phases,issue_intervals=None):
    """Join complete DQ-free work phases, with safe control-only bridges.

    A transfer phase containing any external issue always breaks the window.
    A control-only gap may bridge two work phases only if its complete span is
    physically DQ-idle. Leading/trailing control phases never extend a window.
    Incoming burst tails are removed before attempting any adjacency merge.
    """
    busy=union_intervals(intervals)
    source=intervals if issue_intervals is None else issue_intervals
    source=[dict(e,issue=e.get("issue",e["begin"])) for e in source]
    issues=sorted(e["issue"] for e in source)
    result=[]
    active=None
    pending=[]
    continuity=None
    def flush():
        nonlocal active,pending,continuity
        if active is not None:
            result.append(active)
        active,pending,continuity=None,[],None
    for p in phases:
        left,right=p["begin"],p["end"]
        if left<begin or right>end:
            if left<end and right>begin:
                flush()
            continue
        if active is not None and left!=continuity:
            flush()
        has_external=bisect_left(issues,right)>bisect_left(issues,left)
        if has_external:
            flush();continue
        if p["meaningful"]:
            clear,_=tail_clear_window(source,left,right)
            occupied,_=interval_accounting(busy,clear,right)
            require(occupied==0,"Tail-trimmed complete work phase still carries DQ")
            if clear>=right:
                flush();continue
            if active is not None and clear==continuity:
                active["end"]=right
                active["phases"].extend(pending+[p])
            else:
                flush()
                active=dict(begin=clear,end=right,phases=[p])
            pending=[];continuity=right
        else:
            occupied,_=interval_accounting(busy,left,right)
            if active is not None and occupied==0:
                pending.append(p);continuity=right
            else:
                flush()
    flush()
    return result


def window_record(intervals,begin,end,*,variant,case,stage,scope,channel,tick_us,phases,metadata=None,issue_intervals=None):
    busy,windows=maximal_idle_windows(intervals,begin,end)
    durations=[b-a for a,b in windows]
    # First occurrence wins ties so the reported start/end are deterministic.
    longest=max(windows,key=lambda pair:pair[1]-pair[0],default=None)
    identity=dict(variant=variant,case=case,stage=stage,scope=scope,channel=channel)
    row=dict(identity,scope_begin_tick=begin,scope_end_tick=end,
        scope_begin_us=begin*tick_us,scope_end_us=end*tick_us,scope_duration_us=(end-begin)*tick_us,
        dq_busy_us=busy*tick_us,dq_idle_total_us=sum(durations)*tick_us,
        window_count=len(windows),longest_begin_tick=longest[0] if longest else None,
        longest_end_tick=longest[1] if longest else None,
        longest_begin_us=longest[0]*tick_us if longest else None,
        longest_end_us=longest[1]*tick_us if longest else None,
        longest_begin_relative_scope_us=(longest[0]-begin)*tick_us if longest else None,
        longest_end_relative_scope_us=(longest[1]-begin)*tick_us if longest else None,
        longest_duration_us=(longest[1]-longest[0])*tick_us if longest else 0.0,
        window_duration_p50_us=percentile(durations,.5)*tick_us if durations else None,
        window_duration_p95_us=percentile(durations,.95)*tick_us if durations else None,
        **(metadata or {}))
    details=[]
    for index,(left,right) in enumerate(windows):
        overlaps=[p for p in phases if p["begin"]<right and p["end"]>left]
        details.append(dict(identity,window_index=index,begin_tick=left,end_tick=right,
            begin_us=left*tick_us,end_us=right*tick_us,duration_us=(right-left)*tick_us,
            begin_relative_scope_us=(left-begin)*tick_us,end_relative_scope_us=(right-begin)*tick_us,
            touches_scope_start=left==begin,touches_scope_end=right==end,
            phase_names=[p["name"] for p in overlaps],
            phase_kinds=list(dict.fromkeys(p["kind"] for p in overlaps))))
    if longest:
        best=next(w for w in details if w["begin_tick"]==longest[0] and w["end_tick"]==longest[1])
        row["longest_phase_names"],row["longest_phase_kinds"]=best["phase_names"],best["phase_kinds"]
    else:
        row["longest_phase_names"],row["longest_phase_kinds"]=[],[]
    complete=complete_work_windows(intervals,begin,end,phases,issue_intervals)
    complete_details=[]
    for index,item in enumerate(complete):
        left,right=item["begin"],item["end"]
        complete_details.append(dict(identity,window_index=index,begin_tick=left,end_tick=right,
            begin_us=left*tick_us,end_us=right*tick_us,duration_us=(right-left)*tick_us,
            begin_relative_scope_us=(left-begin)*tick_us,end_relative_scope_us=(right-begin)*tick_us,
            phase_names=[p["name"] for p in item["phases"]],
            phase_kinds=list(dict.fromkeys(p["kind"] for p in item["phases"])),
            work_phase_count=sum(p["meaningful"] for p in item["phases"]),
            control_bridge_phase_count=sum(not p["meaningful"] for p in item["phases"])))
    best=max(complete_details,key=lambda w:w["duration_us"],default=None)
    row.update(complete_work_window_count=len(complete_details),
        complete_work_idle_sum_us=sum(w["duration_us"] for w in complete_details),
        longest_complete_work_window_us=best["duration_us"] if best else 0.0,
        longest_complete_work_begin_tick=best["begin_tick"] if best else None,
        longest_complete_work_end_tick=best["end_tick"] if best else None,
        longest_complete_work_begin_us=best["begin_us"] if best else None,
        longest_complete_work_end_us=best["end_us"] if best else None,
        longest_complete_work_begin_relative_scope_us=best["begin_relative_scope_us"] if best else None,
        longest_complete_work_end_relative_scope_us=best["end_relative_scope_us"] if best else None,
        longest_complete_work_phase_names=best["phase_names"] if best else [],
        longest_complete_work_phase_kinds=best["phase_kinds"] if best else [])
    return row,details,complete_details


def semantic_scopes(manifest,markers,wall):
    """Construct stage and QKT bounds without adding times from separate runs."""
    case=manifest["case"]
    phases=[]
    for s in manifest["segments"]:
        begin,end=markers[int(s["start"])],markers[int(s["end"])]
        require(0<=begin<=end<=wall,"Phase outside native run")
        stage=s.get("stage",case)
        phases.append(dict(name=s["name"],short_name=s["name"].rsplit("/",1)[-1],
            stage=stage,kind=s.get("kind","unclassified"),begin=begin,end=end,meaningful=_has_work(s)))
    require(all(a["end"]==b["begin"] for a,b in zip(phases,phases[1:])),"Noncontiguous semantic phases")
    scopes=[]
    if case=="serial_layer_control":
        # Stage rows stop at actual stage boundaries. The additional full-run
        # row uses this one real timeline, never a sum of independent cases.
        for stage in dict.fromkeys(p["stage"] for p in phases):
            group=[p for p in phases if p["stage"]==stage]
            require(all(a["end"]==b["begin"] for a,b in zip(group,group[1:])),"Noncontiguous serial stage")
            scopes.append(dict(scope="stage",stage=stage,begin=group[0]["begin"],end=group[-1]["end"],phases=group))
        scopes.append(dict(scope="serial_layer",stage="serial_layer_control",begin=0,end=wall,phases=phases))
    else:
        scopes.append(dict(scope="stage",stage=case,begin=0,end=wall,phases=phases))
    qkt=[p for p in phases if p["short_name"].startswith("qk_")]
    if qkt:
        require(all(a["end"]==b["begin"] for a,b in zip(qkt,qkt[1:])),"QKT envelope contains unrelated phases")
        exports=[p for p in qkt if p["short_name"].endswith("_export")]
        metadata=dict(qkt_compute_phase_count=sum(p["short_name"].endswith("_compute") for p in qkt),
            qkt_spill_phase_count=sum(p["short_name"].endswith("_spill") for p in qkt),
            qkt_export_phase_count=len(exports),
            qkt_export_strategy="final_batch" if any(p["short_name"]=="qk_scores_dram_export" for p in qkt) else "per_tile",
            first_export_phase_begin_tick=exports[0]["begin"] if exports else None)
        scopes.append(dict(scope="qkt",stage="chain",begin=qkt[0]["begin"],end=qkt[-1]["end"],phases=qkt,metadata=metadata))
    return scopes


def analyze_case(directory,variant):
    directory=Path(directory)
    paths=[directory/"manifest.json",directory/"simulation/result.json",directory/"simulation/barriers.csv"]
    manifest,result=map(read_json,paths[:2])
    cfg,timings=result["config"],result["resolved_timings_ck"]
    ratio,count=int(cfg["frontend_ratio"]),int(cfg["channels"])
    require(ratio>0 and count==16,"Continuous four-way report requires all 16 channels")
    tick_us=result["tCK_ns"]/(ratio*1000)
    wall=int(result["memory_cycles"])*ratio
    require(abs(wall*tick_us-result["latency_ns"]/1000)<1e-8,"Native wall-time mismatch")
    barriers=read_csv(paths[2])
    markers={int(e["marker"]):int(e["frontend_tick"]) for e in barriers}
    require(len(markers)==len(barriers),"Duplicate phase barrier")
    intervals={}
    external_count=0
    for channel in range(count):
        path=directory/f"simulation/timeline.csv.ch{channel}"
        paths.append(path)
        rows=read_csv(path)
        require(all(int(e["Channel"])==channel for e in rows),"Native channel label mismatch")
        intervals[channel]=dq_intervals(rows,timings,ratio,cfg["pim_data_path"])
        external_count+=len(intervals[channel])
    require(external_count*32==manifest["expected_dq_bytes"]["total"],"Native DQ byte contract mismatch")
    # Busy union across channels is used only for simultaneous-idle windows;
    # occupied times from different channels are never added into a duration.
    all_issues=[e for channel in range(count) for e in intervals[channel]]
    intervals["all_channels"]=union_intervals(all_issues)
    records,windows,complete_windows=[],[],[]
    for scope in semantic_scopes(manifest,markers,wall):
        for channel,dq in intervals.items():
            metadata=dict(scope.get("metadata",{}))
            first_export=None
            issues=all_issues if channel=="all_channels" else dq
            if scope["scope"]=="qkt":
                exports=[p for p in scope["phases"] if p["short_name"].endswith("_export")]
                first_export=min((e["begin"] for e in issues
                    if any(p["begin"]<=e["issue"]<p["end"] for p in exports)),default=None)
                require(first_export is None or scope["begin"]<=first_export<=scope["end"],"First score export outside QKT envelope")
                metadata["first_export_dq_begin_tick"]=first_export
                metadata["first_export_dq_begin_us"]=first_export*tick_us if first_export is not None else None
            row,detail,complete=window_record(dq,scope["begin"],scope["end"],variant=variant,
                case=manifest["case"],stage=scope["stage"],scope=scope["scope"],channel=channel,
                tick_us=tick_us,phases=scope["phases"],metadata=metadata,issue_intervals=issues)
            records.append(row);windows.extend(detail);complete_windows.extend(complete)
            if scope["scope"]=="qkt" and first_export is not None:
                row,detail,complete=window_record(dq,scope["begin"],first_export,variant=variant,
                    case=manifest["case"],stage=scope["stage"],scope="qkt_before_first_export",channel=channel,
                    tick_us=tick_us,phases=scope["phases"],metadata=metadata,issue_intervals=issues)
                records.append(row);windows.extend(detail);complete_windows.extend(complete)
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return records,windows,complete_windows,hashes


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    path=Path(path)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"wt",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def write_report(root,output,*,allow_nested_output=False):
    root,output=Path(root).resolve(strict=True),Path(output).resolve()
    require(allow_nested_output or root!=output and root not in output.parents,"Output must be outside the immutable raw input tree")
    records,windows,complete_windows,hashes=[],[],[],{}
    for variant in VARIANTS:
        for case in CASES:
            directory=root/variant/case
            require((directory/"simulation/result.json").is_file(),f"Missing required native case: {directory}")
            r,w,c,h=analyze_case(directory,variant)
            records.extend(r);windows.extend(w);complete_windows.extend(c)
            hashes.update({f"{variant}/{case}/{p}":sha for p,sha in h.items()})
    result=dict(schema_version=1,input_root=str(root),records=records,source_sha256=hashes,
        complete_work_windows=complete_windows,
        qkt_windows=[w for w in windows if w["scope"]=="qkt" and w["case"]=="chain" and w["channel"]==0],
        files=dict(summary="continuous-dq-summary.csv",physical_windows="continuous-dq-windows.csv.gz",
            complete_work_windows="continuous-dq-complete-work-windows.csv"),
        definitions=dict(intervals="Half-open [begin,end) integer frontend ticks; microseconds are elapsed native-run time.",
            dq_model="AttAcc RDRES/WRGB occupy issue+[0,nBL_min); ordinary RD/WR include nCL/nCWL. Native channel bursts are checked disjoint.",
            longest="Longest single maximal external-DQ-idle interval within the named scope, never a sum of fragments.",
            complete_work="Longest complete-work window merges adjacent meaningful phases with zero external issues after incoming DQ tails are trimmed. DQ-free control-only phases may bridge two work phases; leading/trailing controls are excluded. Any phase issuing external transfers breaks the window, including its idle holes. This semantic metric is distinct from the maximal physical DQ-idle interval.",
            stages="Independent native cases are never concatenated. Serial-control stage rows keep actual run timestamps and stop at stage boundaries. The separate serial_layer scope covers its one real continuous native run and may cross those boundaries.",
            qkt="Envelope from the first qk_* phase through the last qk_* phase, including all score exports and final Q-slot release. DQ-free phase boundaries do not split windows.",
            prefix="qkt_before_first_export ends at the first modeled external score DQ burst, not its command issue or semantic phase boundary. It may include pre-burst CL/row-preparation time.",
            all_channels="Complement of the union of all 16 channels' busy intervals; all channels are idle simultaneously.",
            totals="dq_idle_total_us is secondary accounting only and is never called a continuous window.",
            physical_window_files="The gzip CSV retains every physical window. JSON qkt_windows is a preview restricted to case chain, channel 0, for all four variants.",
            distribution="p50/p95 are unweighted window-duration percentiles using linear interpolation.",
            boundaries="Windows touching scope boundaries are clipped to the scope; their true extent outside that scope is not claimed.",
            limitations="External DQ idle is not MAC active time, an electrical waveform, or measured GPU-usable time. Internal memory waits, refresh and CA use may occur inside it."))
    output.mkdir(parents=True,exist_ok=True)
    (output/"continuous-dq.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    write_csv(output/"continuous-dq-summary.csv",records)
    write_csv(output/"continuous-dq-windows.csv.gz",windows)
    write_csv(output/"continuous-dq-complete-work-windows.csv",complete_windows)
    return result


analyze=write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=write_report(args.raw_root,args.output)
    print(json.dumps(dict(records=len(result["records"]),source_files=len(result["source_sha256"]),output=str(args.output))))


if __name__=="__main__":
    main()

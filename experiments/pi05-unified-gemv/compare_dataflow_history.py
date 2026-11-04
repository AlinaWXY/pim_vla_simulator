"""Compare four existing dataflows; read native logs, never invoke a simulator."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from channel_analysis import dq_intervals, read_csv
from dq_ca_windows import write_report

FLOW_IDS=("direct","two_bank_stream","two_bank_bulk","two_bank_bulk_pim_o")
LABELS=("1 Direct input\nGPU O unmeasured","2 Two banks / stream\nGPU O unmeasured",
        "3 Two banks / bulk\nGPU O unmeasured","4 Two banks / bulk\nPIM O included")
COLORS=("#4979ad","#d9a441","#44a89b","#ad739b")
RESOURCES=("dq","global_ca","dq_global_ca")
HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def write_csv(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare(direct,modern,output,layers=18,steps=10,windows_cache=None):
    direct,modern,output=Path(direct).resolve(),Path(modern).resolve(),Path(output).resolve()
    assert layers>0 and steps>0
    assert all(output!=p and p not in output.parents for p in (direct,modern)),"Use a separate derived output directory"
    output.mkdir(parents=True,exist_ok=False)
    roots=dict(zip(FLOW_IDS,(direct,modern/"neither",modern/"score_only",modern/"both")))
    summaries={flow:read(p/"summary.json") for flow,p in roots.items()}
    cases=[];source_hashes={};native_audits={}
    for flow,root in roots.items():
        s=summaries[flow];assert s["status"]=="passed"
        primary=tuple(s["primary_cases"])
        assert primary==(("chain","v","pv","gate_up_gpu","down_gpu_input") if flow=="direct" else ("chain","pv","gate_up_gpu","down_gpu_input"))
        source_hashes[str(root/"summary.json")]=digest(root/"summary.json")
        prov=root/"provenance.json" if flow=="direct" else modern/"provenance.json"
        source_hashes[str(prov)]=digest(prov)
        for case in (*primary,"serial_layer_control"):
            p=root/case;native=read(p/"simulation/result.json");cfg=native["config"]
            manifest=read(p/"manifest.json")
            assert abs(s["runs"][case]["latency_us"]-native["latency_ns"]/1000)<1e-8
            assert s["runs"][case]["external_DQ_bytes"]==manifest["expected_dq_bytes"]["total"]
            markers={int(b["marker"]):int(b["frontend_tick"]) for b in read_csv(p/"simulation/barriers.csv")}
            phase_us={seg["name"]:(markers[int(seg["end"])]-markers[int(seg["start"])])*native["tCK_ns"]/(1000*cfg["frontend_ratio"]) for seg in manifest["segments"]}
            assert set(phase_us)==set(s["runs"][case]["phase_us"])
            assert all(abs(v-s["runs"][case]["phase_us"][k])<1e-8 for k,v in phase_us.items()),"Phase/O timing must match raw barriers"
            assert cfg.get("pim_rank_staging",False)==(flow!="direct")
            if flow!="direct":assert cfg["pim_staging_banks"]==2
            if case in ("chain","serial_layer_control"):
                assert cfg.get("pim_score_dram_spill",False)==(flow in ("two_bank_bulk","two_bank_bulk_pim_o"))
            if case in ("pv","serial_layer_control"):
                assert cfg.get("pim_o_projection",False)==(flow=="two_bank_bulk_pim_o")
                assert cfg.get("pim_pv_bf16_output",False)==(flow!="direct")
            audit=read(p/"native-audit.json");assert audit["status"]=="passed"
            required={f"timeline.csv.ch{i}" for i in range(16)}
            if flow!="direct":required|={f"rank_ca_events.csv.ch{i}" for i in range(16)}
            assert required<=set(audit["native_sha256"]),"Audit must bind every native resource log"
            checked={}
            for name,sha in audit["native_sha256"].items():
                if name.startswith(("timeline.csv.ch","rank_ca_events.csv.ch")):
                    assert digest(p/"simulation"/name)==sha,(flow,case,name)
                    checked[name]=sha
            native_audits[f"{flow}/{case}"]=dict(audit_sha256=digest(p/"native-audit.json"),checked=checked)
            spec=dict(flow=flow,directory=str(p))
            if flow=="direct":spec["legacy_ca_contract"]=str(HERE/"legacy_direct_ca_contract.json")
            cases.append(spec)
    if windows_cache is None:
        windows=write_report(cases,output/"windows")
    else:
        # Reuse exact interval derivation when only aggregation/figures change.
        cache=Path(windows_cache).resolve()
        validation=read(cache.parent.parent/"analysis-validation.json")
        assert validation["status"]=="passed" and validation["native_simulation_rerun"] is False
        for name in ("dq_ca_windows.py","continuous_dq.py","channel_analysis.py","legacy_direct_ca_contract.json"):
            assert validation["analysis_code_sha256"]["experiments/pi05-unified-gemv/"+name]==digest(HERE/name)
        windows=read(cache)
        assert {(r["flow"],str(Path(r["input_root"]).resolve())) for r in windows["evidence"]}=={(c["flow"],str(Path(c["directory"]).resolve())) for c in cases}
        for evidence in windows["evidence"]:
            for name,sha in evidence["source_sha256"].items():
                assert digest(Path(evidence["input_root"])/name)==sha
        assert {"windows/"+p.name for p in cache.parent.iterdir()}=={n for n in validation["output_sha256"] if n.startswith("windows/")},"Incomplete interval cache"
        for p in cache.parent.iterdir():
            assert digest(p)==validation["output_sha256"]["windows/"+p.name]
        shutil.copytree(cache.parent,output/"windows")
    def record(flow,case,resource="dq",scope="stage",channel=0):
        return next(r for r in windows["records"] if
            (r["flow"],r["case"],r["scope"],r["resource"],r["channel"])==(flow,case,scope,resource,channel))
    rows=[];stage_rows=[];operator_rows=[];channel_rows=[]
    factor=layers*steps/1000
    for index,flow in enumerate(FLOW_IDS,1):
        s=summaries[flow];primary=s["primary_cases"]
        wall=sum(s["runs"][c]["latency_us"] for c in primary)
        assert abs(wall-s["measured_stage_sum_us_per_layer_step"])<1e-8
        o_us=sum(v for k,v in s["runs"]["pv"]["phase_us"].items() if k.startswith("o_projection_"))
        row=dict(flow=index,flow_id=flow,layer_step_us=wall,modeled_ae_ms=wall*factor,
            full_AE_end_to_end_ms=None,o_projection_timed=flow==FLOW_IDS[-1],
            o_time_scope="PIM O read/broadcast/compute, control and BF16 export included" if o_us else "GPU O projection unmeasured",
            pim_o_subflow_us=o_us if o_us else None,pim_o_subflow_ae_ms=o_us*factor if o_us else None,
            serial_control_us=s["serial_layer_memory_control_us"],
            serial_control_ae_ms=s["serial_layer_memory_control_us"]*factor,
            external_dq_KiB=sum(s["runs"][c]["external_DQ_bytes"] for c in primary)/1024,
            pv_output_precision="FP32 complete feature output" if flow=="direct" else "BF16 token-parity partial output")
        for ch in range(16):
            channel=dict(flow=index,channel=ch,layer_step_us=wall)
            for resource in RESOURCES:
                rr=[record(flow,c,resource,channel=ch) for c in primary]
                assert all(r["available"] for r in rr)
                channel[resource+"_busy_us"]=sum(r["resource_busy_us"] for r in rr)
                channel[resource+"_complete_work_sum_us"]=sum(r["complete_work_idle_sum_us"] for r in rr)
                channel[resource+"_complete_work_count"]=sum(r["complete_work_window_count"] for r in rr)
                channel[resource+"_longest_complete_work_us"]=max(r["longest_complete_work_window_us"] for r in rr)
            channel["dq_utilization_percent"]=100*channel["dq_busy_us"]/wall
            channel["global_ca_longest_single_idle_us"]=max(record(flow,c,"global_ca",channel=ch)["longest_duration_us"] for c in primary)
            channel["joint_longest_single_idle_within_dq_work_us"]=max(record(flow,c,"dq_global_ca",channel=ch)["diagnostic_joint_longest_us"] for c in primary)
            channel_rows.append(channel)
            if ch==0:row.update({k:v for k,v in channel.items() if k not in ("flow","channel","layer_step_us")})
        util=[c["dq_utilization_percent"] for c in channel_rows if c["flow"]==index]
        row.update(dq_utilization_16_channel_mean_percent=sum(util)/len(util),
            dq_utilization_16_channel_min_percent=min(util),dq_utilization_16_channel_max_percent=max(util))
        row["all_channels_simultaneous"]=dict(
            dq_longest_complete_work_us=max(record(flow,c,"dq",channel="all_channels")["longest_complete_work_window_us"] for c in primary),
            global_ca_longest_single_idle_us=max(record(flow,c,"global_ca",channel="all_channels")["longest_duration_us"] for c in primary),
            joint_longest_single_idle_within_dq_work_us=max(record(flow,c,"dq_global_ca",channel="all_channels")["diagnostic_joint_longest_us"] for c in primary))
        rows.append(row)
        for case in primary:
            r=record(flow,case)
            stage_rows.append(dict(flow=index,stage=case,latency_us=s["runs"][case]["latency_us"],
                dq_busy_us=r["resource_busy_us"],dq_utilization_percent=100*r["resource_busy_us"]/r["scope_duration_us"],
                o_projection_included=flow==FLOW_IDS[-1] and case=="pv"))
            for resource in RESOURCES:
                scopes=("stage","qkt") if case=="chain" else ("stage",)
                for scope in scopes:
                    r=record(flow,case,resource,scope)
                    operator_rows.append(dict(flow=index,case=case,scope=scope,resource=resource,
                        longest_complete_us=r["longest_complete_work_window_us"],
                        complete_window_count=r["complete_work_window_count"],
                        complete_window_sum_us=r["complete_work_idle_sum_us"],
                        physical_longest_us=r["longest_duration_us"],
                        diagnostic_joint_longest_us=r.get("diagnostic_joint_longest_us")))
    result=dict(schema_version=1,source_native_jobs=[1903036,1924418],simulation_rerun=False,
        layers=layers,denoising_steps=steps,rows=rows,
        latency_scope="Sum of existing independent stages multiplied by layers*steps; modeled AE portion, not complete AE or a continuous simulation of 18 resident layers.",
        idle_scope="Primary DQ: longest complete semantic work window, rejecting transfer phases and their holes. Primary CA: longest actual single shared-CA-free span. Primary joint: longest CA-free span within a complete DQ-work envelope, possibly starting after dispatch; no fragment sums or duration threshold. Strict whole-phase CA/joint fields are supplemental and may discard long idle suffixes. Longest windows are not multiplied by layers*steps.",
        CA_scope="Shared/global channel CA. Direct uses source- and raw-hash-verified legacy flat CA reconstruction; modern runs use explicit scope=global event logs. Rank-private CA is separate.",
        historical_difference="Flow 1 uses separate V, feature-sharded PV, FP32 complete PV outputs, direct double-buffer streaming, and legacy flat dispatch. Flows 2-4 use fused QKV, token-sharded PV, BF16 partial outputs and hierarchical rank dispatch. Flow 1 vs 2 is not an isolated input-staging ablation.",
        missing_latency=["GPU O projection in flows 1-3","GPU softmax/scaling","GPU partial-result reductions","GPU nonlinearities, norm/residual, dispatch and background contention"],
        input_roots={k:str(v) for k,v in roots.items()},source_sha256=source_hashes,native_audits=native_audits,
        derivation_sha256={p.name:digest(p) for p in (Path(__file__),HERE/"dq_ca_windows.py",HERE/"continuous_dq.py",HERE/"channel_analysis.py",HERE/"legacy_direct_ca_contract.json")})
    result["interval_cache"]=dict(path=str(cache),sha256=digest(cache),validation_sha256=digest(cache.parent.parent/"analysis-validation.json")) if windows_cache else None
    dump(output/"comparison.json",result)
    write_csv(output/"comparison.csv",rows);write_csv(output/"stage-comparison.csv",stage_rows)
    write_csv(output/"operator-idle-windows.csv",operator_rows);write_csv(output/"all-channel-summary.csv",channel_rows)
    plots(output,roots,summaries,rows,windows,factor,layers,steps)
    return result


def plots(output,roots,summaries,rows,windows,factor,layers,steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    components=[("QKV / RoPE / QKT / V","#4979ad",[sum(summaries[f]["runs"][c]["latency_us"] for c in ("chain","v") if c in summaries[f]["primary_cases"]) for f in FLOW_IDS]),
        ("PV + result handling","#44a89b",[summaries[f]["runs"]["pv"]["latency_us"]-(r["pim_o_subflow_us"] or 0) for f,r in zip(FLOW_IDS,rows)]),
        ("PIM O + export: INCLUDED","#d66543",[r["pim_o_subflow_us"] or 0 for r in rows]),
        ("Gate / Up","#d9a441",[summaries[f]["runs"]["gate_up_gpu"]["latency_us"] for f in FLOW_IDS]),
        ("Down","#ad739b",[summaries[f]["runs"]["down_gpu_input"]["latency_us"] for f in FLOW_IDS])]
    bottom=np.zeros(4)
    for label,color,values in components:
        values=np.array(values)*factor;axes[0].bar(x,values,bottom=bottom,color=color,label=label)
        if label.startswith("PIM O"):
            axes[0].text(3,bottom[3]+values[3]/2,f"O included\n{values[3]:.3f} ms",ha="center",va="center",color="white",fontsize=9)
        bottom+=values
    for i,r in enumerate(rows):axes[0].text(i,r["modeled_ae_ms"]+.4,f"{r['modeled_ae_ms']:.3f}",ha="center")
    axes[0].set_ylim(0,max(bottom)*1.47);axes[0].set_ylabel(f"Modeled AE portion (ms; {layers} layers × {steps} steps)")
    axes[0].set_title("AE timing scope differs: O timed only in flow 4");axes[0].legend(loc="upper left",fontsize=8)
    values=np.array([r["dq_utilization_16_channel_mean_percent"] for r in rows])
    work=np.array([100*r["dq_complete_work_sum_us"]/r["layer_step_us"] for r in rows])
    other=100-values-work
    assert all(other>=-1e-8),"DQ breakdown must close"
    axes[1].bar(x,values,color="#4979ad",label="DQ busy (utilization)")
    axes[1].bar(x,work,bottom=values,color="#44a89b",label="Complete idle windows (separate windows)")
    axes[1].bar(x,other,bottom=values+work,color="#d8dde3",label="Other idle (excluded from complete-window metric)")
    for i,v in enumerate(values):axes[1].text(i,v/2,f"{v:.3f}%",ha="center",va="center",color="white",fontsize=9)
    axes[1].set_ylim(0,143);axes[1].set_ylabel("Fraction of modeled stage wall time (%)")
    # Exact byte symmetry is required before using ch0 idle fractions with the
    # mean busy fraction. The per-channel CSV remains the authoritative detail.
    assert all(r["dq_utilization_16_channel_max_percent"]-r["dq_utilization_16_channel_min_percent"]<1e-8 for r in rows)
    axes[1].set_title("DQ utilization: 16-channel mean; idle split: ch0")
    axes[1].legend(loc="upper left",fontsize=8)
    for ax in axes:ax.set_xticks(x,LABELS,fontsize=8)
    fig.suptitle("Four existing dataflows — no simulation rerun",fontsize=15)
    fig.supxlabel("Flows 1–3 omit GPU O arithmetic; all omit GPU reductions/nonlinear work. Flow 1 also differs in layout/precision/scheduling.\n"
        "AE values are stage-sum scaling, not measured end-to-end latency. Lower DQ utilization is not a speedup metric.",fontsize=9)
    fig.savefig(output/"ae-latency-dq-utilization.png",dpi=180);fig.savefig(output/"ae-latency-dq-utilization.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    width=.24
    for i,(resource,label,color) in enumerate(zip(RESOURCES,("DQ","Shared CA","DQ + shared CA"),("#4979ad","#d9a441","#44a89b"))):
        for ax,field in zip(axes,("_longest_complete_work_us","_complete_work_sum_us")):
            vals=[r[resource+field] for r in rows];pos=x+(i-1)*width
            ax.bar(pos,vals,width=width,label=label,color=color)
            for p,v in zip(pos,vals):ax.text(p,v+max(vals+[1])*.02,f"{v:.3f}",rotation=90,ha="center",va="bottom",fontsize=8)
    for ax in axes:
        ax.set_xticks(x,("1 Direct","2 Stream","3 Bulk","4 Bulk + O"));ax.set_ylabel("µs / layer / step")
        ax.legend(loc="upper left",fontsize=8);ax.set_ylim(0,ax.get_ylim()[1]*1.28)
    axes[0].set_title("Longest single complete idle window\nNever multiplied by 180")
    axes[1].set_title("Supplement: sum of complete idle windows\nSeparate windows; not continuous time")
    fig.suptitle("Supplement: strict whole-phase idle windows — channel 0",fontsize=14)
    fig.supxlabel("A phase with any issue on the resource is excluded; no arbitrary duration cutoff.\nZero shared-CA/joint time means no whole phase qualifies, not absence of physical bus gaps.",fontsize=9)
    fig.savefig(output/"dq-ca-complete-windows.png",dpi=180);fig.savefig(output/"dq-ca-complete-windows.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,5.8),constrained_layout=True)
    specs=(("chain","qkt"),("pv","stage"),("gate_up_gpu","stage"),("down_gpu_input","stage"))
    for ax,resource,label in zip(axes,RESOURCES,("DQ: complete work window","Shared CA: actual single idle span","DQ + CA: single span within DQ work")):
        for i,(flow,color) in enumerate(zip(FLOW_IDS,COLORS)):
            field={"dq":"longest_complete_work_window_us","global_ca":"longest_duration_us","dq_global_ca":"diagnostic_joint_longest_us"}[resource]
            vals=[next(r[field] for r in windows["records"] if
                (r["flow"],r["case"],r["scope"],r["channel"],r["resource"])==(flow,c,s,0,resource)) for c,s in specs]
            pos=x+(i-1.5)*.20
            ax.bar(pos,vals,width=.19,color=color,label=("1 Direct","2 Stream","3 Bulk","4 Bulk + O")[i])
            for p,v in zip(pos,vals):ax.text(p,v+max(vals+[.001])*.025,f"{v:.3f}",rotation=90,ha="center",va="bottom",fontsize=7)
        ax.set_xticks(x,("QKT","PV / PV + O","Gate / Up","Down"));ax.set_ylabel("Longest single window (µs)")
        ax.set_title(label);ax.legend(loc="upper left",fontsize=8)
        ax.set_ylim(0,ax.get_ylim()[1]*1.30)
    fig.suptitle("Per-operator uninterrupted time — channel 0",fontsize=14)
    fig.supxlabel("Score exports split QKT windows: flow 1 and 2 have 31 separate work windows; flow 3 and 4 have one.\n"
        "CA/joint use the longest actual single gap, including long idle suffixes after dispatch. No panel adds fragmented gaps together.",fontsize=9)
    fig.savefig(output/"operator-idle-windows.png",dpi=180);plt.close(fig)
    # Serial controls are real continuous native runs, unlike the stage sums.
    fig,axes=plt.subplots(4,1,figsize=(14,8),sharex=True,constrained_layout=True)
    serial_max=0
    for ax,flow,label in zip(axes,FLOW_IDS,LABELS):
        case=roots[flow]/"serial_layer_control";native=read(case/"simulation/result.json")
        ratio=native["config"]["frontend_ratio"];scale=native["tCK_ns"]/(1000*ratio)
        wall=native["memory_cycles"]*ratio;serial_max=max(serial_max,wall*scale)
        intervals=dq_intervals(read_csv(case/"simulation/timeline.csv.ch0"),native["resolved_timings_ck"],ratio,native["config"]["pim_data_path"])
        width=max(1,round(.5/scale));edges=np.append(np.arange(0,wall,width),wall);occupied=np.zeros(len(edges)-1)
        for e in intervals:
            lo=max(0,e["begin"]//width);hi=min(len(occupied)-1,e["end"]//width)
            for b in range(lo,hi+1):occupied[b]+=max(0,min(e["end"],edges[b+1])-max(e["begin"],edges[b]))/(edges[b+1]-edges[b])
        ax.stairs(100*occupied,edges*scale,fill=True,color="#4979ad")
        ax.axvline(wall*scale,color="#555555",linestyle="--",linewidth=.8)
        ax.text(.01,.86,f"Continuous native control: {wall*scale:.6f} µs",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor="white",edgecolor="none",alpha=.8))
        ax.set_ylim(0,105);ax.set_ylabel(label+"\nDQ %",fontsize=8)
    axes[-1].set_xlim(0,serial_max+1);axes[-1].set_xlabel("µs from start of real serial-layer control (0.5 µs display bins)")
    fig.suptitle("DQ utilization timeline — existing continuous controls, channel 0",fontsize=14)
    fig.supxlabel("Binning is for display only. Complete DQ/CA windows are derived from exact command reservations, not from these bins.",fontsize=9)
    fig.savefig(output/"dq-utilization-timeline.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct",type=Path,required=True);p.add_argument("--modern",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--layers",type=int,default=18);p.add_argument("--steps",type=int,default=10)
    p.add_argument("--windows-cache",type=Path,help="Reuse a validated dq-ca-windows.json, verifying interval code, contract and every raw hash")
    a=p.parse_args();r=compare(a.direct,a.modern,a.output,a.layers,a.steps,a.windows_cache)
    print(json.dumps(r["rows"],indent=2))

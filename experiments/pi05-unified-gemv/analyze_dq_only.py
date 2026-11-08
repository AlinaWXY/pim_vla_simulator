"""DQ-only sensitivity of the existing 1108 evaluation; no native/GPU rerun."""
import argparse, gzip, json, math
from pathlib import Path
import analyze_evaluation as A

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args()
    root=a.reference;out=root/'dq-only';out.mkdir(exist_ok=True)
    with gzip.open(root/'native-bus-templates.json.gz','rt') as f:stages=json.load(f)
    old=A.read(root/'summary.json');ks=A.kernels(root/'gpu')
    handoffs={k:v['cold32MiB']['device_kernel_median_us'] for k,v in A.read(root/'gpu/handoff-measurements.json')['operators'].items()}
    units=A.read(root/'gpu/prefix-kernels.json')['ordered_mean_units'];rows=[];case_rows=[]
    for v in A.VARIANTS:
        selected={c:stages[v+'/'+c] for c in A.CASES}
        for c,s in selected.items():
            idle=A.gaps(s['dq'],0,s['wall_ticks'])
            assert A.duration(s['dq'])+A.duration(idle)==s['wall_ticks']
            s['joint_idle']=idle # Reuse schedule construction, changing only the eligible resource.
            case_rows.append(dict(variant=v,case=c,longest_raw_DQ_idle_us=max(b-a for a,b in idle)*s['tick_us']))
        _,windows,ae=A.make_schedule(v,selected,ks,handoffs)
        previous=next(r for r in old['models'] if r['variant']==v)
        assert math.isclose(ae,previous['AE_known_model_ms'],abs_tol=1e-9)
        fitted=A.fit(windows,units);launch=A.fit(windows,units,launch=True)
        placements=fitted.pop('placements');A.csvout(out/(v+'-placements.csv'),placements)
        row=dict(variant=v,longest_DQ_idle_us=max(w['duration_us'] for w in windows),
                 DQ_idle_total_ms=sum(w['duration_us'] for w in windows)/1000,
                 joint_longest_us=previous['longest_joint_window_us'],**fitted,
                 overlap_with_host_launch_ms=launch['overlap_ms'],AE_ms=ae,VLM_ms=previous['VLM_measured_ms'],
                 total_ms=ae+previous['VLM_measured_ms']-fitted['overlap_ms'])
        rows.append(row);print(json.dumps(row),flush=True)
    A.csvout(out/'comparison.csv',rows);A.csvout(out/'per-case-raw-windows.csv',case_rows)
    A.dump(out/'summary.json',dict(native_run=old['native_run'],source_sha256=A.sha(root/'native-bus-templates.json.gz'),
        policy='All 16 DQ channels idle; CA ignored. Ordered whole kernels, no splitting/reordering. Memory refresh effects remain in native timing; availability of GPU memory/CA during each window is not guaranteed.',
        rows=rows,raw_case_windows=case_rows))

if __name__=='__main__':main()

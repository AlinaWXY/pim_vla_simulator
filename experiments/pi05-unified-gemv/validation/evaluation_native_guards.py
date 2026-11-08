"""Adversarial native checks for the new direct-input and GPU-RoPE contracts."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reproduce as R

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    backend=R.HERE.parent/'pi05-bgk-gpuq/backend';R.setup(backend)
    import evaluation_workload as E,coarse_workload
    from simulator.backend import run_trace
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for name,case,needle,action,expected in (
        ('GPU_import_without_export','chain','LD ','delete','GPU RoPE import before export'),
        ('duplicate_GPU_Q_import','chain','ST ','duplicate','Duplicate GPU Q tap'),
        ('incomplete_GPU_Q_import','chain','ST ','delete','query'),
        ('direct_input_row_overflow','chain','PIM_WR_GB ','overflow','external row exceeds'),
        ('incomplete_PV_to_O','pv','PIM_RD_RES ','delete','Output export address/count mismatch'),
    ):
        d=a.output/name;d.mkdir();p=d/'workload.trace';E.generate(p,case,'feature_gpu_before')
        lines=p.read_text().splitlines();i=next(i for i,x in enumerate(lines) if x.startswith(needle))
        if action=='delete':lines.pop(i)
        elif action=='duplicate':lines.insert(i,lines[i])
        else:
            f=lines[i].split();f[1]=str(E.addr(0,0,512));lines[i]=' '.join(f)
        p.write_text('\n'.join(lines)+'\n')
        from reproduce_evaluation import evaluation_config
        cfg=replace(evaluation_config(case,'feature_gpu_before'),max_cycles=100000)
        try:run_trace(cfg,p,d/'simulation')
        except Exception as exc:
            error=str(exc);assert expected.lower() in error.lower(),(name,error,expected)
            rows.append(dict(test=name,status='passed',observed_rejection=error))
        else:raise AssertionError('Invalid trace accepted: '+name)
    R.dump(a.output/'validation.json',dict(status='passed',tests=rows))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()

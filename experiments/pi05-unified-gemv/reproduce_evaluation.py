"""Fresh native 1108 matrix. GPU measurements are reused, never recaptured here."""
import argparse,json,os
from dataclasses import replace
from pathlib import Path
import reproduce as R

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variants',nargs='+',default=['feature_gpu_before','token_gpu_before','token_local_before','token_gpu_after','token_local_after'])
    ap.add_argument('--cases',nargs='+',default=['chain','pv','gate_up_gpu','down_gpu_input','serial_layer_control'])
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    backend=R.HERE.parent/'pi05-bgk-gpuq/backend';R.setup(backend)
    import evaluation_workload as E,coarse_workload
    from simulator.backend import run_trace
    sources=[p for base in (backend/'simulator',backend/'ramulator2/src',R.HERE/'source') for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h','.inc')]
    sources += [Path(__file__),backend/'ramulator2/libramulator.so']
    R.dump(a.output/'provenance.json',dict(native_library_sha256=R.digest(sources[-1]),
        source_sha256={str(p.relative_to(R.HERE.parent.parent)):R.digest(p) for p in sources},
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),branch='xinyao_1108',gpu_rerun=False))
    for variant in a.variants:
        v=E.VARIANTS[variant]
        for case in a.cases:
            d=a.output/variant/case;d.mkdir(parents=True,exist_ok=False)
            m=E.generate(d/'workload.trace',case,variant);R.dump(d/'manifest.json',m)
            opts={**coarse_workload.OPTIONS,'rank_descriptors':False,'score_dram_spill':v['after']}
            cfg=replace(R.config_for(case,**opts),pim_coarse_commands=v['after'],
                        pim_stream_inputs=not v['after'],pim_gpu_rope=v['gpu_rope'],pim_feature_layout=v['feature'])
            result=run_trace(cfg,d/'workload.trace',d/'simulation')
            dq=sum(result['issued_commands'].get(c,0)*32 for c in R.EXTERNAL)
            assert dq==m['expected_dq_bytes']['total'],(variant,case,dq,m['expected_dq_bytes'])
            row=dict(variant=variant,case=case,latency_us=result['latency_ns']/1000,
                memory_cycles=result['memory_cycles'],tCK_ns=result['tCK_ns'],external_DQ_bytes=dq,
                trace_sha256=R.digest(d/'workload.trace'),native_library_sha256=R.digest(sources[-1]),
                complete_heterogeneous_E2E=False)
            R.dump(d/'summary.json',row);print(json.dumps(row),flush=True)

if __name__=='__main__':main()

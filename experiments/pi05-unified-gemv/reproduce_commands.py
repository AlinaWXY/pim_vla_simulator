"""Fresh before/after native runs with identical current layout and data traffic."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import reproduce as R


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=('before','after','both'),default='both')
    ap.add_argument('--cases',nargs='+',default=['chain','pv','gate_up_gpu','down_gpu_input','serial_layer_control'])
    a=ap.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    backend=R.HERE.parent/'pi05-bgk-gpuq/backend';R.setup(backend)
    sources=[p for base in (backend/'simulator',backend/'ramulator2/src',R.HERE/'source')
             for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h','.inc')]
    sources += [Path(__file__),backend/'ramulator2/libramulator.so']
    R.dump(root/'source_sha256.json',{str(p.relative_to(R.HERE.parent.parent)):R.digest(p) for p in sorted(sources)})
    from simulator.backend import run_trace
    import workload,coarse_workload
    for mode in (('before','after') if a.mode=='both' else (a.mode,)):
        for case in a.cases:
            directory=root/mode/case;directory.mkdir(parents=True)
            opts=coarse_workload.OPTIONS
            manifest=(workload.generate(directory/'workload.trace',case,**opts) if mode=='before' else
                      coarse_workload.generate(directory/'workload.trace',case))
            R.dump(directory/'manifest.json',manifest)
            cfg=R.config_for(case,**opts)
            if mode=='after': cfg=replace(cfg,pim_coarse_commands=True,pim_rank_descriptors=False)
            result=run_trace(cfg,directory/'workload.trace',directory/'simulation')
            dq=sum(result['issued_commands'].get(c,0)*32 for c in R.EXTERNAL)
            assert dq==manifest['expected_dq_bytes']['total'],(mode,case,dq,manifest['expected_dq_bytes'])
            summary=dict(mode=mode,case=case,latency_us=result['latency_ns']/1000,
                memory_cycles=result['memory_cycles'],tCK_ns=result['tCK_ns'],external_DQ_bytes=dq,
                trace_sha256=R.digest(directory/'workload.trace'),numerical_compute=False)
            R.dump(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

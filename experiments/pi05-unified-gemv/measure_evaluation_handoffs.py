"""Only new heterogeneous boundary operators; does not rerun GPU model or VLM."""
import json,statistics,time
from pathlib import Path
import torch

torch.set_num_threads(1);torch.manual_seed(1108)
device='cuda';bf=torch.bfloat16
score=torch.randn(2,8,16,992,device=device,dtype=torch.float32)
o=torch.randn(16,16,1024,device=device,dtype=bf)
gu=torch.randn(2,16,4096,device=device,dtype=torch.float32)
down=torch.randn(16,1024,device=device,dtype=torch.float32)
tests={
 'feature_score_rank_sum':lambda:torch.add(score[0],score[1]),
 'common_O_rank_sum_cast':lambda:o.sum(dim=0,dtype=torch.float32).to(bf),
 'common_GateUp_fp32_to_bf16':lambda:gu.to(bf),
 'common_Down_fp32_to_bf16':lambda:down.to(bf),
}
flush=torch.empty(32*1024*1024,device=device,dtype=torch.uint8)
out=dict(scope='Supplemental PIM/GPU handoff operators only; GPU/VLM baseline not rerun',
 gpu=torch.cuda.get_device_name(),torch=torch.__version__,measurement='CUDA event elapsed time; eager sequence, host launch gaps included',
 cold_policy='32MiB write before each timed invocation; no claim of identical NCU cache control',operators={})
for name,fn in tests.items():
    for _ in range(20):result=fn()
    torch.cuda.synchronize()
    modes={}
    for mode in ('warm','cold32MiB'):
        samples=[]
        for _ in range(100):
            if mode!='warm':flush.zero_()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();result=fn();b.record();b.synchronize()
            samples.append(a.elapsed_time(b)*1000)
        modes[mode]=dict(mean_us=statistics.mean(samples),median_us=statistics.median(samples),
                         minimum_us=min(samples),samples_us=samples)
    # Separate actual device kernel duration from eager host launch gaps. This
    # matches the retained AE kernel-ledger accounting better than event envelopes.
    for mode in ('warm','cold32MiB'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(30):
                if mode!='warm':flush.zero_()
                with torch.profiler.record_function('boundary/'+name):result=fn()
            torch.cuda.synchronize()
        events=[e for e in prof.events() if e.name=='boundary/'+name and e.device_type==torch.autograd.DeviceType.CPU]
        values=[e.device_time_total for e in events]
        assert len(values)==30 and min(values)>0,(name,mode,values)
        modes[mode]['device_kernel_mean_us']=statistics.mean(values)
        modes[mode]['device_kernel_median_us']=statistics.median(values)
        modes[mode]['device_kernel_samples_us']=values
    out['operators'][name]=modes
out['validation']=dict(score_matches=torch.equal(tests['feature_score_rank_sum'](),score[0]+score[1]),
                       output_shape=list(tests['common_O_rank_sum_cast']().shape),
                       output_dtype=str(tests['common_O_rank_sum_cast']().dtype))
path=Path(__file__).with_name('handoff-measurements.json');path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{m:v['device_kernel_median_us'] for m,v in modes.items()} for k,modes in out['operators'].items()}),flush=True)

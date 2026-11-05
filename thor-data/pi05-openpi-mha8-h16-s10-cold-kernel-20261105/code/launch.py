"""Launch bounded, foreground NCU jobs. No retries or detached continuation."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

mode,policy,tag=sys.argv[1:]
if mode not in ('probe','model') or policy not in ('all','none') or not tag.replace('_','').isalnum():
    raise ValueError('Invalid launch arguments')
root=Path(__file__).resolve().parent.parent
base='/var/tmp/shiyuhuang-vla-baseline'
metrics='gpu__time_duration.sum'
if mode=='probe':
    metrics+=',lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum'
command=['/usr/local/cuda/bin/ncu','--profile-from-start','off','--target-processes','all',
    '--metrics',metrics,'--clock-control','none','--cache-control',policy,'--replay-mode','kernel',
    '--import-sass','off','--nvtx','--nvtx-include','cold_step/',
    '--export',base+'/runs/'+tag,'bash',base+'/code/pi05_h16_cold_kernel_20261105/run.sh',
    '--mode',mode,'--cache-policy',policy,'--tag',tag]
remote='hostname\npwd\n'+f'export TMPDIR={base}/tmp XDG_CACHE_HOME={base}/cache/xdg CUDA_CACHE_PATH={base}/cache/cuda\n'+shlex.join(command)
hop='hostname\npwd\nssh -T -o BatchMode=yes thor0 '+shlex.quote(remote)
with (root/'logs'/(tag+'.command.json')).open('x') as f:
    json.dump(dict(command=command,mode=mode,cache_policy=policy),f,indent=2)
with (root/'logs'/(tag+'.log')).open('x') as log:
    proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','dev0-public',hop],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if 'Profiling "' not in line:
            print(line,end='',flush=True)
    raise SystemExit(proc.wait())

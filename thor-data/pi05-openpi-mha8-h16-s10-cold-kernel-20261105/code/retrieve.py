"""Preserve finished NCU evidence on fact_data and the local workspace."""
from pathlib import Path
import shlex
import subprocess
import sys

tag=sys.argv[1]
if not tag.replace('_','').isalnum():
    raise ValueError('Unsafe tag')
root=Path(__file__).resolve().parent.parent
base='/var/tmp/shiyuhuang-vla-baseline/runs'
dest='/fact_data/shiyuhuang/projects/vla-pi05-latency-v2/results/pi05_h16_cold_kernel_20261105/raw'
thor='cd '+shlex.quote(base)+'\ntar -cf - '+shlex.join([tag,tag+'.ncu-rep'])
remote='set -e\nset -o pipefail\nhostname\npwd\ncd /fact_data/shiyuhuang/projects/vla-pi05-latency-v2\npwd\nmkdir -p '+shlex.quote(dest)+'\nssh -T -o BatchMode=yes thor0 '+shlex.quote(thor)+' | tar --keep-old-files -xf - -C '+shlex.quote(dest)
subprocess.run(['ssh','-T','-o','BatchMode=yes','dev0-public',remote],check=True)
process=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','dev0-public','tar -C '+shlex.quote(dest)+' -cf - '+shlex.join([tag,tag+'.ncu-rep'])],stdout=subprocess.PIPE)
subprocess.run(['tar','--keep-old-files','-xf','-','-C',str(root/'raw')],stdin=process.stdout,check=True)
process.stdout.close()
if process.wait():
    raise RuntimeError('Evidence transfer failed')
command=['/usr/local/cuda/bin/ncu','--import',base+'/'+tag+'.ncu-rep','--page','raw','--csv',
    '--print-units','base','--print-fp','--print-nvtx-rename','kernel']
with (root/'raw'/tag/'kernels.csv').open('x') as out:
    subprocess.run(['ssh','-T','-o','BatchMode=yes','dev0-public','ssh -T -o BatchMode=yes thor0 '+shlex.quote(shlex.join(command))],stdout=out,check=True)
print('RETRIEVED',tag,flush=True)

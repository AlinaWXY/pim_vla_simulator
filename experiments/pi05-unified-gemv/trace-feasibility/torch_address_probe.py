"""One PyTorch operator for tracer compatibility, not a Pi0.5 measurement."""
import os
import subprocess
import torch

others = subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
assert not [int(x) for x in others.splitlines() if x.strip().isdigit() and int(x) != os.getpid()]
a_cpu = torch.arange(64, dtype=torch.float32)
a = a_cpu.cuda()
b = (a_cpu+1).cuda()
c = torch.add(a, b)
torch.cuda.synchronize()
print(f'TORCH_BASE a={a.data_ptr():#x} b={b.data_ptr():#x} c={c.data_ptr():#x} bytes=256')
values = c.cpu().tolist()
assert values == [float(2*i+1) for i in range(64)]
print(f'TORCH_PASS version={torch.__version__} elements=64 sum={sum(values):.0f}')

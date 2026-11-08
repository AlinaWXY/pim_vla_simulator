"""One bounded streaming GPU kernel for discovering MCC counter breakdowns."""
import torch
torch.set_num_threads(1)
x=torch.arange(32*1024*1024,device='cuda',dtype=torch.float32)
y=torch.empty_like(x)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
torch.add(x,1,out=y)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print('logical_read_bytes',x.numel()*x.element_size(),'logical_write_bytes',y.numel()*y.element_size())

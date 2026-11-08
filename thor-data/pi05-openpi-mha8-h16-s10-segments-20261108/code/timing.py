"""Whole-segment timing. Eviction is once at entry, never between its kernels."""
import statistics
import torch
import triton
import triton.language as tl
from pmu import PerfGroup,groups


@triton.jit
def evict_kernel(src,dst,N:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(src+offsets,offsets<N,other=0.,cache_modifier='.ca')
    tl.store(dst+tl.program_id(0),tl.sum(values,0))


def stats(values):
    x=sorted(values)
    def q(p):
        k=(len(x)-1)*p;i=int(k)
        return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(k-i)
    return dict(n=len(x),mean=statistics.mean(x),p50=q(.5),p95=q(.95),minimum=min(x),maximum=max(x))


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,(tuple,list)):
        return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class Eviction:
    def __init__(self):
        self.bytes=256*1024**2
        self.n=self.bytes//4
        self.source=torch.arange(self.n,device='cuda',dtype=torch.float32)
        self.partial=torch.empty(triton.cdiv(self.n,4096),device='cuda')
        self.l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size
        self.run()
        torch.cuda.synchronize()
        if self.bytes<8*self.l2_bytes:
            raise RuntimeError('Eviction footprint too small')

    def run(self):
        evict_kernel[(triton.cdiv(self.n,4096),)](self.source,self.partial,self.n,4096)


class TimedGraph:
    def __init__(self,fn):
        self.fn=fn
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.start=torch.cuda.Event(enable_timing=True,external=True)
        self.end=torch.cuda.Event(enable_timing=True,external=True)
        self.start.record();self.end.record();self.end.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.start.record()
            self.output=fn()
            self.end.record()
        self.graph.replay()
        self.end.synchronize()

    def run(self):
        self.graph.replay()
        return self.output

    def milliseconds(self):
        self.end.synchronize()
        value=self.start.elapsed_time(self.end)
        if value<0 or value>1000:
            raise RuntimeError('Invalid graph event interval')
        return value


def qualify_eviction(eviction,guard):
    """Read-byte probe distinguishes one entry eviction from per-kernel eviction."""
    guard('cache_probe')
    x=torch.arange(256*1024,device='cuda',dtype=torch.float32)
    y=torch.empty_like(x);z=torch.empty_like(x)
    def first():return torch.add(x,1,out=y)
    def second():return torch.mul(y,2,out=z)
    def both():first();return second()
    whole=TimedGraph(both)
    one=TimedGraph(first);two=TimedGraph(second)
    result=dict(input_bytes=x.numel()*x.element_size(),eviction_bytes=eviction.bytes,l2_bytes=eviction.l2_bytes,
        flush_load_cache_modifier='.ca',samples=[],scope='System-wide non-CPU UCF DRAM; qualification only, not timing runs')
    with PerfGroup(groups()['ucf_noncpu']) as counter:
        def counted(fn):
            counter.start();fn();torch.cuda.synchronize();v=counter.stop()
            if v['running_fraction']<.99:raise RuntimeError('Multiplexed probe counter')
            return v
        for i in range(3):
            guard('eviction_calibration_'+str(i))
            torch.cuda.synchronize()
            values=counted(eviction.run)
            ratio=values['values']['local_dram_byte_rd']/eviction.bytes
            if not .85<=ratio<=1.15:raise RuntimeError('Eviction read-byte calibration failed: '+str(ratio))
            result.setdefault('calibration',[]).append(dict(ratio=ratio,pmu=values))
        for i in range(5):
            guard('cache_probe_repeat_'+str(i))
            for _ in range(5):whole.run()
            torch.cuda.synchronize()
            warm=counted(whole.run)
            eviction.run();torch.cuda.synchronize()
            entry=counted(whole.run)
            eviction.run();torch.cuda.synchronize()
            a=counted(one.run)
            eviction.run();torch.cuda.synchronize()
            b=counted(two.run)
            result['samples'].append(dict(repeat=i,reused=warm,entry_once=entry,per_kernel=[a,b]))
    def median(mode):
        return statistics.median(sum(p['values']['local_dram_byte_rd'] for p in row[mode]) if mode=='per_kernel'
            else row[mode]['values']['local_dram_byte_rd'] for row in result['samples'])
    result['median_read_bytes']={mode:median(mode) for mode in ('reused','entry_once','per_kernel')}
    r=result['median_read_bytes'];n=result['input_bytes']
    if not (.7*n<=r['entry_once']<=1.3*n and 1.5*n<=r['per_kernel']<=2.6*n and r['reused']<.3*n):
        raise RuntimeError('Cache regimes not distinguished: '+str(r))
    if not torch.equal(z,(x+1)*2):raise RuntimeError('Probe output differs')
    result['passed']=True
    result['note']='Empirical streaming eviction proxy, not architectural invalidation or measured physical PIM handoff.'
    return result


def measure_case(case,eviction,samples,guard):
    fn=case['fn'];expected=case['expected']
    guard(case['id'])
    for _ in range(4):value=fn()
    torch.cuda.synchronize()
    if not equal(value,expected):raise RuntimeError('Native segment output mismatch: '+case['id'])
    graph=TimedGraph(fn)
    if not equal(graph.output,expected):raise RuntimeError('Graph output mismatch: '+case['id'])
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();b.record();b.synchronize()
    results={k:[] for k in ('eager_reused','eager_entry_cold','graph_reused','graph_entry_cold')}
    # Reverse order on alternate repetitions to reduce systematic drift.
    modes=list(results)
    for i in range(samples):
        for mode in modes if i%2==0 else list(reversed(modes)):
            engine,cold=mode.startswith('graph'),mode.endswith('entry_cold')
            if cold:
                eviction.run();torch.cuda.synchronize()
            else:
                (graph.run if engine else fn)();torch.cuda.synchronize()
            if engine:
                graph.run();ms=graph.milliseconds()
            else:
                a.record();value=fn();b.record();b.synchronize();ms=a.elapsed_time(b)
            results[mode].append(ms)
    if not equal(graph.output,expected) or not equal(value,expected):
        raise RuntimeError('Segment output changed during measurements: '+case['id'])
    return dict(id=case['id'],family=case['family'],suite=case['suite'],layer=case.get('layer'),
        inputs=case.get('inputs'),outputs=case.get('outputs'),note=case.get('note',''),
        bitwise_equal=True,evictions_per_cold_sample=1,internal_evictions=0,
        measurements={k:dict(samples_ms=v,summary_ms=stats(v)) for k,v in results.items()})

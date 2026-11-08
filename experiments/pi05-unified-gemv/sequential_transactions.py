"""Continuous synthetic addresses, with optional LPDDR5X GPU row state."""
import ctypes as C
from online_transactions import OnlineEngine,CK,U

class RowStats(C.Structure):
    _fields_=[(k,C.c_int64) for k in ('act1','act2','prepb','preab','reads','writes','row_hits','row_misses')]

class RowTrace(C.Structure):
    _fields_=[(k,C.c_int64) for k in ('time','command','bank','row','done')]

class SequentialEngine(OnlineEngine):
    def __init__(self,library,rows=True):
        super().__init__(library);self.rows=rows
        self.lib.sequential_enable.argtypes=[C.c_int,C.c_int]
        self.lib.sequential_plan.argtypes=[C.c_int]
        self.lib.sequential_address.argtypes=[C.c_int,C.c_int,C.c_int64]
        self.lib.sequential_stats.argtypes=[C.c_int,C.POINTER(RowStats)]
        self.lib.sequential_trace.argtypes=[C.c_int,C.POINTER(RowTrace),C.c_int]
    def program(self,*args):
        p=super().program(*args);self.check(self.lib.sequential_enable(p,int(self.rows)));return p
    def stream_plan(self,*args):
        h,r=super().stream_plan(*args);self.check(self.lib.sequential_plan(h));return h,r
    def stats(self,context):
        r=RowStats();self.check(self.lib.sequential_stats(context,C.byref(r)))
        return {k:getattr(r,k) for k,_ in r._fields_}
    def address(self,plan,index,sector):self.check(self.lib.sequential_address(plan,index,sector))
    def trace(self,context):
        n=self.check(self.lib.sequential_trace(context,None,0));a=(RowTrace*n)()
        self.check(self.lib.sequential_trace(context,a,n))
        return [{k:getattr(x,k) for k,_ in RowTrace._fields_} for x in a]

import ctypes as C
from uniform_transactions import Engine,U,CK

class Job(C.Structure):
    _fields_=[(k,C.c_int32) for k in ('channel','rank','bg','write','array_access')]+[(k,C.c_int64) for k in ('ready','ca_length','dq_latency','dq_length')]

class OnlineResult(C.Structure):
    _fields_=[(k,C.c_int64) for k in ('finish','first_issue','requests','memory_stall','pim_end','pim_wait','pim_requests')]

class OnlineEngine(Engine):
    def __init__(self,library):
        super().__init__(library)
        self.lib.online_add.argtypes=[C.c_int64,C.c_int,C.c_int,C.POINTER(Job),C.c_int]
        self.lib.online_begin.argtypes=[C.c_int,C.c_int]
        self.lib.online_clone.argtypes=[C.c_int]
        self.lib.online_run.argtypes=[C.c_int,C.c_int,C.c_int64,C.POINTER(OnlineResult)]
        self.lib.online_reference.argtypes=[C.c_int,C.c_int,C.POINTER(OnlineResult)]
        self.lib.online_advance.argtypes=[C.c_int,C.c_int64,C.POINTER(OnlineResult)]
        self.lib.online_drain.argtypes=[C.c_int,C.POINTER(OnlineResult)]
    def program(self,end,mode,refresh_calendar,jobs):
        data=(Job*len(jobs))(*(Job(*j) for j in jobs))
        return self.check(self.lib.online_add(end,mode,refresh_calendar,data,len(data)))
    def begin(self,program,ch):return self.check(self.lib.online_begin(program,ch))
    def clone(self,context):return self.check(self.lib.online_clone(context))
    def execute(self,context,plan,start):
        r=OnlineResult();self.check(self.lib.online_run(context,plan,start,C.byref(r)));return r
    def reference(self,program,plan):
        r=OnlineResult();h=self.check(self.lib.online_reference(program,plan,C.byref(r)));return h,r
    def advance(self,context,until):
        r=OnlineResult();self.check(self.lib.online_advance(context,until,C.byref(r)));return r
    def drain(self,context):
        r=OnlineResult();self.check(self.lib.online_drain(context,C.byref(r)));return r

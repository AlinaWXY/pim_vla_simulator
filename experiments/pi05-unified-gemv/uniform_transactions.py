"""ctypes bindings for the evidence-sized uniform transaction model."""
import ctypes as C
from pathlib import Path

U = 16_000_000
CK = 15008


class Event(C.Structure):
    _fields_ = [(k,C.c_int32) for k in ('channel','kind','rank','bg')] + [
                ('a',C.c_int64),('b',C.c_int64)]


class Result(C.Structure):
    _fields_ = [(k,C.c_int64) for k in ('finish','first_issue','requests','ca_delay',
        'dq_delay','refresh_delay','backpressure','max_inflight','solo_minimum','calibrated_span',
        'last_ca','last_dq_a','last_dq_b','last_write','last_bg')]


class Engine:
    def __init__(self, library):
        self.equivalence={}
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.uniform_error.restype = C.c_char_p
        self.lib.uniform_add.argtypes = [C.c_int64,C.c_int,C.POINTER(Event),C.c_int]
        self.lib.uniform_plan.argtypes = [C.c_int64,C.c_int64,C.c_int64,C.POINTER(Result)]
        self.lib.uniform_stream_plan.argtypes = [C.c_int64,C.c_int64,C.c_int64,C.POINTER(Result)]
        self.lib.uniform_run.argtypes = [C.c_int,C.c_int,C.c_int64,C.c_int,C.POINTER(Result),C.POINTER(Result)]
        self.lib.uniform_probe.argtypes = [C.c_int,C.c_int64,C.c_int,C.c_int,C.c_int,C.POINTER(Result)]
        self.lib.uniform_capacity.argtypes = [C.c_int,C.c_int,C.c_int,C.POINTER(Result)]

    def check(self, value):
        if value < 0:
            raise ValueError(self.lib.uniform_error().decode())
        return value

    def add(self, end, events, channels=16):
        data = (Event*len(events))(*(Event(*e) for e in events))
        handle=self.check(self.lib.uniform_add(end,channels,data,len(data)))
        per=[[] for _ in range(channels)]
        for e in events:per[e[0]].append(tuple(e[1:]))
        groups={};classes=[]
        for ch,items in enumerate(per):
            key=tuple(sorted(items))
            classes.append(groups.setdefault(key,ch))
        self.equivalence[handle]=classes
        return handle

    def plan(self, duration, reads, writes):
        r = Result()
        handle = self.check(self.lib.uniform_plan(duration,reads,writes,C.byref(r)))
        return handle,r

    def stream_plan(self, duration, reads, writes):
        r=Result()
        handle=self.check(self.lib.uniform_stream_plan(duration,reads,writes,C.byref(r)))
        return handle,r

    def run(self, calendar, plan, start=0, channel=0, previous=None):
        r = Result()
        self.check(self.lib.uniform_run(calendar,plan,start,channel,
                   C.byref(previous) if previous is not None else None,C.byref(r)))
        return r

    def probe(self, calendar, ready=0, write=False, bg=0, channel=0):
        r = Result()
        self.check(self.lib.uniform_probe(calendar,ready,int(write),bg,channel,C.byref(r)))
        return r

    def capacity(self, calendar, write=False, channel=0):
        r=Result()
        self.check(self.lib.uniform_capacity(calendar,int(write),channel,C.byref(r)))
        return r

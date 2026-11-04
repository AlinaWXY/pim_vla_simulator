"""Verify a fresh dedicated-bank FFN run against published native evidence."""
import argparse
import hashlib
import json
from pathlib import Path

def read(path): return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()

def verify(run, mode):
    expected=read(Path(__file__).resolve().parent/f"reference/ffn-bank-split/{mode}/expected.json")
    summary=read(run/"summary.json")
    assert summary["status"]=="passed"
    assert summary["ffn_data_banks"]==expected["ffn_data_banks"] and summary["ffn_overlap"] is expected["ffn_overlap"]
    checked=0
    for case,reference in expected["cases"].items():
        actual=summary["runs"][case]
        for key in ("memory_cycles","latency_us","external_DQ_bytes","trace_sha256","phase_us"):
            assert actual[key]==reference[key],(case,key)
        audit=read(run/case/"native-audit.json")
        assert audit["status"]==audit["split_activation"]["status"]=="passed"
        assert audit.get("ffn_split"),(case,"FFN capacity/dataflow audit missing")
        assert audit["native_sha256"]==reference["native_sha256"],(case,"native hash inventory")
        assert digest(run/case/"workload.trace")==reference["trace_sha256"]
        for name,sha in reference["native_sha256"].items():
            assert digest(run/case/"simulation"/name)==sha,(case,name)
            checked+=1
    return dict(status="passed",mode=mode,exact_cycles_traces_phases_and_native_files=True,raw_files_checked=checked)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--mode",choices=("db1","db2","no-overlap1","no-overlap2"),required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.run,args.mode),indent=2))

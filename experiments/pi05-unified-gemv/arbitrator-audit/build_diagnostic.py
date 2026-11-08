"""Observational instrumentation of the corrected causal arbiter; no scheduling changes."""
import shutil,subprocess,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);source=root/'experiments/pi05-unified-gemv'
out.mkdir(parents=True,exist_ok=True)
for name in ('audit_types.h','audit_impl.h'):
    shutil.copyfile(Path(__file__).resolve().parent/name,out/name)
for name in ('uniform_transactions.cpp','sequential_rows.h','sequential_timing.h'):
    shutil.copyfile(source/name,out/name)
code=(source/'sequential_transactions.cpp').read_text()
needle='struct OnlineState {'
assert code.count(needle)==1
code=code.replace(needle,'#include "audit_types.h"\n'+needle)
needle='bool rows_enabled=false; Rows rows;'
assert code.count(needle)==1
code=code.replace(needle,needle+' AuditState audit;')
needle='static void row_sync(OnlineState& s,I t);'
code=code.replace(needle,needle+'\nstatic void audit_observe(OnlineState&,const OnlineJob&,I,bool);')
needle='    s.dq.insert(at,q);\n  }\n}\n#include "sequential_rows.h"'
assert code.count(needle)==1
code=code.replace(needle,'    s.dq.insert(at,q);\n  }\n  audit_observe(s,j,t,gpu);\n}\n#include "audit_impl.h"\n#include "sequential_rows.h"')
(out/'sequential_transactions.cpp').write_text(code)
subprocess.run(['g++','-std=c++20','-O2','-shared','-fPIC',str(out/'sequential_transactions.cpp'),
                '-o',str(out/'libinsertiondiag.so')],check=True)

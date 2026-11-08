"""Build, test, and reproduce the xinyao_arbitrator snapshot on Linux (CPU only)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,default=Path('outputs/arbitrator'))
    parser.add_argument('--cases',default='cdefg')
    parser.add_argument('--tests-only',action='store_true')
    parser.add_argument('--audit-e',action='store_true',help='Also reproduce all-channel e DQ insertion statistics')
    args=parser.parse_args();root=args.root.resolve()
    output=args.output.resolve() if args.output.is_absolute() else root/args.output
    output.mkdir(parents=True,exist_ok=True)
    if set(args.cases)-set('cdefg') or not args.cases:parser.error('--cases must contain c/d/e/f/g')
    if args.audit_e and 'e' not in args.cases:parser.error('--audit-e requires case e')
    src=root/'experiments/pi05-unified-gemv';library=output/'libsequential.so'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(command,log=None):
        print('RUN',*map(str,command),flush=True)
        if log:
            with log.open('w') as stream:subprocess.run(list(map(str,command)),cwd=root,env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        else:subprocess.run(list(map(str,command)),cwd=root,env=env,check=True)
    run(['g++','-std=c++20','-O2','-shared','-fPIC',src/'sequential_transactions.cpp','-o',library])
    env['UNIFORM_LIBRARY']=str(library);env['NATIVE_TIMING_AUDIT']=str(output/'native-transfer-matrix.json')
    # The native differential test also invokes g++ for its test-only bridge.
    run([sys.executable,'-m','unittest','discover','-s',src/'validation','-p','test_*transactions.py','-v'],
        output/'unit-tests.log')
    if args.tests_only:return
    run([sys.executable,src/'run_sequential_transactions.py','--root',root/'reference/evaluation-1108',
        '--library',library,'--output',output/'with-rows','--cases',args.cases])
    run([sys.executable,src/'verify_arbitrator.py','--root',root,'--results',output/'with-rows'])
    if args.audit_e:
        audit=src/'arbitrator-audit';audit_out=output/'e-dq-insertion-audit'
        run([sys.executable,audit/'build_diagnostic.py',root,audit_out])
        run([sys.executable,audit/'replay_insertions.py',root,audit_out,'--reference-run',output/'with-rows'])
        run([sys.executable,src/'verify_arbitrator.py','--root',root,'--audit',audit_out])


if __name__=='__main__':main()

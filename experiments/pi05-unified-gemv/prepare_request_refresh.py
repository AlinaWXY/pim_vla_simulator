"""Bind per-rank refresh epochs to the existing all-channel native trace hashes."""
import argparse,csv,hashlib,json
from pathlib import Path
import analyze_evaluation as A
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cases={};sources={}
    for v in A.VARIANTS:
        for c in A.CASES:
            d=a.runs/v/c;r=A.read(d/'simulation/result.json');ratio=r['config']['frontend_ratio'];groups={}
            for ch in range(16):
                path=d/f'simulation/rank_ca_events.csv.ch{ch}';events=[]
                for row in csv.DictReader(path.open()):
                    if row['scope']=='global' and row['command'] in ('REFab','REFpb'):
                        assert row['command']=='REFab','per-bank traces require bank-specific blackouts'
                        start=(int(row['clock'])-1)*ratio+1;rank=int(row['Rank']);assert rank in (0,1)
                        events.append([rank,start,start+r['resolved_timings_ck']['nRFC']*ratio])
                groups[str(ch)]=events;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            cases[v+'/'+c]=groups
    a.output.parent.mkdir(parents=True,exist_ok=True);A.dump(a.output,dict(cases=cases,source_sha256=sources,
        note='REFab blocks only its target rank; shared CA pulse is recorded separately. No per-bank traces present.'))
    print('Bound refresh epochs for',len(cases),'cases and',len(sources),'channel logs',flush=True)
if __name__=='__main__':main()

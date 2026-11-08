"""Saturated one-direction row-hit bus capacity; NOT GPU kernel progress."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from uniform_transactions import Engine,U


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'uniform-transactions'
    e=Engine(out/'libuniform-stream.so')
    with gzip.open(out/'native-reservations.json.gz','rt') as f:t=json.load(f)
    with gzip.open(a.root/'corrected-overlap/native-bus-templates.json.gz','rt') as f:s=json.load(f)
    rows=[]
    for v in ('token_local_before','token_local_after'):
        for case in ('chain','pv','gate_up_gpu','down_gpu_input'):
            key=v+'/'+case;stage=s[key];duration=stage['wall_ticks']*469
            cal=e.add(duration,t[key]['events'])
            row=dict(variant=v,case=case,duration_us=duration/U)
            for write,label in ((False,'read_only'),(True,'write_only')):
                count=sum(e.capacity(cal,write,ch).requests for ch in range(16))
                row[label+'_burst_capacity']=count
                row[label+'_GiB_per_s']=count*32/(duration/U/1e6)/1024**3
            rows.append(row);print(row,flush=True)
    with (out/'saturated-bus-capacity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()

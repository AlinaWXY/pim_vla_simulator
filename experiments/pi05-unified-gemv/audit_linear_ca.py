"""Audit existing PIM traces; no synthetic GPU requests or latency estimate.

Both barrier phases and first-to-last-MAC issue envelopes are retained. The
envelope ends one CK after the last MAC issue: it excludes the unknown service
tail, and must not be described as the full operator execution time.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from channel_analysis import dq_intervals, issue_tick, read_csv, read_json


def union(items, begin, end):
    out = []
    for a, b in sorted((max(begin, a), min(end, b)) for a, b in items):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def statistics(items, begin, end, ratio, tick_ns):
    merged = union(items, begin, end)
    gaps, cursor = [], begin
    for a, b in merged:
        if cursor < a:
            gaps.append(a - cursor)
        cursor = b
    if cursor < end:
        gaps.append(end - cursor)
    busy = sum(b - a for a, b in merged)
    assert busy + sum(gaps) == end - begin
    ordered = sorted(gaps)
    def quantile(p):
        return ordered[min(len(ordered)-1, int((len(ordered)-1)*p))] / ratio if ordered else 0
    return dict(duration_us=(end-begin)*tick_ns/1000,
                busy_us=busy*tick_ns/1000, busy_percent=100*busy/(end-begin),
                idle_count=len(gaps), idle_p50_ck=quantile(.5), idle_p90_ck=quantile(.9),
                idle_p99_ck=quantile(.99), idle_max_ck=max(gaps, default=0)/ratio,
                busy_run_max_ck=max((b-a for a, b in merged), default=0)/ratio,
                common_idle_gaps_ck=Counter(g/ratio for g in gaps).most_common(6))


def operator(name):
    if 'projection_local_input_compute' in name or name == 'qkv_run_local_spills': return 'QKV'
    if 'o_projection_tile_' in name and name.endswith('_compute') or name == 'o_run_two_tiles_with_direct_export': return 'O'
    if name.endswith('pv_local_input_compute') or name == 'pv_run_local_spill': return 'PV'
    if name.endswith('gate_up_local_input_and_compute') or name == 'gate_up_run_with_direct_export': return 'Gate_Up'
    if name.endswith('down_local_input_and_compute') or name == 'down_run_with_direct_export': return 'Down'
    return None


def audit(directory, variant, records, details, sources, clips):
    m, r = read_json(directory/'manifest.json'), read_json(directory/'simulation/result.json')
    ratio, tick_ns = r['config']['frontend_ratio'], r['tCK_ns']/r['config']['frontend_ratio']
    markers = {int(x['marker']): int(x['frontend_tick']) for x in read_csv(directory/'simulation/barriers.csv')}
    phases = [(s, operator(s['name'])) for s in m['segments'] if operator(s['name'])]
    paths = [directory/'manifest.json', directory/'simulation/result.json', directory/'simulation/barriers.csv']
    for ch in range(r['config']['channels']):
        cp, tp = (directory/f'simulation/{name}.csv.ch{ch}' for name in ['rank_ca_events', 'timeline'])
        paths.extend([cp, tp])
        ca, timeline = read_csv(cp), read_csv(tp)
        resources = {k: [] for k in ['global_ca', 'rank0_ca', 'rank1_ca', 'dq']}
        for e in ca:
            assert int(e['Channel']) == ch
            a, b = issue_tick(e['clock'], ratio), issue_tick(e['ready_at'], ratio)
            keys = ['global_ca'] if e['scope'] == 'global' else [f'rank{i}_ca' for i in ([0, 1] if int(e['Rank']) == -1 else [int(e['Rank'])])]
            for k in keys: resources[k].append((a, b))
        for k, items in resources.items():
            items.sort()
            assert all(b <= c for (a,b),(c,d) in zip(items, items[1:])), (variant,ch,k)
        resources['dq'] = [(x['begin'],x['end']) for x in dq_intervals(timeline, r['resolved_timings_ck'], ratio, r['config']['pim_data_path'])]
        resources['dq_global_ca'] = resources['dq'] + resources['global_ca']
        macs = [(issue_tick(x['clock'],ratio),x['command'],int(x['Rank'])) for x in timeline if x['command'] in ('MACAB','MACQK','MACK')]
        for s, op in phases:
            begin, end = markers[s['start']], markers[s['end']]
            selected = [(t,c,rk) for t,c,rk in macs if begin <= t < end]
            assert selected, (variant,ch,s['name'])
            intervals = [('phase',begin,end), ('mac_issue_envelope',selected[0][0],selected[-1][0]+ratio)]
            for scope, left, right in intervals:
                for resource, items in resources.items():
                    stats = statistics(items,left,right,ratio,tick_ns)
                    identity = dict(variant=variant,channel=ch,operator=op,phase=s['name'],scope=scope,resource=resource,
                                    begin_ck=(left-1)/ratio+1,end_ck=(right-1)/ratio+1,mac_count=len(selected),tCK_ns=r['tCK_ns'])
                    details.append(dict(identity,**stats))
                    records.append(dict(identity,**{k:v for k,v in stats.items() if k!='common_idle_gaps_ck'},
                                        common_idle_gaps_ck=json.dumps(stats['common_idle_gaps_ck'])))
            if ch == 0:
                a, b = selected[0][0], selected[-1][0]+ratio
                # Retain an exact 512 CK sample starting 1/4 through the MAC envelope.
                left = ((a+(b-a)//4-1)//ratio)*ratio+1
                right = min(left+512*ratio,b)
                key = f'{variant}/{op}'
                if key not in clips:
                    clips[key] = dict(variant=variant,operator=op,phase=s['name'],origin_ck=(left-1)/ratio+1,
                                      length_ns=(right-left)*tick_ns,
                                      resources={k:[((x-left)*tick_ns,(y-x)*tick_ns) for x,y in union(items,left,right)]
                                                 for k,items in resources.items() if k!='dq_global_ca'})
    sources[variant] = dict(source=str(directory.resolve()),config=r['config'],tCK_ns=r['tCK_ns'],
        sha256={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def plot(clips, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5,2,figsize=(12,9),sharex=True,sharey=True)
    labels = ['Global CA','Rank 0 CA','Rank 1 CA','DQ']
    resources = ['global_ca','rank0_ca','rank1_ca','dq']
    colors = ['#bc3e35','#476dad','#7487b1','#bd8b27']
    for i, op in enumerate(['QKV','PV','O','Gate_Up','Down']):
        for j, v in enumerate(['token_local_before','token_local_after']):
            c = clips[f'{v}/{op}']; ax=axes[i,j]
            for k, resource in enumerate(resources):
                ax.broken_barh(c['resources'][resource],(3-k-.3,.6),facecolors=colors[k])
            ax.set_title(f"{op} | {'e: before' if j==0 else 'g: after'} | CK {c['origin_ck']:.0f}",fontsize=10)
            ax.set_yticks([3,2,1,0],labels);ax.set_xlim(0,512*.938);ax.grid(axis='x',alpha=.2)
            if i==4: ax.set_xlabel('Time from sample start (ns)')
    fig.suptitle('Native PIM resource reservations during MAC issue envelopes (channel 0)',fontsize=13)
    fig.text(.5,.015,'Exact 512 CK samples; global CA idle is not proof of GPU memory availability. No GPU traffic is generated.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.97])
    fig.savefig(output/'linear-ca-samples.png',dpi=180);fig.savefig(output/'linear-ca-samples.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plot',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records, details, sources, clips = [], [], {}, {}
    for v in ['token_local_before','token_local_after']:
        audit(a.input/v/'serial_layer_control',v,records,details,sources,clips)
    with (a.output/'per-operator-ca.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    result=dict(definition='Native PIM-only trace accounting. No GPU requests, pressure, queue-depth, or read/write ratio.',
                native_rerun=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                envelope='First MAC issue through last MAC issue + 1 CK within each semantic phase. Service tail excluded. Internal input stalls remain included. O after includes inter-tile output.',
                sources=sources,measurements=details,samples=clips)
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not a.no_plot: plot(clips,a.output)
    for x in details:
        if x['channel']==0 and x['scope']=='mac_issue_envelope' and x['resource'] in ['global_ca','dq_global_ca']:
            print(json.dumps(x,ensure_ascii=False))


if __name__=='__main__':main()

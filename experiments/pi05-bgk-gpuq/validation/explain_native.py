"""Report actual command cadence, Q read/relay delays and fresh controls."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main(root):
    case_data = {}
    for name in ('qkv','q','k','v','qk','pv'):
        directory = root/name
        result = load(directory/'simulation/result.json')
        channels = []
        for c in range(16):
            native = read(directory/'simulation'/f'timeline.csv.ch{c}')
            events = read(directory/'simulation'/f'bgk_events.csv.ch{c}')
            macs = [e for e in native if e['command'] in ('MACAB','MACQK')]
            cadence = Counter(int(b['clock'])-int(a['clock']) for a,b in zip(macs,macs[1:])
                              if a['Row']==b['Row'])
            reads = [e for e in events if e['operation']=='PIM_RD_Q']
            relays = [e for e in events if e['operation']=='PIM_RELAY_Q']
            assert len(reads) == len(relays)
            delay = Counter(int(b['clock'])-int(a['ready_at']) for a,b in zip(reads,relays))
            channels.append(dict(channel=c,native_commands=dict(Counter(e['command'] for e in native)),
                MAC_count=len(macs),same_row_MAC_interval_CK_histogram=dict(sorted(cadence.items())),
                Q_reads=len(reads),Q_relays=len(relays),Q_read_payload_bytes=len(reads)*32,
                Q_relay_payload_bytes=len(relays)*32,Q_relay_fanout_delivered_bytes=len(relays)*32*8,
                Q_ready_to_relay_issue_CK_histogram=dict(sorted(delay.items()))))
        case_data[name] = dict(latency_us=result['latency_ns']/1000,channels=channels)
    summaries = {p.name:load(p/'summary.json') for p in root.iterdir()
                 if p.is_dir() and (p/'summary.json').exists()}
    controls = {name:row['latency_us'] for name,row in summaries.items() if 'latency_us' in row}
    output = dict(cases=case_data, measured_controls_us=controls,
        QKV_single_minus_double_us=controls['qkv_serial']-controls['qkv'],
        QK_single_minus_double_us=controls['qk_serial']-controls['qk'],
        QKV_QK_separate_sum_us=controls['qkv']+controls['qk'],
        QKV_QK_chain_us=controls.get('chain_qkv_qk'),
        explanation='Q read and relay are two native commands joined by completion dependency. One 32B controller slot per channel serializes these transfers. QK reads full post-RoPE Q from DRAM once per stage; GPU arithmetic is not measured. Isolated Q/K/V controls are not additive components of fused QKV.')
    (root/'native-diagnostics.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='cases'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)

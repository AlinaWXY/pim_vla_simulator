"""Verify distribution integrity without importing Torch or using a GPU."""
import argparse
import gzip
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references=[]
        self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag=='a' and 'href' in attrs:
            self.references.append(attrs['href'])
        if tag=='img':
            assert attrs.get('alt')
            self.references.append(attrs['src'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.root
    data=json.loads((root/'analysis/summary.json').read_text())
    b,f=data['baseline'],data['fine']
    assert data['configuration']['H']==16 and data['configuration']['steps']==10
    assert data['configuration']['q_heads']==data['configuration']['kv_heads']==8
    assert data['configuration']['adarms_cache'] is True
    for relative,expected in data['provenance'].items():
        path=root/'raw'/relative
        payload=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
        assert hashlib.sha256(payload).hexdigest()==expected,relative
    for tag in ('pi05_h16_latency_baseline_20261105_01','pi05_h16_latency_trace_20261105_01'):
        capture=json.loads((root/'raw'/tag/'capture.json').read_text())
        for name,expected in capture['environment']['code_sha256'].items():
            assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==expected,name
    assert math.isclose(sum(b['phase_share_pct'].values()),100,abs_tol=1e-8)
    assert math.isclose(sum(f['family_active_share_pct'].values()),100,abs_tol=1e-8)
    assert math.isclose(sum(f['category_active_share_pct'].values()),100,abs_tol=1e-8)
    assert f['matched_steps']==50 and f['kernels_per_step']==1033
    for s in json.loads((root/'analysis/step_samples.json').read_text()):
        assert math.isclose(sum(s['category_ms'].values()),s['active_ms'],abs_tol=1e-7)
        assert math.isclose(s['active_ms']+s['gap_ms'],s['event_ms'],abs_tol=1e-7)
    report=Links()
    report.feed((root/'figures/index.html').read_text())
    for ref in report.references:
        if ref.startswith('#'):
            assert ref[1:] in report.ids,ref
        else:
            assert (root/'figures'/ref).is_file(),ref
    images=[]
    for image in sorted((root/'figures').glob('*.png')):
        raw=image.read_bytes()
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',raw[16:24])
        assert width>500 and height>300 and len(raw)>10000
        images.append(dict(file=image.name,width=width,height=height))
    assert len(images)==5
    for image in (root/'figures').glob('*.svg'):
        ET.parse(image)
    result=dict(status='passed',configuration_verified=True,input_hashes_verified=True,
        capture_source_hashes_verified=True,all_percentages_reconcile=True,
        per_step_partition_reconciles=True,html_references_resolve=True,images=images,
        live_browser_layout_verified=False,
        note='Static PNG figures reviewed visually; HTML links/structure checked without a browser renderer.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

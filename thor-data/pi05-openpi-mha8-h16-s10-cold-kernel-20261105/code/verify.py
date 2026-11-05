"""CPU-only checks for the final cold-kernel publication bundle."""
import argparse
import gzip
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from analyze import audit_probe


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs=[]
        self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        if tag=='a':self.refs.append(attrs.get('href',''))
        if tag=='img':
            assert attrs.get('alt')
            self.refs.append(attrs['src'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args()
    root=ns.root
    d=json.loads((root/'analysis_verified/summary.json').read_text())
    assert d['configuration']['H']==16 and d['configuration']['steps']==10
    assert d['configuration']['q_heads']==d['configuration']['kv_heads']==8
    assert d['configuration']['adarms_cache']
    assert d['kernel_count']==3099 and d['kernels_per_repeat']==[1033]*3
    assert d['pass_counts']=={'1':3099}
    assert not d['validation']['warm_signature_mismatches']
    assert audit_probe(root/'raw')['passed']
    total=d['per_step_sum_kernel_ms']['mean']
    for field in ('category_ms','family_ms'):
        assert math.isclose(sum(v['mean'] for v in d[field].values()),total,abs_tol=1e-8)
    for relative,expected in d['input_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
    for path in (root/'raw').glob('*/capture.json'):
        c=json.loads(path.read_text())
        assert c['status']=='complete' and c['output_bitwise_equal']
        assert all(not x['others'] for x in c['occupancy'])
        for name,expected in c['environment']['code_sha256'].items():
            assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==expected,name
    parser=Links()
    parser.feed((root/'figures/index.html').read_text())
    for ref in parser.refs:
        assert ref
        if ref.startswith('#'):assert ref[1:] in parser.ids
        else:assert (root/'figures'/ref).is_file(),ref
    images=[]
    for p in sorted((root/'figures').glob('*.png')):
        raw=p.read_bytes()
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',raw[16:24])
        assert width>500 and height>300
        images.append(dict(path=p.name,width=width,height=height))
    assert len(images)==4
    for p in (root/'figures').glob('*.svg'):ET.parse(p)
    reports=[]
    for p in sorted((root/'raw').glob('*.ncu-rep.gz')):
        raw=gzip.decompress(p.read_bytes())
        if p.with_suffix('').exists():assert raw==p.with_suffix('').read_bytes()
        reports.append(dict(path=p.name,uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    assert len(reports)==3
    result=dict(status='passed',source_hashes_verified=True,raw_hashes_verified=True,cache_probe_passed=True,
        all_model_kernels_single_pass=True,warm_kernel_signatures_match=True,kernel_totals_reconcile=True,
        html_links_resolve=True,plots=images,compressed_reports=reports,
        live_browser_layout_verified=False,figure_visual_review='Four static PNG figures visually inspected')
    with ns.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

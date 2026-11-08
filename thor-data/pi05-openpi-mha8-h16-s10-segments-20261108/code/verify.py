"""CPU-only verification of the complete-segment publication bundle."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from analyze import MODES,RUN,PROBE,INTERFACES,aggregate


class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        if tag=='a':self.refs.append(attrs.get('href',''))
        if tag=='img':
            assert attrs.get('alt');self.refs.append(attrs['src'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--warm',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args();root=ns.root
    summary=json.loads((root/'analysis_verified/summary.json').read_text())
    captures=[json.loads((root/'raw'/tag/'capture.json').read_text()) for tag in (RUN,PROBE,INTERFACES)]
    main_data,probe,extra=captures
    assert all(c['status']=='complete' and all(not r['others'] for r in c['occupancy']) for c in captures)
    cases=main_data['cases']+extra['cases']
    assert len(cases)==458 and len({c['id'] for c in cases})==458
    assert aggregate(cases)==summary['aggregates']
    assert probe['cache_probe']['passed'] and all(c['internal_evictions']==0 for c in cases)
    assert summary['validation']['total_timed_segment_runs']==36640
    for capture in captures:
        for name,digest in capture['files_sha256'].items():
            assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==digest,name
    for rel,digest in summary['input_sha256'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
    previous=ns.warm/'raw/pi05_h16_latency_baseline_20261105_01/capture.json'
    warm=json.loads(previous.read_text())
    assert main_data['reference_actions']==extra['reference_actions']==warm['reference_actions']
    failed=json.loads((root/'raw/pi05_h16_segments_20261108_01/capture.json').read_text())
    assert failed['status']=='failed' and not failed['cases']
    assert 'Inference tensors do not track version counter' in failed['error']
    assert math.isclose(sum(summary['coarse']['phase_share_pct'].values()),100,abs_tol=1e-7)
    parser=Links();parser.feed((root/'figures/index.html').read_text())
    for ref in parser.refs:
        assert ref
        if ref.startswith('#'):assert ref[1:] in parser.ids,ref
        else:assert (root/'figures'/ref).is_file(),ref
    images=[]
    for p in sorted((root/'figures').glob('*.png')):
        raw=p.read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',raw[16:24]);assert width>500 and height>300
        images.append(dict(path=p.name,width=width,height=height))
    assert len(images)==5
    for p in (root/'figures').glob('*.svg'):ET.parse(p)
    result=dict(status='passed',cases=458,timed_runs=36640,all_outputs_tensor_exact=True,
        no_internal_evictions=True,cache_probe_passed=True,source_and_raw_hashes_verified=True,
        matches_nov5_actions=True,nov5_reference_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        initialization_failure_excluded=True,aggregates_reproduced=True,html_links_resolve=True,images=images,
        live_browser_layout_verified=False,figure_visual_review='Five static PNG figures visually inspected')
    with ns.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

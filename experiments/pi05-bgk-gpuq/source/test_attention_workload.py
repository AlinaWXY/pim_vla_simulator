"""Ownership, trace traffic and producer/consumer scheduling contracts."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import attention_layout as L
from attention_workload import generate, OPERATORS, k_suffix_bursts


def read_trace(path):
    result = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        result.append((fields[0], int(fields[1]), int(fields[2]) if len(fields) == 3 else None))
    return result


class AttentionWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cases = {}
        for op in (*OPERATORS, 'chain_k_qk'):
            path = Path(cls.temp.name)/(op+'.trace')
            manifest = generate(path, op)
            cls.cases[op] = manifest, read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_payload_counts_and_no_hidden_external_K_or_Q(self):
        expected = {
            'k': dict(PIM_WR_BG=10240, PIM_MAC_K=65536, PIM_LOCAL_WR=2560),
            'v': dict(PIM_WR_GB=20480, PIM_MAC_AB=4096, PIM_RD_RES=5120, LD=2048, ST=4096),
            'qk': dict(PIM_WR_Q=5120, PIM_MAC_QK=3968, PIM_RD_BG=19840),
            'pv': dict(PIM_WR_GB=19840, PIM_MAC_AB=3968, PIM_RD_RES=5120),
        }
        for op, commands in expected.items():
            manifest, trace = self.cases[op]
            actual = Counter(command for command, _, _ in trace)
            self.assertEqual(dict(actual), manifest['counts'])
            self.assertEqual({k:v for k,v in actual.items() if k not in ('BARRIER','WAIT')}, commands)
        self.assertEqual(self.cases['k'][0]['external_input_bytes'], 320*1024)
        self.assertEqual(self.cases['k'][0]['external_output_bytes'], 0)
        self.assertEqual(self.cases['k'][0]['local_k_write_bytes'], 80*1024)
        self.assertEqual(self.cases['v'][0]['external_input_bytes'], 640*1024)
        self.assertEqual(self.cases['qk'][0]['external_input_bytes'], 160*1024)
        self.assertEqual(self.cases['qk'][0]['external_output_bytes'], 620*1024)
        self.assertEqual(self.cases['pv'][0]['external_input_bytes'], 620*1024)
        self.assertEqual(self.cases['pv'][0]['external_output_bytes'], 160*1024)

    def test_K_BG_X_exact_selected_tokens_and_weight_scan(self):
        manifest, trace = self.cases['k']
        for item in manifest['input_fragments']:
            entries = trace[item['line_start']-1:item['line_end']]
            seen = Counter()
            for command, address, buffer in entries:
                c,r,bg,b,offset = L.decode_address(address)
                self.assertEqual(command, 'PIM_WR_BG')
                self.assertEqual((b, buffer), (0, item['fragment']%2))
                queries = L.k_queries_for_group(c%2,r,bg)
                self.assertIn(offset//32, range(len(queries)))
                seen[c,r,bg,queries[offset//32]] += 1
            expected = Counter((c,r,bg,q) for c in range(16) for r in range(2)
                for bg in range(4) for q in L.k_queries_for_group(c%2,r,bg))
            self.assertEqual(seen, expected)
        by_channel = defaultdict(list)
        for command, address, _ in trace:
            if command == 'PIM_MAC_K':
                c,r,bg,b,offset = L.decode_address(address)
                self.assertEqual((r,bg,b), (0,0,0))
                by_channel[c].append(offset)
        self.assertEqual(len(by_channel), 16)
        for offsets in by_channel.values():
            self.assertEqual(offsets, list(range(0,128*1024,32)))

    def test_local_K_write_ownership_alignment_and_one_row(self):
        locations = list(k_suffix_bursts())
        self.assertEqual(len(locations), len(set(locations)))
        expected = {L.key_location(h,L.PREFIX+q,L.k_feature_inverse(b,f*16))
                    for h in range(8) for q in range(20) for b in range(4) for f in range(4)}
        self.assertEqual(set(locations), expected)
        bank_bytes, bank_rows = Counter(), defaultdict(set)
        for c,r,g,b,offset in locations:
            self.assertEqual(offset%32,0)
            bank_bytes[c,r,g,b] += 32
            bank_rows[c,r,g,b].add(offset//2048)
        self.assertEqual(Counter(bank_bytes.values()), Counter({128:384,256:128}))
        self.assertEqual({len(rows) for rows in bank_rows.values()}, {1})
        manifest, trace = self.cases['k']
        last_mac = max(i for i,event in enumerate(trace) if event[0]=='PIM_MAC_K')
        first_write = min(i for i,event in enumerate(trace) if event[0]=='PIM_LOCAL_WR')
        self.assertIn('BARRIER', [e[0] for e in trace[last_mac+1:first_write]])

    def test_Q_multicast_bank_feature_coverage_and_K_scan(self):
        manifest, trace = self.cases['qk']
        seen_features = defaultdict(list)
        for item in manifest['input_fragments']:
            entries = trace[item['line_start']-1:item['line_end']]
            seen = Counter()
            for command,address,buffer in entries:
                c,r,g,b,offset = L.decode_address(address)
                self.assertEqual(command,'PIM_WR_Q')
                self.assertEqual((r,g,buffer),(0,0,item['fragment']%2))
                seen[c,b,offset//32] += 1
                if c==0 and offset==0:
                    seen_features[b].extend(L.q_input_features(b,item['fragment']))
            self.assertEqual(seen,Counter((c,b,q) for c in range(16) for b in range(4) for q in range(20)))
        self.assertEqual(sorted(d for ds in seen_features.values() for d in ds),list(range(256)))
        by_channel = defaultdict(list)
        for command,address,buffer in trace:
            if command == 'PIM_MAC_QK':
                c,r,g,b,offset = L.decode_address(address)
                relative = offset-L.REGIONS['k_cache']['base']
                slot, fragment = divmod(relative//32,4)
                self.assertEqual(buffer,fragment%2)
                self.assertIn(slot,range(62))
                by_channel[c].append((slot,fragment))
        for entries in by_channel.values():
            self.assertEqual(Counter(entries),Counter((s,f) for s in range(62) for f in range(4)))

    def test_BG_export_is_complete_score_and_bounded_tile(self):
        manifest,trace = self.cases['qk']
        exported = defaultdict(list)
        self.assertEqual(len(manifest['exports']),4)
        for item in manifest['exports']:
            self.assertLessEqual(item['maximum_staged_bytes_per_BG'],1280)
            first,last = item['line_start']-1,item['line_end']
            previous_mac=max(i for i,e in enumerate(trace[:first]) if e[0]=='PIM_MAC_QK')
            self.assertIn('BARRIER',[e[0] for e in trace[previous_mac+1:first]])
            for command,address,buffer in trace[first:last]:
                self.assertEqual(command,'PIM_RD_BG')
                c,r,g,b,offset=L.decode_address(address)
                self.assertEqual(b,0)
                exported[c,r,g].append(offset)
        self.assertEqual(len(exported),128)
        for offsets in exported.values():
            self.assertEqual(offsets,list(range(0,62*20*4,32)))
        self.assertEqual(manifest['buffers']['qk_partial_per_bank'],4960)
        self.assertEqual(manifest['buffers']['qk_export_double_per_BG'],2560)

    def test_V_and_PV_preserve_feature_owners_and_fragment_order(self):
        for op, region, fragments in [('v','w_v',64),('pv','v_cache',62)]:
            _,trace=self.cases[op]
            perchannel=defaultdict(list)
            for command,address,buffer in trace:
                if command=='PIM_MAC_AB':
                    c,r,g,b,offset=L.decode_address(address)
                    perchannel[c].append(offset)
                    self.assertEqual((r,g,b),(0,0,0))
                    fragment=(offset-L.REGIONS[region]['base'])//(4*32)
                    self.assertEqual(buffer,fragment%2)
            for offsets in perchannel.values():
                self.assertEqual(offsets,list(range(L.REGIONS[region]['base'],
                    L.REGIONS[region]['base']+fragments*4*32,32)))
        manifest,trace=self.cases['v']
        reads={L.decode_address(a) for op,a,_ in trace if op=='LD'}
        writes={L.decode_address(a) for op,a,_ in trace if op=='ST'}
        self.assertEqual(reads,{L.value_location(h,960,d) for h in range(8) for d in range(256)})
        self.assertEqual(writes,{L.value_location(h,t,d) for h in range(8) for t in (960,976) for d in range(256)})

    def test_chain_write_completion_boundary_and_phase_coverage(self):
        manifest,trace=self.cases['chain_k_qk']
        last_write=max(i for i,e in enumerate(trace) if e[0]=='PIM_LOCAL_WR')
        first_qk=min(i for i,e in enumerate(trace) if e[0]=='PIM_MAC_QK')
        self.assertLess(last_write,first_qk)
        dependency=manifest['dependencies'][0]
        self.assertIn(('BARRIER',dependency['fence'],None),trace[last_write+1:first_qk])
        self.assertFalse(manifest['integrated_numeric_chain'])
        self.assertIsNone(manifest['unknown_latency_us']['K_local_RoPE_cast_pack'])
        for manifest,trace in self.cases.values():
            markers=[value for op,value,_ in trace if op=='BARRIER']
            self.assertEqual(markers,list(range(len(markers))))
            for i,item in enumerate(manifest['segments']):
                self.assertEqual(item['start'],0 if i==0 else manifest['segments'][i-1]['end'])
                self.assertLess(item['start'],item['end'])

    def test_serial_control_preserves_payload_and_queries(self):
        for op in ('k','qk'):
            path=Path(self.temp.name)/(op+'_serial.trace')
            manifest=generate(path,op,schedule='serial')
            trace=read_trace(path)
            baseline=self.cases[op][0]
            self.assertEqual({k:v for k,v in manifest['counts'].items() if k not in ('BARRIER','WAIT')},
                             {k:v for k,v in baseline['counts'].items() if k not in ('BARRIER','WAIT')})
            self.assertEqual({buffer for _,_,buffer in trace if buffer is not None},{0})
            self.assertGreater(manifest['counts']['BARRIER'],baseline['counts']['BARRIER'])


if __name__=='__main__':
    unittest.main()

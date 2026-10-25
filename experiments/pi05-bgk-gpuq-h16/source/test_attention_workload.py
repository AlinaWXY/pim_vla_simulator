"""Executable ownership, fused-input and roundtrip contracts for xinyao_1025, H=16 action horizon (8 heads)."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import attention_layout as L
from attention_workload import generate, OPERATORS, CONTROL_OPERATORS, k_suffix_bursts


def read_trace(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        result.append((fields[0],int(fields[1]),int(fields[2]) if len(fields)==3 else None))
    return result


class AttentionWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.cases={}
        for op in (*OPERATORS,*CONTROL_OPERATORS,'chain_qkv_qk'):
            path=Path(cls.temp.name)/(op+'.trace')
            cls.cases[op]=generate(path,op),read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_H16_shape_service_and_buffer_contract(self):
        layout=L.manifest()
        self.assertEqual(layout['variant'],'H=16')
        self.assertEqual(layout['action_horizon'],16)
        self.assertEqual(layout['shape'],dict(heads=8,head_dim=256,queries=16,prefix=968,sequence=984,hidden=1024))
        self.assertEqual(L.PADDED_TOKENS,992)
        self.assertEqual({name:(r['base'],r['reserved_bytes_per_bank']) for name,r in L.REGIONS.items()},
                         {'w_qkv':(0,24576),'q_cache':(24576,2048),
                          'k_cache':(26624,8192),'v_cache':(34816,8192)})
        self.assertEqual(layout['v_suffix']['zero_padding_bytes'],32768)
        self.assertEqual(layout['buffers']['qk_partial_per_bank'],3968)
        self.assertEqual(layout['buffers']['qk_partial_reserved_per_bank'],4096)
        for op,(manifest,trace) in self.cases.items():
            self.assertEqual(manifest['action_horizon'],16)
            self.assertEqual(manifest['attention_heads'],8)
            if manifest['matrix']:
                self.assertEqual(manifest['matrix']['M'],16)
                self.assertEqual(manifest['matrix']['mac_service_ck'],16)
            for segment in manifest['segments']:
                if segment['kind']=='pipeline':
                    self.assertEqual(segment['mac_service_ck'],16)
            buffers=manifest['buffers']
            self.assertEqual(buffers['q_per_bank'],1024)
            self.assertEqual(buffers['qkv_input_double_per_rank'],1024)
            self.assertEqual(buffers['qkv_projection_accumulator_per_bank'],768)
            self.assertEqual(buffers['single_projection_accumulator_per_bank'],256)
            self.assertEqual(buffers['qk_partial_per_bank'],3968)
            self.assertEqual(buffers['qk_partial_reservation_per_bank'],4096)
            self.assertEqual(buffers['qk_export_double_per_BG'],2048)
            self.assertEqual(buffers['qk_export_tile_per_BG'],1024)
            self.assertEqual(buffers['q_dram_controller_relay_per_channel'],32)

    def test_payload_counts_and_no_unmodeled_Q_GPU_input(self):
        expected={
            'qkv':dict(PIM_WR_GB=16384,PIM_MAC_AB=12288,PIM_RD_RES=12288,LD=2048,ST=10240),
            'q':dict(PIM_WR_GB=16384,PIM_MAC_AB=4096,PIM_RD_RES=4096,ST=4096),
            'k':dict(PIM_WR_GB=16384,PIM_MAC_AB=4096,PIM_RD_RES=4096,ST=2048),
            'v':dict(PIM_WR_GB=16384,PIM_MAC_AB=4096,PIM_RD_RES=4096,LD=2048,ST=4096),
            'qk':dict(PIM_RD_Q=4096,PIM_MAC_QK=3968,PIM_RD_BG=15872),
            'pv':dict(PIM_WR_GB=15872,PIM_MAC_AB=3968,PIM_RD_RES=4096),
        }
        for op,commands in expected.items():
            manifest,trace=self.cases[op]
            actual=Counter(c for c,_,_ in trace)
            self.assertEqual(dict(actual),manifest['counts'])
            self.assertEqual({k:v for k,v in actual.items() if k not in ('BARRIER','WAIT')},commands)
        qkv=self.cases['qkv'][0]
        self.assertEqual(qkv['external_input_bytes'],512*1024)
        self.assertEqual(qkv['external_output_bytes'],384*1024)
        self.assertEqual(qkv['dram_write_bytes'],320*1024)
        self.assertEqual(qkv['explicit_dram_read_bytes'],64*1024)
        qk=self.cases['qk'][0]
        self.assertEqual(qk['external_input_bytes'],0)
        self.assertEqual(qk['q_dram_read_bytes'],128*1024)
        self.assertEqual(qk['q_internal_broadcast_bytes'],128*1024)
        self.assertEqual(qk['q_internal_delivered_bytes'],1024*1024)
        self.assertEqual(qk['external_output_bytes'],496*1024)

    def test_fused_X_is_loaded_once_and_scans_all_three_matrices(self):
        manifest,trace=self.cases['qkv']
        self.assertEqual(len(manifest['input_fragments']),64)
        for item in manifest['input_fragments']:
            f=item['fragment']
            reads=trace[item['line_start']-1:item['line_end']]
            self.assertEqual(len(reads),256)
            self.assertEqual(Counter((L.decode_address(a)[0],L.decode_address(a)[4]//32)
                for c,a,b in reads if c=='PIM_WR_GB'),Counter((c,q) for c in range(16) for q in range(16)))
            macs=trace[item['mac_line_first']-1:item['mac_line_last']]
            self.assertEqual(len(macs),16*12)
            self.assertEqual({c for c,_,_ in macs},{'PIM_MAC_AB'})
            perchannel=defaultdict(list)
            for _,a,b in macs:
                c,r,g,bank,off=L.decode_address(a)
                self.assertEqual((r,g,bank,b),(0,0,0,f%2))
                perchannel[c].append(off)
            for offsets in perchannel.values():
                self.assertEqual(offsets,[32*(12*f+i) for i in range(12)])
        standalone=sum(self.cases[op][0]['external_input_bytes'] for op in ('q','k','v'))
        self.assertEqual(standalone,3*manifest['external_input_bytes'])
        self.assertEqual(manifest['buffers']['qkv_projection_accumulator_per_bank'],768)

    def test_weight_unique_feature_owners_and_numeric_projection(self):
        # One head covers64banks, four features each; no head-weight replication.
        owners=Counter(L.v_feature_owner(0,d)[0] for d in range(256))
        self.assertEqual(len(owners),64)
        self.assertEqual(set(owners.values()),{4})
        for proj in ('q','k','v'):
            for h in (0,7):
                for d in range(256):
                    owner,slot=L.v_feature_owner(h,d)
                    for k in (0,15,16,511,1023):
                        loc=L.projection_weight_location(proj,h,k,d)
                        self.assertEqual(loc[:4],owner)
                        col,lane=divmod(loc[4]//2,16)
                        f,whichslot=divmod(col,12)
                        which,s=divmod(whichslot,4)
                        self.assertEqual((f,which,s,lane),(k//16,('q','k','v').index(proj),slot,k%16))
                        self.assertEqual(L.decode_address(L.physical_address(*loc)),loc)
            # Deterministic integer calculation reconstructed by physical owner.
            for d in (0,31,32,127,128,255):
                for q in (0,15):
                    expected=sum((q+1+k%7)*(d+1-k%5) for k in range(1024))
                    observed=0
                    for fragment in range(64):
                        for lane in range(16):
                            k=16*fragment+lane
                            loc=L.projection_weight_location(proj,0,k,d)
                            banklocalcol=loc[4]//32
                            observed+=(q+1+(banklocalcol//12*16+lane)%7)*(d+1-k%5)
                    self.assertEqual(observed,expected)
        self.assertEqual(L.manifest()['projection_weight_bytes_each'],4*1024**2)
        self.assertEqual(L.manifest()['qkv_weight_system_bytes'],12*1024**2)

    def test_Q_roundtrip_copies_and_DRAM_source_bursts(self):
        expected={L.query_location(h,c,q,L.k_feature_inverse(b,f*16))
            for h in range(8) for c in range(2) for q in range(16) for b in range(4) for f in range(4)}
        self.assertEqual(len(expected),4096)
        qmanifest,qtrace=self.cases['q']
        stores={L.decode_address(a) for op,a,_ in qtrace if op=='ST'}
        self.assertEqual(stores,expected)
        manifest,trace=self.cases['qk']
        loads=[L.decode_address(a) for op,a,_ in trace if op=='PIM_RD_Q']
        self.assertEqual(set(loads),expected)
        self.assertEqual(len(loads),len(expected))
        seen_features=defaultdict(list)
        for item in manifest['input_fragments']:
            f=item['fragment']
            seen=Counter()
            for op,address,buffer in trace[item['line_start']-1:item['line_end']]:
                c,r,g,b,off=L.decode_address(address)
                self.assertEqual(op,'PIM_RD_Q')
                self.assertEqual((r,g,buffer),(0,0,f%2))
                srcf,q=divmod((off-L.REGIONS['q_cache']['base'])//32,16)
                self.assertEqual(srcf,f)
                seen[c,b,q]+=1
                if c==0 and q==0:
                    seen_features[b].extend(L.q_input_features(b,f))
            self.assertEqual(seen,Counter((c,b,q) for c in range(16) for b in range(4) for q in range(16)))
        self.assertEqual(sorted(d for ds in seen_features.values() for d in ds),list(range(256)))

    def test_K_stores_match_target_without_partial_bursts(self):
        locations=list(k_suffix_bursts())
        expected={L.key_location(h,L.PREFIX+q,L.k_feature_inverse(b,f*16))
            for h in range(8) for q in range(16) for b in range(4) for f in range(4)}
        self.assertEqual(len(locations),len(set(locations)))
        self.assertEqual(set(locations),expected)
        bankbytes=Counter()
        for c,r,g,b,off in locations:
            self.assertEqual(off%32,0)
            bankbytes[c,r,g,b]+=32
        self.assertEqual(Counter(bankbytes.values()),Counter({128:512}))
        manifest,trace=self.cases['k']
        stores={L.decode_address(a) for op,a,_ in trace if op=='ST'}
        self.assertEqual(stores,expected)
        self.assertNotIn('PIM_LOCAL_WR',manifest['counts'])
        self.assertNotIn('PIM_MAC_K',manifest['counts'])

    def test_numeric_Q_K_repacking_and_BG_reduction(self):
        # Distinct logical values survive source feature owners -> GPU -> caches.
        qvalue=lambda q,d: (q+1)*(d%11-5)
        kvalue=lambda t,d: (t%13-6)*(d%7+1)
        for h in (0,7):
            for q in (0,15):
                for t in (0,967,968,983):
                    channel,rank,bg=L.token_group(h,t)
                    expected=sum(qvalue(q,d)*kvalue(t,d) for d in range(256))
                    partials=[]
                    for bank in range(4):
                        part=0
                        for local in range(64):
                            d=L.k_feature_inverse(bank,local)
                            qloc=L.query_location(h,channel%2,q,d)
                            kloc=L.key_location(h,t,d)
                            self.assertEqual(qloc[:4],(channel,0,0,bank))
                            self.assertEqual(kloc[:4],(channel,rank,bg,bank))
                            self.assertEqual(L.projection_owner(h,q,d)[0],L.v_feature_owner(h,d)[0])
                            part+=qvalue(q,d)*kvalue(t,d)
                        partials.append(part)
                    self.assertEqual(sum(partials),expected)

    def test_QK_column_coverage_export_completeness_and_tile_capacity(self):
        manifest,trace=self.cases['qk']
        perchannel=defaultdict(list)
        for op,address,buffer in trace:
            if op=='PIM_MAC_QK':
                c,r,g,b,off=L.decode_address(address)
                slot,f=divmod((off-L.REGIONS['k_cache']['base'])//32,4)
                self.assertEqual(buffer,f%2)
                perchannel[c].append((slot,f))
        for items in perchannel.values():
            self.assertEqual(Counter(items),Counter((s,f) for s in range(62) for f in range(4)))
        exported=defaultdict(list)
        for item in manifest['exports']:
            self.assertLessEqual(item['maximum_staged_bytes_per_BG'],1024)
            first,last=item['line_start']-1,item['line_end']
            previous=max(i for i,e in enumerate(trace[:first]) if e[0]=='PIM_MAC_QK')
            self.assertIn('BARRIER',[e[0] for e in trace[previous+1:first]])
            for op,address,buffer in trace[first:last]:
                self.assertEqual(op,'PIM_RD_BG')
                c,r,g,b,off=L.decode_address(address)
                exported[c,r,g].append(off)
        self.assertEqual(len(exported),128)
        for offsets in exported.values():
            self.assertEqual(offsets,list(range(0,62*16*4,32)))

    def test_V_PV_layout_and_RMW_preserved(self):
        _,trace=self.cases['v']
        reads={L.decode_address(a) for op,a,_ in trace if op=='LD'}
        writes={L.decode_address(a) for op,a,_ in trace if op=='ST'}
        self.assertEqual(reads,{L.value_location(h,960,d) for h in range(8) for d in range(256)})
        self.assertEqual(writes,{L.value_location(h,t,d) for h in range(8) for t in (960,976) for d in range(256)})
        _,trace=self.cases['pv']
        for op,address,buffer in trace:
            if op=='PIM_MAC_AB':
                c,r,g,b,off=L.decode_address(address)
                fragment=(off-L.REGIONS['v_cache']['base'])//128
                self.assertEqual(buffer,fragment%2)

    def test_regions_disjoint_and_source_copies_accounted(self):
        intervals=sorted((r['base'],r['base']+r['reserved_bytes_per_bank']) for r in L.REGIONS.values())
        self.assertEqual(intervals[0][0],0)
        for (_,end),(start,_) in zip(intervals,intervals[1:]):
            self.assertEqual(end,start)
        self.assertEqual(intervals[-1][1],42*1024)
        self.assertEqual(L.REGIONS['q_cache']['used_bytes_per_bank'],2048)
        self.assertEqual(L.manifest()['q_dram_bytes'],128*1024)

    def test_roundtrip_fences_explicit_unknown_GPU_boundary(self):
        manifest,trace=self.cases['chain_qkv_qk']
        lastwrite=max(i for i,e in enumerate(trace) if e[0]=='ST')
        firstq=min(i for i,e in enumerate(trace) if e[0]=='PIM_RD_Q')
        self.assertLess(lastwrite,firstq)
        self.assertIn('BARRIER',[e[0] for e in trace[lastwrite+1:firstq]])
        lastexport=max(i for i,e in enumerate(trace[:lastwrite]) if e[0]=='PIM_RD_RES')
        firstwrite=min(i for i,e in enumerate(trace) if e[0]=='ST')
        self.assertLess(lastexport,firstwrite)
        self.assertIn('BARRIER',[e[0] for e in trace[lastexport+1:firstwrite]])
        self.assertFalse(manifest['integrated_numeric_chain'])
        self.assertIsNone(manifest['unknown_latency_us']['GPU_QK_RoPE_cast_pack'])
        for m,t in self.cases.values():
            markers=[v for op,v,_ in t if op=='BARRIER']
            self.assertEqual(markers,list(range(len(markers))))
            for i,segment in enumerate(m['segments']):
                self.assertEqual(segment['start'],0 if i==0 else m['segments'][i-1]['end'])
                self.assertLess(segment['start'],segment['end'])

    def test_serial_control_preserves_payload_and_fences_reuse(self):
        for op in ('qkv','qk'):
            path=Path(self.temp.name)/(op+'_serial.trace')
            manifest=generate(path,op,schedule='serial')
            trace=read_trace(path)
            baseline=self.cases[op][0]
            strip=lambda c:{k:v for k,v in c.items() if k not in ('BARRIER','WAIT')}
            self.assertEqual(strip(manifest['counts']),strip(baseline['counts']))
            self.assertEqual({b for _,_,b in trace if b is not None},{0})
            self.assertGreater(manifest['counts']['BARRIER'],baseline['counts']['BARRIER'])
            # Every fragment after the first starts only after a completed fence.
            for item in manifest['input_fragments'][1:]:
                previous=trace[:item['line_start']-1]
                lastmac=max(i for i,e in enumerate(previous) if e[0].startswith('PIM_MAC'))
                self.assertIn('BARRIER',[e[0] for e in previous[lastmac+1:]])


if __name__=='__main__':
    unittest.main()

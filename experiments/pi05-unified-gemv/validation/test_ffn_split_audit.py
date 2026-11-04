"""Independent FFN tensor coordinates, exact source rows and tail masks."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import ffn_split_layout as source
import workload
from audit_native import audit_trace,trace_records,audit_slot_events
from ffn_split_audit import (geometry,banks,source_location,weight_address,address,
    audit_mapping_contract,audit_trace_order,overlap_ticks)
from test_audit import event


class FFNSplitTests(unittest.TestCase):
    def test_both_partitions_and_mixed_serial_match_fixed_trace_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            for count in (1,2):
                for case in ('gate_up_gpu','down_gpu_input','combined_timing_chain','serial_layer_control'):
                    path=Path(tmp)/f'{case}-{count}.trace'
                    m=workload.generate(path,case,staging_banks=2,rank_descriptors=True,channel_independent_dispatch=True,
                        v_token_sharding=True,pv_token_sharding=True,qkv_fusion=True,ffn_data_banks=count)
                    records=trace_records(path)
                    audit_trace(records,case,2,m.get('v_token_sharding',False),m.get('pv_token_sharding',False),
                        m.get('qkv_fusion',False),m.get('whole_srf',False),count)
                    mappings=m['ffn_split_mapping'].get('stages',{'single':m['ffn_split_mapping']})
                    for mapping in mappings.values():audit_mapping_contract(mapping,count)
                    if case=='gate_up_gpu':
                        before=path.read_bytes()
                        workload.generate(path,case,staging_banks=2,rank_descriptors=True,channel_independent_dispatch=True,
                            v_token_sharding=True,pv_token_sharding=True,qkv_fusion=True,ffn_data_banks=count,ffn_overlap=False)
                        self.assertEqual(path.read_bytes(),before)
                        wrong=deepcopy(records)
                        next(e for e in wrong if e['operation']=='PIM_STAGE_FRAGMENT')['buffer']=1
                        with self.assertRaisesRegex(AssertionError,'trace order/address'):
                            audit_trace_order(wrong,case,count)

    def test_all_logical_feature_owners_and_weight_boundary_lanes_are_unique(self):
        for count in (1,2):
            data,compute=banks(count)
            for op in ('gate','up','down'):
                g=geometry(op,count);owners=set()
                for feature in range(16*g['features']):
                    ch,local=divmod(feature,g['features']);col,j=divmod(local,len(compute))
                    got=source.feature_location(op,feature,count)
                    self.assertEqual((got['channel'],got['bank'],got['column']),(ch,compute[j],col))
                    owners.add((ch,compute[j],col))
                    for reduction in (0,15,16,g['fragments']*16-16,g['fragments']*16-1):
                        self.assertEqual(source.weight_location(op,reduction,feature,count),weight_address(op,reduction,feature,count))
                self.assertEqual(len(owners),16*g['features'])
                self.assertEqual(sum(mask.bit_count() for mask in g['masks']),g['features'])
                self.assertTrue(all(not(mask & sum(1<<b for b in data)) for mask in g['masks']))
                self.assertLessEqual(g['output_bytes'],2048)

    def test_row_striping_covers_each_DRAM_column_and_both_512B_views(self):
        for count in (1,2):
            for op in ('gate','up','down'):
                g=geometry(op,count);rows={};packets=set()
                for f in range(g['fragments']):
                    for token in range(16):
                        bank,offset=source_location(f,token,count)
                        packet=source.input_packet(op,0,f,token,count)
                        self.assertEqual((packet['source_bank'],packet['offset_bytes'],packet['slot']),(bank,offset,f%2))
                        self.assertEqual(packet['source_byte_offset'],(token*g['fragments']*16+f*16)*2)
                        self.assertNotIn((bank,offset),packets);packets.add((bank,offset))
                        rows.setdefault((bank,offset//2048),set()).add(offset%2048//32)
                self.assertEqual(len(rows),g['fragments']//4)
                self.assertTrue(all(columns==set(range(64)) for columns in rows.values()))
                self.assertEqual(source.input_packet('gate',0,0,0,count)['destination'],source.input_packet('up',0,0,0,count)['destination'])

    def test_export_visits_every_useful_tensor_element_once_without_padding(self):
        for count in (1,2):
            for op in ('gate','up','down'):
                g=geometry(op,count);elements=set()
                for ch in range(16):
                    for offset in range(0,g['features']*64,32):
                        packet=source.result_packet(op,ch,offset,count)
                        self.assertEqual(len(packet),8)
                        for i,e in enumerate(packet):
                            feature,inside=divmod(offset+i*4,64);token=inside//4
                            bank=g['compute'][feature%len(g['compute'])];column=feature//len(g['compute'])
                            self.assertEqual((e['source_bank'],e['source_column'],e['token']),(bank,column,token))
                            self.assertLess((e['accumulator_index']+1)*4,2049)
                            key=ch,feature,token;self.assertNotIn(key,elements);elements.add(key)
                self.assertEqual(len(elements),16*g['features']*16)

    def test_masked_slot_consumption_uses_each_banks_valid_column_count(self):
        count=1;g=geometry('gate',count);mask=sum(1<<b for b in g['compute'])
        fills=[event('GEMV_MULTICAST',1+t,duration=0,banks=mask,tokens=1<<t) for t in range(16)]
        macs=[event('GEMV_MAC',40+16*c,duration=16,banks=m,tokens=65535,columns=c+1) for c,m in enumerate(g['masks'])]
        context={(0,e['clock']):512 for e in macs}
        release=event('GEMV_RELEASE',40+16*g['columns'],duration=0,banks=mask,tokens=65535)
        report=audit_slot_events(fills+macs+[release],'gate_up_gpu',mac_context=context,whole_srf=True,ffn_data_banks=1)
        self.assertEqual(report['physical_bank_releases'],15)
        with self.assertRaisesRegex(AssertionError,'incomplete input/weight'):
            audit_slot_events(fills+macs[:-1]+[release],'gate_up_gpu',mac_context=context,whole_srf=True,ffn_data_banks=1)
        self.assertEqual(overlap_ticks([(0,10),(5,15)],[(8,12)]),4)
        self.assertEqual(overlap_ticks([(0,8)],[(8,12)]),0)


if __name__=='__main__':unittest.main()

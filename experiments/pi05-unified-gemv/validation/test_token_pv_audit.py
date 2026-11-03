"""Independent tensor labels, partial-sum semantics and whole-SRF hazards."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
import workload
import pv_token_layout as P
import qkv_token_layout as Q
from audit_native import audit_trace,trace_records,audit_slot_events
from paired_audit import audit_paired_staging
from token_pv_audit import cache_coordinates,audit_cache_contract,audit_local_cache
from test_audit import event


def whole_fixture(fragments=8):
    stage=[dict(operation="SEAL",Channel=0,stage_generation=1,clock=1)]
    native,gemv=[],[]
    clock=2
    for batch in range(0,fragments,4):
        for f in range(batch,min(batch+4,fragments)):
            for token in range(8):
                offset=524288+(f//2*8+token)*32
                e=dict(Channel=0,Rank=1,BankGroup=f%2,Bank=0,Row=offset//2048,Column=offset%2048//32,
                    clock=clock,ready_at=clock+5,buffer=f%4,stage_generation=1)
                stage += [dict(e,operation="LOCALRD"),dict(e,clock=clock+1,operation="BROADCAST")]
                native.append(dict(e,command="LOCALRD"));clock+=4
        clock+=5
        for f in range(batch,min(batch+4,fragments)):
            for col in range(48):
                offset=393216+(f*48+col)*32
                native.append(dict(Channel=0,Rank=1,BankGroup=-1,Bank=-1,Row=offset//2048,Column=offset%2048//32,
                    clock=clock,command="MACAB"))
                gemv.append(dict(operation="GEMV_MAC",Channel=0,clock=clock,ready_at=clock+8))
                clock+=8
    return stage,native,gemv


def rmw_fixture():
    events,native,gemv=[],[],[]
    for ch in range(16):
        for col in range(1024):
            offset=131072+col*32
            native.append(dict(command="MACAB",Channel=ch,clock=col*8+1,Row=offset//2048,Column=offset%2048//32))
            gemv.append(dict(operation="GEMV_MAC",Channel=ch,clock=col*8+1,ready_at=col*8+9))
        events.append(dict(operation="SOURCE_READY",Channel=ch,clock=8185,ready_at=8193,source_columns=16))
        for bank in range(16):
            for lane in range(16):
                clock=9000+(bank*16+lane)*20
                identity=dict(Channel=ch,Rank=1,BankGroup=bank//4,Bank=bank%4,Row=87,Column=32+lane,
                    request_id=ch*256+bank*16+lane,source_columns=16,source_feature_column=lane,
                    gather_elements=8,preserved_mask=15,updated_mask=4080,zero_mask=61440)
                for op,delta,duration,size in (("LOCALRD",0,5,32),("GATHER",5,2,32),("CAST",7,1,16),("LOCALWR",8,7,32)):
                    e=dict(identity,operation=op,clock=clock+delta,ready_at=clock+delta+duration,bytes=size)
                    events.append(e)
                    if op in ("LOCALRD","LOCALWR"):native.append(dict(e,command=op))
    result=dict(config=dict(gemv_port_cycles=2,pim_cast_cycles=1),resolved_timings_ck=dict(nCL=3,nBL_min=2,nLOCALWR=4,nWR=3))
    return events,native,gemv,result


class TokenPVTests(unittest.TestCase):
    def test_RMW_evidence_requires_dependencies_byte_masks_and_real_array_issues(self):
        events,native,gemv,result=rmw_fixture()
        report=audit_local_cache(events,native,gemv,result,False)
        self.assertEqual(report["packets"],4096)
        for fault,pattern in (("cast","cast precedes"),("mask","prefix"),("missing","omits"),("native","actual native")):
            changed=deepcopy(events);commands=native
            if fault=="cast":next(e for e in changed if e["operation"]=="CAST")["clock"]-=1
            elif fault=="mask":next(e for e in changed if e["operation"]=="LOCALWR")["preserved_mask"]=0
            elif fault=="missing":changed.remove(next(e for e in changed if e["operation"]=="LOCALRD"))
            else:commands=[e for i,e in enumerate(native) if i!=1024]
            with self.subTest(fault=fault),self.assertRaisesRegex(AssertionError,pattern):
                audit_local_cache(changed,commands,gemv,result,False)

    def test_all_new_traces_match_independent_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in workload.CASES:
                path=Path(tmp)/f"{case}.trace"
                m=workload.generate(path,case,staging_banks=2,rank_descriptors=True,v_token_sharding=True,
                    pv_token_sharding=True,qkv_fusion=True)
                audit_trace(trace_records(path),case,2,True,True,True,True)
                if case=="serial_layer_control":
                    self.assertEqual([p["stage"] for p in m["parts"]],["chain","pv","gate_up_gpu","down_gpu_input"])

    def test_QKV_operators_and_feature_lanes_have_disjoint_complete_weight_cells(self):
        for op,index in (("q",0),("k",1),("v",2)):
            for head in range(8):
                for parity in range(2):
                    for feature in range(256):
                        bank=feature%128//8;lane=2*(feature%8)+feature//128
                        for reduction in (0,15,16,1023):
                            self.assertEqual(Q.weight_location(op,head,parity,reduction,feature),
                                (2*head+parity,1,bank//4,bank%4,393216+(reduction//16*48+index*16+lane)*32+reduction%16*2))
        slots=set()
        for op in ("q","k","v"):
            for query in range(0,16,2):
                for feature in range(8):
                    for d in (feature,feature+128):
                        e=Q.output_element(op,0,query,d)
                        slots.add(e["accumulator_index"])
        self.assertEqual(slots,set(range(384)))
        self.assertEqual(len(slots)*4,1536)
        self.assertLessEqual(len(slots)*4,2048)

    def test_local_cache_RMW_preserves_four_prefix_and_inserts_eight_then_zeroes_four(self):
        suffix=set()
        for fused in (False,True):
            for channel in range(16):
                for bank in range(16):
                    for lane in range(16):
                        packet=P.spill_packet(channel,bank,lane,fused)
                        self.assertEqual(packet["offset_bytes"],163840+(480+lane)*32)
                        output=[("prefix",i) for i in range(4)]+[None]*12
                        for i,e in enumerate(packet["elements"]):
                            expected="preserve" if i<4 else "cast_accumulator" if i<12 else "zero_padding"
                            self.assertEqual(e["action"],expected)
                            self.assertEqual(e["token"],2*(480+i)+channel%2)
                            if expected=="cast_accumulator":
                                src=e["producer"]
                                self.assertEqual(src["accumulator_column"],lane+(32 if fused else 0))
                                self.assertEqual(src["accumulator_index"],(i-4)*(48 if fused else 16)+lane+(32 if fused else 0))
                                output[i]=(e["token"],e["feature"])
                                suffix.add((channel//2,e["token"],e["feature"]))
                            elif expected=="zero_padding":output[i]=0
                        self.assertEqual(output[:4],[("prefix",i) for i in range(4)])
                        self.assertTrue(all(v==0 for v in output[12:]))
        self.assertEqual(suffix,{(h,t,d) for h in range(8) for t in range(968,984) for d in range(256)})

    def test_cache_all_tokens_and_features_match_independent_parity_addresses(self):
        for head in range(8):
            for token in (0,1,960,967,968,983,984,991):
                for feature in range(256):
                    self.assertEqual(P.cache_location(head,token,feature),cache_coordinates(head,token,feature))
        contract=P.cache_contract(True)
        audit_cache_contract(contract,48)
        contract["spill"]["per_packet_preserve_bytes"]=[0,16]
        with self.assertRaisesRegex(AssertionError,"byte masks"):
            audit_cache_contract(contract,48)

    def test_PV_P_ownership_and_partial_sum_reconstruct_dense_dot_product(self):
        coverage=Counter()
        # Each tagged P coefficient and V value depends on KV index. Wrong
        # duplicated/full-cache reads or feature concatenation cannot pass.
        for head in range(8):
            for query in (0,7,15):
                for feature in (0,127,128,255):
                    partial=[]
                    for parity in (0,1):
                        total=0
                        for f in range(31):
                            for lane in range(16):
                                e=P.input_coordinates(2*head+parity,f,query,lane,16)
                                self.assertEqual((e["head"],e["query"],e["token"]),(head,query,2*(f*16+lane)+parity))
                                self.assertEqual(e["padding"],e["token"]>=984)
                                if not e["padding"]:
                                    total+=(query+1)*(e["token"]%11-5)*(feature+1+e["token"]%7)
                        partial.append(total)
                    dense=sum((query+1)*(t%11-5)*(feature+1+t%7) for t in range(984))
                    self.assertEqual(sum(partial),dense)
                    reducer=P.reduction_element(head,query,feature,16)
                    self.assertEqual(reducer["operation"],"sum")
                    self.assertEqual([e["channel"] for e in reducer["operands"]],[2*head,2*head+1])
        for channel in range(16):
            for offset in range(0,16384,32):
                packet=P.result_packet(channel,0,offset,16)
                self.assertEqual(len(packet),8)
                for e in packet:
                    coverage[e["head"],e["query"],e["feature"],e["parity"]]+=1
        self.assertEqual(coverage,Counter((h,q,d,p) for h in range(8) for q in range(16) for d in range(256) for p in (0,1)))

    def test_whole_SRF_four_views_fit_and_all_wait_before_refill(self):
        stage,native,gemv=whole_fixture()
        report=audit_paired_staging(stage,native,gemv,"chain",False,2,True,True,True,True)
        self.assertEqual(report["maximum_prefetched_input_bytes_per_bank"],1024)
        self.assertEqual(report["input_batches_checked"],2)
        changed=deepcopy(stage)
        # The first view is already consumed, but the fourth view is still busy.
        first=next(e for e in changed if e["operation"]=="LOCALRD" and e.get("Row")==256 and e.get("Column")==16)
        first["clock"]=gemv[191]["ready_at"]-1
        with self.assertRaisesRegex(AssertionError,"whole previous batch"):
            audit_paired_staging(changed,native,gemv,"chain",False,2,True,True,True,True)

    def test_row_preparation_can_overlap_but_next_whole_SRF_read_cannot(self):
        from native_guards import audit_whole_srf_row_preparation
        native, stage, gemv = [], [], []
        for ch in range(16):
            for batch in range(32):
                issue = batch*100+40
                weight = 262144+((2*batch+1)*32+31)*32
                native.append(dict(Channel=ch, command="MACAB", clock=issue,
                    Row=weight//2048, Column=weight%2048//32))
                gemv.append(dict(Channel=ch, operation="GEMV_MAC", clock=issue, ready_at=issue+16))
                source = 524288+batch*512
                identity = dict(Channel=ch, Rank=1, BankGroup=0, Bank=0,
                    Row=source//2048, Column=source%2048//32)
                stage.append(dict(identity, operation="LOCALRD", clock=batch*100+5, stage_generation=1))
                if batch:
                    native.append(dict(identity, command="PREpb", type=27, clock=(batch-1)*100+48))
        report = audit_whole_srf_row_preparation(native, stage, gemv)
        self.assertEqual(report["batch_transitions_checked"], 496)
        self.assertEqual(report["channels_with_legal_row_preparation_overlap"], 16)
        early = deepcopy(stage)
        next(e for e in early if e["Channel"] == 0 and e["Column"] == 16)["clock"] = 55
        with self.assertRaisesRegex(AssertionError, "LOCALRD precedes"):
            audit_whole_srf_row_preparation(native, early, gemv)
        serialized = [dict(e, clock=e["clock"]+8) if e["command"] == "PREpb" else e for e in native]
        with self.assertRaisesRegex(AssertionError, "never overlaps legal PRE/ACT"):
            audit_whole_srf_row_preparation(serialized, stage, gemv)

    def test_sixteen_row_operator_cannot_use_third_512B_view(self):
        fills=[event("GEMV_MULTICAST",1+2*t,duration=0,slot=2,tokens=1<<t) for t in range(16)]
        fills.append(event("GEMV_MAC",40,duration=16,slot=2,tokens=65535,columns=1))
        with self.assertRaisesRegex(AssertionError,"physical 1KiB"):
            audit_slot_events(fills,"pv",False,pv_token_sharding=True,whole_srf=True)


if __name__ == "__main__":unittest.main()

"""Mutation tests for independent O-projection/score native evidence audits."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"source"))
import workload
import audit_native as A
import audit_o_projection as O
from dispatch_audit import expand_rank_records

COMMON=dict(rank_staging=True,staging_banks=2,rank_descriptors=True,
    channel_independent_dispatch=True,v_token_sharding=True,pv_token_sharding=True,qkv_fusion=True)


def geometry(offset,bank=0):
    return dict(Channel=0,Rank=1,BankGroup=bank//4,Bank=bank%4,Row=offset//2048,Column=offset%2048//32)


def model():
    return dict(config=dict(gemv_port_cycles=2,pim_cast_cycles=1,rank_reduce_cycles=4),
        resolved_timings_ck=dict(nLOCALWR=4,nWR=3,nBL_min=2,nCL=3,nCWL=3,nBL_max=4))


def score_fixture():
    events,native=[],[];clock=1
    for tile in range(31):
        for n in range(16):
            k=tile*16+n;off=65536+k*32 if k<484 else 81920+(16+k-484)*32
            e=dict(geometry(off),clock=clock,ready_at=clock+20,operation="QK_MAC",
                tile=tile,packet=-1,bytes=0,mac_columns=k+1)
            events.append(e);native.append(dict(e,command="MACQK"));clock+=16
        clock+=4
        events.append(dict(geometry(655360+tile*1024),clock=clock,ready_at=clock,
            operation="SPILL_BEGIN",tile=tile,packet=-1,bytes=1024,mac_columns=(tile+1)*16))
        for packet in range(32):
            base=dict(geometry(655360+tile*512+packet//2*32,4*(packet%2)),tile=tile,
                packet=packet,bytes=32,mac_columns=(tile+1)*16)
            events.append(dict(base,operation="TRANSFER",clock=clock,ready_at=clock+2));clock+=2
            e=dict(base,operation="LOCALWR",clock=clock,ready_at=clock+7)
            events.append(e);native.append(dict(e,command="LOCALWR"));clock+=7
        events.append(dict(base,operation="TILE_DONE",clock=clock,ready_at=clock,packet=-1,bytes=1024))
        clock+=1
    for tile in range(31):
        for packet in range(32):
            base=dict(geometry(655360+tile*512+packet//2*32,4*(packet%2)),tile=tile,
                packet=packet,bytes=32,mac_columns=496)
            events.append(dict(base,operation="READ_ACCEPT",clock=clock,ready_at=clock))
            native.append(dict(base,command="RD",clock=clock+1))
            events.append(dict(base,operation="READ_COMPLETE",clock=clock+6,ready_at=clock+6));clock+=7
    return events,native


def bf16_fixture(projection):
    events,native,gemv=[],[],[];clock=1;request=1
    def compute(base,columns):
        nonlocal clock
        for column in range(columns):
            e=dict(geometry(base+column*32),clock=clock,ready_at=clock+16)
            native.append(dict(e,command="MACAB"));gemv.append(dict(e,operation="GEMV_MAC"));clock+=16
    compute(163840,496)
    def packet(op,offset,size,duration,source,destination,query,fragment,tile):
        nonlocal clock
        e=dict(geometry(offset,destination if destination>=0 else 0),operation=op,
            clock=clock,ready_at=clock+duration,bytes=size,source_bank=source,
            destination_bank=destination,query=query,fragment=fragment,tile=tile,
            stage_generation=1,request_id=request,offset_bytes=offset)
        events.append(e);clock+=duration
        return e
    if projection:
        for fragment in range(16):
            for query in range(16):
                dst=4*(fragment%2);off=524288+(fragment//2*16+query)*32
                for op,size,cycles in (("PV_GATHER_FP32",64,4),("PV_CAST_BF16",32,2),
                                      ("PV_TRANSFER_BF16",32,2),("PV_LOCALWR_BF16",32,7)):
                    e=packet(op,off,size,cycles,fragment,dst,query,fragment,-1)
                native.append(dict(e,command="LOCALWR"));request+=1
        events.append(dict(e,operation="PV_INPUT_COMPLETE",ready_at=clock,bytes=8192))
        events.append(dict(geometry(524288),operation="PV_INPUT_SEAL",clock=clock,ready_at=clock,
            bytes=8192,source_bank=-1,destination_bank=-1,query=-1,fragment=-1,tile=-1,
            stage_generation=1,request_id=request,offset_bytes=524288));request+=1;clock+=1
    for tile in ((0,1) if projection else (-1,)):
        if tile==1:
            events.append(dict(geometry(524288),operation="INPUT_REWIND",clock=clock,ready_at=clock,
                bytes=8192,source_bank=-1,destination_bank=-1,query=-1,fragment=-1,tile=-1,
                stage_generation=2,request_id=request,offset_bytes=524288));request+=1;clock+=1
        if projection:
            native.append(dict(geometry(524288),command="LOCALRD",clock=clock))
            clock+=5
            compute(786432+tile*16384,512)
        for p in range(512 if projection else 256):
            per_bank=2 if projection else 1;source=p//per_bank%16;query=p//(16*per_bank)
            for op,size,cycles in (("EXPORT_GATHER_FP32",64,4),("EXPORT_CAST_BF16",32,2),("EXPORT_BF16",32,2)):
                e=packet(op,p*32,size,cycles,source,-1,query,-1,tile)
            native.append(dict(e,command="RDRES"));request+=1
    return events,native,gemv


def stage_fixture():
    stages,op,native,gemv=[],[],[],[]
    for ch in range(16):
        def geo(offset,bank=0):return dict(geometry(offset,bank),Channel=ch)
        clock=1
        stages.append(dict(geo(524288),operation="BEGIN",clock=clock,ready_at=clock,
            bytes=8192,stage_generation=1,buffer=-1));clock+=1
        for f in range(16):
            for q in range(16):
                offset=524288+(f//2*16+q)*32
                e=dict(geo(offset,4*(f%2)),clock=clock,ready_at=clock+7,bytes=32,
                    stage_generation=1,buffer=-1)
                stages.append(dict(e,operation="LOCAL_PRODUCER_WRITE"))
                op.append(dict(e,operation="PV_LOCALWR_BF16"));native.append(dict(e,command="LOCALWR"));clock+=7
        stages.append(dict(geo(524288),operation="SEAL",clock=clock,ready_at=clock,
            bytes=32,stage_generation=1,buffer=-1))
        op.append(dict(geo(524288),operation="PV_INPUT_SEAL",clock=clock,ready_at=clock,
            bytes=8192,stage_generation=1));clock+=1
        for gen in (1,2):
            if gen==2:
                op.append(dict(geo(524288),operation="INPUT_REWIND",clock=clock,ready_at=clock,
                    bytes=8192,stage_generation=gen));clock+=1
            for first in range(0,16,2):
                for f in (first,first+1):
                    for q in range(16):
                        offset=524288+(f//2*16+q)*32
                        e=dict(geo(offset,4*(f%2)),clock=clock,ready_at=clock+5,bytes=32,
                            stage_generation=gen,buffer=f%2)
                        stages.append(dict(e,operation="LOCALRD"));native.append(dict(e,command="LOCALRD"))
                        stages.append(dict(e,operation="BROADCAST",clock=clock+1))
                        gemv.append(dict(e,operation="GEMV_MULTICAST",clock=clock+5,ready_at=clock+5,
                            generation=(gen-1)*8+first//2+1,token_mask=1<<q))
                        clock+=6
                for f in (first,first+1):
                    for col in range(32):
                        e=dict(geo(786432+(gen-1)*16384+(f*32+col)*32),clock=clock,ready_at=clock+16,
                            buffer=f%2,generation=(gen-1)*8+first//2+1)
                        gemv.append(dict(e,operation="GEMV_MAC"));native.append(dict(e,command="MACAB"));clock+=16
    return stages,op,native,gemv


class OProjectionAuditTests(unittest.TestCase):
    def test_independent_oracle_covers_all_variants_and_cases(self):
        with tempfile.TemporaryDirectory(prefix="op-audit-test-",dir=HERE) as tmp:
            path=Path(tmp)/"trace"
            for score,projection,bf16 in ((False,False,False),(False,False,True),
                    (True,False,True),(False,True,True),(True,True,True)):
                for case in workload.CASES:
                    with self.subTest(case=case,score=score,projection=projection,bf16=bf16):
                        workload.generate(path,case,**COMMON,score_dram_spill=score,
                            o_projection=projection,pv_bf16_output=bf16)
                        records=expand_rank_records(A.trace_records(path))
                        actual=Counter(A.trace_key(e) for e in records if e["operation"] not in ("BARRIER","WAIT"))
                        expected=O.expected_trace(case,score,projection,bf16)
                        self.assertEqual(actual,expected,A.counter_difference(expected,actual))

    def test_score_lifecycle_rejects_early_reuse_missing_service_and_wrong_bank(self):
        events,native=score_fixture()
        O.audit_score_lifecycle(events,native,model(),channels=(0,))
        for fault in ("reduction","port","scratch","recovery","tile_reuse","read_early","bank","native"):
            bad=deepcopy(events);commands=native
            if fault=="reduction":next(e for e in bad if e["operation"]=="SPILL_BEGIN")["clock"]-=1
            elif fault=="port":next(e for e in bad if e["operation"]=="TRANSFER")["ready_at"]-=1
            elif fault=="scratch":
                wr=next(e for e in bad if e["operation"]=="LOCALWR")
                transfer=next(e for e in bad if e["operation"]=="TRANSFER" and e["tile"]==0 and e["packet"]==1)
                transfer["clock"]=wr["clock"]+3;transfer["ready_at"]=transfer["clock"]+2
            elif fault=="recovery":next(e for e in bad if e["operation"]=="LOCALWR")["ready_at"]-=1
            elif fault=="tile_reuse":next(e for e in bad if e["operation"]=="QK_MAC" and e["tile"]==1)["clock"]=0
            elif fault=="read_early":next(e for e in bad if e["operation"]=="READ_ACCEPT")["clock"]=0
            elif fault=="bank":next(e for e in bad if e["operation"]=="LOCALWR")["BankGroup"]=3
            else:commands=[e for e in native if not(e["command"]=="LOCALWR" and e["tile"]==0 and e["packet"]==0)]
            with self.subTest(fault=fault),self.assertRaises(AssertionError):
                O.audit_score_lifecycle(bad,commands,model(),channels=(0,))

    def test_BF16_and_local_PV_lifecycle_rejects_faults(self):
        for projection in (False,True):
            events,native,gemv=bf16_fixture(projection)
            O.audit_o_lifecycle(events,native,gemv,model(),projection,channels=(0,))
            for fault in ("gather","cast","source","export")+(("transfer","write","seal_missing","seal_early","read_before_seal","rewind") if projection else ()):
                bad=deepcopy(events)
                commands=native
                op="PV_GATHER_FP32" if projection else "EXPORT_GATHER_FP32"
                if fault=="gather":next(e for e in bad if e["operation"]==op)["ready_at"]-=1
                elif fault=="cast":next(e for e in bad if "CAST" in e["operation"])["clock"]-=1
                elif fault=="source":next(e for e in bad if e["operation"]==op)["source_bank"]+=1
                elif fault=="export":bad.remove(next(e for e in bad if e["operation"]=="EXPORT_BF16"))
                elif fault=="transfer":next(e for e in bad if e["operation"]=="PV_TRANSFER_BF16")["ready_at"]-=1
                elif fault=="write":next(e for e in bad if e["operation"]=="PV_LOCALWR_BF16")["ready_at"]-=1
                elif fault=="seal_missing":bad.remove(next(e for e in bad if e["operation"]=="PV_INPUT_SEAL"))
                elif fault=="seal_early":
                    seal=next(e for e in bad if e["operation"]=="PV_INPUT_SEAL")
                    seal["clock"]-=1;seal["ready_at"]-=1
                elif fault=="read_before_seal":
                    commands=deepcopy(native)
                    seal=next(e for e in bad if e["operation"]=="PV_INPUT_SEAL")
                    next(e for e in commands if e["command"]=="LOCALRD")["clock"]=seal["clock"]-1
                elif fault=="rewind":next(e for e in bad if e["operation"]=="INPUT_REWIND")["clock"]=0
                with self.subTest(projection=projection,fault=fault),self.assertRaises(AssertionError):
                    O.audit_o_lifecycle(bad,commands,gemv,model(),projection,channels=(0,))

    def test_shared_port_checks_interleaved_operators(self):
        events=[dict(Channel=0,clock=1,ready_at=5),dict(Channel=0,clock=5,ready_at=7),dict(Channel=1,clock=1,ready_at=3)]
        O.intervals_disjoint(events,lambda e:e["Channel"],"overlap")
        events[1]["clock"]=4
        with self.assertRaisesRegex(AssertionError,"overlap"):
            O.intervals_disjoint(events,lambda e:e["Channel"],"overlap")

    def test_two_bank_replay_and_whole_SRF_are_evidence_based(self):
        stage,op,native,gemv=stage_fixture()
        result=O.audit_stages(stage,op,native,gemv,model())
        self.assertEqual(result["whole_SRF_batches_checked"],16*2*8)
        for fault in ("missing_rewind","missing_producer_seal","wrong_slot","wrong_weight_fragment","early_MAC","missing_read"):
            s,o,n,g=map(deepcopy,(stage,op,native,gemv))
            if fault=="missing_rewind":o.remove(next(e for e in o if e["operation"]=="INPUT_REWIND"))
            elif fault=="missing_producer_seal":o.remove(next(e for e in o if e["operation"]=="PV_INPUT_SEAL"))
            elif fault=="wrong_slot":next(e for e in s if e["operation"]=="BROADCAST")["buffer"]=1
            elif fault=="wrong_weight_fragment":
                e=next(e for e in n if e["command"]=="MACAB")
                changed=O.event_offset(e)+1024;e["Row"],e["Column"]=changed//2048,changed%2048//32
            elif fault=="early_MAC":
                e=next(e for e in g if e["operation"]=="GEMV_MAC")
                command=next(r for r in n if r["command"]=="MACAB" and r["clock"]==e["clock"])
                e["clock"]-=10;command["clock"]-=10
            else:s.remove(next(e for e in s if e["operation"]=="LOCALRD"))
            with self.subTest(fault=fault),self.assertRaises(AssertionError):
                O.audit_stages(s,o,n,g,model())

    def test_new_local_write_precharge_checks_physical_banks(self):
        t=model()["resolved_timings_ck"]
        # The PRE to bank4 does not close bank0; PREab must wait for both.
        events=[dict(geometry(655360,0),command="LOCALWR",clock=10),
                dict(geometry(524288,4),command="LOCALWR",clock=13),
                dict(geometry(655360,0),command="PREpb",clock=17),
                dict(geometry(524288,4),command="PREpb",clock=20)]
        self.assertEqual(O.audit_new_local_write_precharge(events,t)["physical_precharge_checks"],2)
        unrelated=[dict(geometry(655360,0),command="LOCALWR",clock=10),
                   dict(geometry(524288,4),command="PREpb",clock=11)]
        self.assertEqual(O.audit_new_local_write_precharge(unrelated,t)["physical_precharge_checks"],0)
        for command in ("PREpb","PREab"):
            bad=[dict(geometry(655360,0),command="LOCALWR",clock=10),
                 dict(geometry(655360,0),command=command,clock=16)]
            with self.subTest(command=command),self.assertRaisesRegex(AssertionError,"write recovery"):
                O.audit_new_local_write_precharge(bad,t)

    def test_single_conversion_scratch_cannot_overlap_next_packet(self):
        t=model()["resolved_timings_ck"]
        for local in (False,True):
            gather="PV_GATHER_FP32" if local else "EXPORT_GATHER_FP32"
            consumer="PV_LOCALWR_BF16" if local else "EXPORT_BF16"
            free=12 if local else 8
            events=[dict(Channel=0,request_id=1,operation=gather,clock=0,ready_at=4),
                    dict(Channel=0,request_id=1,operation=consumer,clock=8 if local else 6,ready_at=15 if local else 8),
                    dict(Channel=0,request_id=2,operation=gather,clock=free,ready_at=free+4),
                    dict(Channel=0,request_id=2,operation=consumer,clock=free+8,ready_at=free+15 if local else free+10)]
            result=O.audit_o_packet_scratch(events,t)
            self.assertEqual(result["packets_checked"],2)
            # Both individual packet chains remain valid, and the gather port
            # itself is free. Only the shared conversion scratch is still busy.
            bad=deepcopy(events);bad[2]["clock"]-=1;bad[2]["ready_at"]-=1
            with self.subTest(local=local),self.assertRaisesRegex(AssertionError,"single 64B scratch"):
                O.audit_o_packet_scratch(bad,t)
            with self.assertRaisesRegex(AssertionError,"matching producer or consumer"):
                O.audit_o_packet_scratch(events[:-1],t)

    def test_staging_DRAM_generation_waits_for_prior_operand_scan(self):
        first=[dict(Channel=0,stage_generation=1,operation="BEGIN",clock=0,ready_at=0),
               dict(Channel=0,stage_generation=1,operation="WR",clock=1,ready_at=5),
               dict(Channel=0,stage_generation=1,operation="LOCALRD",clock=6,ready_at=11),
               dict(Channel=0,stage_generation=1,operation="BROADCAST",clock=7,ready_at=11)]
        second=[dict(Channel=0,stage_generation=2,operation="BEGIN",clock=11,ready_at=11),
                dict(Channel=0,stage_generation=2,operation="WR",clock=12,ready_at=16),
                dict(Channel=0,stage_generation=2,operation="LOCALRD",clock=17,ready_at=22),
                dict(Channel=0,stage_generation=2,operation="BROADCAST",clock=18,ready_at=22)]
        result=O.audit_staging_generation_reuse(first+second,[])
        self.assertEqual(result["cross_generation_boundaries_checked"],1)
        # Each generation separately has a coherent write/read order, but the
        # new owner starts while the previous source packet remains in flight.
        bad=deepcopy(second);bad[0]["clock"]=bad[0]["ready_at"]=10
        with self.assertRaisesRegex(AssertionError,"previous staged operand"):
            O.audit_staging_generation_reuse(first+bad,[])
        bad=deepcopy(second);bad[1]["clock"]=10
        with self.assertRaisesRegex(AssertionError,"before its generation"):
            O.audit_staging_generation_reuse(first+bad,[])
        rewind=dict(second[0],operation="INPUT_REWIND")
        O.audit_staging_generation_reuse(first+second[2:],[rewind])
        rewind["clock"]=rewind["ready_at"]=10
        with self.assertRaisesRegex(AssertionError,"previous staged operand"):
            O.audit_staging_generation_reuse(first+second[2:],[rewind])


if __name__=="__main__":
    unittest.main()

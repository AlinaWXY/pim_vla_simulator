"""Independent fixed-geometry and native lifetime audit for the O ablation.

Imports no producer/layout code. Physical address, matrix dimensions and byte
coverage below are derived independently from the experiment contract.
"""
from collections import Counter, defaultdict
from pathlib import Path

import audit_native as A


def expected_trace(case, score, proj, bf16):
    serial = case == "serial_layer_control"
    stages = ("chain", "pv", "gate_up_gpu", "down_gpu_input") if serial else (case,)
    result = Counter()
    def add(op, ch, offset, slot=-1, bank=0, extra=()):
        result[(op, ch, 1, bank//4, bank%4, offset//2048, offset%2048//32, slot, *extra)] += 1
    for stage in stages:
        original = A.expected_addresses(stage, True, 2, True, True, True, True)
        if stage == "chain" and score:
            original = Counter({k:v for k,v in original.items() if k[0] != "PIM_RD_BG"})
            for ch in range(16):
                for tile in range(31):
                    add("PIM_SCORE_SPILL", ch, 655360+tile*1024)
                    for packet in range(32):
                        add("LD", ch, 655360+tile*512+(packet//2)*32, bank=4*(packet%2))
        if stage == "pv" and bf16:
            original = Counter({k:v for k,v in original.items() if k[0] != "PIM_RD_RES"})
            if not proj:
                for ch in range(16):
                    for offset in range(0, 8192, 32):
                        add("PIM_RD_RES", ch, offset)
            else:
                for ch in range(16):
                    add("PIM_STAGE_BEGIN", ch, 524288, extra=(8192,16))
                    for f in range(16):
                        for q in range(16):
                            add("PIM_PV_SPILL", ch, 524288+(f//2*16+q)*32, bank=4*(f%2))
                    add("PIM_STAGE_SEAL", ch, 524288)
                    for tile in range(2):
                        if tile:
                            add("PIM_O_REWIND", ch, 524288)
                        for f in range(16):
                            add("PIM_STAGE_FRAGMENT", ch, 524288+(f//2)*512, f%2, 4*(f%2))
                            for column in range(32):
                                add("PIM_MAC_AB", ch, 786432+tile*16384+(f*32+column)*32, f%2)
                        for slot in (0,1):
                            add("PIM_GEMV_RELEASE", ch, 0, slot)
                        for offset in range(0, 16384, 32):
                            add("PIM_RD_RES", ch, offset)
        result += original
    return result


def commands(records):
    expected = A.expected_native_commands(records)
    for e in records:
        op = e["operation"]
        ch, rank, bg, bank, row, col = A.decode(e["address"])
        if op == "PIM_PV_SPILL":
            expected[A.native_key("LOCALWR", (ch,rank,bg,bank,row,col))] += 1
        elif op == "PIM_SCORE_SPILL":
            tile = (row*2048+col*32-655360)//1024
            for packet in range(32):
                offset = 655360+tile*512+(packet//2)*32
                expected[A.native_key("LOCALWR", (ch,rank,packet%2,0,offset//2048,offset%2048//32))] += 1
    return expected


def event_offset(e):
    return e["Row"]*2048+e["Column"]*32


def event_key(e):
    return e["Channel"],e["clock"],*(e[k] for k in A.GEOMETRY[1:])


def intervals_disjoint(events, resource, message):
    groups=defaultdict(list)
    for e in events:
        A.require(e["ready_at"]>=e["clock"],"Negative service interval")
        groups[resource(e)].append(e)
    for rows in groups.values():
        ordered=sorted(rows,key=lambda e:(e["clock"],e["ready_at"]))
        A.require(all(a["ready_at"]<=b["clock"] for a,b in zip(ordered,ordered[1:])),message)


def audit_new_local_write_precharge(native,timings):
    """Independently check new score/PV stores against physical PRE targets."""
    pending={};writes=checks=0
    recovery=timings["nLOCALWR"]+timings["nWR"]
    for e in sorted(native,key=lambda e:(e["Channel"],e["clock"])):
        if e["command"]=="LOCALWR" and (524288<=event_offset(e)<528384 or
                                             655360<=event_offset(e)<671744):
            A.require(e["Rank"]==1 and e["BankGroup"] in (0,1) and e["Bank"]==0,
                "New local result write has invalid physical bank")
            pending[e["Channel"],e["Rank"],e["BankGroup"],e["Bank"]]=e["clock"]
            writes+=1
        elif e["command"] in ("PREpb","PREab"):
            targets=[key for key in pending if key[0]==e["Channel"] and
                (e["Rank"]<0 or key[1]==e["Rank"]) and
                (e["command"]=="PREab" or
                 (e["BankGroup"]<0 or key[2]==e["BankGroup"]) and (e["Bank"]<0 or key[3]==e["Bank"]))]
            for key in targets:
                A.require(e["clock"]-pending[key]>=recovery,
                    "PRE closes score/PV destination before LOCALWR service plus write recovery")
                checks+=1;del pending[key]
    return dict(new_LOCALWR_commands=writes,physical_precharge_checks=checks,
        minimum_write_to_precharge_ck=recovery,scope="score DRAM and two-bank PV-result staging only")


def audit_o_packet_scratch(events,timings):
    """Prove inter-packet lifetime of the one 64B FP32/BF16 conversion scratch.

    This small check can augment frozen native audits without rerunning the
    complete address and command analysis. It reads only o_projection events.
    """
    relevant={"PV_GATHER_FP32","PV_LOCALWR_BF16","EXPORT_GATHER_FP32","EXPORT_BF16"}
    packets=defaultdict(dict)
    for e in events:
        if e["operation"] not in relevant:continue
        key=e["Channel"],e["request_id"]
        A.require(e["operation"] not in packets[key],"Duplicate conversion-scratch packet event")
        packets[key][e["operation"]]=e
    intervals=[]
    for key,packet in packets.items():
        local="PV_GATHER_FP32" in packet
        expected={"PV_GATHER_FP32","PV_LOCALWR_BF16"} if local else {"EXPORT_GATHER_FP32","EXPORT_BF16"}
        A.require(set(packet)==expected,"Conversion scratch lacks its matching producer or consumer")
        gather=packet["PV_GATHER_FP32" if local else "EXPORT_GATHER_FP32"]
        consumer=packet["PV_LOCALWR_BF16" if local else "EXPORT_BF16"]
        # DRAM write recovery can overlap subsequent scratch use, but its
        # nLOCALWR data service cannot. External export consumes through DQ end.
        free=consumer["clock"]+timings["nLOCALWR"] if local else consumer["ready_at"]
        A.require(consumer["clock"]>=gather["ready_at"] and free>=consumer["clock"],
            "Conversion scratch consumer precedes its producer")
        intervals.append(dict(Channel=key[0],clock=gather["clock"],ready_at=free))
    intervals_disjoint(intervals,lambda e:e["Channel"],
        "Next PV/O packet overwrites the single 64B scratch before LOCALWR or DQ consumption finishes")
    return dict(packets_checked=len(intervals),channels_checked=len({e["Channel"] for e in intervals}),
        scratch_bytes_per_rank=64,local_write_service_before_reuse=True,external_DQ_completion_before_reuse=True)


def audit_staging_generation_reuse(staging,op_events):
    """Prove DRAM source residency across operator BEGIN and O input REWIND.

    Intra-generation source coverage is checked separately. This augmentation
    rejects a later generation before the prior scan's physical reads and SRF
    broadcasts complete, even if each generation's own address multiset fits.
    """
    starts={};groups=defaultdict(list)
    for e in staging:
        key=e["Channel"],e["stage_generation"]
        groups[key].append(e)
        if e["operation"]=="BEGIN":
            A.require(key not in starts,"Duplicate staging generation BEGIN")
            starts[key]=e["clock"]
    for e in op_events:
        if e["operation"]!="INPUT_REWIND":continue
        key=e["Channel"],e["stage_generation"]
        A.require(key not in starts,"Staging generation has both BEGIN and input REWIND")
        starts[key]=e["clock"]
    A.require(set(starts)==set(groups),"Staging generation start/operand evidence differs")
    last_complete={};checks=0
    for (ch,generation),begin in sorted(starts.items()):
        A.require(generation==last_complete.get((ch,"generation"),0)+1,"Staging generation sequence skips prior storage owner")
        if ch in last_complete:
            A.require(begin>=last_complete[ch],
                "New staging generation begins before previous staged operand reads/broadcasts complete")
            checks+=1
        rows=groups[ch,generation]
        reads=[e for e in rows if e["operation"]=="LOCALRD"]
        broadcasts=[e for e in rows if e["operation"]=="BROADCAST"]
        A.require(reads and broadcasts,"Staging storage owner has no complete operand scan")
        A.require(all(e["clock"]>=begin for e in rows if e["operation"] in ("WR","LOCAL_PRODUCER_WRITE","LOCALRD","BROADCAST")),
            "Staging producer/consumer accesses storage before its generation begins")
        last_complete[ch]=max(e["ready_at"] for e in reads+broadcasts)
        last_complete[ch,"generation"]=generation
    return dict(generations_checked=len(starts),cross_generation_boundaries_checked=checks,
        prior_DRAM_read_and_broadcast_completion_before_reuse=True)


def audit_score_lifecycle(events,native,result,channels=range(16)):
    """Reconstruct the one-tile lifetime, real writes, and delayed DRAM reads."""
    timings,cfg=result["resolved_timings_ck"],result["config"]
    allowed={"QK_MAC","SPILL_BEGIN","TRANSFER","LOCALWR","TILE_DONE","READ_ACCEPT","READ_COMPLETE"}
    A.require(all(e["operation"] in allowed for e in events),"Unknown score lifecycle event")
    A.require({e["Channel"] for e in events}==set(channels),"Missing score lifecycle channel")
    expected_wr,expected_mac,expected_rd=Counter(),Counter(),Counter()
    for ch in channels:
        rows=[e for e in events if e["Channel"]==ch]
        groups=defaultdict(list)
        for e in rows:
            A.require(e["Rank"]==1 and 0<=e["tile"]<31,"Invalid score tile owner")
            groups[e["tile"],e["operation"]].append(e)
        previous_done=0
        for tile in range(31):
            macs=sorted(groups[tile,"QK_MAC"],key=lambda e:e["clock"])
            starts,done=groups[tile,"SPILL_BEGIN"],groups[tile,"TILE_DONE"]
            A.require(len(macs)==16 and len(starts)==len(done)==1,"Score tile lacks complete compute/store lifecycle")
            A.require(macs[0]["clock"]>=previous_done,"Score tile reused before prior spill completes")
            for n,e in enumerate(macs):
                k=tile*16+n
                expected_offset=65536+k*32 if k<484 else 81920+(16+k-484)*32
                A.require(event_offset(e)==expected_offset and e["mac_columns"]==k+1,"Score MAC does not cover the next K token")
                A.require(e["ready_at"]==e["clock"]+16+cfg["rank_reduce_cycles"],"Score completion omits MAC or rank reduction service")
                expected_mac[A.native_key("MACQK",tuple(e[k] for k in A.GEOMETRY))+(e["clock"],)]+=1
            A.require(starts[0]["clock"]>=max(e["ready_at"] for e in macs),"Score spill precedes complete rank reduction")
            transfers={e["packet"]:e for e in groups[tile,"TRANSFER"]}
            writes={e["packet"]:e for e in groups[tile,"LOCALWR"]}
            A.require(len(groups[tile,"TRANSFER"])==len(groups[tile,"LOCALWR"])==32 and
                set(transfers)==set(writes)==set(range(32)),"Score tile omits or repeats a stored packet")
            for packet in range(32):
                transfer,wr=transfers[packet],writes[packet]
                for e in (transfer,wr):
                    A.require(e["BankGroup"]==packet%2 and e["Bank"]==0 and
                        event_offset(e)==655360+tile*512+(packet//2)*32 and e["bytes"]==32,
                        "Score packet uses the wrong striped DRAM destination")
                A.require(transfer["clock"]>=starts[0]["clock"] and
                    transfer["ready_at"]==transfer["clock"]+cfg["gemv_port_cycles"],"Score transfer omits local-port service")
                A.require(wr["clock"]>=transfer["ready_at"] and
                    wr["ready_at"]==wr["clock"]+timings["nLOCALWR"]+timings["nWR"],
                    "Score write precedes transfer or omits write recovery")
                expected_wr[event_key(wr)]+=1
                if packet:
                    A.require(transfer["clock"]>=writes[packet-1]["clock"]+timings["nLOCALWR"],
                        "Score next packet overwrites the single 32B transfer scratch before LOCALWR service completes")
            previous_done=done[0]["clock"]
            A.require(previous_done>=max(e["ready_at"] for e in writes.values()),"Score tile retired before final write recovery")
        for op in ("READ_ACCEPT","READ_COMPLETE"):
            reads=[e for e in rows if e["operation"]==op]
            A.require(Counter((e["tile"],e["packet"]) for e in reads)==Counter({(t,p):1 for t in range(31) for p in range(32)}),
                "Score export read coverage differs from complete stored scores")
            A.require(all(e["clock"]>=previous_done for e in reads),"Score export begins before all tile stores complete")
        accepted={(e["tile"],e["packet"]):e for e in rows if e["operation"]=="READ_ACCEPT"}
        issued={tuple(e[k] for k in A.GEOMETRY):e for e in native if e["Channel"]==ch and
                e["command"] in ("RD","RDA") and 655360<=event_offset(e)<671744}
        for e in (e for e in rows if e["operation"]=="READ_COMPLETE"):
            a=accepted[e["tile"],e["packet"]]
            A.require(e["clock"]>=a["clock"],"Score read completes before acceptance")
            geometry=tuple(e[k] for k in A.GEOMETRY)
            A.require(geometry in issued and tuple(a[k] for k in A.GEOMETRY)==geometry,
                "Score read completion references wrong physical packet")
            rd=issued[geometry]
            A.require(rd["clock"]>=a["clock"] and e["clock"]>=rd["clock"]+timings["nCL"]+timings["nBL_min"],
                "Score read completion precedes native DRAM read latency")
            expected_rd[A.native_key("RD",tuple(e[k] for k in A.GEOMETRY))]+=1
    actual_wr=Counter(event_key(e) for e in native if e["command"]=="LOCALWR" and 655360<=event_offset(e)<671744)
    actual_mac=Counter(A.native_key("MACQK",tuple(e[k] for k in A.GEOMETRY))+(e["clock"],) for e in native if e["command"]=="MACQK")
    actual_rd=Counter(A.native_key("RD",tuple(e[k] for k in A.GEOMETRY)) for e in native if e["command"] in ("RD","RDA") and 655360<=event_offset(e)<671744)
    A.require(actual_wr==expected_wr and actual_mac==expected_mac and actual_rd==expected_rd,"Score lifecycle log lacks matching native array commands")
    intervals_disjoint([e for e in events if e["operation"]=="TRANSFER"],lambda e:e["Channel"],"Score transfers overlap the shared local port")
    return dict(tiles_per_rank=31,bytes_per_rank=31744,one_live_score_tile=True,
        complete_MAC_and_rank_reduction_before_spill=True,final_read_after_all_tiles=True,write_recovery_checked=True)


def audit_o_lifecycle(events,native,gemv,result,projection,channels=range(16)):
    """Check FP32->BF16 conversion cells, real local producer writes and exports."""
    cfg,t=result["config"],result["resolved_timings_ck"]
    local_ops=("PV_GATHER_FP32","PV_CAST_BF16","PV_TRANSFER_BF16","PV_LOCALWR_BF16")
    export_ops=("EXPORT_GATHER_FP32","EXPORT_CAST_BF16","EXPORT_BF16")
    allowed=set(local_ops+export_ops+("PV_INPUT_COMPLETE","PV_INPUT_SEAL","INPUT_REWIND"))
    A.require(all(e["operation"] in allowed for e in events),"Unknown O projection event")
    A.require({e["Channel"] for e in events}==set(channels),"Missing BF16 output channel")
    completions={(e["Channel"],e["clock"]):e["ready_at"] for e in gemv if e["operation"]=="GEMV_MAC"}
    source_ready={}
    for ch in channels:
        for base,end in ((163840,180224),(786432,802816),(802816,819200)):
            macs=[e for e in native if e["Channel"]==ch and e["command"]=="MACAB" and base<=event_offset(e)<end]
            if not macs:continue
            A.require(len(macs)==(496 if base==163840 else 512),"PV/O source MAC count is incomplete")
            source_ready[ch,base]=max(completions[ch,e["clock"]] for e in macs)
    expected_native_writes=Counter();expected_exports=Counter();last_tile_export={}
    for ch in channels:
        rows=[e for e in events if e["Channel"]==ch]
        groups=defaultdict(dict)
        for e in rows:
            if e["operation"] in local_ops+export_ops:
                key=e["request_id"]
                A.require(e["operation"] not in groups[key],"Duplicate O packet service")
                groups[key][e["operation"]]=e
        local_count=0;export_offsets=defaultdict(list)
        for packet in groups.values():
            local=bool(set(packet)&set(local_ops))
            names=local_ops if local else export_ops
            A.require(set(packet)==set(names),"O packet omits gather/cast/transfer or final movement")
            ordered=[packet[op] for op in names]
            A.require(all(tuple(e[k] for k in A.GEOMETRY)==tuple(ordered[0][k] for k in A.GEOMETRY)
                          for e in ordered),"O packet service changes address identity")
            for a,b in zip(ordered,ordered[1:]):
                A.require(b["clock"]>=a["ready_at"],"O packet consumes gather/cast/transfer before completion")
            gather,cast=ordered[:2]
            A.require(gather["bytes"]==64 and gather["ready_at"]==gather["clock"]+2*cfg["gemv_port_cycles"],"O FP32 gather byte count or service differs")
            A.require(cast["bytes"]==32 and cast["ready_at"]==cast["clock"]+2*cfg["pim_cast_cycles"],"O BF16 cast byte count or service differs")
            if local:
                A.require(projection,"PV local result write occurred without O projection")
                transfer,wr=ordered[2:]
                bank=wr["BankGroup"]*4+wr["Bank"]
                packet_index=(event_offset(wr)-524288)//32
                fragment=2*(packet_index//16)+bank//4;query=packet_index%16
                A.require(bank in (0,4) and 0<=fragment<16 and
                    all(e["source_bank"]==e["fragment"]==fragment and e["query"]==query and
                        e["destination_bank"]==bank for e in ordered),"PV gather selects incorrect accumulator cells or destination")
                A.require(gather["clock"]>=source_ready[ch,163840],"PV gather begins before its full reduction completes")
                A.require(transfer["bytes"]==32 and transfer["ready_at"]==transfer["clock"]+cfg["gemv_port_cycles"],"PV local transfer service differs")
                A.require(wr["bytes"]==32 and wr["ready_at"]==wr["clock"]+t["nLOCALWR"]+t["nWR"],"PV local write omits array service/recovery")
                expected_native_writes[event_key(wr)]+=1;local_count+=1
            else:
                export=ordered[-1];tile=export["tile"]
                A.require(tile in ((0,1) if projection else (-1,)),"Unexpected BF16 output owner")
                base=786432+tile*16384 if projection else 163840
                A.require(gather["clock"]>=source_ready[ch,base],"BF16 export begins before its full matrix reduction")
                offset=event_offset(export);p=offset//32;per_bank=2 if projection else 1
                A.require(all(e["source_bank"]==(p//per_bank)%16 and e["query"]==p//(16*per_bank)
                    and e["tile"]==tile for e in ordered),"BF16 export selects wrong output bank/query")
                A.require(export["bytes"]==32 and export["ready_at"]==export["clock"]+t["nBL_min"],"BF16 export does not preserve external DQ burst completion")
                expected_exports[ch,export["clock"],offset]+=1
                export_offsets[tile].append(offset)
                last_tile_export[ch,tile]=max(last_tile_export.get((ch,tile),0),export["ready_at"])
        A.require(local_count==(256 if projection else 0),"PV local spill packet count differs")
        for tile in ((0,1) if projection else (-1,)):
            A.require(export_offsets[tile]==list(range(0,16384 if projection else 8192,32)),"BF16 output packet order/coverage differs")
        if projection:
            completion=[e for e in rows if e["operation"]=="PV_INPUT_COMPLETE"]
            seals=[e for e in rows if e["operation"]=="PV_INPUT_SEAL"]
            rewinds=[e for e in rows if e["operation"]=="INPUT_REWIND"]
            writes=[e for e in rows if e["operation"]=="PV_LOCALWR_BF16"]
            A.require(len(completion)==len(seals)==len(rewinds)==1,"Local PV input/seal/rewind lifecycle incomplete")
            A.require(completion[0]["ready_at"]>=max(e["ready_at"] for e in writes),"PV input marked complete before local writes recover")
            seal=seals[0]
            A.require(seal["clock"]==seal["ready_at"] and seal["bytes"]==8192 and
                seal["BankGroup"]==seal["Bank"]==0 and event_offset(seal)==524288,
                "PV input seal has incorrect address or payload")
            A.require(seal["clock"]>=completion[0]["ready_at"],"PV input sealed before its complete local producer")
            gather_begin=min(e["clock"] for e in rows if e["operation"]=="PV_GATHER_FP32")
            first_o_mac=min(e["clock"] for e in native if e["Channel"]==ch and
                e["command"]=="MACAB" and 786432<=event_offset(e)<802816)
            first_reads=[e for e in native if e["Channel"]==ch and e["command"]=="LOCALRD" and
                524288<=event_offset(e)<528384 and gather_begin<=e["clock"]<first_o_mac]
            A.require(first_reads and min(e["clock"] for e in first_reads)>=seal["clock"],
                "O operand read precedes PV_INPUT_SEAL or lacks native DRAM read")
            A.require(rewinds[0]["clock"]>=last_tile_export[ch,0],"O input rewound before tile0 output DQ completes")
    actual_writes=Counter(event_key(e) for e in native if e["command"]=="LOCALWR" and 524288<=event_offset(e)<528384)
    A.require(actual_writes==expected_native_writes,"PV result producer log lacks actual native LOCALWR")
    native_exports=Counter((e["Channel"],e["clock"],event_offset(e)) for e in native if e["command"]=="RDRES")
    A.require(not (expected_exports-native_exports),"BF16 export lacks matching native RDRES")
    intervals_disjoint([e for e in events if e["operation"] in ("PV_CAST_BF16","EXPORT_CAST_BF16")],
        lambda e:(e["Channel"],e["source_bank"]),"BF16 casts overlap their source bank conversion unit")
    return dict(local_PV_packets=sum(expected_native_writes.values()),BF16_export_packets=sum(expected_exports.values()),
        FP32_source_coverage=True,cast_and_transfer_service_checked=True,local_write_recovery_checked=True,
        output_DQ_before_rewind_checked=projection,packet_scratch=audit_o_packet_scratch(events,t))


def audit_ca(ca, native, stage, controller, gemv, cache, score_events, op_events):
    # Reuse strict collision/duration checking, adding only evidenced pseudo ops.
    from staging_audit import audit_rank_ca
    extra = Counter()
    names = {"PV_GATHER_FP32":"PV_GATHER", "PV_CAST_BF16":"PV_CAST",
        "PV_TRANSFER_BF16":"PV_TRANSFER", "INPUT_REWIND":"O_REWIND",
        "EXPORT_GATHER_FP32":"RESULT_GATHER", "EXPORT_CAST_BF16":"RESULT_CAST"}
    for e in op_events:
        if e["operation"] in names:
            extra[e["Channel"], e["clock"], e["Rank"], names[e["operation"]]] += 1
    # Score port naming is recorded directly and separately from array writes.
    for e in score_events:
        if e["operation"] in ("TRANSFER", "SCORE_TRANSFER"):
            extra[e["Channel"], e["clock"], 1, "SCORE_TRANSFER"] += 1
    remaining, removed = [], Counter()
    for e in ca:
        key = (e["Channel"],e["clock"],e["Rank"],e["command"])
        if e["scope"] == "rank" and extra[key]:
            A.require(e["ready_at"]-e["clock"] == 1, "Pseudo command must occupy one rank CA cycle")
            extra[key] -= 1
            removed[key] += 1
        else:
            remaining.append(e)
    A.require(not +extra, "Missing native CA reservation for local conversion/transfer")
    # Check full resource intervals before the legacy exact-coverage checker.
    resources = defaultdict(list)
    for e in ca:
        ranks=(-1,) if e["scope"]=="global" else (0,1) if e["Rank"]<0 else (e["Rank"],)
        for rank in ranks:resources[e["Channel"],e["scope"],rank].append(e)
    for rows in resources.values():
        ordered = sorted(rows,key=lambda e:e["clock"])
        A.require(all(a["ready_at"]<=b["clock"] for a,b in zip(ordered,ordered[1:])), "Local command overlaps CA reservation")
    result = audit_rank_ca(remaining,native,stage,controller,None,gemv,cache)
    result["additional_local_pseudo_commands"] = sum(removed.values())
    return result


def audit_stages(stage, op_events, native, gemv, result):
    # Reconstruct each data generation, including a second read of retained O input.
    storage_reuse=audit_staging_generation_reuse(stage,op_events)
    generations = {}
    for e in stage:
        A.require(e["operation"] in ("BEGIN","WR","SEAL","LOCALRD","BROADCAST","LOCAL_PRODUCER_WRITE"),"Unknown staging evidence operation")
        A.require(e["Rank"]==1 and e["Bank"]==0 and e["BankGroup"] in (0,1),"Invalid two-bank staging physical owner")
        if e["operation"] == "BEGIN":
            rows = 8 if e["bytes"] == 16384 else 16
            generations[e["Channel"],e["stage_generation"]] = dict(bytes=e["bytes"],rows=rows,begin=e["clock"],rewind=False)
    for e in op_events:
        if e["operation"] == "INPUT_REWIND":
            generations[e["Channel"],e["stage_generation"]] = dict(bytes=8192,rows=16,begin=e["clock"],rewind=True)
    grouped = defaultdict(list)
    for e in stage:
        grouped[e["Channel"],e["stage_generation"]].append(e)
    A.require(set(grouped)==set(generations),"Missing or undeclared staging generation")
    for ch in range(16):
        actual=sorted(g for c,g in generations if c==ch)
        A.require(actual==list(range(1,len(actual)+1)) and actual,"Staging generation skipped or channel absent")
    timings=result["resolved_timings_ck"]
    source_at_fill={};fragment_reads=defaultdict(list);fragment_broadcasts=defaultdict(list)
    native_reads=Counter(event_key(e) for e in native if e["command"]=="LOCALRD" and 524288<=event_offset(e)<589824)
    logged_reads=Counter(event_key(e) for e in stage if e["operation"]=="LOCALRD")
    A.require(native_reads==logged_reads,"Staging read log lacks matching native LOCALRD")
    A.require(Counter(event_key(e) for e in stage if e["operation"]=="WR")==
        Counter(event_key(e) for e in native if e["command"] in ("WR","WRA") and 524288<=event_offset(e)<589824),
        "Staging GPU producer log lacks matching native WR")
    checked = 0
    for key, g in generations.items():
        entries = grouped[key]
        reads = [e for e in entries if e["operation"] == "LOCALRD"]
        broadcasts = [e for e in entries if e["operation"] == "BROADCAST"]
        expected = Counter((4*(f%2),524288+(f//2*g["rows"]+q)*32)
            for f in range(g["bytes"]//(g["rows"]*32)) for q in range(g["rows"]))
        coords = lambda e:(e["BankGroup"]*4+e["Bank"],e["Row"]*2048+e["Column"]*32)
        A.require(Counter(map(coords,reads))==expected, "Staging array read omitted/duplicated a source packet")
        A.require(Counter(map(coords,broadcasts))==expected, "Staging broadcast omitted/duplicated a source packet")
        seals = [e for e in entries if e["operation"] == "SEAL"]
        if not g["rewind"]:
            A.require(len(seals)==1, "Each stage input must be sealed exactly once")
            seal = seals[0]["clock"]
        else:
            A.require(not seals and not any(e["operation"] in ("BEGIN","WR","LOCAL_PRODUCER_WRITE") for e in entries),
                "O rewind unexpectedly rewrites or reseals retained input")
            seal = g["begin"]
        by_source = {coords(e):e for e in reads}
        for e in broadcasts:
            rd = by_source[coords(e)]
            A.require(rd["clock"]>=seal and rd["ready_at"]==rd["clock"]+timings["nCL"]+timings["nBL_min"],
                "Staged input read precedes seal or omits column-read service")
            A.require(e["clock"]>=rd["clock"] and e["ready_at"]==max(rd["ready_at"],e["clock"]), "Broadcast precedes producer/read completion")
            bank,offset=coords(e)
            fragment=2*((offset-524288)//32//g["rows"])+bank//4
            token=((offset-524288)//32)%g["rows"]
            views=1024//(g["rows"]*32)
            A.require(e["buffer"]==rd["buffer"]==fragment%views,"Staging source fragment aliases wrong physical SRF view")
            identity=(key[0],e["ready_at"],e["buffer"],1<<token)
            A.require(identity not in source_at_fill,"Ambiguous staged source identity")
            source_at_fill[identity]=(key[1],fragment)
            fragment_reads[key[0],key[1],fragment].append(rd)
            fragment_broadcasts[key[0],key[1],fragment].append(e)
        if g["bytes"] == 8192:
            writes = [e for e in op_events if e["Channel"]==key[0] and e["operation"]=="PV_LOCALWR_BF16"]
            A.require(Counter(map(coords,writes))==expected, "PV local producer packet coverage differs from O input")
            A.require(max(e["ready_at"] for e in writes)<=seal, "O consumer begins before all PV writes recover")
            if not g["rewind"]:
                produced=[e for e in entries if e["operation"]=="LOCAL_PRODUCER_WRITE"]
                A.require(Counter(event_key(e) for e in produced)==Counter(event_key(e) for e in writes),
                    "Local staging producer registration differs from actual PV writes")
                producer_seals=[e for e in op_events if e["operation"]=="PV_INPUT_SEAL" and
                    (e["Channel"],e["stage_generation"])==key]
                A.require(len(producer_seals)==1 and producer_seals[0]["clock"]==seal,
                    "PV_INPUT_SEAL differs from the actual staging seal")
        else:
            writes=[e for e in entries if e["operation"]=="WR"]
            A.require(Counter(map(coords,writes))==expected,"GPU input write coverage differs from staged tensor")
            A.require(all(e["clock"]>=g["begin"] and e["ready_at"]==e["clock"]+timings["nCWL"]+timings["nBL_max"]+timings["nWR"]
                for e in writes),"GPU staged write omits arrival or write recovery")
            A.require(max(e["ready_at"] for e in writes)<=seal,"GPU input sealed before write recovery")
        checked += len(reads)
    # Bind every SRF generation to its producer fragment, then independently
    # require the entire 1KiB batch to be filled before any MAC consumes it.
    fills=Counter();owners={};macs=defaultdict(list)
    native_macs={(e["Channel"],e["clock"]):e for e in native if e["command"]=="MACAB"}
    regions=((131072,163840,16),(163840,180224,16),(262144,327680,32),
             (327680,360448,4),(393216,491520,48),(786432,802816,32),(802816,819200,32))
    generation_base={}
    for e in gemv:
        if e["operation"]=="GEMV_MULTICAST":
            identity=e["Channel"],e["clock"],e["buffer"],e["token_mask"]
            A.require(identity in source_at_fill,"Physical SRF fill lacks staged producer")
            fills[identity]+=1
            slot=e["Channel"],e["buffer"],e["generation"]
            source=source_at_fill[identity]
            A.require(owners.get(slot,source)==source,"One SRF generation mixes input fragments")
            owners[slot]=source
        elif e["operation"]=="GEMV_MAC":
            slot=e["Channel"],e["buffer"],e["generation"]
            A.require(slot in owners,"MAC lacks staged input owner")
            gen,fragment=owners[slot];g=generations[e["Channel"],gen]
            address=event_offset(native_macs[e["Channel"],e["clock"]])
            layout=next(((base,columns) for base,end,columns in regions if base<=address<end),None)
            A.require(layout is not None,"MAC has no independently defined weight region")
            base,columns=layout
            A.require((address-base)//(columns*32)==fragment,"MAC consumes wrong reduction fragment")
            expected_base=(802816 if g["rewind"] else 786432) if g["bytes"]==8192 else (
                163840 if g["bytes"]==15872 else 327680 if g["bytes"]==131072 else
                262144 if g["bytes"]==32768 else base)
            A.require(base==expected_base and (g["bytes"]!=16384 or base in (131072,393216)),"MAC consumes wrong operator input generation")
            generation_base[e["Channel"],gen]=(base,columns)
            macs[e["Channel"],gen,fragment].append(e)
    A.require(fills==Counter({key:1 for key in source_at_fill}),"Staged broadcasts do not exactly match physical SRF fills")
    A.require(set(macs)==set(fragment_reads)==set(fragment_broadcasts),"Staged fragment lacks reads, broadcasts, or MACs")
    batches=0
    for key,g in sorted(generations.items()):
        ch,gen=key;columns=generation_base[key][1];views=1024//(g["rows"]*32)
        previous_end=0
        for first in range(0,g["bytes"]//(g["rows"]*32),views):
            batch=[(ch,gen,f) for f in range(first,min(first+views,g["bytes"]//(g["rows"]*32)))]
            reads=[e for k in batch for e in fragment_reads[k]]
            ready=max(e["ready_at"] for k in batch for e in fragment_broadcasts[k])
            compute=[e for k in batch for e in macs[k]]
            A.require(all(len(macs[k])==columns for k in batch),"SRF fragment lacks complete weight-column consumption")
            A.require(min(e["clock"] for e in compute)>=ready,"MAC begins before the whole SRF batch is filled")
            A.require(min(e["clock"] for e in reads)>=previous_end,"Whole SRF overwritten before previous batch completes")
            previous_end=max(e["ready_at"] for e in compute);batches+=1
    return dict(generations=len(generations),source_packets_checked=checked,two_bank_source_coverage=True,
                local_producer_recovery_before_consumption=True,whole_srf_lifetime=True,
                whole_SRF_batches_checked=batches,source_to_MAC_identity=True,DRAM_storage_reuse=storage_reuse)


def audit_experiment(directory):
    directory=Path(directory)
    manifest=A.read_json(directory/"manifest.json")
    result=A.read_json(directory/"simulation/result.json")
    cfg=result["config"]
    flags=tuple(cfg.get(k,False) for k in ("pim_score_dram_spill","pim_o_projection","pim_pv_bf16_output"))
    from dispatch_audit import expand_rank_records, audit_ingress, audit_rank_descriptors
    raw=A.trace_records(directory/"workload.trace")
    records=expand_rank_records(raw)
    actual=Counter(A.trace_key(e) for e in records if e["operation"] not in ("BARRIER","WAIT"))
    expected=expected_trace(manifest["case"],*flags)
    A.require(actual==expected,"Independent experiment trace oracle differs: "+A.counter_difference(expected,actual))
    sim=directory/"simulation"
    files=[]
    def collect(pattern, enabled=True):
        if not enabled:return []
        paths=[sim/f"{pattern}.ch{ch}" for ch in range(16)]
        A.require(all(p.is_file() for p in paths),f"Missing {pattern} native evidence")
        files.extend(paths)
        return [e for p in paths for e in A.read_csv(p)]
    native=collect("timeline.csv")
    gemv=collect("unified_gemv.csv")
    stage=collect("staging_events.csv")
    ca=collect("rank_ca_events.csv")
    controller=collect("bgk_events.csv")
    cache=collect("v_cache.csv",cfg["pim_token_pv"])
    score=collect("score_spill.csv",flags[0])
    op=collect("o_projection.csv",flags[1] or flags[2])
    expected_commands=commands(records)
    actual_commands=Counter(A.native_key({"RDA":"RD","WRA":"WR"}.get(e["command"],e["command"]),tuple(e[k] for k in A.GEOMETRY))
        for e in native if e["command"] in set(A.NATIVE.values())|{"RDA","WRA"})
    A.require(expected_commands==actual_commands,"Native array/external address coverage differs: "+A.counter_difference(expected_commands,actual_commands))
    for ch in range(16):
        for trace_op,native_op in (("PIM_MAC_AB","MACAB"),("PIM_MAC_QK","MACQK")):
            lhs=[A.decode(e["address"])[4:] for e in records if e["operation"]==trace_op and A.decode(e["address"])[0]==ch]
            rhs=[(e["Row"],e["Column"]) for e in native if e["Channel"]==ch and e["command"]==native_op]
            A.require(lhs==rhs,"Native MAC order differs from complete reduction order")
    slots=A.audit_slot_events(gemv,manifest["case"],mac_context={(e["Channel"],e["clock"]):e["Row"] for e in native if e["command"]=="MACAB"},
        v_token_sharding=True,pv_token_sharding=True,qkv_fusion=True,whole_srf=True)
    counts=Counter(e["command"] for e in native)
    dq=sum(counts[c]*32 for c in A.EXTERNAL)
    A.require(dq==manifest["expected_dq_bytes"]["total"],"DQ byte contract differs")
    A.require(Counter(result["issued_commands"])==counts,"Native counters differ from trace")
    from act2_audit import audit_act2
    report=dict(status="passed",case=manifest["case"],trace_and_native_address_coverage=True,
        independent_trace_commands=sum(expected.values()),native_data_commands_checked=sum(actual_commands.values()),
        array_open_row_checks=A.audit_array_rows(native),unified_gemv=slots,
        split_activation=audit_act2(native,result["resolved_timings_ck"],staging_events=stage),
        new_local_write_precharge=audit_new_local_write_precharge(native,result["resolved_timings_ck"]),
        staging=audit_stages(stage,op,native,gemv,result),
        rank_controller=audit_ca(ca,native,stage,controller,gemv,cache,score,op),
        native_external_DQ_bytes=dq,native_command_counts=dict(counts),numerical_compute=False)
    if flags[0]:
        report["score_DRAM"]=audit_score_lifecycle(score,native,result)
    if flags[2]:
        report["BF16_output"]=audit_o_lifecycle(op,native,gemv,result,flags[1])
    # Check the one physical rank port across all modeled users, not merely
    # within each new operator's own event stream.
    port_events=[e for e in score if e["operation"]=="TRANSFER"]
    port_events += [e for e in op if e["operation"] in ("PV_GATHER_FP32","PV_TRANSFER_BF16","EXPORT_GATHER_FP32")]
    port_events += [e for e in cache if e["operation"]=="GATHER"]
    port_events += [e for e in gemv if e["operation"] in ("GEMV_Q_LOCAL","GEMV_Q_REMOTE")]
    intervals_disjoint(port_events,lambda e:e["Channel"],"Score/PV/O/Q/V users overlap the shared rank local port")
    report["shared_local_port_intervals_checked"]=len(port_events)
    dispatch=A.read_csv(sim/"frontend_dispatch.csv")
    report["channel_ingress"]=audit_ingress(raw,dispatch,A.read_csv(sim/"channel_ingress.csv"),A.read_csv(sim/"barriers.csv"),cfg["channel_queue_capacity"])
    report["rank_descriptors"]=audit_rank_descriptors(raw,collect("rank_descriptors.csv"),controller,ca,dispatch,cfg["frontend_ratio"])
    if manifest["case"]=="serial_layer_control":
        report["serial_continuity"]=A.audit_serial_continuity(manifest,result,sim,native,gemv)
    if manifest.get("qkv_fusion"):
        from token_pv_audit import audit_qkv,audit_local_cache
        report["qkv_fusion"]=audit_qkv(manifest,native,gemv)
        report["local_V_cache"]=audit_local_cache(cache,native,gemv,result,True)
    elif manifest["case"]=="v":
        from token_pv_audit import audit_local_cache
        report["local_V_cache"]=audit_local_cache(cache,native,gemv,result,False)
    report["trace_sha256"]=A.digest(directory/"workload.trace")
    report["native_sha256"]={p.name:A.digest(p) for p in files}
    return report


def augment_frozen_runs(root,output):
    """Read-only augmentation of 20 frozen runs; write a separate evidence file."""
    import json
    root,output=Path(root).resolve(strict=True),Path(output).resolve()
    A.require(not output.is_relative_to(root),"Augmentation report must be outside frozen result tree")
    A.require(not output.exists(),"Use a fresh augmentation report path")
    report=dict(status="passed",native_rerun=False,frozen_evidence_modified=False,runs={},raw_sha256={},
        audit_sha256={Path(__file__).name:A.digest(__file__),Path(A.__file__).name:A.digest(A.__file__)})
    for variant in ("neither","score_only","oproj_only","both"):
        for case in ("chain","pv","gate_up_gpu","down_gpu_input","serial_layer_control"):
            sim=root/variant/case/"simulation"
            result_path=sim/"result.json";result=A.read_json(result_path)
            stage_paths=[sim/f"staging_events.csv.ch{ch}" for ch in range(16)]
            A.require(all(p.is_file() for p in stage_paths),"Missing frozen staging evidence")
            op_paths=sorted(sim.glob("o_projection.csv.ch*"))
            expected_op=bool(result["config"].get("pim_o_projection") or result["config"].get("pim_pv_bf16_output"))
            A.require(len(op_paths)==(16 if expected_op else 0),"Frozen BF16 evidence channel count differs")
            if expected_op:
                A.require({p.name for p in op_paths}=={f"o_projection.csv.ch{ch}" for ch in range(16)},
                    "Frozen BF16 evidence omits or duplicates a channel")
            stage=[e for p in stage_paths for e in A.read_csv(p)]
            op=[e for p in op_paths for e in A.read_csv(p)]
            report["runs"][f"{variant}/{case}"]=dict(
                packet_scratch=audit_o_packet_scratch(op,result["resolved_timings_ck"]),
                DRAM_storage_reuse=audit_staging_generation_reuse(stage,op))
            for path in (result_path,*stage_paths,*op_paths):
                report["raw_sha256"][path.relative_to(root).as_posix()]=A.digest(path)
    A.require(len(report["runs"])==20,"Frozen augmentation lacks a requested run")
    provenance=root/"provenance.json"
    if provenance.is_file():report["raw_sha256"]["provenance.json"]=A.digest(provenance)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    return report


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Augment frozen native runs without modifying or rerunning them")
    parser.add_argument("--augment-frozen",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=augment_frozen_runs(args.augment_frozen,args.output)
    print(f"Passed frozen augmentation: {len(result['runs'])} runs, {len(result['raw_sha256'])} raw evidence hashes")

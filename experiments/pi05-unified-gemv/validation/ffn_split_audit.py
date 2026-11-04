"""Independent FFN bank partition, physical layout and streaming evidence."""
from collections import Counter, defaultdict

BASES = dict(gate=1048576, up=1087488, down=2097152)
INPUT_BASE = 524288


def require(value, message):
    if not value: raise AssertionError(message)


def banks(count):
    require(count in (1, 2), "FFN dedicated data-bank count must be one or two")
    data = [0] if count == 1 else [0, 4]
    return data, [b for b in range(16) if b not in data]


def geometry(operator, count):
    data, compute = banks(count)
    require(operator in BASES, "Unknown independent FFN operator")
    features, fragments = (64, 256) if operator == "down" else (256, 64)
    columns = (features+len(compute)-1)//len(compute)
    masks = [sum(1 << bank for ordinal, bank in enumerate(compute)
                 if column*len(compute)+ordinal < features) for column in range(columns)]
    return dict(base=BASES[operator], features=features, fragments=fragments, columns=columns,
        data=data, compute=compute, masks=masks, output_bytes=16*columns*4,
        reserved_weight_bytes=(20 if operator == "down" else 19)*2048)


def source_location(fragment, token, count):
    data, _ = banks(count)
    logical_row, within = divmod(fragment, 4)
    return data[logical_row%count], INPUT_BASE+(logical_row//count)*2048+within*512+token*32


def address(channel, bank, offset):
    return ((offset//2048)<<20) | ((bank%4)<<18) | ((bank//4)<<16) | (1<<15) | ((offset%2048//32)<<9) | (channel<<5) | (offset%32)


def weight_address(operator, reduction, feature, count):
    g=geometry(operator,count)
    channel, local=divmod(feature,g['features'])
    column, ordinal=divmod(local,len(g['compute']))
    fragment,lane=divmod(reduction,16)
    return address(channel,g['compute'][ordinal],g['base']+(fragment*g['columns']+column)*32+lane*2)


def passes_for(case):
    require(case in ('gate_up_gpu','down_gpu_input','combined_timing_chain','serial_layer_control'), 'Unknown FFN case')
    return ('down',) if case=='down_gpu_input' else ('gate','up') if case=='gate_up_gpu' else ('gate','up','down')


def expected_channel_pass(operator, count, channel):
    g=geometry(operator,count)
    out=[]
    def emit(op,offset,slot=-1,bank=0,extra=()):
        out.append((op,address(channel,bank,offset),slot,*extra))
    emit('PIM_FFN_PASS',g['base'])
    if operator!='up':
        emit('PIM_STAGE_BEGIN',INPUT_BASE,extra=(g['fragments']*512,16))
        for row in range(g['fragments']//4//count):
            for column in range(64):
                for bank in g['data']:emit('ST',INPUT_BASE+row*2048+column*32,bank=bank)
        emit('PIM_STAGE_SEAL',INPUT_BASE)
    def fill(f):
        bank,offset=source_location(f,0,count)
        emit('PIM_STAGE_FRAGMENT',offset,f%2,bank)
    fill(0)
    for fragment in range(g['fragments']):
        if fragment+1<g['fragments']:fill(fragment+1)
        for column in range(g['columns']):
            emit('PIM_MAC_AB',g['base']+(fragment*g['columns']+column)*32,fragment%2)
    for slot in (0,1):emit('PIM_GEMV_RELEASE',0,slot)
    for offset in range(0,g['features']*64,32):emit('PIM_FFN_RDRES',offset)
    return out


def expected_addresses(case, count):
    from audit_native import decode
    result=Counter()
    for channel in range(16):
        for operator in passes_for(case):
            for operation,addr,slot,*extra in expected_channel_pass(operator,count,channel):
                result[operation,*decode(addr),slot,*extra]+=1
    return result


def audit_trace_order(records, case, count):
    actual=defaultdict(list);active=set()
    for e in records:
        op=e['operation'];ch=e['address']>>5&15
        if op in ('BARRIER','WAIT'):continue
        if op=='PIM_FFN_PASS':active.add(ch)
        if ch not in active:continue
        extra=(e['stage_bytes'],e['stage_rows']) if op=='PIM_STAGE_BEGIN' else ()
        actual[ch].append((op,e['address'],e['buffer'],*extra))
    for channel in range(16):
        expected=[e for operator in passes_for(case) for e in expected_channel_pass(operator,count,channel)]
        require(actual[channel]==expected, f'FFN independent trace order/address differs on channel{channel}')
    return sum(map(len,actual.values()))


def audit_mapping_contract(mapping, count):
    data,compute=banks(count)
    require(mapping['data_banks']==data and mapping['compute_banks']==compute, 'FFN physical bank partition differs')
    require((mapping['input_srf_bytes_per_bank'],mapping['input_slots_per_bank'],mapping['input_slot_bytes'],
             mapping['output_capacity_bytes_per_bank'])==(1024,2,512,2048), 'FFN physical buffer capacity changed')
    for op in BASES:
        actual=mapping['passes'][op];g=geometry(op,count)
        expected=dict(weight_base=g['base'],input_fragments=g['fragments'],columns_per_fragment=g['columns'],
            logical_features_per_rank=g['features'],valid_bank_masks=g['masks'],output_bytes_per_compute_bank=g['output_bytes'],
            weight_used_bytes_per_compute_bank=g['fragments']*g['columns']*32,
            weight_reserved_bytes_per_compute_bank=g['reserved_weight_bytes'],valid_export_bytes_per_rank=g['features']*64)
        require(all(actual[k]==v for k,v in expected.items()), f'{op} layout differs from independent matrix geometry')
        require(g['output_bytes']<=2048 and g['fragments']*g['columns']*32<=g['reserved_weight_bytes'], 'FFN storage capacity exceeded')
        valid=[dict(bank=b,features=list(range(j,g['features'],len(compute)))) for j,b in enumerate(compute)]
        require(actual['valid_features_per_bank']==valid and actual['padded_bank_MACs_issued'] is False,
                'FFN padding aliases useful outputs or is computed')
    regions=[(0,65536),(65536,81408),(81920,83968),(83968,116736),(131072,180224),
             (262144,360448),(393216,491520),(524288,655360)]
    for op in BASES:
        g=geometry(op,count);region=(g['base'],g['base']+g['reserved_weight_bytes'])
        require(all(region[1]<=a or b<=region[0] for a,b in regions), 'New FFN weight region overlaps reserved legacy/cache/input storage')
        regions.append(region)
    return dict(data_banks=data,compute_banks=compute,input_SRF_bytes_per_bank=1024,
        input_views=2,input_view_bytes=512,output_capacity_bytes_per_bank=2048,
        no_extra_row_SRAM=True,GU_output_bytes_per_bank=geometry('gate',count)['output_bytes'],
        Down_output_bytes_per_bank=320,padding_MACs=0,physical_regions_disjoint=True)


def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result


def overlap_ticks(left,right):
    left,right=merged(left),merged(right);i=j=total=0
    while i<len(left) and j<len(right):
        total+=max(0,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<=right[j][1]:i+=1
        else:j+=1
    return total


def audit_runtime(manifest,records,native,gemv,stage,split,result):
    """Reconstruct tensor coverage, DRAM residency, both SRFs and true overlap."""
    from audit_native import decode
    count=manifest['ffn_data_banks'];data,compute=banks(count);mask=sum(1<<b for b in compute)
    operators=passes_for(manifest['case']);by_base={BASES[op]:op for op in operators}
    mapping=manifest['ffn_split_mapping']
    contracts=mapping.get('stages',{'single':mapping})
    for contract in contracts.values():audit_mapping_contract(contract,count)
    pass_events=[e for e in split if e['operation']=='PASS']
    require(Counter((e['Channel'],e['pass']) for e in pass_events)==Counter((ch,BASES[op]) for ch in range(16) for op in operators),
            'FFN PASS evidence omits/duplicates an operator or channel')
    requests={e['line']-1:e for e in records if e['operation'] not in ('BARRIER','WAIT')}
    first_pass=min(e['line'] for e in records if e['operation']=='PIM_FFN_PASS')
    attention_stages=sum(e['operation']=='PIM_STAGE_BEGIN' and decode(e['address'])[0]==0 and e['line']<first_pass for e in records)
    groups=defaultdict(list)
    for e in stage:
        if e['stage_generation']>attention_stages:groups[e['Channel'],e['stage_generation']].append(e)
    expected_groups={(ch,attention_stages+i+1) for ch in range(16) for i in range(len(operators))}
    require(set(groups)==expected_groups,'FFN staged input usage/generation coverage differs')
    timeline={(e['Channel'],e['clock']):e for e in native if e['command'] in ('MACMASK','RDRES')}
    physical=lambda e:tuple(e[k] for k in ('Channel','clock','Rank','BankGroup','Bank','Row','Column'))
    native_reads=Counter(physical(e) for e in native if e['command']=='LOCALRD' and e['Row']>=256)
    native_writes=Counter(physical(e) for e in native if e['command'] in ('WR','WRA') and e['Row']>=256)
    expected_reads,expected_writes=Counter(),Counter()
    fills=Counter((e['Channel'],e['clock'],e['buffer'],e['token_mask'],e['bank_mask']) for e in gemv if e['operation']=='GEMV_MULTICAST' and e['bank_mask']==mask)
    expected_fills=Counter();source_identity={}
    stage_read={};stage_broadcast={};read_records=[];row_reports=[]
    timings=result['resolved_timings_ck'];read_latency=timings['nCL']+timings['nBL_min']
    write_latency=timings['nCWL']+timings['nBL_max']+timings['nWR']
    input_bytes=read_bytes=0
    for (ch,gen),entries in sorted(groups.items()):
        op=operators[gen-attention_stages-1];g=geometry(op,count)
        byop=defaultdict(list)
        for e in entries:byop[e['operation']].append(e)
        require(set(byop)==({'REPLAY','LOCALRD','BROADCAST'} if op=='up' else {'BEGIN','WR','SEAL','LOCALRD','BROADCAST'}),'FFN input usage has unexpected staging operations')
        pass_event=next(e for e in pass_events if e['Channel']==ch and e['pass']==BASES[op])
        require(requests[pass_event['request_id']]['operation']=='PIM_FFN_PASS' and
                requests[pass_event['request_id']]['address']==address(ch,0,BASES[op]),'FFN PASS request/address identity differs')
        if op=='up':
            require(len(byop['REPLAY'])==1 and byop['REPLAY'][0]['bytes']==32768,'Up must replay the same complete staged X exactly once')
            sealed=byop['REPLAY'][0]['ready_at']
            require(sealed==byop['REPLAY'][0]['clock']==pass_event['clock'],'Up replay does not atomically preserve sealed input')
        else:
            require(len(byop['BEGIN'])==len(byop['SEAL'])==1,'FFN input requires one BEGIN and one SEAL')
            begin,seal=byop['BEGIN'][0],byop['SEAL'][0]
            require(pass_event['clock']<=begin['clock'],'FFN input begins before its matching PASS')
            require(begin['bytes']==g['fragments']*512 and begin['Row']==seal['Row']==256 and begin['Column']==seal['Column']==0,'FFN staging payload/base differs')
            require(all(e['clock']>=begin['clock'] and e['ready_at']==e['clock']+write_latency for e in byop['WR']),'FFN WR omits actual write recovery')
            require(seal['clock']>=max(e['ready_at'] for e in byop['WR']),'FFN input seal precedes a completed input write')
            sealed=seal['ready_at'];input_bytes+=g['fragments']*512
        locations={source_location(f,t,count):(f,t) for f in range(g['fragments']) for t in range(16)}
        for kind in ('LOCALRD','BROADCAST')+(() if op=='up' else ('WR',)):
            require(Counter((e['BankGroup']*4+e['Bank'],e['Row']*2048+e['Column']*32) for e in byop[kind])==Counter(locations.keys()),'FFN '+kind+' physical source coverage differs')
            require(all(e['Rank']==1 and e['bytes']==32 for e in byop[kind]),'FFN source packet rank/size differs')
        reads={(e['BankGroup']*4+e['Bank'],e['Row']*2048+e['Column']*32):e for e in byop['LOCALRD']}
        for e in byop['LOCALRD']:
            key=e['BankGroup']*4+e['Bank'],e['Row']*2048+e['Column']*32;f,t=locations[key]
            require(e['buffer']==f%2 and e['clock']>=sealed and e['ready_at']==e['clock']+read_latency,'FFN LOCALRD slot, seal dependency or column latency differs')
            stage_read[ch,BASES[op],f,t]=e;expected_reads[physical(e)]+=1
            read_records.append((op,e));read_bytes+=32
        for e in byop['WR']:expected_writes[physical(e)]+=1
        for e in byop['BROADCAST']:
            key=e['BankGroup']*4+e['Bank'],e['Row']*2048+e['Column']*32;f,t=locations[key];rd=reads[key]
            require(e['buffer']==f%2 and e['clock']>=rd['clock'] and e['ready_at']==max(rd['ready_at'],e['clock']),'FFN broadcast has wrong source slot/read dependency')
            stage_broadcast[ch,BASES[op],f,t]=e
            expected_fills[ch,e['ready_at'],e['buffer'],1<<t,mask]+=1
            source_identity[ch,e['ready_at'],e['buffer'],1<<t]=(op,f)
        row_reports.append(dict(channel=ch,operator=op,fully_read_rows=g['fragments']//4,
            fully_written_rows=0 if op=='up' else g['fragments']//4,columns_per_row=64,bytes_per_row=2048))
    require(not(expected_reads-native_reads) and not(expected_writes-native_writes),'FFN staging events lack real native array reads/writes')
    require(fills==expected_fills,'FFN broadcast completions lack exact compute-bank SRF fills')
    owners={};macs=defaultdict(list)
    for e in gemv:
        if e['operation']=='GEMV_MULTICAST' and e['bank_mask']==mask:
            identity=source_identity[e['Channel'],e['clock'],e['buffer'],e['token_mask']]
            key=e['Channel'],e['buffer'],e['generation']
            require(owners.get(key,identity)==identity,'FFN SRF generation mixes source fragments/passes');owners[key]=identity
        elif e['operation']=='GEMV_MAC' and (e['Channel'],e['clock']) in timeline and timeline[e['Channel'],e['clock']]['command']=='MACMASK':
            n=timeline[e['Channel'],e['clock']];offset=n['Row']*2048+n['Column']*32
            op=next((op for op in operators if BASES[op]<=offset<BASES[op]+geometry(op,count)['reserved_weight_bytes']),None)
            require(op is not None,'FFN MAC uses unknown weight region');g=geometry(op,count)
            f,column=divmod((offset-g['base'])//32,g['columns'])
            require(owners[e['Channel'],e['buffer'],e['generation']]==(op,f),'FFN MAC source/weight fragment identity differs')
            require(e['buffer']==f%2 and e['bank_mask']==g['masks'][column] and e['token_mask']==e['lane_mask']==65535 and e['service_cycles']==16,'FFN useful MAC coverage or service differs')
            macs[e['Channel'],op].append(e)
    expected_mac_events=Counter()
    for (ch,op),work in macs.items():
        g=geometry(op,count)
        for e in work:
            actual=timeline[ch,e['clock']];offset=actual['Row']*2048+actual['Column']*32
            f=(offset-g['base'])//(g['columns']*32)
            expected_mac_events[ch,e['clock'],e['ready_at'],g['base'],f,e['buffer'],e['bank_mask'],e['bank_mask'].bit_count()*32]+=1
    actual_mac_events=Counter((e['Channel'],e['clock'],e['ready_at'],e['pass'],e['fragment'],e['buffer'],e['bank_mask'],e['bytes'])
        for e in split if e['operation']=='MAC')
    require(actual_mac_events==expected_mac_events,'FFN MAC service records differ from actual masked compute')
    source_ops=defaultdict(dict);export_ops=defaultdict(dict);pass_by={(e['Channel'],e['pass']):e for e in pass_events}
    allowed={'PASS','SOURCE_READ','SOURCE_BROADCAST','MAC','EXPORT','EXPORT_COMPLETE'}
    for e in split:
        require(e['operation'] in allowed and e['Rank']==1 and e['pass'] in by_base,'Unknown FFN split native operation')
        require(e['request_id'] in requests,'FFN native event lacks original trace request')
        if e['operation'].startswith('SOURCE_'):
            key=e['Channel'],e['pass'],e['fragment'],e['feature']
            require(e['operation'] not in source_ops[key],'Duplicate data-bank SRF source-cell event');source_ops[key][e['operation']]=e
        elif e['operation'].startswith('EXPORT'):
            require(e['operation'] not in export_ops[e['request_id']],'Duplicate FFN result callback');export_ops[e['request_id']][e['operation']]=e
    require(set(source_ops)==set(stage_read),'Data-bank SRF events omit real input cells')
    cell_free={};source_holds=defaultdict(list)
    for key,ops in sorted(source_ops.items(),key=lambda item:item[1].get('SOURCE_READ',{}).get('clock',-1)):
        require(set(ops)=={'SOURCE_READ','SOURCE_BROADCAST'},'Data-bank SRF must record read and broadcast ownership')
        rd,bc=ops['SOURCE_READ'],ops['SOURCE_BROADCAST'];actual_rd,actual_bc=stage_read[key],stage_broadcast[key]
        bank=actual_rd['BankGroup']*4+actual_rd['Bank'];ch,base,f,t=key
        for e,actual in ((rd,actual_rd),(bc,actual_bc)):
            require((e['clock'],e['ready_at'],e['buffer'],e['bank_mask'],e['bytes'])==(actual['clock'],actual['ready_at'],f%2,1<<bank,32),'Data-bank SRF event differs from physical source operation')
            request=requests[e['request_id']]
            require(request['operation']=='PIM_STAGE_FRAGMENT' and decode(request['address'])==(ch,1,bank//4,bank%4,actual_rd['Row'],actual_rd['Column']-t),'Source SRF cell request/fragment identity differs')
        cell=(ch,bank,f%2,t)
        require(rd['clock']>=cell_free.get(cell,0),'Data-bank SRF cell overwritten before broadcast completes')
        cell_free[cell]=bc['ready_at'];source_holds[ch,bank].extend(((rd['clock'],1),(bc['ready_at'],-1)))
    peak_source_cells=0
    for changes in source_holds.values():
        live=0
        for clock,delta in sorted(changes,key=lambda item:(item[0],item[1])):
            live+=delta;peak_source_cells=max(peak_source_cells,live)
            require(0<=live<=32,'Data-bank reads require more than physical 1KiB SRF')
        require(live==0,'Data-bank SRF holds survive final broadcast')
    expected_exports={e['line']-1:e for e in records if e['operation']=='PIM_FFN_RDRES'}
    require(set(export_ops)==set(expected_exports),'FFN result stream omits/duplicates useful output packets')
    export_done=defaultdict(int)
    for request_id,ops in export_ops.items():
        require(set(ops)=={'EXPORT','EXPORT_COMPLETE'},'FFN output packet lacks real completion')
        e,done=ops['EXPORT'],ops['EXPORT_COMPLETE'];op=by_base[e['pass']];g=geometry(op,count)
        req=expected_exports[request_id];ch,rank,bg,bank,row,col=decode(req['address']);feature=(row*2048+col*32)//64
        require(feature<g['features'] and e['feature']==feature and e['bank_mask']==1<<compute[feature%len(compute)] and e['bytes']==32,'FFN export includes padding or wrong feature bank')
        actual=timeline[e['Channel'],e['clock']]
        require(actual['command']=='RDRES' and (actual['Row'],actual['Column'])==(row,col),'FFN output event lacks actual RDRES')
        require(e['clock']>=max(m['ready_at'] for m in macs[ch,op]) and done['clock']>=e['ready_at'],'FFN export precedes complete useful output')
        require(done['pass']==e['pass'] and done['feature']==feature and done['bytes']==32 and done['ready_at']==done['clock'],
                'FFN output callback changes feature/pass identity or completion time')
        export_done[ch,op]=max(export_done[ch,op],done['clock'])
    rows=[];tick_us=result['tCK_ns']/1000
    for ch in range(16):
        for index,op in enumerate(operators):
            g=geometry(op,count);work=macs[ch,op]
            require(len(work)==g['fragments']*g['columns'],'FFN MAC command coverage differs')
            scalar=sum(e['bank_mask'].bit_count()*256 for e in work)
            require(scalar==g['fragments']*g['features']*256,'FFN useful scalar MAC work changed or padding computed')
            if index:require(pass_by[ch,BASES[op]]['clock']>=export_done[ch,operators[index-1]],'FFN pass overwrites undrained output pool')
            reads=[e for name,e in read_records if name==op and e['Channel']==ch]
            intervals=merged((e['clock'],e['ready_at']) for e in work)
            ticks=overlap_ticks([(e['clock'],e['ready_at']) for e in reads],intervals)
            issues=sum(any(start<=e['clock']<end for start,end in intervals) for e in reads)
            broadcasts=[e for (c,base,f,t),e in stage_broadcast.items() if c==ch and base==BASES[op]]
            broadcast_issues=sum(any(start<=e['clock']<end for start,end in intervals) for e in broadcasts)
            if result['config']['ffn_overlap']:
                require(ticks>0 and issues>0 and broadcast_issues>0,'Streaming FFN never overlaps actual source reads/broadcasts with MAC service')
            else:require(ticks==issues==broadcast_issues==0,'Disabled FFN overlap still overlaps data transfer and MAC service')
            rows.append(dict(channel=ch,operator=op,scalar_MACs=scalar,MAC_commands=len(work),
                LOCALRD_issues_during_MAC=issues,BCAST_issues_during_MAC=broadcast_issues,
                LOCALRD_MAC_overlap_ck=ticks,LOCALRD_MAC_overlap_us=ticks*tick_us))
    return dict(status='passed',**audit_mapping_contract(next(iter(contracts.values())),count),
        passes=list(operators),ffn_overlap=result['config']['ffn_overlap'],whole_input_write_bytes=input_bytes,
        local_input_read_bytes=read_bytes,full_source_row_coverage=row_reports,overlap=rows,
        useful_scalar_MACs=sum(e['scalar_MACs'] for e in rows),source_SRF_peak_inflight_bytes=peak_source_cells*32,
        data_source_cells_checked=len(source_ops),output_packets_checked=len(export_ops),
        input_source_identity_verified=True,Gate_X_stored_once_read_twice='gate' in operators,
        data_compute_bank_isolation=True,numerical_compute=False)


def audit_attention_prefix(manifest,native,gemv,stage,split,result):
    """Keep all existing attention staging checks in a mixed FFN simulation.

    Native PASS clocks bound the prefix; input generations separate its
    independently known chain/V/PV layouts. Global CA, ACT and row checks
    always receive the full unfiltered simulation in audit_run.
    """
    from staging_audit import audit_staging
    from paired_audit import audit_paired_staging
    if manifest['case']!='serial_layer_control':return {}
    qkv=manifest.get('qkv_fusion',False);vtoken=manifest.get('v_token_sharding',False)
    pvtoken=manifest.get('pv_token_sharding',False);whole=result['config'].get('pim_whole_srf',False)
    names=('chain','pv') if qkv else ('chain','v','pv')
    pass_begin={ch:min(e['clock'] for e in split if e['Channel']==ch and e['operation']=='PASS') for ch in range(16)}
    starts={(e['Channel'],e['stage_generation']):e['clock'] for e in stage if e['operation']=='BEGIN'}
    reports={}
    for generation,name in enumerate(names,1):
        bounds={ch:(starts[ch,generation],starts[ch,generation+1] if generation<len(names) else pass_begin[ch]) for ch in range(16)}
        select=lambda e:bounds[e['Channel']][0]<=e['clock']<bounds[e['Channel']][1]
        commands=[e for e in native if select(e)];events=[e for e in gemv if select(e)]
        staging=[dict(e,stage_generation=1) for e in stage if e['stage_generation']==generation]
        count=result['config'].get('pim_staging_banks',1)
        reports[name]=dict(staging=audit_staging(staging,commands,events,name,result['resolved_timings_ck'],count,vtoken,pvtoken,qkv,whole),
            batching=audit_paired_staging(staging,commands,events,name,staging_banks=count,v_token_sharding=vtoken,
                pv_token_sharding=pvtoken,qkv_fusion=qkv,whole_srf=whole))
    return reports

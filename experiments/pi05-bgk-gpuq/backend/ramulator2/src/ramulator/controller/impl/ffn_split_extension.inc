// Dedicated FFN data-bank storage and compute-bank SIMD. No extra SRAM:
// compute banks retain two 512B views and at most 2KiB of FP32 outputs.
int m_ffn_data_banks=0,m_split_base=-1,m_split_columns=0,m_split_features=0,m_split_fragments=0;
bool m_ffn_overlap=true,m_split_stage_mode=false;
int m_actmask=-1,m_premask=-1,m_macmask=-1;
int m_split_macs=0,m_split_exports=0,m_split_export_done=0;
Clk_t m_split_ready=0;
std::string m_masked_path,m_ffn_split_path;
std::ofstream m_masked_events,m_ffn_split_events;
size_t s_masked_weight_bytes=0;
// Occupancy timestamps of exactly two 16x32B source SRF views per bank.
std::array<std::array<std::array<Clk_t,16>,2>,16> m_split_source_hold{};
struct BankMaskScope {
  DRAMDevice& device; uint64_t previous;
  BankMaskScope(DRAMDevice& d,uint64_t mask):device(d),previous(d.active_bank_mask) { device.active_bank_mask=mask; }
  ~BankMaskScope() { device.active_bank_mask=previous; }
};
bool split_active() const { return m_ffn_data_banks && m_split_stage_mode; }
int split_compute_mask() const { return m_ffn_data_banks==1?65534:65518; }
int split_compute_count() const { return 16-m_ffn_data_banks; }
int split_base_at(int offset) const {
  if(offset>=1048576 && offset<1048576+19*2048) return 1048576;
  if(offset>=1087488 && offset<1087488+19*2048) return 1087488;
  if(offset>=2097152 && offset<2097152+20*2048) return 2097152;
  return -1;
}
bool is_split_mac(const Request& req) const {
  return m_ffn_data_banks && req.type_id==Request::Type::PIM_MAC_AB && split_base_at(gemv_offset(req))>=0;
}
bool is_split_control(const Request& req) const {
  return req.type_id==Request::Type::PIM_FFN_PASS || req.type_id==Request::Type::PIM_FFN_RDRES;
}
int split_bank_columns(int bank) const {
  int ordinal=0;
  for(int b=0;b<bank;++b) if(split_compute_mask()&(1<<b)) ++ordinal;
  return (m_split_features-ordinal+split_compute_count()-1)/split_compute_count();
}
int split_column_mask(int column) const {
  int ordinal=0,mask=0;
  for(int b=0;b<16;++b) if(split_compute_mask()&(1<<b)) {
    if(column*split_compute_count()+ordinal<m_split_features) mask|=1<<b;
    ++ordinal;
  }
  return mask;
}
void init_ffn_split() {
  if(m_ffn_data_banks<0 || m_ffn_data_banks>2) throw std::runtime_error("ffn_data_banks must be 0, 1 or 2");
  if(!m_ffn_data_banks) return;
  if(!m_rank_controller || !m_rank_staging || !m_unified_gemv || m_ffn_dram)
    throw std::runtime_error("FFN bank split requires staged unified rank controller");
  m_actmask=m_device.m_spec->get_command_id("ACTMASK");
  m_premask=m_device.m_spec->get_command_id("PREMASK");
  m_macmask=m_device.m_spec->get_command_id("MACMASK");
}
void setup_ffn_split() {
  if(!m_ffn_data_banks) return;
  m_stats.add("masked_weight_bytes",s_masked_weight_bytes);
  if(!m_masked_path.empty()) {
    m_masked_events.open(m_masked_path);
    if(!m_masked_events) throw std::runtime_error("Cannot open masked events");
    m_masked_events<<"clock,ready_at,command,Channel,Rank,bank_mask,Row,Column,buffer,pass,fragment,request_id\n";
  }
  if(!m_ffn_split_path.empty()) {
    m_ffn_split_events.open(m_ffn_split_path);
    if(!m_ffn_split_events) throw std::runtime_error("Cannot open FFN split events");
    m_ffn_split_events<<"clock,ready_at,operation,Channel,Rank,pass,fragment,buffer,bank_mask,bytes,feature,request_id\n";
  }
}
void record_split(const Request& req,const char* operation,Clk_t ready,int bytes=0,int feature=-1,int mask=-1) {
  if(!m_ffn_split_events) return;
  m_ffn_split_events<<m_clk<<','<<ready<<','<<operation<<','<<m_channel_id<<",1,"<<m_split_base<<','
    <<(is_split_mac(req)?(gemv_offset(req)-m_split_base)/(m_split_columns*32):
       is_stage(req)?(stage_logical_offset(req)-m_stage.base)/512:-1)<<','<<req.pim_buffer<<','
    <<(mask<0?split_compute_mask():mask)<<','<<bytes<<','<<feature<<','<<req.ingress_id<<'\n';
}
void record_masked(const Request& req,int mask) {
  if(!m_masked_events) return;
  const auto& a=req.addr_vec;
  const int duration=std::max(1,m_device.m_spec->command_cycles[req.command]);
  m_masked_events<<m_clk<<','<<m_clk+duration<<','<<m_device.m_spec->command_names[req.command]<<','
    <<m_channel_id<<",1,"<<mask<<','<<a[m_device.m_spec->get_level_id("Row")]<<','
    <<a[m_device.m_spec->get_level_id("Column")]<<','<<req.pim_buffer<<','<<m_split_base<<','
    <<(gemv_offset(req)-m_split_base)/(m_split_columns*32)<<','<<req.ingress_id<<'\n';
}
bool send_split(Request& req) {
  if(!m_ffn_data_banks) throw std::runtime_error("FFN split request requires dedicated data banks");
  if(req.addr_vec[m_rank_level]!=1 || req.addr_vec[m_bg_level]!=0 || req.addr_vec[m_bank_level]!=0)
    throw std::runtime_error("FFN split requires canonical rank1 address");
  if(is_split_control(req) && req.pim_buffer!=-1) throw std::runtime_error("FFN control does not take input slot");
  if(is_split_mac(req) && (req.pim_buffer<0 || req.pim_buffer>1))
    throw std::runtime_error("FFN split requires two 512B SRF views");
  req.pim_rank_mask=2;req.pim_bank_mask=uint64_t(split_compute_mask())<<16;
  req.final_command=is_split_mac(req)?m_macmask:req.type_id==Request::Type::PIM_FFN_RDRES?m_rdres:-1;
  req.addr_vec[m_bg_level]=req.addr_vec[m_bank_level]=-1;
  req.arrive=m_clk;m_pim_buffer.enqueue(req);++s_pim_requests;return true;
}
bool split_outputs_retired() const {
  return m_split_base<0 || (m_split_macs==m_split_columns*m_split_fragments &&
    m_split_export_done==m_split_features*2 && m_clk>=m_split_ready);
}
bool issue_split_pass(ReqBuffer::iterator it) {
  const int base=gemv_offset(*it);
  if(split_base_at(base)!=base) throw std::runtime_error("FFN PASS requires exact weight base");
  if(m_split_base>=0 && !split_outputs_retired()) {
    if(m_split_exports==m_split_features*2 && m_split_export_done<m_split_exports) return false;
    throw std::runtime_error("FFN PASS before previous output drain");
  }
  if(!output_other_ready()) return false;
  for(int b=16;b<32;++b) for(int s=0;s<4;++s)
    if(m_gemv_slots[b][s].kind!=GemvSlot::Empty)
      throw std::runtime_error("FFN PASS requires all SRF views released");
  if(base==1087488) {
    if(m_split_base!=1048576 || !m_stage.sealed || m_stage.rows!=16 || m_stage.bytes!=32768 ||
        m_stage.loads.size()!=static_cast<size_t>(m_stage.bytes/32))
      throw std::runtime_error("FFN Up replay requires completed Gate input");
    m_stage.loads.clear();++m_stage.generation;
    record_stage(*it,"REPLAY",m_clk,m_clk,m_stage.bytes);
  }
  m_split_base=base;m_split_features=base==2097152?64:256;m_split_fragments=base==2097152?256:64;
  m_split_columns=(m_split_features+split_compute_count()-1)/split_compute_count();
  if(16*m_split_columns*4>2048) throw std::runtime_error("FFN output exceeds 2KiB per compute bank");
  m_split_macs=m_split_exports=m_split_export_done=0;m_split_ready=m_clk;m_split_stage_mode=true;
  record_split(*it,"PASS",m_clk);enqueue_bgk_done(it,m_clk);return true;
}
void split_source_read(const Request& req,Clk_t ready) {
  const int b=req.addr_vec[m_bg_level]*m_banks_per_bg+req.addr_vec[m_bank_level];
  const int row=((stage_logical_offset(req)-m_stage.base)%512)/32;
  if(m_clk<m_split_source_hold[b][req.pim_buffer][row])
    throw std::runtime_error("FFN source SRF overwrite before broadcast");
  m_split_source_hold[b][req.pim_buffer][row]=ready;
  record_split(req,"SOURCE_READ",ready,32,row,1<<b);
}
bool split_fill_ready(const Request& req) const {
  const int offset=stage_logical_offset(req)-m_stage.base;
  const int f=offset/512,row=(offset%512)/32,slot=req.pim_buffer;
  if(m_stage.rows!=16 || slot<0 || slot>1 || slot!=f%2)
    throw std::runtime_error("FFN split source/slot geometry mismatch");
  if(!m_ffn_overlap && (m_split_macs<f*m_split_columns || m_clk<m_split_ready)) return false;
  const int source=req.addr_vec[m_bg_level]*m_banks_per_bg+req.addr_vec[m_bank_level];
  if(m_clk<m_split_source_hold[source][slot][row]) return false;
  if(row && (m_stage_slot_generation[slot]!=m_stage.generation || m_stage_slot_fragment[slot]!=f))
    throw std::runtime_error("staging mixed source fragments");
  for(int b=0;b<16;++b) if(split_compute_mask()&(1<<b)) {
    const auto& state=m_gemv_slots[b+16][slot];const auto& input=m_bank_inputs[b+16][slot];
    if(!row) {
      if(m_clk<std::max({input.ready_at,input.in_use_until,input.local_in_use_until})) return false;
      if(state.kind!=GemvSlot::Empty && (!state.expected_columns || state.mac_columns!=state.expected_columns)) return false;
    } else if(state.kind!=GemvSlot::External || state.lanes[row])
      throw std::runtime_error("staging duplicate row or missing first fragment row");
  }
  return true;
}
void split_fill(const Request& req,Clk_t ready) {
  const int offset=stage_logical_offset(req)-m_stage.base,row=(offset%512)/32,slot=req.pim_buffer;
  if(row==0) {m_stage_slot_fragment[slot]=offset/512;m_stage_slot_generation[slot]=m_stage.generation;}
  const int source=req.addr_vec[m_bg_level]*m_banks_per_bg+req.addr_vec[m_bank_level];
  m_split_source_hold[source][slot][row]=ready;
  record_split(req,"SOURCE_BROADCAST",ready,32,row,1<<source);
  int generation=0;
  for(int b=0;b<16;++b) if(split_compute_mask()&(1<<b)) {
    const int bank=b+16;
    if(row==0) gemv_start_bank(bank,slot,GemvSlot::External);
    auto& state=m_gemv_slots[bank][slot];state.lanes[row]=65535;generation=state.generation;
    auto& input=m_bank_inputs[bank][slot];input.written=true;input.ready_at=std::max(input.ready_at,ready);
  }
  record_gemv(req,"GEMV_MULTICAST",ready,ready,split_compute_mask(),1<<row,65535,32,generation);
}
bool issue_split_mac(ReqBuffer::iterator it) {
  if(!split_active() || split_base_at(gemv_offset(*it))!=m_split_base)
    throw std::runtime_error("FFN MAC requires matching PASS");
  const int index=(gemv_offset(*it)-m_split_base)/32,column=index%m_split_columns,fragment=index/m_split_columns;
  if(index!=m_split_macs || index>=m_split_fragments*m_split_columns)
    throw std::runtime_error("FFN MAC column order mismatch");
  const int mask=split_column_mask(column),slot=it->pim_buffer;
  it->pim_bank_mask=uint64_t(mask)<<16;
  BankMaskScope selection(m_device,it->pim_bank_mask);
  it->command=get_preq_command(m_macmask,it->addr_vec);
  if(it->command==m_macmask) {
    if(m_stage_slot_generation[slot]!=m_stage.generation || m_stage_slot_fragment[slot]!=fragment) return false;
    if(m_clk<m_split_ready) return false;
    for(int b=0;b<16;++b) if(mask&(1<<b)) {
      const auto& state=m_gemv_slots[b+16][slot];
      if(state.kind!=GemvSlot::External) return false;
      for(int row=0;row<16;++row) if(state.lanes[row]!=65535) return false;
      if(m_clk<m_bank_inputs[b+16][slot].ready_at) return false;
      if(state.mac_columns!=column) throw std::runtime_error("FFN input column coverage mismatch");
    }
  }
  if(!check_timing(it->command,it->addr_vec)) return false;
  m_device.issue_command(it->command,it->addr_vec,m_clk);rr_notify_command(*it);record_masked(*it,mask);
  if(it->command!=m_macmask) return true;
  const Clk_t ready=m_clk+16;
  for(int b=0;b<16;++b) if(mask&(1<<b)) {
    auto& state=m_gemv_slots[b+16][slot];state.expected_columns=split_bank_columns(b);++state.mac_columns;
    state.weight_block=gemv_offset(*it)-column*32;
    m_bank_inputs[b+16][slot].in_use_until=ready;m_bank_compute_ready[b+16]=ready;
  }
  ++m_split_macs;m_split_ready=ready;s_masked_weight_bytes+=std::popcount(static_cast<unsigned>(mask))*32;
  const int first=std::countr_zero(static_cast<unsigned>(mask))+16;
  record_gemv(*it,"GEMV_MAC",m_clk,ready,mask,65535,65535,32,m_gemv_slots[first][slot].generation,column+1);
  record_split(*it,"MAC",ready,std::popcount(static_cast<unsigned>(mask))*32,-1,mask);
  enqueue_bgk_done(it,ready);return true;
}
bool issue_split_export(ReqBuffer::iterator it) {
  if(!split_active() || m_split_macs!=m_split_fragments*m_split_columns)
    throw std::runtime_error("FFN result export requires complete pass");
  if(gemv_offset(*it)!=m_split_exports*32 || m_split_exports>=m_split_features*2)
    throw std::runtime_error("FFN result export address/count mismatch");
  if(m_clk<m_split_ready || !check_timing(m_rdres,it->addr_vec)) return false;
  it->command=m_rdres;m_device.issue_command(m_rdres,it->addr_vec,m_clk);rr_notify_command(*it);
  const int feature=m_split_exports/2,ordinal=feature%split_compute_count();
  int source=-1,n=0;for(int b=0;b<16;++b) if(split_compute_mask()&(1<<b)) {if(n++==ordinal)source=b;}
  const Clk_t ready=m_clk+std::max(m_pim_rd_latency,m_bgk_input_cycles);
  ++m_split_exports;record_split(*it,"EXPORT",ready,32,feature,1<<source);
  enqueue_bgk_done(it,ready);return true;
}
// One fine command per rank CA slot. Give the earliest reduction-order MAC
// first chance, then progress the earliest prefetch. A busy MAC leaves its
// 16-cycle execution window available to source reads and broadcasts.
bool issue_split_ready() {
  auto first=m_pim_buffer.begin();
  if(first->type_id==Request::Type::PIM_FFN_PASS) return issue_split_pass(first);
  if(first->type_id==Request::Type::PIM_FFN_RDRES) return issue_split_export(first);
  if(!split_active()) throw std::runtime_error("FFN MAC requires matching PASS");
  if(is_gemv_release(*first)) return issue_gemv_release(first);
  auto load=m_pim_buffer.end(),mac=m_pim_buffer.end();
  for(auto it=m_pim_buffer.begin();it!=m_pim_buffer.end();++it) {
    if(is_stage(*it)) {
      if(it->type_id==Request::Type::PIM_STAGE_BEGIN || it->type_id==Request::Type::PIM_STAGE_SEAL) {
        if(it==first) return issue_stage(it);
        break;
      }
      if(load==m_pim_buffer.end()) load=it;
    } else if(is_split_mac(*it)) {if(mac==m_pim_buffer.end()) mac=it;}
    else break;
  }
  if(mac!=m_pim_buffer.end() && issue_split_mac(mac)) return true;
  if(load!=m_pim_buffer.end() && issue_stage(load)) return true;
  return false;
}

// Hierarchical rank controller and whole-operator input storage.
// Only coverage/timing metadata is kept here; payloads reside in source-bank DRAM.
bool m_rank_staging=false, m_rank_controller=false, m_rank_fine_context=false;
int m_staging_banks=1;
Clk_t m_global_ca_ready=0;
std::vector<Clk_t> m_rank_ca_ready;
std::string m_rank_ca_path, m_staging_path;
std::ofstream m_rank_ca_events, m_staging_events;
struct RankFineScope {
  bool& flag; bool previous;
  RankFineScope(bool& f,bool enabled):flag(f),previous(f) { if(enabled) flag=true; }
  ~RankFineScope() { flag=previous; }
};
struct StagePayload {
  bool active=false,sealed=false;
  int base=0,bytes=0,rows=0,generation=0;
  std::unordered_map<int,Clk_t> writes;
  std::unordered_map<int,int> loads;
} m_stage;
std::array<int,2> m_stage_slot_fragment{-1,-1},m_stage_slot_generation{0,0};
bool is_stage(const Request& req) const {
  return req.type_id>=Request::Type::PIM_STAGE_BEGIN && req.type_id<=Request::Type::PIM_STAGE_FRAGMENT;
}
int stage_offset(const Request& req) const {
  const auto& a=req.pim_origin_addr_vec.empty()?req.addr_vec:req.pim_origin_addr_vec;
  return a[m_device.m_spec->get_level_id("Row")]*2048+a[m_device.m_spec->get_level_id("Column")]*32;
}
void stage_address(const Request& req) const {
  const bool control=req.type_id==Request::Type::PIM_STAGE_BEGIN || req.type_id==Request::Type::PIM_STAGE_SEAL;
  const int rank=req.addr_vec[m_rank_level],bg=req.addr_vec[m_bg_level],bank=req.addr_vec[m_bank_level];
  if(m_staging_banks==1) {
    if(rank!=1 || bg!=0 || bank!=0)
      throw std::runtime_error("staging requires rank1 bank0");
    return;
  }
  if(control) {
    if(rank!=1 || bg!=0 || bank!=0)
      throw std::runtime_error("staging control requires rank1 bank0");
  } else {
    if(rank!=1 || (bg!=0 && bg!=1) || bank!=0)
      throw std::runtime_error("staging requires rank1 BG0/BG1 bank0");
    if(stage_offset(req)>=256*2048+65536)
      throw std::runtime_error("staging source bank capacity exceeded");
  }
}
int stage_logical_offset(const Request& req) const {
  const int physical=stage_offset(req);
  if(m_staging_banks==1) return physical;
  if(!m_stage.active || m_stage.rows<=0)
    throw std::runtime_error("staging write outside active payload");
  // Alternating logical fragments occupy BG0/B0 and BG1/B0. Each
  // descriptor remains on its physical source bank for all token reads.
  if(physical<m_stage.base) return physical; // Keep below-region addresses out of the logical payload.
  const int packet=(physical-m_stage.base)/32;
  const int fragment=2*(packet/m_stage.rows)+req.addr_vec[m_bg_level];
  return m_stage.base+(fragment*m_stage.rows+packet%m_stage.rows)*32;
}
void record_ca(Clk_t clock,Clk_t ready,int rank,const std::string& command,const char* scope) {
  if(m_rank_ca_events) m_rank_ca_events<<clock<<','<<ready<<','<<m_channel_id<<','<<rank<<','<<command<<','<<scope<<'\n';
}
void record_stage(const Request& req,const char* operation,Clk_t clock,Clk_t ready,int bytes=32) {
  if(!m_staging_events) return;
  const auto& a=req.pim_origin_addr_vec.empty()?req.addr_vec:req.pim_origin_addr_vec;
  m_staging_events<<clock<<','<<ready<<','<<operation<<','<<m_channel_id<<','<<a[m_rank_level]<<','<<a[m_bg_level]<<','<<a[m_bank_level]<<','<<a[m_device.m_spec->get_level_id("Row")]<<','<<a[m_device.m_spec->get_level_id("Column")]<<','<<req.pim_buffer<<','<<bytes<<','<<m_stage.generation<<'\n';
}
bool rank_ca_available(int rank,Clk_t clock) const {
  if(!m_rank_controller) return true;
  if(rank>=0) return clock>=m_rank_ca_ready.at(rank);
  return std::all_of(m_rank_ca_ready.begin(),m_rank_ca_ready.end(),[&](Clk_t ready){return clock>=ready;});
}
bool rank_pseudo(const char* command,int rank=1) {
  if(!m_rank_controller) return true;
  if(!rank_ca_available(rank,m_clk)) return false;
  m_rank_ca_ready[rank]=m_clk+1;
  record_ca(m_clk,m_clk+1,rank,command,"rank");
  return true;
}
void init_rank_staging() {
  if(m_staging_banks!=1 && m_staging_banks!=2)
    throw std::runtime_error("pim_staging_banks must be 1 or 2");
  if(m_staging_banks!=1 && !m_rank_staging)
    throw std::runtime_error("multiple staging banks require rank staging");
  if(m_rank_staging && (!m_rank_controller || !m_unified_gemv || m_ffn_dram))
    throw std::runtime_error("rank staging requires rank controller, unified GEMV and GPU FFN");
  m_rank_ca_ready.resize(m_inputs.size(),0);
  if(!m_rank_controller) return;
  m_device.command_admission=[this](int command,const AddrVec_t& a,Clk_t clock) {
    return rank_ca_available(a[m_rank_level],clock) && (m_rank_fine_context || clock>=m_global_ca_ready);
  };
  m_device.command_observer=[this](int command,const AddrVec_t& a,Clk_t clock) {
    const int rank=a[m_rank_level];
    if(!rank_ca_available(rank,clock) || (!m_rank_fine_context && clock<m_global_ca_ready))
      throw std::runtime_error("rank/global command bus collision");
    const int cycles=std::max(1,m_device.m_spec->command_cycles[command]);
    const auto& name=m_device.m_spec->command_names[command];
    for(int r=0;r<static_cast<int>(m_rank_ca_ready.size());++r) if(rank<0 || rank==r) {
      m_rank_ca_ready[r]=clock+cycles;
      record_ca(clock,clock+cycles,r,name,"rank");
    }
    if(!m_rank_fine_context) {
      m_global_ca_ready=clock+cycles;
      record_ca(clock,clock+cycles,rank,name,"global");
    }
  };
}
const char* coarse_name(const Request& req) const {
  if(req.pim_rank_batch) return rank_descriptor_name(req.type_id);
  switch(req.type_id) {
    case Request::Type::PIM_STAGE_BEGIN:return "PIM_STAGE_BEGIN";
    case Request::Type::PIM_STAGE_SEAL:return "PIM_STAGE_SEAL";
    case Request::Type::PIM_STAGE_LOAD:return "PIM_STAGE_LOAD";
    case Request::Type::PIM_STAGE_FRAGMENT:return "PIM_STAGE_FRAGMENT";
    case Request::Type::PIM_GEMV_RELEASE:return "PIM_GEMV_RELEASE";
    case Request::Type::PIM_MAC_AB:return "PIM_MAC_AB";
    case Request::Type::PIM_RD_RES:return "PIM_RD_RES";
    case Request::Type::PIM_WR_GB:return "PIM_WR_GB";
    case Request::Type::PIM_MAC_QK:return "PIM_MAC_QK";
    case Request::Type::PIM_RD_BG:return "PIM_RD_BG";
    default:return is_rank_rope(req)?rr_name(req.type_id):"PIM_DISPATCH";
  }
}
bool send_stage(Request& req) {
  if(!m_rank_staging) throw std::runtime_error("staging requests require pim_rank_staging");
  stage_address(req);
  const bool load=req.type_id==Request::Type::PIM_STAGE_LOAD || req.type_id==Request::Type::PIM_STAGE_FRAGMENT;
  if(load && req.pim_buffer<0)
    throw std::runtime_error("staging load requires explicit slot");
  if(!load && req.pim_buffer>=0)
    throw std::runtime_error("staging control does not take a slot");
  req.pim_rank_mask=2;req.pim_bank_mask=uint64_t{65535}<<16;
  req.final_command=load?m_localrd:-1;
  req.arrive=m_clk;m_pim_buffer.enqueue(req);++s_pim_requests;
  return true;
}
bool stage_fill_ready(const Request& req) {
  const int row=((stage_logical_offset(req)-m_stage.base)/32)%m_stage.rows,slot=req.pim_buffer;
  // Distinct rows pipeline at native nCCDL. Only a new fragment waits for
  // the previous slot consumer; earlier reads of this fragment may be in flight.
  if(row==0 && !gemv_slot_idle(slot)) return false;
  const int fragment=(stage_logical_offset(req)-m_stage.base)/(m_stage.rows*32);
  if(row!=0 && (m_stage_slot_generation[slot]!=m_stage.generation || m_stage_slot_fragment[slot]!=fragment))
    throw std::runtime_error("staging mixed source fragments");
  for(int bank=16;bank<32;++bank) {
    const auto& state=m_gemv_slots[bank][slot];
    if(state.kind==GemvSlot::Q) throw std::runtime_error("Unified GEMV Q slot held until PIM_GEMV_RELEASE");
    if(row==0) {
      if(state.kind!=GemvSlot::Empty && (!state.expected_columns || state.mac_columns!=state.expected_columns))
        throw std::runtime_error("staging overwrite before full consumption");
    } else if(state.kind!=GemvSlot::External || state.lanes[row])
      throw std::runtime_error("staging duplicate row or missing first fragment row");
  }
  return true;
}
void stage_fill(const Request& req,Clk_t ready) {
  const int row=((stage_logical_offset(req)-m_stage.base)/32)%m_stage.rows,slot=req.pim_buffer;
  if(row==0) {
    m_stage_slot_fragment[slot]=(stage_logical_offset(req)-m_stage.base)/(m_stage.rows*32);
    m_stage_slot_generation[slot]=m_stage.generation;
  }
  for(int bank=16;bank<32;++bank) {
    if(row==0) gemv_start_bank(bank,slot,GemvSlot::External);
    auto& state=m_gemv_slots[bank][slot];state.lanes[row]=65535;
    auto& input=m_bank_inputs[bank][slot];input.written=true;input.ready_at=std::max(input.ready_at,ready);
  }
  record_gemv(req,"GEMV_MULTICAST",ready,ready,65535,1<<row,65535,32,m_gemv_slots[16][slot].generation);
}
bool issue_stage(ReqBuffer::iterator it) {
  stage_address(*it);
  const int physical_offset=stage_offset(*it);
  if(it->type_id==Request::Type::PIM_STAGE_BEGIN) {
    if(physical_offset!=256*2048 || it->pim_stage_bytes<=0 || it->pim_stage_bytes>131072 ||
       (it->pim_stage_rows!=8 && it->pim_stage_rows!=16) || it->pim_stage_bytes%(it->pim_stage_rows*32))
      throw std::runtime_error("invalid staging region/size/rows");
    if(m_stage.active) {
      if(!m_stage.sealed || m_stage.loads.size()!=static_cast<size_t>(m_stage.bytes/32))
        throw std::runtime_error("staging overwrite before full consumption");
      for(int slot=0;slot<2;++slot) for(int bank=16;bank<32;++bank) {
        const auto& state=m_gemv_slots[bank][slot];
        if(state.kind==GemvSlot::External && (!state.expected_columns || state.mac_columns!=state.expected_columns))
          throw std::runtime_error("staging overwrite before full consumption");
      }
    }
    if(!gemv_slot_idle(0) || !gemv_slot_idle(1)) return false;
    const int generation=m_stage.generation+1;
    m_stage=StagePayload{};m_stage.active=true;m_stage.base=physical_offset;m_stage.bytes=it->pim_stage_bytes;m_stage.rows=it->pim_stage_rows;m_stage.generation=generation;
    record_stage(*it,"BEGIN",m_clk,m_clk,m_stage.bytes);enqueue_bgk_done(it,m_clk);return true;
  }
  if(it->type_id==Request::Type::PIM_STAGE_SEAL) {
    if(!m_stage.active || physical_offset!=m_stage.base || m_stage.sealed)
      throw std::runtime_error("invalid staging seal");
    if(m_stage.writes.size()!=static_cast<size_t>(m_stage.bytes/32))
      throw std::runtime_error("staging seal requires every declared input write");
    for(int i=0;i<m_stage.bytes;i+=32) {
      auto found=m_stage.writes.find(m_stage.base+i);
      if(found==m_stage.writes.end()) throw std::runtime_error("staging seal requires every declared input write");
      if(found->second<0 || m_clk<found->second) return false;
    }
    m_stage.sealed=true;record_stage(*it,"SEAL",m_clk,m_clk,m_stage.bytes);enqueue_bgk_done(it,m_clk);return true;
  }
  if(!m_stage.active || !m_stage.sealed) throw std::runtime_error("staging load requires sealed input");
  const int offset=stage_logical_offset(*it);
  if(offset<m_stage.base || offset>=m_stage.base+m_stage.bytes || !m_stage.writes.count(offset))
    throw std::runtime_error("staging load address outside payload");
  if(it->type_id==Request::Type::PIM_STAGE_FRAGMENT && it->pim_stage_cursor==0 && ((offset-m_stage.base)/32)%m_stage.rows)
    throw std::runtime_error("staging fragment must begin at a fragment boundary");
  if(it->pim_q_stage==1) {
    if(!rank_pseudo("BCAST")) return false;
    const Clk_t ready=std::max(it->depart,m_clk);
    stage_fill(*it,ready);record_stage(*it,"BROADCAST",m_clk,ready);
    ++m_stage.loads[offset];
    if(it->type_id==Request::Type::PIM_STAGE_FRAGMENT && ++it->pim_stage_cursor<m_stage.rows) {
      const int next=physical_offset+32;
      it->addr_vec[m_device.m_spec->get_level_id("Row")]=next/2048;
      it->addr_vec[m_device.m_spec->get_level_id("Column")]=(next%2048)/32;
      it->pim_origin_addr_vec=it->addr_vec;
      it->pim_q_stage=0;
    } else enqueue_bgk_done(it,ready);
    return true;
  }
  if(m_stage.loads.count(offset)) throw std::runtime_error("duplicate staging load");
  if(!stage_fill_ready(*it)) return false;
  it->command=get_preq_command(m_localrd,it->addr_vec);
  if(!issue_internal_array_command(it)) return false;
  if(it->command!=m_localrd) return true;
  it->depart=m_clk+m_device.m_spec->get_timing_value("nCL")+m_bgk_input_cycles;
  it->pim_q_stage=1;record_stage(*it,"LOCALRD",m_clk,it->depart);
  return true;
}

void stage_mac_identity(const Request& req) const {
  if(!m_rank_staging) return;
  const int slot=req.pim_buffer,offset=gemv_offset(req);
  if(slot<0 || m_stage_slot_generation[slot]!=m_stage.generation)
    throw std::runtime_error("staging source/weight fragment mismatch");
  const int base=offset<65536?0:offset<100352?83968:offset<116736?100352:offset<327680?262144:327680;
  const int fragment=(offset-base)/(gemv_columns(req)*32);
  if(fragment!=m_stage_slot_fragment[slot])
    throw std::runtime_error("staging source/weight fragment mismatch");
  if(m_stage.rows!=gemv_rows(req))
    throw std::runtime_error("staging token rows do not match operator");
}

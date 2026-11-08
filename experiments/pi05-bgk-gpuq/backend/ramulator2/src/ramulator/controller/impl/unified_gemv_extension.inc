// H16 opt-in input storage: two 512B slots in each PIM bank. State below is
// ownership/coverage metadata, not another rank-sized data buffer.
bool m_unified_gemv = false;
bool m_stream_inputs=false,m_gpu_rope=false,m_feature_layout=false;
bool direct_weight(const Request& r) const { return m_stream_inputs && gemv_offset(r)<786432; }
int m_gemv_port_cycles = 2;
std::string m_gemv_events_path;
std::ofstream m_gemv_events;
struct GemvSlot {
  enum Kind { Empty, External, Z, Q } kind = Empty;
  std::array<uint16_t,16> lanes{};
  int generation = 0, mac_columns = 0, expected_columns = 0, weight_block = -1;
};
std::vector<std::array<GemvSlot,4>> m_gemv_slots;
// A single 32B external ingress staging entry per rank/channel. Ordinary
// WRGB fills all bank GEMV slots when its DQ burst arrives; the rank broadcast
// has no additional latency or separately charged internal-port service.
// Bank-local Q delivery retains its explicitly modeled local-port service.
Request* m_gemv_ingress_owner = nullptr;
Clk_t m_gemv_ingress_free = 0;

bool is_gemv_release(const Request& r) const { return r.type_id == Request::Type::PIM_GEMV_RELEASE; }
int gemv_offset(const Request& r) const {
  const auto& a=r.pim_origin_addr_vec.empty()?r.addr_vec:r.pim_origin_addr_vec;
  return a[m_device.m_spec->get_level_id("Row")]*2048+a[m_device.m_spec->get_level_id("Column")]*32;
}
int gemv_columns(const Request& r) const {
  const int off=gemv_offset(r);
  if(off>=786432 && off<819200) {
    if(!m_o_projection) throw std::runtime_error("O weights require pim_o_projection");
    return 32;
  }
  if (off>=393216 && off<491520) {
    if(!m_qkv_fusion) throw std::runtime_error("Fused QKV address requires pim_qkv_fusion");
    return m_feature_layout?24:48;
  }
  if (off>=163840 && off<180224) {
    if(!m_token_pv) throw std::runtime_error("Token V cache requires pim_token_pv");
    return m_feature_layout?8:16;
  }
  if (off<65536 || (off>=262144 && off<327680)) return 32;
  if (off>=83968 && off<116736) return 8;
  // Token-sharded V: each rank computes eight tokens and all head features.
  // Its separate Wv copy has sixteen output columns per input fragment.
  if (off>=131072 && off<163840) return 16;
  if (off>=327680 && off<360448) {
    // GPU-fed Down shards 1024 output features across16 ranks/banks:
    // each bank owns4 complete outputs and consumes all256 input fragments.
    // Preserve the separate legacy DRAM-Z layout only when explicitly enabled.
    return m_ffn_dram?16:4;
  }
  throw std::runtime_error("Unified GEMV MAC weight/cache address outside H16 layout");
}
int gemv_weight_base(const Request& r) const {
  const int off=gemv_offset(r);
  if(off>=786432 && off<819200) return 786432+(off-786432)/o_tile_bytes()*o_tile_bytes();
  if (off>=393216 && off<491520) return 393216;
  if (off>=163840 && off<180224) return 163840;
  if (off>=131072 && off<163840) return 131072;
  return off<65536?0:off<100352?83968:off<116736?100352:off<327680?262144:327680;
}
int gemv_mac_column(const Request& r) const {
  return ((gemv_offset(r)-gemv_weight_base(r))/32)%gemv_columns(r);
}
int gemv_rows(const Request& r) const {
  const int off=gemv_offset(r);
  if(m_feature_layout && off>=393216 && off<491520) return 16;
  return (off<65536 || (off>=131072 && off<163840) || (off>=393216 && off<491520))?8:16;
}
int gemv_service(const Request& r) const { return m_unified_gemv?gemv_rows(r):m_mac_interval; }
void init_unified_gemv() {
  if (m_gemv_port_cycles<1) throw std::runtime_error("gemv_port_cycles must be positive");
  if (m_unified_gemv && (m_pim_horizon!=16 || m_inputs.size()!=2 || m_bgs*m_banks_per_bg!=16 ||
      m_data_path!="attacc" || m_input_broadcast || m_mac_scope!="rank"))
    throw std::runtime_error("Unified GEMV requires H16, rank1-only rank-scoped AttAcc");
  if (m_unified_gemv) m_gemv_slots.resize(m_bank_inputs.size());
}
void record_gemv(const Request& req,const char* operation,Clk_t start,Clk_t ready,
                 int banks,int tokens,int lanes,int bytes,int generation,int columns=0) {
  if (!m_gemv_events) return;
  m_gemv_events<<start<<','<<ready<<','<<operation<<','<<m_channel_id<<",1,"<<
      (req.pim_buffer<0?0:req.pim_buffer)<<','<<banks<<','<<tokens<<','<<lanes<<','<<bytes<<','<<
      ready-start<<','<<generation<<','<<columns<<'\n';
  m_gemv_events.flush();
}
void gemv_clear_bank(int bank,int slot) {
  const int generation=m_gemv_slots[bank][slot].generation;
  m_gemv_slots[bank][slot]=GemvSlot{};
  m_gemv_slots[bank][slot].generation=generation;
  m_bank_inputs[bank][slot]=InputBuffer{};
  if(slot==0) m_rope_banks[bank].query_mask=0;
}
bool gemv_slot_idle(int slot) const {
  for(int bank=16;bank<32;++bank) {
    const auto& input=m_bank_inputs[bank][slot];
    if(m_clk<std::max({input.ready_at,input.in_use_until,input.local_in_use_until})) return false;
  }
  return true;
}
void gemv_start_bank(int bank,int slot,GemvSlot::Kind kind) {
  auto& state=m_gemv_slots[bank][slot];
  if(state.kind!=GemvSlot::Empty) {
    if(state.kind==GemvSlot::Q) throw std::runtime_error("Unified GEMV Q slot held until PIM_GEMV_RELEASE");
    if(state.expected_columns==0 || state.mac_columns!=state.expected_columns)
      throw std::runtime_error("Unified GEMV overwrite before full fragment consumption");
    gemv_clear_bank(bank,slot);
  }
  state.kind=kind;
  ++state.generation;
}
bool gemv_external_ready(const Request& req) {
  if(m_whole_srf && !m_stream_inputs) throw std::runtime_error("Whole SRF input requires complete DRAM staging");
  if(req.pim_buffer<0) throw std::runtime_error("Unified GEMV requires explicit input slot");
  if(m_gemv_ingress_owner || m_clk<m_gemv_ingress_free) return false;
  if(!gemv_slot_idle(req.pim_buffer)) return false;
  const int row=gemv_offset(req)/32;
  if(row<0 || row>=16) throw std::runtime_error("Unified GEMV external row exceeds 512B slot");
  for(int bank=16;bank<32;++bank) {
    const auto& state=m_gemv_slots[bank][req.pim_buffer];
    if(state.kind==GemvSlot::Q) throw std::runtime_error("Unified GEMV Q slot held until PIM_GEMV_RELEASE");
    if(row==0) {
      if(state.kind!=GemvSlot::Empty && (!state.expected_columns || state.mac_columns!=state.expected_columns))
        throw std::runtime_error("Unified GEMV overwrite before full fragment consumption");
    } else if(state.kind!=GemvSlot::External || state.lanes[row])
      throw std::runtime_error("Unified GEMV duplicate/missing first external input row");
  }
  return true;
}
Clk_t gemv_external_issued(const Request& req) {
  const int slot=req.pim_buffer,row=gemv_offset(req)/32;
  const Clk_t ready=m_clk+m_bgk_input_cycles;
  m_gemv_ingress_free=ready;
  for(int bank=16;bank<32;++bank) {
    if(row==0) gemv_start_bank(bank,slot,GemvSlot::External);
    auto& state=m_gemv_slots[bank][slot]; state.lanes[row]=65535;
    auto& input=m_bank_inputs[bank][slot]; input.written=true; input.ready_at=ready;
  }
  // This event marks simultaneous destination writes at DQ completion.
  // The external WRGB timeline already charges the two-cycle data burst.
  record_gemv(req,"GEMV_MULTICAST",ready,ready,65535,1<<row,65535,32,m_gemv_slots[16][slot].generation);
  return ready;
}
bool gemv_mac_ready(const Request& req) {
  const int slot=req.pim_buffer,rows=gemv_rows(req),columns=gemv_columns(req);
  if(slot<0) throw std::runtime_error("Unified GEMV MAC requires explicit input slot");
  stage_mac_identity(req);
  if(!whole_mac_ready(req)) return false;
  if(!gemv_slot_idle(slot)) return false;
  for(int bank=16;bank<32;++bank) {
    if(m_clk<m_bank_compute_ready[bank]) return false;
    const auto& state=m_gemv_slots[bank][slot];
    const auto expected_kind=is_ffn_down(req)?GemvSlot::Z:GemvSlot::External;
    if(state.kind!=expected_kind) throw std::runtime_error("Unified GEMV MAC input owner mismatch or empty slot");
    for(int row=0;row<rows;++row) if(state.lanes[row]!=65535)
      throw std::runtime_error("Unified GEMV MAC requires every token and lane in every target bank");
    for(int row=rows;row<16;++row) if(state.lanes[row])
      throw std::runtime_error("Unified GEMV MAC has extra input rows for this operator");
    const int weight_block=gemv_offset(req)-32*gemv_mac_column(req);
    if(state.mac_columns!=gemv_mac_column(req) || (state.expected_columns && state.expected_columns!=columns) ||
        (state.weight_block>=0 && state.weight_block!=weight_block))
      throw std::runtime_error("Unified GEMV MAC column order or operator switch before release");
  }
  return true;
}
void gemv_mac_issued(const Request& req,Clk_t ready) {
  const int slot=req.pim_buffer;
  for(int bank=16;bank<32;++bank) {
    auto& state=m_gemv_slots[bank][slot];
    state.expected_columns=gemv_columns(req);
    state.weight_block=gemv_offset(req)-32*gemv_mac_column(req);
    ++state.mac_columns;
    m_bank_inputs[bank][slot].in_use_until=ready;
    m_bank_compute_ready[bank]=ready;
  }
  record_gemv(req,"GEMV_MAC",m_clk,ready,65535,(1<<gemv_rows(req))-1,65535,32,
      m_gemv_slots[16][slot].generation,m_gemv_slots[16][slot].mac_columns);
}
bool gemv_q_ready(int bank,int query) {
  whole_q_ready(bank);
  const auto& state=m_gemv_slots[bank][0];
  const auto& input=m_bank_inputs[bank][0];
  if(m_clk<std::max(input.in_use_until,input.ready_at)) return false;
  if(state.kind!=GemvSlot::Empty && state.kind!=GemvSlot::Q)
    throw std::runtime_error("Unified GEMV Q fill requires explicit release of projection slot");
  if(state.lanes[query]) throw std::runtime_error("Unified GEMV duplicate Q token row");
  return true;
}
Clk_t gemv_q_write(const Request& req,int bank,int query,bool remote) {
  const Clk_t start=m_clk+(remote?m_bgk_input_cycles:0),ready=start+m_gemv_port_cycles;
  auto& state=m_gemv_slots[bank][0];
  if(state.kind==GemvSlot::Empty) gemv_start_bank(bank,0,GemvSlot::Q);
  state.lanes[query]=65535;
  auto& input=m_bank_inputs[bank][0]; input.written=true; input.ready_at=ready;
  m_local_ports[1]=m_gemv_ingress_free=ready;
  record_gemv(req,remote?"GEMV_Q_REMOTE":"GEMV_Q_LOCAL",start,ready,1<<(bank-16),1<<query,65535,32,state.generation);
  return ready;
}
bool gemv_z_start(const Request& req,bool new_batch) {
  const int slot=req.pim_buffer;
  if(!gemv_slot_idle(slot)) return false;
  if(new_batch) for(int bank=16;bank<32;++bank) gemv_start_bank(bank,slot,GemvSlot::Z);
  return true;
}
Clk_t gemv_z_scatter(const Request& req) {
  const int slot=req.pim_buffer,lane=z_bank(req);
  const Clk_t ready=m_clk+m_gemv_port_cycles;
  for(int bank=16;bank<32;++bank) {
    auto& state=m_gemv_slots[bank][slot];
    if(state.kind!=GemvSlot::Z) throw std::runtime_error("Unified GEMV Z scatter owner mismatch");
    for(auto& mask:state.lanes) {
      if(mask&(1<<lane)) throw std::runtime_error("Unified GEMV duplicate Z lane");
      mask|=1<<lane;
    }
    auto& input=m_bank_inputs[bank][slot]; input.written=true; input.ready_at=ready;
  }
  m_local_ports[1]=ready;
  record_gemv(req,"GEMV_Z_SCATTER",m_clk,ready,65535,65535,1<<lane,32,m_gemv_slots[16][slot].generation);
  return ready;
}
bool send_gemv_release(Request& req) {
  if(!m_unified_gemv || req.pim_buffer<0 || req.addr_vec[m_rank_level]!=1 ||
      req.addr_vec[m_bg_level]!=0 || req.addr_vec[m_bank_level]!=0)
    throw std::runtime_error("PIM_GEMV_RELEASE requires unified mode, canonical rank1 and slot0/1");
  req.pim_rank_mask=2;req.pim_bank_mask=uint64_t{65535}<<16;
  req.final_command=-1;req.arrive=m_clk;m_pim_buffer.enqueue(req);++s_pim_requests;
  return true;
}
bool issue_gemv_release(ReqBuffer::iterator it) {
  const int slot=it->pim_buffer;
  if(!gemv_slot_idle(slot) || m_clk<m_local_ports[1] || m_clk<m_rank_reduction_ready) return false;
  for(int bank=16;bank<32;++bank) {
    const auto& state=m_gemv_slots[bank][slot];
    if(state.kind==GemvSlot::Q && state.mac_columns!=(m_feature_layout?992:496))
      throw std::runtime_error("Unified GEMV Q release requires completed 496-token K scan");
    if(state.kind!=GemvSlot::Empty && state.kind!=GemvSlot::Q &&
        (!state.expected_columns || state.mac_columns!=state.expected_columns))
      throw std::runtime_error("Unified GEMV release of unconsumed input fragment");
  }
  record_gemv(*it,"GEMV_RELEASE",m_clk,m_clk,65535,0,0,0,m_gemv_slots[16][slot].generation);
  for(int bank=16;bank<32;++bank) gemv_clear_bank(bank,slot);
  enqueue_bgk_done(it,m_clk);return true;
}

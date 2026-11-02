// Optional Z-in-DRAM timing model. No tensor arithmetic or NL latency is implied.
bool m_ffn_dram = false, m_ffn_z_preinstalled = false;
std::string m_ffn_events_path;
std::ofstream m_ffn_events;
struct ZBatch {
  int column = -1, generation = 0, mac_columns = 0;
  uint64_t requested = 0, delivered = 0;
};
std::array<ZBatch, 2> m_z_batches;
std::array<Request*, 16> m_z_bank_owner{};
std::array<Clk_t, 16> m_z_bank_free{};
std::array<std::array<Clk_t, 64>, 16> m_z_written{};
Request* m_z_material_owner = nullptr;
Clk_t m_z_material_free = 0, m_ffn_compute_ready = 0;
size_t s_z_stores = 0, s_z_loads = 0, s_z_gather_bytes = 0, s_z_return_bytes = 0, s_z_load_port_bytes = 0;

bool is_ffn_z(const Request& req) const {
  return req.type_id == Request::Type::PIM_Z_STORE || req.type_id == Request::Type::PIM_Z_LOAD;
}
int z_bank(const Request& req) const { return req.addr_vec[m_bg_level]*m_banks_per_bg+req.addr_vec[m_bank_level]; }
int z_column(const Request& req) const { return req.addr_vec[m_device.m_spec->get_level_id("Column")]; }
int z_token_blocks() const { return (m_pim_horizon+15)/16; }
uint64_t z_source_mask() const {
  const int count=16*z_token_blocks();
  return count==64 ? ~uint64_t{0} : (uint64_t{1}<<count)-1;
}
uint64_t z_source_bit(const Request& req) const {
  return uint64_t{1} << (z_bank(req)*z_token_blocks()+z_column(req)%z_token_blocks());
}
int z_gather_bursts(const Request& req) const {
  const int valid=std::min(16,m_pim_horizon-16*(z_column(req)%z_token_blocks()));
  return (valid*8+31)/32; // FP32 G/U for valid tokens; charged whole32B port bursts.
}
int z_offset(const Request& req) const {
  const auto& a = req.pim_origin_addr_vec.empty() ? req.addr_vec : req.pim_origin_addr_vec;
  return a[m_device.m_spec->get_level_id("Row")]*2048+a[m_device.m_spec->get_level_id("Column")]*32;
}
bool is_ffn_down(const Request& req) const {
  return m_ffn_dram && req.type_id == Request::Type::PIM_MAC_AB && z_offset(req) >= 327680 && z_offset(req) < 360448;
}
void init_ffn_dram() {
  if (m_ffn_z_preinstalled && !m_ffn_dram) throw std::runtime_error("ffn_z_preinstalled requires pim_ffn_dram");
  if (m_ffn_dram && (m_inputs.size()!=2 || m_bgs*m_banks_per_bg!=16 || (m_rank_rope && !m_unified_gemv) ||
      m_input_broadcast || m_mac_scope!="rank" || m_data_path!="attacc"))
    throw std::runtime_error("FFN Z DRAM requires rank-scoped AttAcc, rank1 only, without rank RoPE");
}
void record_ffn(const Request& req, const char* operation, Clk_t ready, int bytes,
                int service, int generation=0, int bank_mask=0, int columns=0) {
  if (!m_ffn_events) return;
  const auto& a = req.pim_origin_addr_vec.empty() ? req.addr_vec : req.pim_origin_addr_vec;
  m_ffn_events << m_clk << ',' << ready << ',' << operation << ',' << m_channel_id
      << ',' << a[m_rank_level] << ',' << a[m_bg_level] << ',' << a[m_bank_level]
      << ',' << a[m_device.m_spec->get_level_id("Row")] << ',' << a[m_device.m_spec->get_level_id("Column")]
      << ',' << req.pim_buffer << ',' << bytes << ',' << service << ',' << generation
      << ',' << bank_mask << ',' << columns << '\n';
  m_ffn_events.flush();
}
bool send_ffn_z(Request& req) {
  if (!m_ffn_dram) throw std::runtime_error("PIM_Z request requires pim_ffn_dram=true");
  if (req.addr_vec[m_rank_level]!=1 || req.size_bytes!=32)
    throw std::runtime_error("PIM_Z requires physical rank1 and a32B request");
  if (req.addr_vec[m_device.m_spec->get_level_id("Row")]!=176 || z_column(req)>=16*z_token_blocks())
    throw std::runtime_error("PIM_Z address must be row176 column0.." + std::to_string(16*z_token_blocks()-1));
  if ((req.type_id==Request::Type::PIM_Z_LOAD && req.pim_buffer<0) ||
      (req.type_id==Request::Type::PIM_Z_STORE && req.pim_buffer!=-1))
    throw std::runtime_error("Z_LOAD requires BUFFER0/1; Z_STORE has no input buffer");
  req.pim_rank_mask=2;
  req.pim_bank_mask=uint64_t{1} << (16+z_bank(req));
  req.final_command=spec_command(req.type_id);
  req.arrive=m_clk;
  m_pim_buffer.enqueue(req);
  ++s_pim_requests;
  return true;
}
bool z_bypass(const Request& later, const Request& earlier) const {
  if (is_ffn_z(later) && is_ffn_z(earlier)) {
    // Bank stages are independent; the rank port is arbitrated at issue.
    return !(later.pim_bank_mask & earlier.pim_bank_mask);
  }
  const Request& z = is_ffn_z(later) ? later : earlier;
  const Request& other = is_ffn_z(later) ? earlier : later;
  if (other.type_id == Request::Type::PIM_MAC_AB) {
    if (z.type_id == Request::Type::PIM_Z_STORE) return false;
    // An unread Z request still needs ACT1/ACT2/LOCALRD on the same arrays
    // as the rank MAC. Letting either prepare rows past the other can make
    // ACT1(Z) and PREab(weights) repeatedly cancel each other. Only an
    // already-issued LOCALRD may overlap another slot's MAC: its remaining
    // work is read completion and the independent rank-local SRAM port.
    return z.pim_q_stage == 1 && other.pim_buffer >= 0 && other.pim_buffer != z.pim_buffer;
  }
  return false;
}
bool prepare_z_batch(Request& req) {
  auto& batch=m_z_batches[req.pim_buffer];
  auto& input=m_unified_gemv?m_bank_inputs[16][req.pim_buffer]:m_inputs[1][req.pim_buffer];
  const int column=z_column(req)/z_token_blocks();
  const uint64_t bit=z_source_bit(req);
  if (m_clk < std::max(input.in_use_until,input.local_in_use_until)) return false;
  const bool new_batch=batch.column<0 || batch.mac_columns==16;
  if (m_unified_gemv && !gemv_z_start(req,new_batch)) return false;
  if (new_batch) {
    const int generation=batch.generation+1;
    batch=ZBatch{}; batch.column=column; batch.generation=generation;
    input.written=false; input.ready_at=0;
  }
  if (column!=batch.column) throw std::runtime_error("Z_LOAD mixed columns before prior fragment consumed");
  if (batch.requested & bit) throw std::runtime_error("Duplicate Z_LOAD source bank in input fragment");
  batch.requested |= bit;
  return true;
}
bool ffn_mac_ready(const Request& req) {
  if (!is_ffn_down(req)) return true;
  if (req.addr_vec[m_rank_level]!=1 || req.pim_buffer<0)
    throw std::runtime_error("FFN Down requires rank1 and an explicit Z input buffer");
  const int relative=(z_offset(req)-327680)%8192;
  const int fragment=relative/512, column=(relative%512)/32;
  const auto& batch=m_z_batches[req.pim_buffer];
  if (batch.column!=fragment || req.pim_buffer!=fragment%2)
    throw std::runtime_error("Down MAC references missing or mismatched Z fragment");
  if (batch.delivered!=z_source_mask()) {
    for (const auto& pending:m_pim_buffer)
      if (pending.type_id==Request::Type::PIM_Z_LOAD && pending.pim_buffer==req.pim_buffer) return false;
    throw std::runtime_error("Down MAC requires all16 source banks in Z input fragment");
  }
  if (column!=batch.mac_columns) throw std::runtime_error("Down MAC output column order/coverage mismatch");
  return m_unified_gemv ? gemv_slot_idle(req.pim_buffer) : m_clk >= m_inputs[1][req.pim_buffer].ready_at;
}
void ffn_mac_issued(const Request& req) {
  if (!m_ffn_dram || req.type_id!=Request::Type::PIM_MAC_AB) return;
  m_ffn_compute_ready=m_clk+gemv_service(req);
  int generation=0, columns=0;
  if (is_ffn_down(req)) {
    auto& batch=m_z_batches[req.pim_buffer];
    generation=batch.generation; columns=++batch.mac_columns;
  }
  record_ffn(req,"PIM_MAC_AB",m_clk+gemv_service(req),32,gemv_service(req),generation,65535,columns);
}
bool issue_ffn_z(ReqBuffer::iterator it, bool local) {
  const int bank=z_bank(*it), column=z_column(*it), bit=1<<bank;
  const bool store=it->type_id==Request::Type::PIM_Z_STORE;
  auto& port=m_local_ports[1];
  const int port_cycles=m_unified_gemv?m_gemv_port_cycles:m_local_cycles;
  if (store && it->pim_q_stage==0) {
    if (!local || m_clk<port || m_clk<m_ffn_compute_ready || m_z_bank_owner[bank] || m_clk<m_z_bank_free[bank]) return false;
    if (m_z_material_owner && m_clk>=m_z_material_free) m_z_material_owner=nullptr;
    if (m_z_material_owner) return false;
    m_z_bank_owner[bank]=&*it; m_z_material_owner=&*it;
    m_z_material_free=std::numeric_limits<Clk_t>::max();
    const int bursts=z_gather_bursts(*it);
    it->depart=port=m_clk+bursts*port_cycles;
    it->pim_q_stage=1;
    record_ffn(*it,"PIM_Z_GATHER",it->depart,bursts*32,bursts*port_cycles,0,bit);
    s_z_gather_bytes+=bursts*32;
    return true;
  }
  if (store && it->pim_q_stage==1) {
    if (!local || m_clk<it->depart || m_clk<port) return false;
    // The nonlinearity is an explicitly excluded value-ready boundary.
    // This event measures only movement of its assumed-ready BF16 output.
    it->depart=port=m_clk+port_cycles;
    m_z_material_free=it->depart;
    it->pim_q_stage=2;
    record_ffn(*it,"PIM_Z_RETURN",it->depart,32,port_cycles,0,bit);
    s_z_return_bytes+=32;
    return true;
  }
  if (store && m_clk<it->depart) return false;
  if (!store && it->pim_q_stage==1) {
    if (!local || m_clk<it->depart || m_clk<port) return false;
    auto& batch=m_z_batches[it->pim_buffer];
    auto& input=m_unified_gemv?m_bank_inputs[16][it->pim_buffer]:m_inputs[1][it->pim_buffer];
    if (m_clk<input.in_use_until || (m_unified_gemv && !gemv_slot_idle(it->pim_buffer))) return false;
    const Clk_t ready=port=m_unified_gemv?gemv_z_scatter(*it):m_clk+port_cycles;
    batch.delivered |= z_source_bit(*it);
    input.ready_at=std::max(input.ready_at,ready);
    input.local_in_use_until=ready;
    record_ffn(*it,"PIM_Z_LOAD_PORT",ready,32,port_cycles,batch.generation,bit);
    if (batch.delivered==z_source_mask()) {
      input.written=true;
      record_ffn(*it,"PIM_Z_BATCH_READY",input.ready_at,m_pim_horizon*32,port_cycles,batch.generation,65535);
    }
    m_z_bank_owner[bank]=nullptr; m_z_bank_free[bank]=ready;
    s_z_load_port_bytes+=32;
    enqueue_bgk_done(it,ready);
    return true;
  }
  if (!store) {
    if (!m_z_written[bank][column] && !m_ffn_z_preinstalled)
      throw std::runtime_error("Z_LOAD has no completed Z_STORE or explicit preinstalled Z");
    if (m_clk<m_z_written[bank][column] || m_z_bank_owner[bank] || m_clk<m_z_bank_free[bank]) return false;
  }
  it->command=get_preq_command(it->final_command,it->addr_vec);
  const bool final=it->command==it->final_command;
  if (local!=final || !check_timing(it->command,it->addr_vec)) return false;
  if (final && !store && !prepare_z_batch(*it)) return false;
  m_device.issue_command(it->command,it->addr_vec,m_clk);
  rr_notify_command(*it);
  if (!final) return true;
  if (store) {
    const int service=m_bank_write_cycles+m_local_recovery;
    const Clk_t ready=m_clk+service;
    record_ffn(*it,"PIM_Z_STORE",ready,32,service,0,bit);
    m_z_written[bank][column]=ready;
    m_z_bank_owner[bank]=nullptr; m_z_bank_free[bank]=m_clk+m_bank_write_cycles;
    ++s_z_stores;
    enqueue_bgk_done(it,ready);
  } else {
    const int service=m_device.m_spec->get_timing_value("nCL")+m_bgk_input_cycles;
    it->depart=m_clk+service; it->pim_q_stage=1;
    m_z_bank_owner[bank]=&*it;
    record_ffn(*it,"PIM_Z_LOAD",it->depart,32,service,m_z_batches[it->pim_buffer].generation,bit);
    ++s_z_loads;
  }
  return true;
}
void issue_ffn_z_local() {
  if (!m_ffn_dram) return;
  for (auto it=m_pim_buffer.begin();it!=m_pim_buffer.end();) {
    auto current=it++;
    if (!is_ffn_z(*current)) continue;
    bool independent=true;
    for (auto before=m_pim_buffer.begin();before!=current;++before)
      if (!can_bypass(*current,*before,false)) {independent=false;break;}
    if (independent) issue_ffn_z(current,true);
  }
}

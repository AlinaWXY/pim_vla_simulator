// Optional PV BF16 boundary and local head-partial output projection.
// All payloads remain modeled data: these states enforce ownership, coverage,
// conversion/transfer service and real destination-array command timing.
bool m_o_projection=false,m_pv_bf16_output=false;
bool m_o_completed=false;
bool m_o_gpu_input_ready=false;
bool direct_o_input() const { return m_stream_inputs && !m_rank_controller; }
std::string m_o_projection_path;
std::ofstream m_o_projection_events;
static constexpr int O_WEIGHT_BASE=786432,O_TILE_BYTES=16384,O_STAGE_BASE=524288;
int o_tile_bytes() const { return m_feature_layout?8192:16384; }
int o_input_bytes() const { return m_feature_layout?4096:8192; }
std::array<uint16_t,16> m_o_pv_spills{};
int m_o_next_tile=0,m_o_pv_spill_count=0,m_o_export_phase=0;
// One 64B conversion packet per rank. Reuse waits for the prior LOCALWR data
// service or DQ export to consume it; DRAM write recovery may overlap reuse.
Clk_t m_o_export_ready=0,m_o_packet_free=0;
size_t s_o_gather_bytes=0,s_o_cast_bytes=0,s_o_transfer_bytes=0,s_o_local_write_bytes=0;
bool is_o_command(const Request& req) const {
  return req.type_id==Request::Type::PIM_PV_SPILL || req.type_id==Request::Type::PIM_O_REWIND;
}
bool is_o_weight_base(int base) const { return base==O_WEIGHT_BASE || base==O_WEIGHT_BASE+o_tile_bytes(); }
bool output_bf16() const {
  return (m_pv_bf16_output && m_output_base==163840) || is_o_weight_base(m_output_base);
}
int output_export_packets() const { return m_output_bytes*16/(output_bf16()?64:32); }
void init_o_projection() {
  if(m_pv_bf16_output && (!m_token_pv || !m_whole_srf))
    throw std::runtime_error("PV BF16 output requires token PV and whole SRF");
  if(m_o_projection && (!m_pv_bf16_output || (!direct_o_input() && (!m_rank_staging || m_staging_banks!=2))))
    throw std::runtime_error("O projection requires PV BF16 and direct GPU inputs or two staging banks");
}
void setup_o_projection() {
  if(!m_pv_bf16_output && !m_o_projection) return;
  m_stats.add("o_gather_bytes",s_o_gather_bytes);
  m_stats.add("o_cast_bytes",s_o_cast_bytes);
  m_stats.add("o_transfer_bytes",s_o_transfer_bytes);
  m_stats.add("o_local_write_bytes",s_o_local_write_bytes);
  if(m_o_projection_path.empty()) return;
  m_o_projection_events.open(m_o_projection_path);
  if(!m_o_projection_events) throw std::runtime_error("Cannot open O projection events");
  m_o_projection_events<<"clock,ready_at,operation,Channel,Rank,BankGroup,Bank,Row,Column,source_bank,destination_bank,query,fragment,tile,bytes,stage_generation,request_id,offset_bytes\n";
}
void record_o(const Request& req,const char* op,Clk_t ready,int bytes,int source=-1,int destination=-1,int query=-1,int fragment=-1) {
  if(!m_o_projection_events) return;
  const auto& a=req.pim_origin_addr_vec.empty()?req.addr_vec:req.pim_origin_addr_vec;
  const int tile=is_o_weight_base(m_output_base)?(m_output_base-O_WEIGHT_BASE)/o_tile_bytes():-1;
  m_o_projection_events<<m_clk<<','<<ready<<','<<op<<','<<m_channel_id<<",1,"<<a[m_bg_level]<<','<<a[m_bank_level]<<','
    <<a[m_device.m_spec->get_level_id("Row")]<<','<<a[m_device.m_spec->get_level_id("Column")]<<','<<source<<','<<destination<<','
    <<query<<','<<fragment<<','<<tile<<','<<bytes<<','<<m_stage.generation<<','<<req.ingress_id<<','<<gemv_offset(req)<<'\n';
}
void o_output_begin(int base) {
  m_o_export_phase=0;m_o_export_ready=0;
  if(base==163840) {
    m_o_pv_spills.fill(0);m_o_pv_spill_count=0;m_o_next_tile=0;m_o_completed=false;m_o_gpu_input_ready=false;
  }
  if(is_o_weight_base(base)) {
    if(direct_o_input()) {
      if(!m_o_projection || !m_o_gpu_input_ready || (base-O_WEIGHT_BASE)/o_tile_bytes()!=m_o_next_tile)
        throw std::runtime_error("Direct O MAC requires exported PV input and ordered tiles");
      return;
    }
    if(!m_o_projection || !output_all(m_o_pv_spills) || !m_stage.active || !m_stage.sealed || m_stage.bytes!=o_input_bytes())
      throw std::runtime_error("O MAC requires complete locally produced PV input");
    if((base-O_WEIGHT_BASE)/o_tile_bytes()!=m_o_next_tile)
      throw std::runtime_error("O tile order requires explicit input rewind");
  }
}
// A BF16 result packet contains sixteen FP32 accumulators before conversion.
// One packet is packed at a time, with explicit shared-port and bank-cast use.
bool o_bf16_export_ready(const Request& req) {
  if(!output_bf16()) return true;
  // One packet has 16 BF16 values. Feature-wise PV contributes eight per
  // bank, so a packet gathers an adjacent bank pair instead of half a packet.
  const int source=(m_output_exports%m_output_columns)*16/m_output_columns;
  const int query=m_output_exports/m_output_columns;
  if(m_o_export_phase==0) {
    if(m_clk<m_o_packet_free || m_clk<m_local_ports[1] || !rank_pseudo("RESULT_GATHER")) return false;
    m_o_export_ready=m_local_ports[1]=m_clk+2*m_gemv_port_cycles;
    m_o_export_phase=1;s_o_gather_bytes+=64;
    record_o(req,"EXPORT_GATHER_FP32",m_o_export_ready,64,source,-1,query);
    return false;
  }
  if(m_clk<m_o_export_ready) return false;
  if(m_o_export_phase==1) {
    if(m_output_columns==8 && m_clk<m_rope_banks[source+17].cast_busy) return false;
    if(m_clk<m_rope_banks[source+16].cast_busy || !rank_pseudo("RESULT_CAST")) return false;
    m_o_export_ready=m_rope_banks[source+16].cast_busy=m_clk+2*m_cast_cycles;
    if(m_output_columns==8) m_rope_banks[source+17].cast_busy=m_o_export_ready;
    m_o_export_phase=2;s_o_cast_bytes+=32;
    record_o(req,"EXPORT_CAST_BF16",m_o_export_ready,32,source,-1,query);
    return false;
  }
  return true;
}
void o_bf16_export_issued(const Request& req) {
  if(!output_bf16()) return;
  if(m_o_export_phase!=2 || m_clk<m_o_export_ready)
    throw std::runtime_error("BF16 export before complete packing");
  record_o(req,"EXPORT_BF16",req.depart,32,(m_output_exports%m_output_columns)*16/m_output_columns,-1,
    m_output_exports/m_output_columns);
  m_o_packet_free=req.depart;
  m_o_export_phase=0;m_o_export_ready=0;
}
bool send_o_command(Request& req) {
  if(!m_o_projection || req.addr_vec[m_rank_level]!=1 || req.pim_buffer!=-1 || req.size_bytes!=32)
    throw std::runtime_error("O local command requires enabled mode, rank1 and no SRF view");
  stage_address(req);
  if(req.type_id==Request::Type::PIM_O_REWIND &&
     (stage_offset(req)!=O_STAGE_BASE || req.addr_vec[m_bg_level]!=0 || req.addr_vec[m_bank_level]!=0))
    throw std::runtime_error("O rewind requires canonical staging address");
  req.pim_rank_mask=2;
  req.pim_bank_mask=uint64_t{1}<<(16+4*req.addr_vec[m_bg_level]+req.addr_vec[m_bank_level]);
  req.final_command=req.type_id==Request::Type::PIM_PV_SPILL?m_localwr:-1;
  req.arrive=m_clk;
  if(!m_pim_buffer.enqueue(req)) return false;
  ++s_pim_requests;++s_bgk_requests;return true;
}
bool issue_o_command(ReqBuffer::iterator it) {
  if(it->type_id==Request::Type::PIM_O_REWIND) {
    if(m_output_base!=O_WEIGHT_BASE || m_o_next_tile!=0 || !m_stage.active || !m_stage.sealed ||
       m_stage.bytes!=o_input_bytes() || m_stage.rows!=16 || m_stage.loads.size()!=size_t(o_input_bytes()/32))
      throw std::runtime_error("O rewind requires complete first tile and full first input scan");
    if(!rank_ca_available(1,m_clk) || !output_other_ready()) return false;
    if(!rank_pseudo("O_REWIND")) return false;
    whole_stage_begin(16);
    m_stage.loads.clear();++m_stage.generation;m_o_next_tile=1;
    record_o(*it,"INPUT_REWIND",m_clk,o_input_bytes());
    enqueue_bgk_done(it,m_clk);return true;
  }
  if(!m_stage.active || m_stage.sealed || m_stage.base!=O_STAGE_BASE || m_stage.bytes!=o_input_bytes() || m_stage.rows!=16)
    throw std::runtime_error("PV spill requires active unsealed 8192B two-bank staging");
  const int offset=stage_logical_offset(*it),packet=(offset-O_STAGE_BASE)/32;
  if(offset<O_STAGE_BASE || offset>=O_STAGE_BASE+o_input_bytes() || offset%32)
    throw std::runtime_error("PV spill outside local input payload");
  const int source=(packet/16)*(m_feature_layout?2:1),query=packet%16;
  const int destination=4*it->addr_vec[m_bg_level]+it->addr_vec[m_bank_level];
  if(it->pim_q_stage==0) {
    if(m_output_base!=163840 || m_output_macs!=m_output_expected_macs)
      throw std::runtime_error("PV spill requires complete PV accumulators");
    if((m_o_pv_spills[source]&(1<<query)) || m_stage.writes.count(offset))
      throw std::runtime_error("Duplicate PV local input packet");
    if(m_clk<m_output_compute_ready || m_clk<m_o_packet_free || m_clk<m_local_ports[1] || !rank_pseudo("PV_GATHER")) return false;
    it->depart=m_local_ports[1]=m_clk+2*m_gemv_port_cycles;
    it->pim_q_stage=1;s_o_gather_bytes+=64;
    m_stage.writes.emplace(offset,-1);
    record_o(*it,"PV_GATHER_FP32",it->depart,64,source,destination,query,source);return true;
  }
  if(m_clk<it->depart) return false;
  if(it->pim_q_stage==1) {
    if(m_clk<m_rope_banks[source+16].cast_busy || !rank_pseudo("PV_CAST")) return false;
    it->depart=m_rope_banks[source+16].cast_busy=m_clk+2*m_cast_cycles;
    it->pim_q_stage=2;s_o_cast_bytes+=32;
    record_o(*it,"PV_CAST_BF16",it->depart,32,source,destination,query,source);return true;
  }
  if(it->pim_q_stage==2) {
    if(m_clk<m_local_ports[1] || !rank_pseudo("PV_TRANSFER")) return false;
    it->depart=m_local_ports[1]=m_clk+m_gemv_port_cycles;
    it->pim_q_stage=3;s_o_transfer_bytes+=32;
    record_o(*it,"PV_TRANSFER_BF16",it->depart,32,source,destination,query,source);return true;
  }
  it->command=get_preq_command(m_localwr,it->addr_vec);
  if(!issue_internal_array_command(it)) return false;
  if(it->command!=m_localwr) return true;
  const Clk_t ready=m_clk+m_bank_write_cycles+m_local_recovery;
  m_o_packet_free=m_clk+m_bank_write_cycles;
  m_stage.writes.at(offset)=ready;m_o_pv_spills[source]|=1<<query; if(m_feature_layout) m_o_pv_spills[source+1]|=1<<query; ++m_o_pv_spill_count;
  m_output_ready=std::max(m_output_ready,ready);
  s_o_local_write_bytes+=32;++s_bank_local_writes;
  record_o(*it,"PV_LOCALWR_BF16",ready,32,source,destination,query,source);
  record_stage(*it,"LOCAL_PRODUCER_WRITE",m_clk,ready);
  if(m_o_pv_spill_count==o_input_bytes()/32) record_o(*it,"PV_INPUT_COMPLETE",ready,o_input_bytes());
  enqueue_bgk_done(it,ready);return true;
}
void finish_o_projection() {
  if(m_o_projection && direct_o_input() && m_o_gpu_input_ready && (!m_o_completed || m_o_next_tile!=2))
    throw std::runtime_error("Incomplete GPU-broadcast PV to O projection chain");
  if(m_o_projection && m_o_pv_spill_count &&
     (m_o_pv_spill_count!=o_input_bytes()/32 || m_o_next_tile!=1 || !m_o_completed))
    throw std::runtime_error("Incomplete local PV to O projection chain");
  if(m_o_export_phase) throw std::runtime_error("Incomplete BF16 result packing");
  if(m_clk<m_o_packet_free) throw std::runtime_error("BF16 packet reused before consumer service finished");
}

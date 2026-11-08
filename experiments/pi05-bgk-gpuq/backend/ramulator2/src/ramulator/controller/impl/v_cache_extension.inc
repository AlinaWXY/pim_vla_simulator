// Token-owned V cache and shared 2KiB FP32 output storage. These are ownership,
// coverage and timing states, not numerical tensors or an extra output array.
bool m_token_pv=false,m_qkv_fusion=false,m_v_cache_preinstalled=false;
std::string m_v_cache_path;
std::ofstream m_v_cache_events;
int m_output_base=-1,m_output_columns=0,m_output_macs=0,m_output_expected_macs=0;
int m_output_bytes=0,m_output_exports=0,m_output_export_done=0;
Clk_t m_output_ready=0,m_output_compute_ready=0;
std::array<uint16_t,16> m_output_qk_spills{},m_v_spills{};
std::array<Clk_t,16> m_v_pack_free{};
Clk_t m_v_cache_ready=0;
int m_v_source_columns=0;
size_t s_v_cache_reads=0,s_v_cache_writes=0,s_v_gather_bytes=0,s_v_cast_bytes=0;
bool is_v_cache_spill(const Request& req) const {
  const int off=gemv_offset(req);
  return req.type_id==Request::Type::PIM_SPILL && off>=163840 && off<180224;
}
bool output_all(const std::array<uint16_t,16>& masks) const {
  return std::all_of(masks.begin(),masks.end(),[](uint16_t mask){return mask==65535;});
}
void init_v_cache() {
  if(m_whole_srf && (!m_unified_gemv || !m_rank_staging))
    throw std::runtime_error("Whole SRF requires unified GEMV and rank staging");
  if(m_token_pv && !m_whole_srf)
    throw std::runtime_error("Token PV requires whole SRF");
  if(m_qkv_fusion && (!m_token_pv || !m_rank_rope))
    throw std::runtime_error("QKV fusion requires token PV and rank RoPE");
  if(m_v_cache_preinstalled && !m_token_pv)
    throw std::runtime_error("Preinstalled token V cache requires token PV");
  if(m_v_cache_preinstalled) m_v_spills.fill(65535);
}
void setup_v_cache() {
  if(!m_token_pv) return;
  m_stats.add("v_cache_array_reads",s_v_cache_reads);
  m_stats.add("v_cache_array_writes",s_v_cache_writes);
  m_stats.add("v_cache_gather_bytes",s_v_gather_bytes);
  m_stats.add("v_cache_cast_bytes",s_v_cast_bytes);
  if(m_v_cache_path.empty()) return;
  m_v_cache_events.open(m_v_cache_path);
  if(!m_v_cache_events) throw std::runtime_error("Cannot open V cache events");
  m_v_cache_events<<"clock,ready_at,operation,Channel,Rank,BankGroup,Bank,Row,Column,bytes,request_id,source_columns,source_feature_column,gather_elements,preserved_mask,updated_mask,zero_mask\n";
}
void record_v_cache(const Request& req,const char* op,Clk_t ready,int bytes,int lane=-1) {
  if(!m_v_cache_events) return;
  const auto& a=req.pim_origin_addr_vec.empty()?req.addr_vec:req.pim_origin_addr_vec;
  m_v_cache_events<<m_clk<<','<<ready<<','<<op<<','<<m_channel_id<<','<<a[m_rank_level]<<','
    <<a[m_bg_level]<<','<<a[m_bank_level]<<','<<a[m_device.m_spec->get_level_id("Row")]<<','
    <<a[m_device.m_spec->get_level_id("Column")]<<','<<bytes<<','<<req.ingress_id<<','
    <<m_v_source_columns<<','<<(lane<0?-1:m_feature_layout?16+lane%8:lane+(m_v_source_columns==48?32:0))<<','
    <<(lane<0?0:8)<<','<<(lane<0?0:m_feature_layout?(lane<8?255:0):15)<<','
    <<(lane<0?0:m_feature_layout?(lane<8?65280:255):4080)<<','
    <<(lane<0?0:m_feature_layout?(lane<8?0:65280):61440)<<'\n';
}
int output_fragments(int base) const {
  if(is_o_weight_base(base)) return m_feature_layout?8:16;
  if(base==163840) return m_feature_layout?62:31;
  if(base==100352) return 62;
  if(base==327680) return m_ffn_dram?64:256;
  return 64;
}
bool output_retired() const {
  if(m_output_base<0) return true;
  if(m_output_macs!=m_output_expected_macs) return false;
  const bool qk=m_output_base==0 || m_output_base==393216;
  const bool local_v=m_token_pv && (m_output_base==131072 || m_output_base==393216);
  const bool local_pv=m_o_projection && m_output_base==163840;
  if(qk && !output_all(m_output_qk_spills)) return false;
  if(local_v && !output_all(m_v_spills)) return false;
  if(local_pv && !output_all(m_o_pv_spills)) return false;
  if(!qk && !local_v && !local_pv && m_output_export_done!=output_export_packets()) return false;
  return true;
}
bool output_other_ready() {
  if(!m_whole_srf || m_output_base<0) return true;
  if(!output_retired()) {
    // Every result packet may be issued while its final DQ completion is still
    // pending. Wait for that legal tail; missing packets remain an error.
    const bool local=m_output_base==0 || m_output_base==393216 ||
      (m_token_pv && m_output_base==131072) || (m_o_projection && m_output_base==163840);
    if(!local && m_output_macs==m_output_expected_macs &&
       m_output_exports==output_export_packets() && m_output_export_done<m_output_exports) return false;
    throw std::runtime_error("Output owner change before complete result drain");
  }
  if(m_clk<m_output_ready) return false;
  m_output_base=-1;
  return true;
}
bool output_mac_ready(const Request& req) {
  if(!m_whole_srf) return true;
  const int base=gemv_weight_base(req),columns=gemv_columns(req),bytes=gemv_rows(req)*columns*4;
  if(bytes>2048) throw std::runtime_error("Bank output exceeds shared 2KiB capacity");
  if(m_output_base>=0 && m_output_base!=base && !output_other_ready()) return false;
  if(m_output_base<0) {
    m_output_base=base;m_output_columns=columns;m_output_bytes=bytes;
    m_output_macs=0;m_output_expected_macs=output_fragments(base)*columns;
    m_output_exports=m_output_export_done=0;m_output_ready=m_output_compute_ready=0;m_output_qk_spills.fill(0);
    o_output_begin(base);
    if(m_token_pv && (base==131072 || base==393216)) {
      m_v_source_columns=columns;m_v_spills.fill(0);m_v_cache_ready=0;
    }
  }
  if(base==163840) {
    if(!output_all(m_v_spills)) throw std::runtime_error("PV requires complete local V cache");
    if(m_clk<m_v_cache_ready) return false;
  }
  if((gemv_offset(req)-base)/32!=m_output_macs || m_output_macs>=m_output_expected_macs)
    throw std::runtime_error("Projection/output MAC order mismatch");
  return true;
}
void output_mac_issued(const Request& req) {
  if(!m_whole_srf) return;
  ++m_output_macs;m_output_ready=m_output_compute_ready=m_clk+gemv_service(req);
  if(m_token_pv && (m_output_base==131072 || m_output_base==393216) && m_output_macs==m_output_expected_macs)
    record_v_cache(req,"SOURCE_READY",m_output_ready,0);
}
bool output_qk_spill_ready(const Request& req) const {
  if(!m_whole_srf) return true;
  if((m_output_base!=0 && m_output_base!=393216) || m_output_macs!=m_output_expected_macs)
    throw std::runtime_error("QK spill requires complete projection output");
  const int lane=rr_column(req)-rr_local_queries(),bank=rr_bank(req)-16;
  if(m_output_qk_spills[bank]&(1<<lane)) throw std::runtime_error("Duplicate QK output spill");
  return m_clk>=m_output_compute_ready;
}
void output_qk_spill_issued(const Request& req,Clk_t ready) {
  if(!m_whole_srf) return;
  m_output_qk_spills[rr_bank(req)-16]|=1<<(rr_column(req)-rr_local_queries());
  m_output_ready=std::max(m_output_ready,ready);
}
bool output_export_ready(const Request& req) {
  if(!m_whole_srf) return true;
  if(m_output_base<0 || m_output_macs!=m_output_expected_macs)
    throw std::runtime_error("Result export requires complete output owner");
  if(m_output_base==0 || m_output_base==393216 || (m_token_pv && m_output_base==131072) ||
     (m_o_projection && m_output_base==163840))
    throw std::runtime_error("Local projection output must use spill instead of GPU export");
  if(gemv_offset(req)!=m_output_exports*32 || m_output_exports>=output_export_packets())
    throw std::runtime_error("Output export address/count mismatch");
  return m_clk>=m_output_ready && o_bf16_export_ready(req);
}
void output_export_issued(const Request& req) {
  if(!m_whole_srf) return;
  o_bf16_export_issued(req);
  ++m_output_exports;
  record_v_cache(req,"RESULT_EXPORT",req.depart,32);
}
void output_complete(const Request& req) {
  if(!m_whole_srf || req.type_id!=Request::Type::PIM_RD_RES) return;
  if(++m_output_export_done>m_output_exports) throw std::runtime_error("Output completed before export issue");
  if(m_output_base==786432+o_tile_bytes() && m_output_export_done==output_export_packets()) m_o_completed=true;
}
void prepare_v_cache_spill(Request& req) {
  if(!m_token_pv) throw std::runtime_error("V cache spill requires pim_token_pv");
  if(req.addr_vec[m_rank_level]!=1 || req.size_bytes!=32 || req.pim_buffer!=-1)
    throw std::runtime_error("V cache spill requires rank1 aligned32B without input view");
  const int off=gemv_offset(req);
  if(off<163840+30*512 || off>=163840+31*512)
    throw std::runtime_error("V cache spill requires final KV block");
  req.pim_bank_mask=uint64_t{1}<<rr_bank(req);req.pim_rank_mask=2;
  req.final_command=m_localrd;req.arrive=m_clk;
}
bool send_v_cache_spill(Request& req) {
  prepare_v_cache_spill(req);
  if(!m_pim_buffer.enqueue(req)) return false;
  ++s_pim_requests;++s_bgk_requests;return true;
}
bool issue_v_cache_spill(ReqBuffer::iterator it) {
  const int b=rr_bank(*it)-16,lane=(gemv_offset(*it)-(163840+30*512))/32;
  if(it->pim_q_stage==0) {
    if((m_output_base!=131072 && m_output_base!=393216) || m_output_macs!=m_output_expected_macs)
      throw std::runtime_error("V cache spill requires complete V projection");
    if(m_v_spills[b]&(1<<lane)) throw std::runtime_error("duplicate V cache spill");
    if(m_clk<m_output_compute_ready || m_clk<m_v_pack_free[b]) return false;
    it->command=get_preq_command(m_localrd,it->addr_vec);
    if(!issue_internal_array_command(it)) return false;
    if(it->command!=m_localrd) return true;
    it->depart=m_clk+m_device.m_spec->get_timing_value("nCL")+m_bgk_input_cycles;
    it->pim_q_stage=1;++s_v_cache_reads;
    record_v_cache(*it,"LOCALRD",it->depart,32,lane);return true;
  }
  if(m_clk<it->depart) return false;
  if(it->pim_q_stage==1) {
    if(m_clk<m_local_ports[1] || !rank_pseudo("V_GATHER")) return false;
    it->depart=m_local_ports[1]=m_clk+m_gemv_port_cycles;it->pim_q_stage=2;
    s_v_gather_bytes+=32;record_v_cache(*it,"GATHER",it->depart,32,lane);return true;
  }
  if(it->pim_q_stage==2) {
    if(m_clk<m_rope_banks[b+16].cast_busy || !rank_pseudo("V_CAST")) return false;
    it->depart=m_rope_banks[b+16].cast_busy=m_clk+m_cast_cycles;it->pim_q_stage=3;
    it->final_command=m_localwr;
    s_v_cast_bytes+=16;record_v_cache(*it,"CAST",it->depart,16,lane);return true;
  }
  it->command=get_preq_command(m_localwr,it->addr_vec);
  if(!issue_internal_array_command(it)) return false;
  if(it->command!=m_localwr) return true;
  const Clk_t ready=m_clk+m_bank_write_cycles+m_local_recovery;
  m_v_pack_free[b]=m_clk+m_bank_write_cycles;m_v_spills[b]|=1<<lane;
  m_v_cache_ready=std::max(m_v_cache_ready,ready);m_output_ready=std::max(m_output_ready,ready);
  ++s_v_cache_writes;++s_bank_local_writes;
  record_v_cache(*it,"LOCALWR",ready,32,lane);
  record_bgk(*it,ready,m_bank_write_cycles+m_local_recovery,"PIM_SPILL");
  enqueue_bgk_done(it,ready);return true;
}

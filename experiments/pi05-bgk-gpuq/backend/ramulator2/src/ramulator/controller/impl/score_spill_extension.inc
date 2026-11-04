// One bounded FP32 QKT score tile is drained through a rank-local port to
// bank0/4 DRAM. The score store has no external data transfer until ordinary LD.
bool m_score_dram_spill=false;
std::string m_score_spill_path;
std::ofstream m_score_spill_events;
static constexpr int SCORE_BASE=655360, SCORE_TILES=31, SCORE_PACKETS=32;
int m_score_mac_columns=0,m_score_spilled_tiles=0,m_score_read_completions=0;
Clk_t m_score_store_ready=0;
std::unordered_map<int,Clk_t> m_score_writes;
std::unordered_map<int,bool> m_score_reads;
bool is_score_spill(const Request& req) const { return req.type_id==Request::Type::PIM_SCORE_SPILL; }
void record_score(const Request& req,const char* operation,Clk_t ready,int tile,int packet,int bytes) {
  if(!m_score_spill_events) return;
  const auto& a=req.addr_vec;
  m_score_spill_events<<m_clk<<','<<ready<<','<<operation<<','<<m_channel_id<<','<<a[m_rank_level]<<','
      <<a[m_bg_level]<<','<<a[m_bank_level]<<','<<a[m_device.m_spec->get_level_id("Row")]<<','
      <<a[m_device.m_spec->get_level_id("Column")]<<','<<tile<<','<<packet<<','<<bytes<<','<<m_score_mac_columns<<'\n';
}
void init_score_spill() {
  if(m_score_dram_spill && (!m_rank_rope || !m_rank_controller || !m_unified_gemv || m_pim_horizon!=16))
    throw std::runtime_error("Score DRAM spill requires H16 rank RoPE, rank controller and unified GEMV");
}
void setup_score_spill() {
  if(!m_score_dram_spill || m_score_spill_path.empty()) return;
  m_score_spill_events.open(m_score_spill_path);
  if(!m_score_spill_events) throw std::runtime_error("Cannot open score spill events");
  m_score_spill_events<<"clock,ready_at,operation,Channel,Rank,BankGroup,Bank,Row,Column,tile,packet,bytes,mac_columns\n";
}
bool send_score_spill(Request& req) {
  if(!m_score_dram_spill) throw std::runtime_error("PIM_SCORE_SPILL requires pim_score_dram_spill");
  const int offset=stage_offset(req);
  if(req.addr_vec[m_rank_level]!=1 || req.addr_vec[m_bg_level]!=0 || req.addr_vec[m_bank_level]!=0 ||
     req.pim_buffer!=-1 || req.size_bytes!=32 || offset<SCORE_BASE || offset>=SCORE_BASE+SCORE_TILES*1024 || (offset-SCORE_BASE)%1024)
    throw std::runtime_error("Score spill requires canonical rank1 tile descriptor");
  req.pim_rank_mask=2;req.pim_bank_mask=uint64_t{65535}<<16;
  req.final_command=m_localwr;req.arrive=m_clk;
  if(!m_pim_buffer.enqueue(req)) return false;
  ++s_pim_requests;return true;
}
bool score_mac_ready() const {
  if(!m_score_dram_spill) return true;
  if(m_score_mac_columns>=SCORE_TILES*16) throw std::runtime_error("Score K scan exceeds declared tiles");
  if(m_score_mac_columns/16!=m_score_spilled_tiles)
    throw std::runtime_error("Score tile overwritten before complete local spill");
  return m_clk>=m_score_store_ready;
}
void score_mac_issued(const Request& req,Clk_t ready) {
  if(!m_score_dram_spill) return;
  ++m_score_mac_columns;
  record_score(req,"QK_MAC",ready,(m_score_mac_columns-1)/16,-1,0);
}
void score_set_packet(Request& req,int tile,int packet) {
  const int offset=SCORE_BASE+tile*512+(packet/2)*32;
  req.addr_vec[m_rank_level]=1;req.addr_vec[m_bg_level]=packet%2;req.addr_vec[m_bank_level]=0;
  req.addr_vec[m_device.m_spec->get_level_id("Row")]=offset/2048;
  req.addr_vec[m_device.m_spec->get_level_id("Column")]=(offset%2048)/32;
}
bool issue_score_spill(ReqBuffer::iterator it) {
  const auto& origin=it->pim_origin_addr_vec;
  const int logical=origin[m_device.m_spec->get_level_id("Row")]*2048+origin[m_device.m_spec->get_level_id("Column")]*32;
  const int tile=(logical-SCORE_BASE)/1024,packet=it->pim_stage_cursor;
  if(it->pim_q_stage==0) {
    if(tile!=m_score_spilled_tiles || m_score_mac_columns!=(tile+1)*16)
      throw std::runtime_error("Score spill tile order or complete MAC coverage mismatch");
    if(m_clk<m_rank_reduction_ready || m_clk<m_score_store_ready) return false;
    record_score(*it,"SPILL_BEGIN",m_clk,tile,-1,1024);
    score_set_packet(*it,tile,0);it->pim_q_stage=1;
  }
  if(it->pim_q_stage==1) {
    if(m_clk<m_local_ports[1] || !rank_pseudo("SCORE_TRANSFER")) return false;
    it->depart=m_local_ports[1]=m_clk+m_gemv_port_cycles;
    record_score(*it,"TRANSFER",it->depart,tile,packet,32);
    it->pim_q_stage=2;return true;
  }
  if(m_clk<it->depart) return false;
  if(it->pim_q_stage==2) {
    it->command=get_preq_command(m_localwr,it->addr_vec);
    if(!issue_internal_array_command(it)) return false;
    if(it->command!=m_localwr) return true;
    const Clk_t ready=m_clk+m_bank_write_cycles+m_local_recovery;
    const int key=tile*32+packet;
    if(m_score_writes.count(key)) throw std::runtime_error("Duplicate score DRAM write");
    m_score_writes[key]=ready;m_score_store_ready=std::max(m_score_store_ready,ready);
    record_score(*it,"LOCALWR",ready,tile,packet,32);++s_bank_local_writes;
    // The single 32B transfer buffer is reusable once LOCALWR consumes its
    // payload. Bank recovery remains in the native timing graph and in the
    // tile completion fence; it must not block a transfer to the other bank.
    it->depart=m_clk+m_bank_write_cycles;it->pim_q_stage=3;return true;
  }
  if(it->pim_stage_cursor+1<SCORE_PACKETS) {
    ++it->pim_stage_cursor;
    score_set_packet(*it,tile,it->pim_stage_cursor);it->pim_q_stage=1;return true;
  }
  if(m_clk<m_score_store_ready) return false;
  ++it->pim_stage_cursor;
  ++m_score_spilled_tiles;
  record_score(*it,"TILE_DONE",m_clk,tile,-1,1024);
  enqueue_bgk_done(it,m_clk);return true;
}
int score_read_key(const Request& req) const {
  if(!m_score_dram_spill || req.addr_vec[m_rank_level]!=1) return -1;
  const int offset=stage_offset(req);
  if(offset<SCORE_BASE || offset>=SCORE_BASE+16384) return -1;
  if(req.type_id!=Request::Type::Read || req.addr_vec[m_bank_level]!=0 || req.addr_vec[m_bg_level]>1)
    throw std::runtime_error("Score DRAM region permits only bank0/4 reads");
  const int rel=offset-SCORE_BASE;
  const int tile=rel/512,packet=2*((rel%512)/32)+req.addr_vec[m_bg_level];
  if(tile>=SCORE_TILES) throw std::runtime_error("Score read outside stored tile coverage");
  return tile*32+packet;
}
bool score_read_ready(int key) const {
  if(key<0) return true;
  if(m_score_spilled_tiles!=SCORE_TILES || m_score_writes.size()!=SCORE_TILES*SCORE_PACKETS)
    throw std::runtime_error("Score export before complete DRAM store");
  if(m_score_reads.count(key)) throw std::runtime_error("Duplicate score DRAM read");
  return m_clk>=m_score_store_ready;
}
void score_read_accepted(const Request& req,int key) {
  if(key<0) return;
  m_score_reads[key]=true;record_score(req,"READ_ACCEPT",m_clk,key/32,key%32,32);
}
void score_read_complete(const Request& req,int key) {
  if(key<0) return;
  ++m_score_read_completions;record_score(req,"READ_COMPLETE",m_clk,key/32,key%32,32);
}
void finish_score_spill() const {
  if(!m_score_dram_spill || !m_score_mac_columns) return;
  if(m_score_mac_columns!=SCORE_TILES*16 || m_score_spilled_tiles!=SCORE_TILES ||
     m_score_read_completions!=SCORE_TILES*SCORE_PACKETS || m_clk<m_score_store_ready)
    throw std::runtime_error("Incomplete QKT score spill/export at end of trace");
}

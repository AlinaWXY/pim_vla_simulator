// Rank descriptors reuse the existing fine request queue and completion rules.
// Only the parent is admitted over the external command bus. The unchanged
// bank micro-requests arbitrate on rank CA and retain all data/array hazards.
bool m_rank_descriptors=false;
std::string m_rank_descriptor_path;
std::ofstream m_rank_descriptor_events;
uint64_t m_rank_descriptor_sequence=0;
struct RankDescriptorCompletion {
  Request parent;
  uint64_t id=0;
  uint32_t completed_mask=0;
  Clk_t max_ready=0;
};
const char* rank_descriptor_name(int type) const {
  static const char* names[]={"PIM_RANK_SPILL","PIM_RANK_ROPE_COEF","PIM_RANK_ROPE_LOAD",
    "PIM_RANK_ROPE_EXEC","PIM_RANK_ROPE_STORE","PIM_RANK_Q_EXPORT","PIM_RANK_Q_REMOTE"};
  if(type<Request::Type::PIM_SPILL || type>Request::Type::PIM_Q_REMOTE)
    throw std::runtime_error("unsupported rank descriptor operation");
  return names[type-Request::Type::PIM_SPILL];
}
void init_rank_descriptors() {
  if(m_rank_descriptors && (!m_rank_controller || !m_unified_gemv))
    throw std::runtime_error("rank descriptors require rank controller and unified GEMV");
  if(m_rank_descriptors && (m_bgs*m_banks_per_bg!=16 || m_pim_buffer.max_size<16))
    throw std::runtime_error("rank descriptors require sixteen banks and sixteen fine queue entries");
}
void setup_rank_descriptors() {
  if(!m_rank_descriptors || m_rank_descriptor_path.empty()) return;
  m_rank_descriptor_events.open(m_rank_descriptor_path);
  if(!m_rank_descriptor_events) throw std::runtime_error("Cannot open rank descriptor events");
  m_rank_descriptor_events<<"clock,event,descriptor_id,trace_index,operation,Channel,Rank,BankGroup,Bank,Row,Column,buffer,children,ready_at\n";
}
void record_rank_descriptor(const Request& req,uint64_t id,const char* event,Clk_t clock,Clk_t ready,int children) {
  if(!m_rank_descriptor_events) return;
  const auto& a=req.pim_origin_addr_vec.empty()?req.addr_vec:req.pim_origin_addr_vec;
  m_rank_descriptor_events<<clock<<','<<event<<','<<id<<','<<req.ingress_id<<','
    <<(req.pim_rank_batch?rank_descriptor_name(req.type_id):rr_name(req.type_id))<<','
    <<m_channel_id<<','<<a[m_rank_level]<<','<<a[m_bg_level]<<','<<a[m_bank_level]<<','
    <<a[m_device.m_spec->get_level_id("Row")]<<','<<a[m_device.m_spec->get_level_id("Column")]<<','
    <<req.pim_buffer<<','<<children<<','<<ready<<'\n';
}
bool send_rank_descriptor(Request& parent) {
  if(!m_rank_descriptors) throw std::runtime_error("rank descriptor requires pim_rank_descriptors");
  rank_descriptor_name(parent.type_id); // Validate the allowed operation before expansion.
  if(parent.addr_vec[m_rank_level]!=1 || parent.addr_vec[m_bg_level]!=0 || parent.addr_vec[m_bank_level]!=0)
    throw std::runtime_error("rank descriptor requires canonical rank1 BG0 bank0");
  if(parent.channel_count<1 || (parent.channel_count&(parent.channel_count-1)) ||
     m_channel_id>=parent.channel_count || parent.size_bytes!=32 || parent.addr%32)
    throw std::runtime_error("rank descriptor requires aligned 32B address and valid channel geometry");
  if(m_pim_buffer.size()+16>m_pim_buffer.max_size) return false;
  const auto& spec=*m_device.m_spec;
  const int row_level=spec.get_level_id("Row"),column_level=spec.get_level_id("Column");
  const Addr_t columns=spec.get_level_size("Column")/spec.internal_prefetch_size;
  const Addr_t ranks=spec.get_level_size("Rank"),tx=spec.get_tx_bytes();
  const auto encode_intra=[&](const AddrVec_t& a) {
    return (((static_cast<Addr_t>(a[row_level])*m_banks_per_bg+a[m_bank_level])*m_bgs+a[m_bg_level])*ranks+a[m_rank_level])*columns*tx+a[column_level]*tx;
  };
  const auto encode_full=[&](Addr_t intra) {
    return ((intra/tx)*parent.channel_count+m_channel_id)*tx+intra%tx;
  };
  if(encode_intra(parent.addr_vec)!=parent.intra_channel_addr || encode_full(parent.intra_channel_addr)!=parent.addr)
    throw std::runtime_error("rank descriptor requires RoBaRaCoCh and cache-line channel interleave");
  std::array<Request,16> children;
  for(int index=0;index<16;++index) {
    auto& child=children[index];child=parent;child.pim_rank_batch=false;child.pim_rank_micro=true;
    auto expected=parent.addr_vec;
    expected[m_bg_level]=index/m_banks_per_bg;expected[m_bank_level]=index%m_banks_per_bg;
    child.intra_channel_addr=encode_intra(expected);child.addr=encode_full(child.intra_channel_addr);
    m_addr_mapper->apply(child);child.addr_vec[0]=m_channel_id;
    if(child.addr_vec!=expected) throw std::runtime_error("rank descriptor child address mapping mismatch");
    child.pim_origin_addr_vec=child.addr_vec;
    if(is_v_cache_spill(child)) prepare_v_cache_spill(child);
    else prepare_rank_rope(child); // Validate every child before enqueuing any of them.
  }
  auto completion=std::make_shared<RankDescriptorCompletion>();
  completion->parent=parent;completion->parent.arrive=m_clk;
  completion->id=++m_rank_descriptor_sequence;
  const Clk_t dispatch=std::max<Clk_t>(1,m_clk);
  record_rank_descriptor(parent,completion->id,"DISPATCH",dispatch,dispatch,16);
  for(int index=0;index<16;++index) {
    auto& child=children[index];
    child.callback=[this,completion,index](Request& done) {
      const uint32_t bit=uint32_t{1}<<index;
      if(completion->completed_mask&bit || rr_bank(done)!=16+index || done.depart>m_clk)
        throw std::runtime_error("invalid or premature rank descriptor child completion");
      completion->completed_mask|=bit;
      completion->max_ready=std::max(completion->max_ready,done.depart);
      record_rank_descriptor(done,completion->id,"CHILD_COMPLETE",m_clk,done.depart,1);
      if(completion->completed_mask==65535) {
        auto& parent_done=completion->parent;parent_done.depart=completion->max_ready;
        record_rank_descriptor(parent_done,completion->id,"COMPLETE",m_clk,parent_done.depart,16);
        if(parent_done.callback) parent_done.callback(parent_done);
      }
    };
    if(!m_pim_buffer.enqueue(child)) throw std::runtime_error("rank descriptor atomic queue reservation violated");
    ++s_pim_requests;++s_bgk_requests;
  }
  return true;
}

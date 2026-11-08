// Four metadata views do not allocate four physical 512B buffers. A whole-SRF
// stage partitions exactly 1024B into four 256B or two 512B views.
bool m_whole_srf=false;
int m_whole_rows=0,m_whole_batch=-1;
int whole_views() const { return m_whole_rows?1024/(m_whole_rows*32):0; }
void whole_stage_begin(int rows) {
  if(!m_whole_srf) return;
  for(int b=16;b<32;++b) for(int s=0;s<4;++s)
    if(m_gemv_slots[b][s].kind!=GemvSlot::Empty)
      throw std::runtime_error("Whole SRF operator change requires all views released");
  m_whole_rows=rows;m_whole_batch=-1;
}
bool whole_fill_ready(const Request& req,int row) {
  if(!m_whole_srf) return true;
  const int views=whole_views(),fragment=(stage_logical_offset(req)-m_stage.base)/(m_stage.rows*32);
  if(!views || req.pim_buffer!=fragment%views)
    throw std::runtime_error("Whole SRF fragment/view identity mismatch");
  if(row) return true;
  const int batch=fragment/views;
  if(batch==m_whole_batch) return true;
  if(batch!=m_whole_batch+1 || fragment%views)
    throw std::runtime_error("Whole SRF batch order mismatch");
  for(int s=0;s<4;++s) {
    if(!gemv_slot_idle(s)) return false;
    for(int b=16;b<32;++b) {
      const auto& state=m_gemv_slots[b][s];
      if(state.kind!=GemvSlot::Empty && (state.kind!=GemvSlot::External ||
          !state.expected_columns || state.mac_columns!=state.expected_columns))
        throw std::runtime_error("Whole SRF overwrite before complete batch consumption");
    }
  }
  return true;
}
void whole_fill_start(const Request& req) {
  if(!m_whole_srf) return;
  const int batch=(stage_logical_offset(req)-m_stage.base)/(m_stage.rows*32*whole_views());
  if(batch==m_whole_batch) return;
  // Old views were fully consumed before the first read of this batch.
  for(int b=16;b<32;++b) for(int s=0;s<4;++s) gemv_clear_bank(b,s);
  m_whole_batch=batch;
}
bool whole_mac_ready(const Request& req) const {
  if(!m_whole_srf) return true;
  const int rows=gemv_rows(req),views=1024/(rows*32);
  if(direct_weight(req)) {
    const int fragment=(gemv_offset(req)-gemv_weight_base(req))/(gemv_columns(req)*32);
    const int valid=std::min(views,output_fragments(gemv_weight_base(req))-(fragment/views)*views);
    if(req.pim_buffer<0 || req.pim_buffer>=views) throw std::runtime_error("Direct input exceeds 1KiB SRF");
    for(int s=0;s<valid;++s) for(int b=16;b<32;++b) {
      const auto& state=m_gemv_slots[b][s];
      if(state.kind!=GemvSlot::External) return false;
      for(int q=0;q<rows;++q) if(state.lanes[q]!=65535) return false;
      for(int q=rows;q<16;++q) if(state.lanes[q]) throw std::runtime_error("Direct batch excess row");
      if(m_clk<m_bank_inputs[b][s].ready_at) return false;
    }
    return true;
  }
  if(rows!=m_whole_rows || req.pim_buffer<0 || req.pim_buffer>=views || m_whole_batch<0)
    throw std::runtime_error("SRF view exceeds physical 1KiB capacity");
  const int fragments=m_stage.bytes/(rows*32);
  const int valid=std::min(views,fragments-m_whole_batch*views);
  for(int s=0;s<valid;++s) for(int b=16;b<32;++b) {
    const auto& state=m_gemv_slots[b][s];
    if(m_stage_slot_generation[s]!=m_stage.generation ||
       m_stage_slot_fragment[s]!=m_whole_batch*views+s || state.kind!=GemvSlot::External)
      throw std::runtime_error("Whole SRF compute before full batch fill");
    for(int row=0;row<rows;++row) if(state.lanes[row]!=65535)
      throw std::runtime_error("Whole SRF compute before full batch fill");
    if(m_clk<m_bank_inputs[b][s].ready_at) return false;
  }
  return true;
}
void whole_q_ready(int bank) const {
  if(!m_whole_srf) return;
  // QKT uses one 512B Q tile, after releasing all projection views.
  for(int s=1;s<4;++s) if(m_gemv_slots[bank][s].kind!=GemvSlot::Empty)
    throw std::runtime_error("Q tile overlaps unreleased SRF views");
}

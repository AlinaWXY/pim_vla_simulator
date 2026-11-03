// Shared split-activation ownership for internal PIM array accesses. Payloads
// and their completion callbacks remain in the existing PIM request queue.
bool m_internal_act2_pending=false;
ReqBuffer::iterator m_internal_act2_owner;

bool issue_internal_array_command(ReqBuffer::iterator it) {
  // A PIM access must not take an ordinary or internal activation's owner.
  // The pending internal ACT2 is issued exclusively by issue_internal_act2().
  if(has_pending_act2()) return false;
  if(it->command==m_cmd_act2)
    throw std::runtime_error("internal ACT2 without its activation owner");
  if(!can_start_act1(*it) || !check_timing(it->command,it->addr_vec)) return false;
  m_device.issue_command(it->command,it->addr_vec,m_clk);
  rr_notify_command(*it);
  if(it->command==m_cmd_act1) {
    const int bank=m_device.get_flat_bank_id(it->addr_vec);
    m_internal_act2_owner=it;
    m_internal_act2_pending=true;
    m_act2_owner_valid[bank]=true;
    m_act2_deadline[bank]=m_clk+m_nAAD;
  }
  return true;
}

void issue_internal_act2() {
  if(!m_internal_act2_pending)
    throw std::runtime_error("internal ACT2 without a pending request");
  RankFineScope fine(m_rank_fine_context,m_rank_controller);
  auto it=m_internal_act2_owner;
  const int bank=m_device.get_flat_bank_id(it->addr_vec);
  if(!m_act2_owner_valid[bank] ||
     get_preq_command(it->final_command,it->addr_vec)!=m_cmd_act2)
    throw std::runtime_error("internal ACT2 lost its activation owner");
  if(m_clk>m_act2_deadline[bank])
    throw std::runtime_error("internal ACT2 deadline exceeded");
  if(!check_timing(m_cmd_act2,it->addr_vec)) {
    if(m_clk==m_act2_deadline[bank])
      throw std::runtime_error("internal ACT2 cannot meet its deadline");
    return;
  }
  it->command=m_cmd_act2;
  m_device.issue_command(m_cmd_act2,it->addr_vec,m_clk);
  rr_notify_command(*it);
  m_act2_owner_valid[bank]=false;
  m_act2_deadline[bank]=-1;
  m_internal_act2_pending=false;
}

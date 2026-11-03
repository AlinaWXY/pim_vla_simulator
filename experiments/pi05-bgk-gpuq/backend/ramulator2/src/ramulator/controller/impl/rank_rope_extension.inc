// Opt-in timing model. Included inside LPDDR5PIMController; no numerical arithmetic.
bool m_rank_rope = false;
int m_rope_ii = 1, m_rope_latency = 4, m_cast_cycles = 1, m_rank_reduce_cycles = 4;
int m_localrd = -1;
Clk_t m_projection_ready = 0, m_rank_reduction_ready = 0;
struct RopeInput { Addr_t address = -1; Clk_t ready = 0, busy = 0; bool valid = false; };
struct RopeOutput { Addr_t address; Clk_t ready; Clk_t release = std::numeric_limits<Clk_t>::max(); };
struct RopeBank {
  std::array<RopeInput, 2> inputs;
  int coefficient_token = -1, coefficient_uses = 3;
  Clk_t coefficient_ready = 0, coefficient_busy = 0, next_pair_issue = 0, cast_busy = 0;
  std::vector<RopeOutput> outputs; // two 32B output FIFO entries, distinct from input buffers
  std::unordered_map<Addr_t, Clk_t> spill_ready, rotated_ready;
  uint64_t query_mask = 0;
};
std::vector<RopeBank> m_rope_banks;

bool is_rank_rope(const Request& r) const { return r.type_id >= Request::Type::PIM_SPILL && r.type_id <= Request::Type::PIM_Q_REMOTE; }
int rr_bank(const Request& r) const {
  return (r.addr_vec[m_rank_level] * m_bgs + r.addr_vec[m_bg_level]) * m_banks_per_bg + r.addr_vec[m_bank_level];
}
// Work storage begins at row40. The logical column may cross a DRAM row;
// native events and array commands retain the actual Row/Column address.
int rr_column(const Request& r) const {
  return (r.addr_vec[m_device.m_spec->get_level_id("Row")]-40)*64 + r.addr_vec[m_device.m_spec->get_level_id("Column")];
}
int rr_local_queries() const { return m_pim_horizon/2; }
uint64_t rr_full_query_mask() const {
  return m_pim_horizon==64 ? ~uint64_t{0} : (uint64_t{1}<<m_pim_horizon)-1;
}
const char* rr_name(int type) const {
  static const char* names[] = {"PIM_SPILL", "PIM_ROPE_COEF", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC", "PIM_ROPE_STORE", "PIM_Q_EXPORT", "PIM_Q_REMOTE"};
  return names[type - Request::Type::PIM_SPILL];
}
void init_rank_rope() {
  if (m_rope_ii < 1 || m_rope_latency < 1 || m_cast_cycles < 1 || m_rank_reduce_cycles < 1)
    throw std::runtime_error("Rank RoPE latencies and II must be positive");
  if (m_rank_rope && (m_inputs.size() != 2 || m_data_path != "attacc" || m_input_broadcast || m_mac_scope != "rank"))
    throw std::runtime_error("Rank RoPE requires rank1 PIM, rank0 reserved, rank-scoped AttAcc");
  m_localrd = m_device.m_spec->get_command_id("LOCALRD");
  m_rope_banks.resize(m_bank_inputs.size());
}
void prepare_rank_rope(Request& req) {
  if (!m_rank_rope) throw std::runtime_error("New rank/RoPE request requires pim_rank_rope=true");
  if (req.addr_vec[m_rank_level] != 1 || req.size_bytes != 32)
    throw std::runtime_error("Rank RoPE request requires rank1 and exactly 32B");
  const int col = rr_column(req);
  const int local_queries = rr_local_queries();
  if ((req.type_id == Request::Type::PIM_ROPE_COEF && (col < 0 || col >= local_queries)) ||
      (req.type_id != Request::Type::PIM_ROPE_COEF && (col < local_queries || col >= 3*local_queries)) ||
      ((req.type_id == Request::Type::PIM_Q_EXPORT || req.type_id == Request::Type::PIM_Q_REMOTE) && col >= 2*local_queries))
    throw std::runtime_error("Rank RoPE working-row column violates coefficient/Q/K contract");
  if ((req.type_id == Request::Type::PIM_ROPE_LOAD || req.type_id == Request::Type::PIM_ROPE_EXEC ||
       req.type_id == Request::Type::PIM_Q_REMOTE) && req.pim_buffer < 0)
    throw std::runtime_error("Rank RoPE load/execute/remote requires buffer 0/1");
  if (req.type_id == Request::Type::PIM_Q_REMOTE && req.pim_buffer != 0)
    throw std::runtime_error("Rank Q batch is the bounded " + std::to_string(32*m_pim_horizon) + "B buffer0");
  req.pim_bank_mask = uint64_t{1} << rr_bank(req);
  req.pim_rank_mask = 2;
  req.final_command = spec_command(req.type_id);
  req.arrive = m_clk;
}
bool send_rank_rope(Request& req) {
  prepare_rank_rope(req);
  if(!m_pim_buffer.enqueue(req)) return false;
  ++s_pim_requests; ++s_bgk_requests;
  return true;
}
bool rr_bypass(const Request& later, const Request& earlier) const {
  if (!(later.pim_bank_mask & earlier.pim_bank_mask)) return true;
  // Preserve all same-bank memory issue order. A pending compute may allow
  // a load into the other input slot; its array prerequisites can run early.
  return is_rank_rope(later) && is_rank_rope(earlier) &&
      later.type_id == Request::Type::PIM_ROPE_LOAD && earlier.type_id == Request::Type::PIM_ROPE_EXEC &&
      later.pim_buffer != earlier.pim_buffer;
}
void rr_notify_command(Request& req) {
  m_rowpolicy->on_issue(req);
  for (auto* p : m_plugins) p->on_issue(req);
}
bool rr_export(ReqBuffer::iterator it) {
  auto& bank = m_rope_banks[rr_bank(*it)];
  if (it->pim_q_stage == 0) {
    const auto ready = bank.rotated_ready.find(it->addr);
    if (ready == bank.rotated_ready.end()) throw std::runtime_error("Q export precedes RoPE store");
    if (m_clk < ready->second) return false;
    if (m_unified_gemv) {
      const int q=2*(rr_column(*it)-rr_local_queries())+m_channel_id%2;
      if(m_gemv_ingress_owner || m_clk<m_gemv_ingress_free || !gemv_q_ready(rr_bank(*it),q)) return false;
    }
    Request* owner = &*it;
    Request rd = *it;
    rd.type_id = Request::Type::Read; rd.pim_buffer = -1;
    rd.callback = [this, owner](Request& done) {
      if (done.command != m_cmd_rd && done.command != m_cmd_rda)
        throw std::runtime_error("Q export requires a real DRAM RD");
      auto& b = m_rope_banks[rr_bank(*owner)];
      const int query = 2 * (rr_column(*owner) - rr_local_queries()) + m_channel_id % 2;
      if (b.query_mask & (uint64_t{1} << query)) throw std::runtime_error("Duplicate local Q fill");
      if (!m_unified_gemv) b.query_mask |= uint64_t{1} << query;
      auto& input = m_bank_inputs[rr_bank(*owner)][0];
      if (!m_unified_gemv) { input.written = true; input.ready_at = std::max(input.ready_at, done.depart); }
      owner->depart=done.depart;
      owner->pim_q_stage = 2;
      record_bgk(*owner, done.depart, m_device.m_spec->read_latency, "PIM_Q_EXPORT",
                 done.depart - m_device.m_spec->read_latency, 1);
    };
    if (!LPDDR5Controller::send(rd)) return false;
    it->pim_q_stage = 1;
    if (m_unified_gemv) m_gemv_ingress_owner=&*it;
    return true;
  }
  if (it->pim_q_stage == 2) {
    if(m_unified_gemv) {
      const int b=rr_bank(*it),q=2*(rr_column(*it)-rr_local_queries())+m_channel_id%2;
      if(m_clk<m_local_ports[1] || !gemv_q_ready(b,q)) return false;
      if(!rank_pseudo("Q_LOCAL_FILL")) return false;
      const Clk_t ready=gemv_q_write(*it,b,q,false);
      m_rope_banks[b].query_mask|=uint64_t{1}<<q;
      m_gemv_ingress_owner=nullptr;
      enqueue_bgk_done(it,ready);
    } else enqueue_bgk_done(it,m_clk);
    return true;
  }
  return false;
}
bool issue_rank_rope(ReqBuffer::iterator it, bool local_port) {
  if (it->type_id == Request::Type::PIM_Q_EXPORT) return !local_port && rr_export(it);
  const int b = rr_bank(*it), col = rr_column(*it), type = it->type_id;
  auto& bank = m_rope_banks[b];
  std::erase_if(bank.outputs, [this](const RopeOutput& out) { return out.release <= m_clk; });
  if (type == Request::Type::PIM_ROPE_EXEC) {
    if (!local_port) return false;
    auto& input = bank.inputs[it->pim_buffer];
    if (!input.valid || input.address != it->addr) throw std::runtime_error("RoPE execute needs matching loaded 32B input");
    if (bank.coefficient_token != col % rr_local_queries()) throw std::runtime_error("RoPE coefficient token mismatch");
    if (m_clk < input.ready || m_clk < bank.coefficient_ready || m_clk < bank.next_pair_issue || bank.outputs.size() >= 2) return false;
    if(!rank_pseudo("PIM_ROPE_EXEC")) return false;
    const int use = col < 2*rr_local_queries() ? 1 : 2;
    if (bank.coefficient_uses & use) throw std::runtime_error("Duplicate RoPE vector for coefficients");
    bank.coefficient_uses |= use;
    const Clk_t ready = m_clk + 7 * m_rope_ii + m_rope_latency;
    input.busy = m_clk + 8 * m_rope_ii;
    input.valid = false;
    bank.coefficient_busy = std::max(bank.coefficient_busy, input.busy);
    bank.next_pair_issue = input.busy;
    bank.outputs.push_back({it->addr, ready});
    record_bgk(*it, ready, 7 * m_rope_ii + m_rope_latency, rr_name(type));
    enqueue_bgk_done(it, ready);
    return true;
  }
  if (type == Request::Type::PIM_SPILL && it->pim_q_stage == 0) {
    if (m_clk < m_projection_ready || m_clk < bank.cast_busy) return false;
    if (col == rr_local_queries()) {
      if (m_clk < m_bank_inputs[b][0].in_use_until) return false;
      bank.query_mask = 0;
      m_bank_inputs[b][0].written = false;
    }
    bank.rotated_ready.erase(it->addr);
    it->depart = bank.cast_busy = m_clk + m_cast_cycles;
    it->pim_q_stage = 1;
    record_bgk(*it, it->depart, m_cast_cycles, "PIM_CAST");
  }
  if (type == Request::Type::PIM_ROPE_COEF &&
      (bank.coefficient_uses != 3 || m_clk < bank.coefficient_busy)) return false;
  if (type == Request::Type::PIM_ROPE_LOAD) {
    auto found = bank.spill_ready.find(it->addr);
    if (found == bank.spill_ready.end()) throw std::runtime_error("RoPE input requires projection spill");
    auto& input = bank.inputs[it->pim_buffer];
    if (m_clk < found->second || m_clk < input.busy || input.valid) return false;
  }
  auto out = bank.outputs.end();
  if (type == Request::Type::PIM_ROPE_STORE) {
    out = std::find_if(bank.outputs.begin(), bank.outputs.end(), [&](const RopeOutput& o) { return o.address == it->addr; });
    if (out == bank.outputs.end()) throw std::runtime_error("RoPE store requires computed output FIFO entry");
    if (m_clk < out->ready) return false;
  }
  it->command = get_preq_command(it->final_command, it->addr_vec);
  const bool final = it->command == it->final_command;
  const bool internal = type != Request::Type::PIM_Q_REMOTE;
  if (local_port != (final && internal)) return false;
  if (final && type == Request::Type::PIM_SPILL && m_clk < it->depart) return false;
  if (final && type == Request::Type::PIM_Q_REMOTE && m_clk < m_bank_inputs[b][0].in_use_until) return false;
  if(m_unified_gemv && final && type==Request::Type::PIM_Q_REMOTE) {
    const int q=2*(col-rr_local_queries())+(1-m_channel_id%2);
    if(m_gemv_ingress_owner || m_clk<m_gemv_ingress_free || m_clk<m_local_ports[1] || !gemv_q_ready(b,q)) return false;
  }
  if (!issue_internal_array_command(it)) return false;
  if (!final) return true;
  int service = (type == Request::Type::PIM_ROPE_COEF || type == Request::Type::PIM_ROPE_LOAD)
      ? m_device.m_spec->get_timing_value("nCL") + m_bgk_input_cycles
      : type == Request::Type::PIM_Q_REMOTE ? m_bgk_input_cycles : m_bank_write_cycles + m_local_recovery;
  Clk_t ready = m_clk + service;
  if (type == Request::Type::PIM_SPILL) bank.spill_ready[it->addr] = ready;
  if (type == Request::Type::PIM_ROPE_COEF) {
    bank.coefficient_token = col; bank.coefficient_ready = ready; bank.coefficient_uses = 0;
  }
  if (type == Request::Type::PIM_ROPE_LOAD) {
    auto& input = bank.inputs[it->pim_buffer];
    input.address = it->addr; input.ready = ready; input.valid = true;
  }
  if (type == Request::Type::PIM_ROPE_STORE) {
    bank.rotated_ready[it->addr] = ready;
    out->release = m_clk + m_bank_write_cycles;
  }
  if (type == Request::Type::PIM_Q_REMOTE) {
    const int query = 2 * (col - rr_local_queries()) + (1 - m_channel_id % 2);
    if (bank.query_mask & (uint64_t{1} << query)) throw std::runtime_error("Duplicate remote Q fill");
    bank.query_mask |= uint64_t{1} << query;
    auto& input = m_bank_inputs[b][0];
    if(m_unified_gemv) {
      ready=gemv_q_write(*it,b,query,true);service=ready-m_clk;
    } else { input.written = true; input.ready_at = std::max(input.ready_at, ready); }
  }
  record_bgk(*it, ready, service, rr_name(type));
  enqueue_bgk_done(it, ready);
  return true;
}
void issue_rank_rope_local() {
  if (!m_rank_rope) return;
  for (auto it = m_pim_buffer.begin(); it != m_pim_buffer.end();) {
    auto current = it++;
    if (!is_rank_rope(*current)) continue;
    bool independent = true;
    for (auto before = m_pim_buffer.begin(); before != current; ++before)
      if (!can_bypass(*current, *before, false)) { independent = false; break; }
    if (independent) issue_rank_rope(current, true);
  }
}

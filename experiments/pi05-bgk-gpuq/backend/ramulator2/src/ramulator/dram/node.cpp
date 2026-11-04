#include "ramulator/dram/node.h"

namespace Ramulator {

DRAMNode::DRAMNode(DRAMSpec* spec, DRAMNode* parent, int level, int id)
    : m_spec(spec), m_parent_node(parent), m_level(level), m_node_id(id) {
  int num_cmds = spec->command_count;
  m_cmd_ready_clk.resize(num_cmds, -1);
  m_cmd_history.resize(num_cmds);
  int shared_window_group_count = 0;
  for (int cmd = 0; cmd < num_cmds; cmd++) {
    int window = 0;
    for (const auto& t : spec->timing_cons[level][cmd]) {
      window = std::max(window, t.window);
      if (t.history_group >= 0) {
        shared_window_group_count = std::max(shared_window_group_count, t.history_group + 1);
      }
    }
    if (window != 0) {
      m_cmd_history[cmd].resize(window, -1);
    } else {
      m_cmd_history[cmd].clear();
    }
  }
  m_shared_window_history.resize(shared_window_group_count);
  for (int cmd = 0; cmd < num_cmds; cmd++) {
    for (const auto& t : spec->timing_cons[level][cmd]) {
      if (t.history_group >= 0 && m_shared_window_history[t.history_group].size() < t.window) {
        m_shared_window_history[t.history_group].resize(t.window, -1);
      }
    }
  }

  m_state = spec->init_states[m_level];

  // Recursively construct next levels
  int next_level = level + 1;
  int last_level = spec->get_level_id("Row");
  if (next_level == last_level) {
    return;
  } else {
    int next_level_size = m_spec->organization.level_sizes[next_level];
    if (next_level_size == 0) {
      return;
    } else {
      for (int i = 0; i < next_level_size; i++) {
        m_child_nodes.push_back(std::make_unique<DRAMNode>(spec, this, next_level, i));
      }
    }
  }
}

void DRAMNode::update_timing(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  const bool half = m_spec->bank_targets[command] == BankTarget::Half;
  const int bg_level = half ? m_spec->get_level_id("BankGroup") : -1;
  const bool half_group = half && m_level == bg_level;
  const bool matches = half && m_level == m_spec->get_level_id("Bank") ? true : half_group
      ? m_node_id / (m_spec->organization.level_sizes[bg_level] / 2) ==
            addr_vec[m_level] / (m_spec->organization.level_sizes[bg_level] / 2)
      : m_node_id == addr_vec[m_level] || addr_vec[m_level] == -1;
  /************************************************
   *         Update Sibling Node Timing
   ***********************************************/
  if (!matches) {
    for (const auto& t : m_spec->timing_cons[m_level][command]) {
      if (!t.sibling) {
        continue;
      }
      Clk_t future = clk + t.val;
      m_cmd_ready_clk[t.cmd] = std::max(m_cmd_ready_clk[t.cmd], future);
    }
    return;
  }

  /************************************************
   *          Update Target Node Timing
   ***********************************************/
  if (m_cmd_history[command].size()) {
    m_cmd_history[command].pop_back();
    m_cmd_history[command].push_front(clk);
  }

  std::vector<bool> shared_window_updated(m_shared_window_history.size(), false);
  for (const auto& t : m_spec->timing_cons[m_level][command]) {
    if (t.history_group < 0 || shared_window_updated[t.history_group]) {
      continue;
    }
    auto& history = m_shared_window_history[t.history_group];
    history.pop_back();
    history.push_front(clk + m_spec->command_cycles[command] - 1);
    shared_window_updated[t.history_group] = true;
  }

  for (const auto& t : m_spec->timing_cons[m_level][command]) {
    if (t.sibling) {
      continue;
    }
    Clk_t past = t.history_group >= 0
                     ? m_shared_window_history[t.history_group][t.window - 1]
                     : m_cmd_history[command][t.window - 1];
    if (past < 0) {
      continue;
    }
    Clk_t future = past + t.val;
    m_cmd_ready_clk[t.cmd] = std::max(m_cmd_ready_clk[t.cmd], future);
  }

  if (m_child_nodes.empty()) {
    return;
  }

  // Skip traversal that does not have sibling timing constraints
  int child_level = m_level + 1;
  int target_child_id = half && child_level == m_spec->get_level_id("Bank") ? -1 : addr_vec[child_level];
  bool children_have_sibling_cons = m_spec->has_sibling_cons[child_level][command];

  if (children_have_sibling_cons || target_child_id == -1 || (half && child_level == bg_level)) {
    for (auto& child : m_child_nodes) {
      child->update_timing(command, addr_vec, clk);
    }
  } else {
    m_child_nodes[target_child_id]->update_timing(command, addr_vec, clk);
  }
}

bool DRAMNode::check_timing(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  if (m_cmd_ready_clk[command] != -1 && clk < m_cmd_ready_clk[command]) {
    return false;
  }

  int child_id = m_spec->bank_targets[command] == BankTarget::Half &&
                         m_level + 1 == m_spec->get_level_id("Bank")
                     ? -1 : addr_vec[m_level + 1];
  if (m_child_nodes.empty()) {
    return true;
  }

  if (m_spec->bank_targets[command] == BankTarget::Half &&
      m_level + 1 == m_spec->get_level_id("BankGroup")) {
    const int groups_per_half = m_child_nodes.size() / 2;
    if (groups_per_half == 0 || child_id < 0) {
      throw std::runtime_error("Half-bank commands require an explicit BankGroup");
    }
    const int first = child_id / groups_per_half * groups_per_half;
    for (int i = first; i < first + groups_per_half; ++i) {
      if (!m_child_nodes[i]->check_timing(command, addr_vec, clk)) return false;
    }
    return true;
  }

  if (child_id == -1) {
    bool timing_ok = true;
    for (auto& child : m_child_nodes) {
      timing_ok = timing_ok && child->check_timing(command, addr_vec, clk);
    }
    return timing_ok;
  } else {
    return m_child_nodes[child_id]->check_timing(command, addr_vec, clk);
  }
}

}  // namespace Ramulator

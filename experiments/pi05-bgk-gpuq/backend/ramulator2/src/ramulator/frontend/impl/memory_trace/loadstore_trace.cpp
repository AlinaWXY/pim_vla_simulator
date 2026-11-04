#include <filesystem>
#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <map>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class LoadStoreTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, LoadStoreTrace, "LoadStoreTrace")

 private:
  static constexpr int Barrier = -2;
  static constexpr int Wait = -3;
  struct Trace {
    int type_id; Addr_t addr; int pim_buffer; int stage_bytes=0, stage_rows=0;
    bool rank_batch=false;
    size_t peer_source=std::numeric_limits<size_t>::max();
  };
  std::vector<Trace> m_trace;
  size_t m_curr_trace_idx = 0;
  size_t m_trace_count = 0;
  size_t m_completed = 0;
  std::unordered_set<Addr_t> m_q_exports_issued, m_q_exports_completed;
  size_t m_barriers = 0;
  uint64_t m_wait_cycles = 0;
  std::string m_trace_path;
  std::string m_events_path;
  std::ofstream m_events;
  bool m_wait_for_completion = false;
  int m_write_tail_cycles = 0;
  int m_last_pim = -1;
  bool m_phase_has_write = false;
  Clk_t m_write_ready_at = 0;
  Clk_t m_wait_until = -1;
  uint64_t m_max_frontend_ticks = 0;
  bool m_channel_independent_dispatch = false;
  std::string m_dispatch_events_path;
  std::ofstream m_dispatch_events;
  std::vector<std::vector<size_t>> m_channel_streams;
  std::vector<size_t> m_channel_positions;
  std::vector<bool> m_trace_completed, m_peer_wait_logged;
  bool m_epoch_active = false;

  void record_dispatch(size_t index, int channel, const char* operation) {
    if (!m_dispatch_events) return;
    const auto& t = m_trace[index];
    m_dispatch_events << m_clk << ',' << channel << ',' << operation << ',' << index << ','
      << t.addr << ',' << t.type_id << ',' << t.rank_batch << '\n';
  }

  // Trace records are predecoded instruction sources, not unbounded hardware
  // request queues. Only the current dependency epoch is exposed; each channel
  // advances its own source cursor into a bounded memory-system ingress FIFO.
  bool prepare_epoch() {
    int last = m_last_pim;
    bool has_write = m_phase_has_write;
    size_t end = m_curr_trace_idx;
    for (auto& stream : m_channel_streams) stream.clear();
    std::fill(m_channel_positions.begin(), m_channel_positions.end(), 0);
    while (end < m_trace.size()) {
      const auto& t = m_trace[end];
      if (t.type_id == Barrier || t.type_id == Wait) break;
      const int pim = t.type_id == Request::Type::PIM_MAC_HB ||
          (t.type_id == Request::Type::Read && last == 2) ? 2 :
          t.type_id >= Request::Type::PIM_MAC_AB ? 1 : 0;
      const bool reads_to_half = pim == 2 && last == 0 && !has_write;
      if (m_wait_for_completion && last >= 0 && pim != last && !reads_to_half) {
        if (end != m_curr_trace_idx) break;
        if (!drained()) return false;
      }
      m_channel_streams.at(m_memory_system->get_channel_id(t.addr)).push_back(end);
      if (pim != last) has_write = false;
      if (t.type_id == Request::Type::Write) has_write = true;
      last = pim;
      ++end;
    }
    if (end == m_curr_trace_idx) throw std::runtime_error("Empty independent trace epoch");
    m_last_pim = last;
    m_phase_has_write = has_write;
    m_curr_trace_idx = end;
    m_epoch_active = true;
    return true;
  }

  void tick_independent() {
    while (!m_epoch_active && m_curr_trace_idx < m_trace.size()) {
      const auto& t = m_trace[m_curr_trace_idx];
      if (t.type_id != Barrier && t.type_id != Wait) {
        if (!prepare_epoch()) return;
        break;
      }
      if (!drained()) return;
      if (t.type_id == Wait) {
        if (m_wait_until < 0) {
          m_wait_until = m_clk + memory_to_frontend(t.addr);
          m_wait_cycles += t.addr;
        }
        if (m_clk < m_wait_until) return;
        m_wait_until = -1;
      } else {
        ++m_barriers;
        if (m_events) {
          m_events << t.addr << ',' << m_clk << ',' << m_completed << '\n';
          m_events.flush();
        }
      }
      ++m_curr_trace_idx;
    }
    if (!m_epoch_active) return;
    bool remaining = false;
    for (size_t channel = 0; channel < m_channel_streams.size(); ++channel) {
      auto& position = m_channel_positions[channel];
      const auto& stream = m_channel_streams[channel];
      if (position == stream.size()) continue;
      const size_t index = stream[position];
      const auto& t = m_trace[index];
      if (t.type_id == Request::Type::PIM_Q_REMOTE && !m_trace_completed.at(t.peer_source)) {
        if (!m_peer_wait_logged[index]) {
          record_dispatch(index, static_cast<int>(channel), "PEER_WAIT");
          m_peer_wait_logged[index] = true;
        }
        remaining = true;
        continue;
      }
      Request req(t.addr, t.type_id);
      req.ingress_id = static_cast<int>(index);
      req.pim_buffer = t.pim_buffer;
      req.pim_stage_bytes = t.stage_bytes;
      req.pim_stage_rows = t.stage_rows;
      req.pim_rank_batch = t.rank_batch;
      req.size_bytes = m_memory_system->get_tx_bytes();
      req.callback = [this, index, channel](Request& completed) {
        ++m_completed;
        m_trace_completed[index] = true;
        record_dispatch(index, static_cast<int>(channel), "COMPLETE");
        if (completed.type_id == Request::Type::Write)
          m_write_ready_at = std::max(m_write_ready_at,
              m_clk + memory_to_frontend(m_write_tail_cycles) + 1);
      };
      if (m_memory_system->send(req)) {
        ++position;
        ++m_trace_count;
        record_dispatch(index, static_cast<int>(channel), "ACCEPT");
      }
      if (position != stream.size()) remaining = true;
    }
    m_epoch_active = remaining;
  }

  bool drained() const {
    return m_completed == m_trace_count && m_clk >= m_write_ready_at;
  }

  Clk_t memory_to_frontend(Addr_t cycles) const {
    int ratio = m_memory_system->get_clock_ratio();
    if (cycles > (std::numeric_limits<Clk_t>::max() - ratio) / m_clock_ratio) {
      throw std::runtime_error("LoadStoreTrace: WAIT duration overflow");
    }
    return (cycles * m_clock_ratio + ratio - 1) / ratio;
  }

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_wait_for_completion, bool, "wait_for_completion").default_val(false);
    RAMULATOR_PARSE_PARAM(m_write_tail_cycles, int, "write_tail_cycles").default_val(0);
    RAMULATOR_PARSE_PARAM(m_max_frontend_ticks, uint64_t, "max_frontend_ticks").default_val(0);
    RAMULATOR_PARSE_PARAM(m_events_path, std::string, "events_path").default_val("");
    RAMULATOR_PARSE_PARAM(m_channel_independent_dispatch, bool, "channel_independent_dispatch").default_val(false);
    RAMULATOR_PARSE_PARAM(m_dispatch_events_path, std::string, "dispatch_events_path").default_val("");
    if (m_write_tail_cycles < 0) {
      throw std::runtime_error("LoadStoreTrace: write_tail_cycles must be nonnegative");
    }
    if (m_channel_independent_dispatch && !m_wait_for_completion)
      throw std::runtime_error("Independent channel traces require wait_for_completion=true");
    m_stats.add("requests_accepted", m_trace_count);
    m_stats.add("requests_completed", m_completed);
    m_stats.add("barriers_completed", m_barriers);
    m_stats.add("explicit_wait_cycles", m_wait_cycles);
    init_trace();
    if (!m_events_path.empty()) {
      m_events.open(m_events_path);
      if (!m_events) throw std::runtime_error("LoadStoreTrace: cannot open events_path");
      m_events << "marker,frontend_tick,requests_completed\n";
    }
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    if (!m_channel_independent_dispatch) return;
    if (!memory_system->has_independent_channel_ingress())
      throw std::runtime_error("Independent trace streams require bounded independent channel ingress");
    if (m_trace.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("Trace has too many records for ingress identifiers");
    m_channel_streams.resize(memory_system->get_num_channels());
    m_channel_positions.resize(m_channel_streams.size(), 0);
    m_trace_completed.resize(m_trace.size(), false);
    m_peer_wait_logged.resize(m_trace.size(), false);
    std::map<std::pair<Addr_t, bool>, size_t> exports;
    for (size_t i = 0; i < m_trace.size(); ++i) {
      auto& t = m_trace[i];
      if (t.type_id == Request::Type::PIM_Q_EXPORT) exports[{t.addr, t.rank_batch}] = i;
      if (t.type_id == Request::Type::PIM_Q_REMOTE) {
        const auto found = exports.find({t.addr ^ Addr_t{32}, t.rank_batch});
        if (found == exports.end()) throw std::runtime_error("Q_REMOTE has no preceding peer Q_EXPORT");
        t.peer_source = found->second;
      }
    }
    if (!m_dispatch_events_path.empty()) {
      m_dispatch_events.open(m_dispatch_events_path);
      if (!m_dispatch_events) throw std::runtime_error("Cannot open dispatch_events_path");
      m_dispatch_events << "frontend_tick,channel,operation,trace_index,address,type_id,rank_batch\n";
    }
  }

  void tick() override {
    m_clk++;
    if (m_max_frontend_ticks && static_cast<uint64_t>(m_clk) > m_max_frontend_ticks) {
      throw std::runtime_error("LoadStoreTrace: simulation tick limit exceeded");
    }
    if (m_channel_independent_dispatch) {
      tick_independent();
      return;
    }
    // Control records do not consume an artificial frontend issue slot.
    while (m_curr_trace_idx < m_trace.size()) {
      const auto& t = m_trace[m_curr_trace_idx];
      if (t.type_id == Barrier || t.type_id == Wait) {
        if (!drained()) return;
        if (t.type_id == Wait) {
          if (m_wait_until < 0) {
            m_wait_until = m_clk + memory_to_frontend(t.addr);
            m_wait_cycles += t.addr;
          }
          if (m_clk < m_wait_until) return;
          m_wait_until = -1;
        } else {
          m_barriers++;
          if (m_events) {
            m_events << t.addr << ',' << m_clk << ',' << m_completed << '\n';
            m_events.flush();
          }
        }
        m_curr_trace_idx++;
        continue;
      }
      // Phase 2 allows HB compute and LD to overlap. The controller checks
      // the actual rank/half before admitting either request. ST and other PIM
      // operations retain completion fences; BARRIER always drains globally.
      int pim = t.type_id == Request::Type::PIM_MAC_HB ||
                        (t.type_id == Request::Type::Read && m_last_pim == 2)
                    ? 2 : t.type_id >= Request::Type::PIM_MAC_AB ? 1 : 0;
      const bool reads_to_half = pim == 2 && m_last_pim == 0 && !m_phase_has_write;
      if (m_wait_for_completion && m_last_pim >= 0 && pim != m_last_pim &&
          !reads_to_half && !drained()) return;
      if (t.type_id == Request::Type::PIM_Q_REMOTE) {
        const Addr_t source = t.addr ^ Addr_t{32}; // paired adjacent channels, 32B interleave
        if (!m_q_exports_issued.count(source)) throw std::runtime_error("Q_REMOTE has no preceding peer Q_EXPORT");
        if (!m_q_exports_completed.count(source)) return;
      }
      Request req(t.addr, t.type_id);
      req.pim_buffer = t.pim_buffer;
      req.ingress_id = static_cast<int>(m_curr_trace_idx);
      req.pim_stage_bytes = t.stage_bytes; req.pim_stage_rows = t.stage_rows;
      req.pim_rank_batch = t.rank_batch;
      req.size_bytes = m_memory_system->get_tx_bytes();
      if (m_wait_for_completion) {
        req.callback = [this](Request& completed) {
          m_completed++;
          if (completed.type_id == Request::Type::PIM_Q_EXPORT) m_q_exports_completed.insert(completed.addr);
          if (completed.type_id == Request::Type::Write) {
            m_write_ready_at = m_clk + memory_to_frontend(m_write_tail_cycles) + 1;
          }
        };
      }
      if (m_memory_system->send(req)) {
        m_curr_trace_idx++;
        m_trace_count++;
        if (t.type_id == Request::Type::PIM_Q_EXPORT) {
          m_q_exports_issued.insert(t.addr);
          m_q_exports_completed.erase(t.addr);
        }
        if (pim != m_last_pim) m_phase_has_write = false;
        if (t.type_id == Request::Type::Write) m_phase_has_write = true;
        m_last_pim = pim;
      }
      return;
    }
  }

 private:
  void init_trace() {
    std::ifstream input(m_trace_path);
    if (!input) throw std::runtime_error("LoadStoreTrace: cannot open trace " + m_trace_path);
    std::string line;
    int line_num = 0;
    while (std::getline(input, line)) {
      line_num++;
      std::istringstream fields(line);
      std::string op, value, buffer, extra, overflow;
      if (!(fields >> op >> value)) {
        throw std::runtime_error(fmt::format("Trace line {}: expected OP ADDRESS [BUFFER]", line_num));
      }
      fields >> buffer;
      fields >> extra;
      const bool rank_batch = op.starts_with("PIM_RANK_");
      if (rank_batch) op = "PIM_" + op.substr(9);
      if (fields >> overflow || (!extra.empty() && op != "PIM_STAGE_BEGIN")) throw std::runtime_error("LoadStoreTrace: too many fields");
      int type = op == "LD" ? Request::Type::Read :
          op == "ST" ? Request::Type::Write :
          op == "PIM_MAC_AB" ? Request::Type::PIM_MAC_AB :
          op == "PIM_MAC_HB" ? Request::Type::PIM_MAC_HB :
          op == "PIM_WR_GB" ? Request::Type::PIM_WR_GB :
          op == "PIM_RD_RES" ? Request::Type::PIM_RD_RES :
          op == "PIM_LOCAL_GB" ? Request::Type::PIM_LOCAL_GB :
          op == "PIM_WR_BG" ? Request::Type::PIM_WR_BG :
          op == "PIM_WR_Q" ? Request::Type::PIM_WR_Q :
          op == "PIM_MAC_K" ? Request::Type::PIM_MAC_K :
          op == "PIM_MAC_QK" ? Request::Type::PIM_MAC_QK :
          op == "PIM_RD_BG" ? Request::Type::PIM_RD_BG :
          op == "PIM_LOCAL_WR" ? Request::Type::PIM_LOCAL_WR :
          op == "PIM_RD_Q" ? Request::Type::PIM_RD_Q :
          op == "PIM_SPILL" ? Request::Type::PIM_SPILL :
          op == "PIM_ROPE_COEF" ? Request::Type::PIM_ROPE_COEF :
          op == "PIM_ROPE_LOAD" ? Request::Type::PIM_ROPE_LOAD :
          op == "PIM_ROPE_EXEC" ? Request::Type::PIM_ROPE_EXEC :
          op == "PIM_ROPE_STORE" ? Request::Type::PIM_ROPE_STORE :
          op == "PIM_Q_EXPORT" ? Request::Type::PIM_Q_EXPORT :
          op == "PIM_Q_REMOTE" ? Request::Type::PIM_Q_REMOTE :
          op == "PIM_Z_STORE" ? Request::Type::PIM_Z_STORE :
          op == "PIM_Z_LOAD" ? Request::Type::PIM_Z_LOAD :
          op == "PIM_GEMV_RELEASE" ? Request::Type::PIM_GEMV_RELEASE :
          op == "PIM_STAGE_BEGIN" ? Request::Type::PIM_STAGE_BEGIN :
          op == "PIM_STAGE_SEAL" ? Request::Type::PIM_STAGE_SEAL :
          op == "PIM_STAGE_LOAD" ? Request::Type::PIM_STAGE_LOAD :
          op == "PIM_STAGE_FRAGMENT" ? Request::Type::PIM_STAGE_FRAGMENT :
          op == "PIM_FFN_PASS" ? Request::Type::PIM_FFN_PASS :
          op == "PIM_FFN_RDRES" ? Request::Type::PIM_FFN_RDRES :
          op == "BARRIER" ? Barrier : op == "WAIT" ? Wait : -1;
      if (type == -1) throw std::runtime_error("LoadStoreTrace: unknown operation " + op);
      if (rank_batch && (type < Request::Type::PIM_SPILL || type > Request::Type::PIM_Q_REMOTE))
        throw std::runtime_error("Unsupported rank descriptor operation " + op);
      int pim_buffer = -1, stage_bytes=0, stage_rows=0;
      if (type == Request::Type::PIM_STAGE_BEGIN) {
        size_t used=0;
        stage_bytes=std::stoi(buffer,&used);
        if(used!=buffer.size()) throw std::runtime_error("Invalid staging byte count");
        stage_rows=std::stoi(extra,&used);
        if(used!=extra.size() || stage_bytes<=0 || stage_bytes>131072 || stage_bytes%32 || (stage_rows!=8 && stage_rows!=16) || stage_bytes%(stage_rows*32)) throw std::runtime_error("Invalid staging size/rows");
      } else if (!buffer.empty()) {
        if ((type != Request::Type::PIM_WR_GB && type != Request::Type::PIM_MAC_AB &&
             type != Request::Type::PIM_LOCAL_GB && type != Request::Type::PIM_WR_BG &&
             type != Request::Type::PIM_WR_Q && type != Request::Type::PIM_MAC_K &&
             type != Request::Type::PIM_MAC_QK && type != Request::Type::PIM_RD_Q &&
             type != Request::Type::PIM_ROPE_LOAD && type != Request::Type::PIM_ROPE_EXEC && type != Request::Type::PIM_Q_REMOTE && type != Request::Type::PIM_Z_LOAD && type != Request::Type::PIM_GEMV_RELEASE && type != Request::Type::PIM_STAGE_LOAD && type != Request::Type::PIM_STAGE_FRAGMENT) ||
            (buffer != "0" && buffer != "1" && buffer != "2" && buffer != "3")) {
          throw std::runtime_error("LoadStoreTrace: BUFFER must be 0..3; controller validates physical SRF views");
        }
        pim_buffer = std::stoi(buffer);
      }
      if ((type == Request::Type::PIM_STAGE_FRAGMENT || type == Request::Type::PIM_STAGE_LOAD || type == Request::Type::PIM_GEMV_RELEASE || type == Request::Type::PIM_Z_LOAD || type == Request::Type::PIM_ROPE_LOAD || type == Request::Type::PIM_ROPE_EXEC || type == Request::Type::PIM_Q_REMOTE || type == Request::Type::PIM_LOCAL_GB || type == Request::Type::PIM_RD_Q || (type >= Request::Type::PIM_WR_BG &&
          type <= Request::Type::PIM_MAC_QK)) && pim_buffer < 0)
        throw std::runtime_error("PIM_LOCAL_GB requires BUFFER=0/1");
      size_t consumed = 0;
      int base = value.starts_with("0x") || value.starts_with("0X") ? 16 : 10;
      Addr_t addr = std::stoll(value, &consumed, base);
      if (consumed != value.size() || addr < 0) {
        throw std::runtime_error("LoadStoreTrace: invalid nonnegative address/duration");
      }
      if (type != Request::Type::Read && type != Request::Type::Write && !m_wait_for_completion) {
        throw std::runtime_error("PIM/control traces require wait_for_completion=true");
      }
      if (type == Request::Type::Write && m_wait_for_completion && m_write_tail_cycles == 0) {
        throw std::runtime_error("LoadStoreTrace: writes require explicit write_tail_cycles");
      }
      m_trace.push_back({type, addr, pim_buffer, stage_bytes, stage_rows, rank_batch});
    }
    if (m_trace.empty()) throw std::runtime_error("LoadStoreTrace: empty trace");
  }

  bool is_finished() override {
    return !m_epoch_active && m_curr_trace_idx >= m_trace.size() && (!m_wait_for_completion || drained());
  }
};

}  // namespace Ramulator

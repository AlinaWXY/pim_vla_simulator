#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class LoadStoreTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, LoadStoreTrace, "LoadStoreTrace")

 private:
  static constexpr int Barrier = -2;
  static constexpr int Wait = -3;
  struct Trace { int type_id; Addr_t addr; int pim_buffer; };
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
    if (m_write_tail_cycles < 0) {
      throw std::runtime_error("LoadStoreTrace: write_tail_cycles must be nonnegative");
    }
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

  void tick() override {
    m_clk++;
    if (m_max_frontend_ticks && static_cast<uint64_t>(m_clk) > m_max_frontend_ticks) {
      throw std::runtime_error("LoadStoreTrace: simulation tick limit exceeded");
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
      std::string op, value, buffer, extra;
      if (!(fields >> op >> value)) {
        throw std::runtime_error(fmt::format("Trace line {}: expected OP ADDRESS [BUFFER]", line_num));
      }
      fields >> buffer;
      if (fields >> extra) throw std::runtime_error("LoadStoreTrace: too many fields");
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
          op == "BARRIER" ? Barrier : op == "WAIT" ? Wait : -1;
      if (type == -1) throw std::runtime_error("LoadStoreTrace: unknown operation " + op);
      int pim_buffer = -1;
      if (!buffer.empty()) {
        if ((type != Request::Type::PIM_WR_GB && type != Request::Type::PIM_MAC_AB &&
             type != Request::Type::PIM_LOCAL_GB && type != Request::Type::PIM_WR_BG &&
             type != Request::Type::PIM_WR_Q && type != Request::Type::PIM_MAC_K &&
             type != Request::Type::PIM_MAC_QK && type != Request::Type::PIM_RD_Q &&
             type != Request::Type::PIM_ROPE_LOAD && type != Request::Type::PIM_ROPE_EXEC && type != Request::Type::PIM_Q_REMOTE && type != Request::Type::PIM_Z_LOAD && type != Request::Type::PIM_GEMV_RELEASE) ||
            (buffer != "0" && buffer != "1")) {
          throw std::runtime_error("LoadStoreTrace: BUFFER must be 0/1 on WRGB or MACAB");
        }
        pim_buffer = buffer == "0" ? 0 : 1;
      }
      if ((type == Request::Type::PIM_GEMV_RELEASE || type == Request::Type::PIM_Z_LOAD || type == Request::Type::PIM_ROPE_LOAD || type == Request::Type::PIM_ROPE_EXEC || type == Request::Type::PIM_Q_REMOTE || type == Request::Type::PIM_LOCAL_GB || type == Request::Type::PIM_RD_Q || (type >= Request::Type::PIM_WR_BG &&
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
      m_trace.push_back({type, addr, pim_buffer});
    }
    if (m_trace.empty()) throw std::runtime_error("LoadStoreTrace: empty trace");
  }

  bool is_finished() override {
    return m_curr_trace_idx >= m_trace.size() && (!m_wait_for_completion || drained());
  }
};

}  // namespace Ramulator

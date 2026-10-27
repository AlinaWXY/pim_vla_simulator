#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unordered_map>
#include "ramulator/base/base.h"
#include "ramulator/controller/impl/lpddr_controller_base.h"
#include "ramulator/controller/refresh/i_refresh_manager.h"
#include "ramulator/controller/rowpolicy/i_row_policy.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

class LPDDR5Controller : public LPDDRControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, LPDDR5Controller, LPDDRControllerBase, "LPDDR5")

 public:
  void init() override {
    LPDDRControllerBase::init();

    const auto& spec = *m_device.m_spec;
    if (spec.get_level_size("Rank") > 1 && m_wck_sync_mode != WCKSyncMode::AlwaysOn) {
      throw std::runtime_error(
          "LPDDR5 multi-rank calibration requires wck_sync_mode=always_on "
          "with all ranks pre-synchronized; per-rank WCK startup is not modeled");
    }
    m_cmd_act1 = spec.get_command_id("ACT1");
    m_cmd_act2 = spec.get_command_id("ACT2");
    m_cmd_cas_rd = spec.get_command_id("CAS_RD");
    m_cmd_cas_wr = spec.get_command_id("CAS_WR");
    m_cmd_rd = spec.get_command_id("RD");
    m_cmd_wr = spec.get_command_id("WR");
    m_cmd_rda = spec.get_command_id("RDA");
    m_cmd_wra = spec.get_command_id("WRA");

    m_nAAD = spec.get_timing_value("nAAD");
    m_read_latency = spec.get_timing_value("nCL");
    m_write_latency = spec.get_timing_value("nCWL");
    m_burst_cycles = spec.get_timing_value("nBL_max");
    m_nWCKPST = spec.get_timing_value("nWCKPST");
    m_cas_deadline_guard = 1;
    m_wck_expiry_inclusive = true;
  }
};

// Legacy PIM traces remain ordered. Annotated all-bank requests may overlap
// writes to one input SRAM with MACs reading the other SRAM.
class LPDDR5PIMController final : public LPDDR5Controller {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, LPDDR5PIMController, LPDDR5Controller, "LPDDR5PIM")

  ReqBuffer m_pim_buffer;
  std::deque<Request> m_pim_pending;
  Clk_t m_normal_ready_at = 0;
  int m_mac_interval = 0;
  int m_pim_horizon = 16;
  struct InputBuffer {
    Clk_t ready_at = 0;
    Clk_t in_use_until = 0;
    bool written = false;
    Clk_t local_in_use_until = 0;
  };
  std::vector<std::array<InputBuffer, 2>> m_inputs;
  std::vector<std::array<InputBuffer, 2>> m_bank_inputs;
  std::vector<Clk_t> m_bank_compute_ready;
  int m_bgs = 0, m_banks_per_bg = 0, m_mack = -1, m_macqk = -1, m_localwr = -1;
  int m_k_cycles = 2, m_qk_cycles = 20, m_bank_write_cycles = 4, m_local_recovery = 0;
  int m_bgk_input_cycles = 2;
  // One physical 32B controller staging entry. It remains reserved through
  // the return burst and the charged WRGB multicast, including its tail.
  bool m_q_staging_busy = false;
  Clk_t m_q_staging_free_at = 0;
  size_t s_q_dram_reads = 0, s_q_relays = 0;
  std::string m_bgk_events_path;
  std::ofstream m_bgk_events;
  size_t s_bank_local_writes = 0, s_bgk_requests = 0;

  int m_local_cycles = 2;
  std::string m_local_trace_path;
  std::ofstream m_local_trace;
  std::vector<Clk_t> m_local_ports;
  size_t s_local_transfers = 0;
  int m_pim_wr_latency = 0;
  int m_pim_rd_latency = 0;
  int m_write_recovery = 0;
  int m_rank_level = -1;
  int m_bg_level = -1;
  int m_macab = -1;
  int m_machb = -1;
  int m_groups_per_half = 0;
  int m_wrgb = -1;
  int m_rdres = -1;
  bool m_input_broadcast = false;
  std::string m_mac_scope = "rank";
  std::string m_data_path = "lpddr";
  size_t s_pim_requests = 0;
  size_t s_pim_completed = 0;
  size_t s_pim_latency = 0;

#include "rank_rope_extension.inc"
#include "ffn_dram_extension.inc"

 public:
  void init() override {
    // The registered LPDDR5Controller constructor has already initialized the
    // device and children. Do not call it again from the derived constructor.
    RAMULATOR_PARSE_PARAM(m_bgk_events_path, std::string, "bgk_events_path").default_val("");
    RAMULATOR_PARSE_PARAM(m_input_broadcast, bool, "pim_input_broadcast").default_val(false);
    RAMULATOR_PARSE_PARAM(m_mac_scope, std::string, "pim_mac_scope").default_val("rank");
    RAMULATOR_PARSE_PARAM(m_data_path, std::string, "pim_data_path").default_val("lpddr");
    RAMULATOR_PARSE_PARAM(m_local_cycles, int, "pim_local_transfer_cycles").default_val(2);
    RAMULATOR_PARSE_PARAM(m_pim_horizon, int, "pim_horizon").default_val(16);
    if (m_pim_horizon < 2 || m_pim_horizon > 64 || m_pim_horizon % 2)
      throw std::runtime_error("pim_horizon must be even and in2..64");
    RAMULATOR_PARSE_PARAM(m_local_trace_path, std::string, "local_input_trace_path").default_val("");
    if (m_local_cycles < 1) throw std::runtime_error("Local transfer cycles must be positive");
    if (m_data_path != "lpddr" && m_data_path != "attacc") {
      throw std::runtime_error("pim_data_path must be lpddr or attacc");
    }
    if (m_mac_scope != "rank" && m_mac_scope != "channel") {
      throw std::runtime_error("pim_mac_scope must be rank or channel");
    }
    const auto& spec = *m_device.m_spec;
    if (spec.standard_name != "LPDDR5PIM" && spec.standard_name != "LPDDR5XPIM") {
      throw std::runtime_error("LPDDR5PIM controller requires LPDDR5PIM DRAM");
    }
    m_mac_interval = spec.get_timing_value("nCCDAB");
    m_inputs.resize(spec.get_level_size("Rank"));
    m_local_ports.resize(m_inputs.size(), 0);
    m_pim_wr_latency = spec.get_timing_value("nPIMWrLatency");
    m_pim_rd_latency = spec.get_timing_value("nPIMRdLatency");
    if ((m_data_path == "attacc") != (m_pim_wr_latency == 0 && m_pim_rd_latency == 0)) {
      throw std::runtime_error("PIM controller and DRAM data-path modes must match");
    }
    m_write_recovery = spec.get_timing_value("nCWL") + spec.get_timing_value("nBL_max") + spec.get_timing_value("nWR");
    m_rank_level = spec.get_level_id("Rank");
    m_bg_level = spec.get_level_id("BankGroup");
    m_macab = spec.get_command_id("MACAB");
    m_machb = spec.get_command_id("MACHB");
    m_groups_per_half = spec.get_level_size("BankGroup") / 2;
    m_wrgb = spec.get_command_id("WRGB");
    m_rdres = spec.get_command_id("RDRES");
    m_bgs = spec.get_level_size("BankGroup");
    m_banks_per_bg = spec.get_level_size("Bank");
    m_bank_inputs.resize(m_inputs.size() * m_bgs * m_banks_per_bg);
    if (m_bank_inputs.size() > 64) throw std::runtime_error("BG-K bank mask supports <=64 banks/channel");
    m_bank_compute_ready.resize(m_bank_inputs.size(),0);
    m_mack = spec.get_command_id("MACK");
    m_macqk = spec.get_command_id("MACQK");
    m_localwr = spec.get_command_id("LOCALWR");
    m_k_cycles = spec.get_timing_value("nCCDK");
    m_qk_cycles = spec.get_timing_value("nCCDQK");
    m_bank_write_cycles = spec.get_timing_value("nLOCALWR");
    m_local_recovery = spec.get_timing_value("nWR");
    m_bgk_input_cycles = spec.get_timing_value("nBL_min");
    // AttAcc's no-CWL movement abstraction still consumes a complete burst.
    // Legacy WRGB consumers previously became ready at issue, one burst early.
    if (m_data_path == "attacc") m_pim_wr_latency = m_bgk_input_cycles;
    RAMULATOR_PARSE_PARAM(m_rank_rope, bool, "pim_rank_rope").default_val(false);
    RAMULATOR_PARSE_PARAM(m_rope_ii, int, "rope_ii_cycles").default_val(1);
    RAMULATOR_PARSE_PARAM(m_rope_latency, int, "rope_latency_cycles").default_val(4);
    RAMULATOR_PARSE_PARAM(m_cast_cycles, int, "pim_cast_cycles").default_val(1);
    RAMULATOR_PARSE_PARAM(m_rank_reduce_cycles, int, "rank_reduce_cycles").default_val(4);
    init_rank_rope();
    RAMULATOR_PARSE_PARAM(m_ffn_dram, bool, "pim_ffn_dram").default_val(false);
    RAMULATOR_PARSE_PARAM(m_ffn_z_preinstalled, bool, "ffn_z_preinstalled").default_val(false);
    RAMULATOR_PARSE_PARAM(m_ffn_events_path, std::string, "ffn_dram_events_path").default_val("");
    init_ffn_dram();
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    LPDDR5Controller::setup(frontend, memory_system);
    m_stats.add("num_pim_requests", s_pim_requests);
    m_stats.add("num_pim_completed", s_pim_completed);
    m_stats.add("pim_request_latency_cycles", s_pim_latency);
    m_stats.add("local_input_transfers", s_local_transfers);
    m_stats.add("bgk_requests", s_bgk_requests);
    m_stats.add("bank_local_writes", s_bank_local_writes);
    m_stats.add("q_dram_reads", s_q_dram_reads);
    m_stats.add("q_internal_relays", s_q_relays);
    if (m_ffn_dram) {
      m_stats.add("ffn_z_stores", s_z_stores);
      m_stats.add("ffn_z_loads", s_z_loads);
      m_stats.add("ffn_gu_gather_bytes", s_z_gather_bytes);
      m_stats.add("ffn_z_return_bytes", s_z_return_bytes);
      m_stats.add("ffn_z_load_port_bytes", s_z_load_port_bytes);
      if (!m_ffn_events_path.empty()) {
        m_ffn_events.open(m_ffn_events_path);
        if (!m_ffn_events) throw std::runtime_error("Cannot open ffn_dram_events_path");
        m_ffn_events << "clock,ready_at,operation,Channel,Rank,BankGroup,Bank,Row,Column,buffer,bytes,service_cycles,generation,bank_mask,mac_columns\n";
      }
    }
    if (!m_bgk_events_path.empty()) {
      m_bgk_events.open(m_bgk_events_path);
      if (!m_bgk_events) throw std::runtime_error("Cannot open bgk_events_path");
      m_bgk_events << "clock,ready_at,operation,Channel,Rank,BankGroup,Bank,Row,Column,buffer,target_banks,bytes,service_cycles\n";
    }
    if (!m_local_trace_path.empty()) {
      m_local_trace.open(m_local_trace_path);
      if (!m_local_trace) throw std::runtime_error("Cannot open local_input_trace_path");
      m_local_trace << "clock,ready_at,Channel,Rank,pim_buffer,offset_bytes,bytes,port_cycles,buffer_in_use_until,port_ready_before\n";
      m_local_trace.flush();
    }
  }

  bool send(Request& req) override {
    if (req.pim_buffer < -1 || req.pim_buffer > 1 ||
        (req.pim_buffer >= 0 && req.type_id != Request::Type::PIM_WR_GB &&
         req.type_id != Request::Type::PIM_MAC_AB && req.type_id != Request::Type::PIM_LOCAL_GB &&
         req.type_id != Request::Type::PIM_WR_BG && req.type_id != Request::Type::PIM_WR_Q &&
         req.type_id != Request::Type::PIM_MAC_K && req.type_id != Request::Type::PIM_MAC_QK &&
         req.type_id != Request::Type::PIM_RD_Q && req.type_id != Request::Type::PIM_ROPE_LOAD &&
         req.type_id != Request::Type::PIM_ROPE_EXEC && req.type_id != Request::Type::PIM_Q_REMOTE &&
         req.type_id != Request::Type::PIM_Z_LOAD)) {
      throw std::runtime_error("PIM buffer annotation requires WRGB/MACAB and slot 0/1");
    }
    // Map before admission so overlapping-bank requests can apply backpressure
    // without blocking independent reads behind a global PIM completion fence.
    m_addr_mapper->apply(req);
    req.addr_vec[0] = m_channel_id;
    req.pim_origin_addr_vec = req.addr_vec;
    if ((m_rank_rope || m_ffn_dram) && req.type_id >= Request::Type::PIM_MAC_AB && req.addr_vec[m_rank_level] != 1)
      throw std::runtime_error("rank0 is reserved for GPU: PIM requests must address physical rank1");
    if (req.type_id < Request::Type::PIM_MAC_AB) {
      for (const auto& buffers : m_inputs) {
        for (const auto& buffer : buffers) {
          if (m_clk < buffer.in_use_until) return false;
        }
      }
      for (const auto& pim : m_pim_buffer) {
        if (!can_overlap(pim, req)) return false;
      }
      for (const auto& pim : m_pim_pending) {
        if (!can_overlap(pim, req)) return false;
      }
      auto callback = req.callback;
      req.callback = [this, callback](Request& done) {
        if (done.type_id == Request::Type::Write) {
          m_normal_ready_at = std::max(m_normal_ready_at, m_clk + m_write_recovery);
        }
        if (callback) {
          callback(done);
        }
      };
      bool accepted = LPDDR5Controller::send(req);
      req.callback = callback;
      return accepted;
    }
    if (m_pim_buffer.size() >= m_pim_buffer.max_size) {
      return false;
    }
    if (is_ffn_z(req)) return send_ffn_z(req);
    if (is_rank_rope(req)) return send_rank_rope(req);
    if (m_rank_rope && (req.type_id == Request::Type::PIM_MAC_AB || req.type_id == Request::Type::PIM_WR_GB))
      req.pim_bank_mask = ((uint64_t{1} << (m_bgs*m_banks_per_bg)) - 1) << (m_bgs*m_banks_per_bg);
    if (is_bgk(req)) {
      if (m_data_path != "attacc") throw std::runtime_error("BG-K extension requires AttAcc movement path");
      if (req.size_bytes != 32) throw std::runtime_error("BG-K extension requires 32B requests");
      if ((is_bgk_input(req) || is_bgk_mac(req)) && req.pim_buffer < 0)
        throw std::runtime_error("BG-K input/MAC requires explicit double-buffer slot");
      const int rank = req.addr_vec[m_rank_level];
      const int bg = req.addr_vec[m_bg_level];
      const int bank = req.addr_vec[m_bank_level];
      if (req.type_id == Request::Type::PIM_RD_Q && (rank != 0 || bg != 0))
        throw std::runtime_error("PIM_RD_Q source must be rank0/BG0, with an explicit bank index");
      for (size_t b = 0; b < m_bank_inputs.size(); ++b) {
        const int br = b / (m_bgs * m_banks_per_bg);
        const int bgid = (b / m_banks_per_bg) % m_bgs;
        const int bb = b % m_banks_per_bg;
        const bool selected = is_bgk_mac(req) ||
            ((req.type_id == Request::Type::PIM_WR_Q || req.type_id == Request::Type::PIM_RD_Q) ? bb == bank :
             req.type_id == Request::Type::PIM_LOCAL_WR ? br == rank && bgid == bg && bb == bank :
             br == rank && bgid == bg);
        if (selected && (!m_rank_rope || br == 1)) req.pim_bank_mask |= uint64_t{1} << b;
      }
      if (is_bgk_mac(req)) {
        if (rank != (m_rank_rope ? 1 : 0) || bg != 0 || bank != 0) throw std::runtime_error("BG-K MAC must use canonical active-rank/BG0/bank0 address");
        req.addr_vec[m_bg_level] = req.addr_vec[m_bank_level] = -1;
        req.addr_vec[m_rank_level] = m_rank_rope ? 1 : -1;
      } else if (req.type_id == Request::Type::PIM_WR_BG || req.type_id == Request::Type::PIM_RD_BG) {
        req.addr_vec[m_bank_level] = -1;
      }
      if (m_rank_rope && req.type_id == Request::Type::PIM_RD_BG) {
        req.pim_bank_mask = ((uint64_t{1} << (m_bgs*m_banks_per_bg)) - 1) << (m_bgs*m_banks_per_bg);
        req.addr_vec[m_bg_level] = -1;
      }
      req.pim_rank_mask = m_rank_rope ? 2 : is_bgk_mac(req) || req.type_id == Request::Type::PIM_WR_Q || req.type_id == Request::Type::PIM_RD_Q
          ? (1 << m_inputs.size()) - 1 : 1 << rank;
      req.final_command = spec_command(req.type_id);
      req.arrive = m_clk;
      m_pim_buffer.enqueue(req);
      ++s_pim_requests; ++s_bgk_requests;
      return true;
    }
    if (is_local(req)) {
      if (req.pim_buffer < 0 || req.size_bytes != 32)
        throw std::runtime_error("Local fill requires a buffer and 32 bytes");
      const int rank = req.addr_vec[m_rank_level];
      if (rank < 0 || rank >= static_cast<int>(m_inputs.size()))
        throw std::runtime_error("Local fill requires one explicit rank");
      req.pim_rank_mask = 1 << rank;
      req.final_command = -1;  // Local SRAM event; never sent to the DRAM device.
      req.arrive = m_clk;
      m_pim_buffer.enqueue(req);
      ++s_pim_requests;
      return true;
    }
    const bool half = req.type_id == Request::Type::PIM_MAC_HB;
    if (m_data_path == "attacc" && half) {
      throw std::runtime_error("AttAcc data path supports all-bank traces only");
    }
    if ((m_input_broadcast || m_mac_scope == "channel") && half) {
      throw std::runtime_error("Input broadcast/channel MAC supports all-bank traces only");
    }
    if ((m_input_broadcast || m_mac_scope == "channel") &&
        (req.type_id == Request::Type::PIM_WR_GB || req.type_id == Request::Type::PIM_MAC_AB) &&
        req.pim_buffer < 0) {
      throw std::runtime_error("Broadcast mode requires explicit BUFFER=0/1 on WRGB and MACAB");
    }
    if (m_input_broadcast && req.type_id == Request::Type::PIM_WR_GB && req.addr_vec[m_rank_level] != 0) {
      throw std::runtime_error("Broadcast WRGB requires a rank-0 address");
    }
    const bool channel_mac = m_mac_scope == "channel" && req.type_id == Request::Type::PIM_MAC_AB;
    if (channel_mac && req.addr_vec[m_rank_level] != 0) {
      throw std::runtime_error("Channel MACAB requires a rank-0 address; one request covers all ranks");
    }
    // Keep buffer/result/all-bank phases separated from the concurrent phase,
    // including when requests arrive through frontends other than LoadStoreTrace.
    for (const auto& pim : m_pim_buffer) {
      if (half != (pim.type_id == Request::Type::PIM_MAC_HB)) return false;
    }
    for (const auto& pim : m_pim_pending) {
      if (half != (pim.type_id == Request::Type::PIM_MAC_HB)) return false;
    }
    if (half) {
      if (m_groups_per_half == 0 || req.addr_vec[m_bg_level] < 0) {
        throw std::runtime_error("PIM_MAC_HB requires an explicit bank group");
      }
      if (m_clk < m_normal_ready_at) return false;
      for (const auto* buffer : {&m_read_buffer, &m_write_buffer, &m_active_buffer, &m_activating_buffer}) {
        for (const auto& normal : buffer->buffer) {
          if (!can_overlap(req, normal)) return false;
        }
      }
      for (const auto& normal : m_pending) {
        if (!can_overlap(req, normal)) return false;
      }
      req.addr_vec[m_bg_level] = req.addr_vec[m_bg_level] / m_groups_per_half * m_groups_per_half;
    } else {
      req.addr_vec[m_bg_level] = -1;
    }
    req.addr_vec[m_bank_level] = -1;
    // Trace addresses are explicit. Channel MAC uses canonical rank 0, then
    // expands the DRAM command (including ACT/PRE prerequisites) to every rank.
    if (req.addr_vec[m_rank_level] < 0) {
      throw std::runtime_error("PIM requires an explicit rank");
    }
    req.pim_rank_mask = channel_mac || (m_input_broadcast && req.type_id == Request::Type::PIM_WR_GB)
        ? (1 << m_inputs.size()) - 1 : 1 << req.addr_vec[m_rank_level];
    if (channel_mac) req.addr_vec[m_rank_level] = -1;
    req.final_command = spec_command(req.type_id);
    req.arrive = m_clk;
    m_pim_buffer.enqueue(req);
    s_pim_requests++;
    return true;
  }

  void tick() override {
    if ((m_pim_buffer.size() && m_pim_buffer.begin()->type_id == Request::Type::PIM_MAC_HB) ||
        (!m_pim_pending.empty() && m_pim_pending.front().type_id == Request::Type::PIM_MAC_HB)) {
      LPDDR5Controller::tick();
      complete_pim();
      return;
    }
    if (m_read_buffer.size() || m_write_buffer.size() || m_active_buffer.size() || m_activating_buffer.size() ||
        !m_pending.empty() || m_cas_issued || m_clk < m_normal_ready_at ||
        (!m_pim_buffer.size() && m_pim_pending.empty())) {
      LPDDR5Controller::tick();
      complete_pim();
      issue_local_ready();
      return;
    }
    tick_prologue();
    complete_pim();
    issue_local_ready();
    m_refresh->tick();
    m_rowpolicy->pre_schedule();
    for (auto* p : m_plugins) {
      p->pre_schedule();
    }

    // Weights are latched when MAC issues; refresh obeys the normal array
    // recovery rules. SRAM ownership is tracked independently below.
    Candidate cand = pick_priority_if();
    if (cand.valid) {
      issue_standard_candidate(cand);
    } else if (m_priority_buffer.size() == 0 && m_pim_buffer.size()) {
      issue_pim_ready();
    }
    m_rowpolicy->post_schedule();
    for (auto* p : m_plugins) {
      p->post_schedule();
    }
  }

  void reset_stats() override {
    LPDDR5Controller::reset_stats();
    s_pim_requests = s_pim_completed = s_pim_latency = 0;
    s_local_transfers = 0;
    s_bank_local_writes = s_bgk_requests = 0;
    s_q_dram_reads = s_q_relays = 0;
    s_z_stores = s_z_loads = s_z_gather_bytes = s_z_return_bytes = s_z_load_port_bytes = 0;
  }

 private:
  int spec_command(int type) const {
    const auto& commands = m_device.m_spec->supported_requests;
    if (type < 0 || type >= static_cast<int>(commands.size())) {
      throw std::runtime_error("Unsupported PIM request type");
    }
    return commands[type];
  }

  bool can_overlap(const Request& pim, const Request& normal) const {
    if (pim.type_id != Request::Type::PIM_MAC_HB || normal.type_id != Request::Type::Read) return false;
    return pim.addr_vec[m_rank_level] != normal.addr_vec[m_rank_level] ||
           pim.addr_vec[m_bg_level] / m_groups_per_half != normal.addr_vec[m_bg_level] / m_groups_per_half;
  }

  void complete_pim() {
    while (!m_pim_pending.empty() && m_pim_pending.front().depart <= m_clk) {
      auto& done = m_pim_pending.front();
      s_pim_completed++;
      s_pim_latency += done.depart - done.arrive;
      if (done.callback) done.callback(done);
      m_pim_pending.pop_front();
    }
  }

  bool try_issue_pim() override {
    if (!m_pim_buffer.size() || m_pim_buffer.begin()->type_id != Request::Type::PIM_MAC_HB) return false;
    return issue_pim_front();
  }

  bool can_bypass(const Request& later, const Request& earlier, bool preparing_row) const {
    if (m_ffn_dram && (is_ffn_z(later) || is_ffn_z(earlier))) return z_bypass(later, earlier);
    if (m_rank_rope && (is_rank_rope(later) || is_rank_rope(earlier))) return rr_bypass(later, earlier);
    // RD_Q uses the ordinary bank scheduler before its multicast. Keep its
    // row state stable until the relay is consumed; no speculative later MAC
    // ACT/PRE may prepare a conflicting row while this composite is queued.
    if (earlier.type_id == Request::Type::PIM_RD_Q) return false;
    if (later.type_id == Request::Type::PIM_RD_Q) return false;
    if (is_bgk(later) || is_bgk(earlier)) {
      if (!is_bgk(later) || !is_bgk(earlier)) return false;
      if (is_bgk_input(later) && is_bgk_input(earlier)) return false; // one external movement path
      if (!(later.pim_bank_mask & earlier.pim_bank_mask)) return true;
      if (is_bgk_mac(later) && is_bgk_mac(earlier)) return false;
      if (later.type_id == Request::Type::PIM_LOCAL_WR || earlier.type_id == Request::Type::PIM_LOCAL_WR ||
          later.type_id == Request::Type::PIM_RD_BG || earlier.type_id == Request::Type::PIM_RD_BG) return false;
      if (preparing_row && is_bgk_input(earlier)) return true;
      return later.pim_buffer >= 0 && earlier.pim_buffer >= 0 && later.pim_buffer != earlier.pim_buffer;
    }
    if (is_local(later) || is_local(earlier)) {
      if (!(later.pim_rank_mask & earlier.pim_rank_mask)) return true;
      if (is_local(later) && is_local(earlier)) return false;
      if (earlier.type_id == Request::Type::PIM_RD_RES ||
          later.type_id == Request::Type::PIM_RD_RES) return false;
      if (preparing_row && is_local(earlier)) return true;
      return later.pim_buffer >= 0 && earlier.pim_buffer >= 0 &&
             later.pim_buffer != earlier.pim_buffer;
    }
    if (m_data_path == "attacc") {
      // Like AttAcc, the trace/barriers express result dependencies. Within
      // a window, a previous tile's result movement and the next tile's MAC
      // may pass each other. Transfers retain their requested order and share
      // the DRAM timing model's movement path.
      const bool later_result = later.type_id == Request::Type::PIM_RD_RES;
      const bool earlier_result = earlier.type_id == Request::Type::PIM_RD_RES;
      if (later_result || earlier_result) {
        return later_result ? earlier.type_id == Request::Type::PIM_MAC_AB
                            : later.type_id == Request::Type::PIM_MAC_AB;
      }
    }
    // Unannotated requests, result reads, and mode transitions remain fences.
    if (later.pim_buffer < 0 || earlier.pim_buffer < 0) return false;
    // Keep the requested DQ/rank write sequence. Overlap must not silently
    // gain extra bandwidth by batching writes to one rank.
    if (later.type_id == Request::Type::PIM_WR_GB &&
        earlier.type_id == Request::Type::PIM_WR_GB) return false;
    // A broadcast depends on matching slots in EVERY destination rank.
    if (!(later.pim_rank_mask & earlier.pim_rank_mask)) return true;
    // Preserve MAC/weight order and every dependency on the same SRAM slot.
    if (later.type_id == Request::Type::PIM_MAC_AB &&
        earlier.type_id == Request::Type::PIM_MAC_AB) return false;
    // ACT/PRE use the DRAM array, not input SRAM. They may pass even a
    // same-slot write, but never an earlier MAC targeting the same banks.
    if (preparing_row && earlier.type_id == Request::Type::PIM_WR_GB) return true;
    return later.pim_buffer != earlier.pim_buffer;
  }



  bool is_bgk(const Request& req) const { return req.type_id >= Request::Type::PIM_WR_BG && req.type_id <= Request::Type::PIM_RD_Q; }
  bool is_bgk_input(const Request& req) const {
    return req.type_id == Request::Type::PIM_WR_BG || req.type_id == Request::Type::PIM_WR_Q ||
           req.type_id == Request::Type::PIM_RD_Q;
  }
  bool is_bgk_mac(const Request& req) const {
    return req.type_id == Request::Type::PIM_MAC_K || req.type_id == Request::Type::PIM_MAC_QK;
  }
  void record_bgk(const Request& req, Clk_t ready, int service,
                  const char* name = nullptr, Clk_t clock = -1, int target_banks = -1) {
    if (!m_bgk_events) return;
    static const char* names[] = {"PIM_WR_BG","PIM_WR_Q","PIM_MAC_K","PIM_MAC_QK","PIM_RD_BG","PIM_LOCAL_WR","PIM_RD_Q"};
    const auto& spec = *m_device.m_spec;
    const auto& address = m_rank_rope && !req.pim_origin_addr_vec.empty() ? req.pim_origin_addr_vec : req.addr_vec;
    m_bgk_events << (clock < 0 ? m_clk : clock) << ',' << ready << ','
        << (name ? name : names[req.type_id-Request::Type::PIM_WR_BG])
        << ',' << m_channel_id << ',' << address[m_rank_level] << ',' << address[m_bg_level]
        << ',' << address[m_bank_level] << ',' << address[spec.get_level_id("Row")]
        << ',' << address[spec.get_level_id("Column")] << ',' << req.pim_buffer
        << ',' << (target_banks < 0 ? std::popcount(req.pim_bank_mask) : target_banks)
        << ",32," << service << '\n';
  }
  void enqueue_bgk_done(ReqBuffer::iterator it, Clk_t ready) {
    it->depart = ready;
    auto pos = std::upper_bound(m_pim_pending.begin(), m_pim_pending.end(), ready,
        [](Clk_t clk, const Request& pending) { return clk < pending.depart; });
    m_pim_pending.insert(pos,*it);
    m_pim_buffer.remove(it);
  }
  bool bgk_targets_ready(const Request& req, bool actual_mac) {
    const int slot = req.pim_buffer < 0 ? 0 : req.pim_buffer;
    for (size_t b=0;b<m_bank_inputs.size();++b) {
      if (!(req.pim_bank_mask & (uint64_t{1} << b))) continue;
      const auto& input = m_bank_inputs[b][slot];
      if (is_bgk_input(req)) {
        if (m_clk < input.in_use_until) return false;
      } else if (actual_mac) {
        if (m_rank_rope && req.type_id == Request::Type::PIM_MAC_QK && m_rope_banks[b].query_mask != rr_full_query_mask())
          throw std::runtime_error(m_pim_horizon==16 ? "Rank QK requires all sixteen query slices in every bank" :
              "Rank QK requires all horizon query slices in every bank");
        if (!input.written) throw std::runtime_error("BG-K MAC references unfilled bank-local input buffer");
        if (m_clk < input.ready_at) return false;
      } else if (req.type_id == Request::Type::PIM_RD_BG || req.type_id == Request::Type::PIM_LOCAL_WR) {
        if (m_clk < m_bank_compute_ready[b]) return false;
      }
    }
    return true;
  }
  bool issue_bgk(ReqBuffer::iterator it) {
    if (it->type_id == Request::Type::PIM_RD_Q) return issue_q_dram_read(it);
    if (m_rank_rope && it->type_id == Request::Type::PIM_RD_BG && m_clk < m_rank_reduction_ready) return false;
    it->command = get_preq_command(it->final_command,it->addr_vec);
    // Payload issues through independent local ports; only ACT/PRE uses CA.
    if (it->command == m_localwr) return false;
    const bool actual_mac = is_bgk_mac(*it) && it->command == it->final_command;
    if (!bgk_targets_ready(*it,actual_mac) || !check_timing(it->command,it->addr_vec)) return false;
    m_device.issue_command(it->command,it->addr_vec,m_clk);
    m_rowpolicy->on_issue(*it);
    for (auto* p:m_plugins) p->on_issue(*it);
    if (it->command != it->final_command) return true;
    int service = is_bgk_mac(*it) ? (it->type_id == Request::Type::PIM_MAC_K ? m_k_cycles : m_qk_cycles)
                                : m_bgk_input_cycles;
    const Clk_t ready=m_clk+service;
    const int slot=it->pim_buffer<0?0:it->pim_buffer;
    for(size_t b=0;b<m_bank_inputs.size();++b) {
      if(!(it->pim_bank_mask & (uint64_t{1} << b))) continue;
      auto& input=m_bank_inputs[b][slot];
      if(is_bgk_input(*it)) { input.written=true; input.ready_at=ready; }
      if(actual_mac) { input.in_use_until=ready; m_bank_compute_ready[b]=ready; }
    }
    if (m_rank_rope && actual_mac) {
      m_rank_reduction_ready = ready + m_rank_reduce_cycles;
      record_bgk(*it, m_rank_reduction_ready, m_rank_reduce_cycles, "PIM_RANK_REDUCE", ready, 16);
    }
    record_bgk(*it,ready,service);
    enqueue_bgk_done(it, m_rank_rope && actual_mac ? m_rank_reduction_ready : ready);
    return true;
  }

  bool issue_q_dram_read(ReqBuffer::iterator it) {
    if (it->pim_q_stage == 0) {
      if (m_q_staging_busy || m_clk < m_q_staging_free_at) return false;
      // Preserve the composite in the ordered PIM queue until its multicast
      // retires. The normal controller owns ACT1/ACT2/PRE/RD, CL, and BL.
      // ReqBuffer is a std::list, so this pointer survives other insertions.
      Request* owner = &*it;
      Request read = *it;
      read.type_id = Request::Type::Read;
      read.pim_buffer = -1;
      read.callback = [this, owner](Request& done) {
        if (done.command != m_cmd_rd && done.command != m_cmd_rda)
          throw std::runtime_error("RD_Q must complete a real DRAM read, not forwarding");
        const int latency = m_device.m_spec->read_latency;
        record_bgk(*owner, done.depart, latency, "PIM_RD_Q", done.depart - latency, 1);
        owner->pim_q_stage = 2;
        ++s_q_dram_reads;
      };
      if (!LPDDR5Controller::send(read)) return false;
      it->pim_q_stage = 1;
      m_q_staging_busy = true;
      return true;
    }
    if (it->pim_q_stage != 2 || !bgk_targets_ready(*it, false)) return false;
    // The staged payload must traverse the same movement path as GPU WRGB.
    // Broadcast the bank-index fragment to all BGs/ranks in this channel.
    // The DRAM source address stays intact in the event for independent audit.
    AddrVec_t target = it->addr_vec;
    target[m_rank_level] = target[m_bg_level] = -1;
    if (!check_timing(m_wrgb, target)) return false;
    it->command = m_wrgb;
    m_device.issue_command(m_wrgb, target, m_clk);
    Request multicast = *it;
    multicast.addr_vec = target;
    m_rowpolicy->on_issue(multicast);
    for (auto* p : m_plugins) p->on_issue(multicast);
    const Clk_t ready = m_clk + m_bgk_input_cycles;
    for (size_t b = 0; b < m_bank_inputs.size(); ++b) {
      if (!(it->pim_bank_mask & (uint64_t{1} << b))) continue;
      auto& input = m_bank_inputs[b][it->pim_buffer];
      input.written = true;
      input.ready_at = ready;
    }
    record_bgk(*it, ready, m_bgk_input_cycles, "PIM_RELAY_Q");
    ++s_q_relays;
    m_q_staging_busy = false;
    m_q_staging_free_at = ready;
    enqueue_bgk_done(it, ready);
    return true;
  }
  void issue_bank_local_writes() {
    // Scan all queued banks. Multiple bank payloads can start in the same CK.
    // Native row prerequisites are issued separately by issue_bgk on shared CA.
    for(auto it=m_pim_buffer.begin();it!=m_pim_buffer.end();) {
      if(it->type_id!=Request::Type::PIM_LOCAL_WR) {++it;continue;}
      bool independent=true;
      for(auto before=m_pim_buffer.begin();before!=it;++before) {
        if(!can_bypass(*it,*before,false)) {independent=false;break;}
      }
      if(!independent || !bgk_targets_ready(*it,false) ||
         get_preq_command(m_localwr,it->addr_vec)!=m_localwr ||
         !m_device.check_timing(m_localwr,it->addr_vec,m_clk)) {++it;continue;}
      it->command=m_localwr;
      m_device.issue_command(m_localwr,it->addr_vec,m_clk);
      const Clk_t ready=m_clk+m_bank_write_cycles+m_local_recovery;
      record_bgk(*it,ready,m_bank_write_cycles);
      ++s_bank_local_writes;
      // ReqBuffer is a std::list; removal leaves the following iterator valid.
      auto next=std::next(it);
      enqueue_bgk_done(it,ready);
      it=next;
    }
  }

  bool is_local(const Request& req) const {
    return req.type_id == Request::Type::PIM_LOCAL_GB;
  }

  void issue_local_ready() {
    issue_ffn_z_local();
    issue_rank_rope_local();
    issue_bank_local_writes();
    // These ports are rank-local SRAM links, not CA/DQ commands. Up to one
    // transfer per rank may start in this memory cycle, independently of DQ.
    for (size_t rank = 0; rank < m_inputs.size(); ++rank) {
      if (m_clk < m_local_ports[rank]) continue;
      for (auto it = m_pim_buffer.begin(); it != m_pim_buffer.end(); ++it) {
        if (!is_local(*it) || it->pim_rank_mask != (1 << rank)) continue;
        bool independent = true;
        for (auto before = m_pim_buffer.begin(); before != it; ++before) {
          if (!can_bypass(*it, *before, false)) { independent = false; break; }
        }
        if (!independent) break;
        auto& input = m_inputs[rank][it->pim_buffer];
        if (m_clk < std::max(input.in_use_until, input.ready_at)) break;
        const Clk_t ready = m_clk + m_local_cycles;
        if (m_local_trace) {
          const auto& spec = *m_device.m_spec;
          const auto offset = it->addr_vec[spec.get_level_id("Row")] * 2048
                            + it->addr_vec[spec.get_level_id("Column")] * 32;
          m_local_trace << m_clk << ',' << ready << ',' << m_channel_id << ','
                        << rank << ',' << it->pim_buffer << ',' << offset
                        << ",32," << m_local_cycles << ',' << input.in_use_until
                        << ',' << m_local_ports[rank] << '\n';
          m_local_trace.flush();
        }
        m_local_ports[rank] = ready;
        input.written = true;
        input.ready_at = ready;
        input.local_in_use_until = ready;
        it->depart = ready;
        auto pos = std::upper_bound(m_pim_pending.begin(), m_pim_pending.end(), ready,
            [](Clk_t clk, const Request& pending) { return clk < pending.depart; });
        m_pim_pending.insert(pos, *it);
        m_pim_buffer.remove(it);
        ++s_local_transfers;
        break;
      }
    }
  }

  bool issue_pim_ready() {
    for (auto it = m_pim_buffer.begin(); it != m_pim_buffer.end(); ++it) {
      if (is_local(*it) || (is_rank_rope(*it) && it->type_id == Request::Type::PIM_ROPE_EXEC)) continue;
      const int command = get_preq_command(it->final_command, it->addr_vec);
      if (it->type_id == Request::Type::PIM_LOCAL_WR && command == m_localwr) continue;
      const bool preparing_row = (it->final_command == m_macab || is_bgk_mac(*it)) &&
          (m_device.m_spec->command_meta[command].is_opening ||
           m_device.m_spec->command_meta[command].is_closing);
      bool independent = true;
      for (auto before = m_pim_buffer.begin(); before != it; ++before) {
        if (!can_bypass(*it, *before, preparing_row)) { independent = false; break; }
      }
      if (independent && issue_pim(it)) return true;
    }
    return false;
  }

  bool issue_pim_front() {
    return issue_pim(m_pim_buffer.begin());
  }

  bool issue_pim(ReqBuffer::iterator it) {
    if (is_ffn_z(*it)) return issue_ffn_z(it, false);
    if (is_rank_rope(*it)) return issue_rank_rope(it, false);
    if (is_bgk(*it)) return issue_bgk(it);
    it->command = get_preq_command(it->final_command, it->addr_vec);
    const int slot = it->pim_buffer < 0 ? 0 : it->pim_buffer;
    if (it->final_command == m_wrgb) {
      for (size_t rank = 0; rank < m_inputs.size(); ++rank) {
        if ((it->pim_rank_mask & (1 << rank)) &&
            m_clk < std::max(m_inputs[rank][slot].in_use_until,
                             m_inputs[rank][slot].local_in_use_until)) return false;
      }
    }
    // Only the actual MAC consumes input. ACT/PRE may prepare its weight row
    // while the input SRAM is still being filled.
    if (it->command == m_macab) {
      if (!ffn_mac_ready(*it)) return false;
      for (size_t rank = 0; rank < m_inputs.size(); ++rank) {
        if (!(it->pim_rank_mask & (1 << rank))) continue;
        const auto& input = m_inputs[rank][slot];
        if (it->pim_buffer >= 0 && !input.written) {
          throw std::runtime_error("PIM_MAC_AB references an input buffer that was never filled");
        }
        if (m_clk < input.ready_at) return false;
      }
    }
    if (it->final_command == m_rdres && m_data_path != "attacc") {
      const auto& buffers = m_inputs.at(it->addr_vec[m_rank_level]);
      if (m_clk < std::max(buffers[0].in_use_until, buffers[1].in_use_until)) return false;
    }
    if (!check_timing(it->command, it->addr_vec)) return false;
    m_device.issue_command(it->command, it->addr_vec, m_clk);
    m_rowpolicy->on_issue(*it);
    for (auto* p : m_plugins) p->on_issue(*it);
    if (it->command == it->final_command) {
      if (it->final_command == m_wrgb) {
        // One DQ burst, simultaneous delivery to all selected rank SRAMs.
        // Broadcast fanout has no extra modeled latency. Keep write completion
        // and every destination's input ownership checks explicit.
        for (size_t rank = 0; rank < m_inputs.size(); ++rank) {
          if (it->pim_rank_mask & (1 << rank)) {
            auto& target = m_inputs[rank][it->pim_buffer < 0 ? 0 : it->pim_buffer];
            target.written = true;
            target.ready_at = m_clk + m_pim_wr_latency;
          }
        }
      } else if (it->final_command == m_macab && it->pim_buffer >= 0) {
        // The fused operation consumes this SRAM throughout its issue slot.
        // This is SRAM ownership, not an extra MAC completion latency or a
        // channel-wide lock: the other buffer can be written immediately.
        for (size_t rank = 0; rank < m_inputs.size(); ++rank) {
          if (it->pim_rank_mask & (1 << rank)) {
            m_inputs[rank][slot].in_use_until = m_clk + m_mac_interval;
          }
        }
      }
      if (it->final_command == m_macab) ffn_mac_issued(*it);
      if (m_rank_rope && it->final_command == m_macab) m_projection_ready = m_clk + m_mac_interval;
      const int latency = (it->final_command == m_macab || it->final_command == m_machb) ? (m_rank_rope ? m_mac_interval : 0) :
          (it->final_command == m_wrgb ? m_pim_wr_latency : m_pim_rd_latency);
      it->depart = m_clk + latency;
      if (m_rank_rope) record_bgk(*it, it->depart, latency, it->final_command == m_macab ? "PIM_MAC_AB" : it->final_command == m_wrgb ? "PIM_WR_GB" : "PIM_RD_RES", -1, 16);
      auto pos = std::upper_bound(m_pim_pending.begin(), m_pim_pending.end(), it->depart,
          [](Clk_t clk, const Request& pending) { return clk < pending.depart; });
      m_pim_pending.insert(pos, *it);
      m_pim_buffer.remove(it);
      // AttAcc-style MAC retirement at issue; no independent compute tail.
      if (latency == 0) complete_pim();
    }
    return true;
  }
};

}  // namespace Ramulator

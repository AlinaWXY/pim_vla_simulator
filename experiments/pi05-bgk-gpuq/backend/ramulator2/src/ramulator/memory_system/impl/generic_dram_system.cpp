#include <stdexcept>
#include <deque>
#include <fstream>

#include <fmt/format.h>

#include "ramulator/base/param.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator/translation/i_translation.h"

namespace Ramulator {

class GenericDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, GenericDRAMSystem, "GenericDRAM");

 protected:
  IChannelMapper* m_channel_mapper;
  std::vector<IController*> m_controllers;
  unsigned int m_clock_ratio = 1;
  int m_tx_bytes = 0;
  bool m_channel_independent_dispatch = false;
  int m_channel_queue_capacity = 32;
  std::string m_ingress_events_path;
  std::ofstream m_ingress_events;
  std::vector<std::deque<Request>> m_ingress;
  Clk_t m_ingress_clock = 0;

  void record_ingress(const Request& req, int channel, const char* operation) {
    if (m_ingress_events) {
      m_ingress_events << m_ingress_clock << ',' << channel << ',' << operation << ','
        << req.ingress_id << ',' << req.addr << ',' << req.type_id << ','
        << req.pim_rank_batch << ',' << m_ingress[channel].size() << ','
        << m_channel_queue_capacity << '\n';
    }
  }

 public:
  int s_num_read_requests = 0;
  int s_num_write_requests = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_channel_independent_dispatch, bool, "channel_independent_dispatch").default_val(false);
    RAMULATOR_PARSE_PARAM(m_channel_queue_capacity, int, "channel_queue_capacity").default_val(32);
    RAMULATOR_PARSE_PARAM(m_ingress_events_path, std::string, "ingress_events_path").default_val("");
    if (m_channel_queue_capacity < 1) throw std::runtime_error("channel_queue_capacity must be positive");
    RAMULATOR_CREATE_CHILD(m_channel_mapper, IChannelMapper);

    // Each controller = one channel. DRAM config lives inside each controller.
    RAMULATOR_CREATE_CHILD_LIST(m_controllers, IController);
    if (m_controllers.empty()) {
      throw std::runtime_error("GenericDRAM requires at least one controller");
    }
    for (size_t i = 0; i < m_controllers.size(); i++) {
      dynamic_cast<Implementation*>(m_controllers[i])->set_id(fmt::format("Channel {}", i));
      m_controllers[i]->set_channel_id(static_cast<int>(i));
      m_controllers[i]->m_clock_ratio = m_clock_ratio;
    }

    // Setup channel mapper with controller info
    m_tx_bytes = m_controllers[0]->get_tx_bytes();
    m_channel_mapper->setup(static_cast<int>(m_controllers.size()), calc_log2(m_tx_bytes));
    m_ingress.resize(m_controllers.size());
    if (m_channel_independent_dispatch && !m_ingress_events_path.empty()) {
      m_ingress_events.open(m_ingress_events_path);
      if (!m_ingress_events) throw std::runtime_error("Cannot open ingress_events_path");
      m_ingress_events << "clock,channel,operation,request_id,address,type_id,rank_batch,queue_depth,capacity\n";
    }

    m_stats.add("total_num_read_requests", s_num_read_requests);
    m_stats.add("total_num_write_requests", s_num_write_requests);
  };

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
  }

  bool send(Request& req) override {
    // Validate request size: must be set and fit within one transaction.
    if (req.size_bytes <= 0 || req.size_bytes > m_tx_bytes) {
      throw std::runtime_error(fmt::format(
          "Request size_bytes must be set by the frontend (got {}, tx_bytes = {}).",
          req.size_bytes, m_tx_bytes));
    }

    // Channel mapper sets req.addr_vec[0] and req.intra_channel_addr.
    // Controller::send() handles address mapping internally.
    m_channel_mapper->apply(req);
    req.channel_count = static_cast<int>(m_controllers.size());
    int channel_id = req.addr_vec[0];
    bool is_success = false;
    if (m_channel_independent_dispatch) {
      auto& queue = m_ingress[channel_id];
      if (queue.size() >= static_cast<size_t>(m_channel_queue_capacity)) return false;
      queue.push_back(req);
      record_ingress(req, channel_id, "ENQUEUE");
      is_success = true;
    } else {
      is_success = m_controllers[channel_id]->send(req);
    }

    if (is_success) {
      switch (req.type_id) {
        case Request::Type::Read: {
          s_num_read_requests++;
          break;
        }
        case Request::Type::Write: {
          s_num_write_requests++;
          break;
        }
      }
    }
    return is_success;
  };

  void tick() override {
    for (size_t channel = 0; channel < m_controllers.size(); ++channel) {
      auto* controller = m_controllers[channel];
      // Admission to the bounded FIFO is independent of the downstream CA.
      // One dispatch opportunity per channel per memory CK; the controller
      // retains all CA arbitration, buffer backpressure and dependency checks.
      if (m_channel_independent_dispatch && !m_ingress[channel].empty()) {
        Request req = m_ingress[channel].front();
        if (controller->send(req)) {
          m_ingress[channel].pop_front();
          record_ingress(req, static_cast<int>(channel), "DISPATCH");
        }
      }
      controller->tick();
    }
    ++m_ingress_clock;
  };

  int get_num_channels() const override { return static_cast<int>(m_controllers.size()); }
  int get_channel_id(Addr_t addr) const override {
    Request req(addr, Request::Type::Read);
    m_channel_mapper->apply(req);
    return req.addr_vec[0];
  }
  bool has_independent_channel_ingress() const override { return m_channel_independent_dispatch; }

  void reset_stats() override {
    s_num_read_requests = 0;
    s_num_write_requests = 0;
  }

  int get_clock_ratio() override {
    return m_clock_ratio;
  }

  float get_tCK() override {
    if (!m_controllers.empty()) {
      return m_controllers[0]->get_tCK();
    }
    return -1.0f;
  }

  int get_tx_bytes() override {
    return m_tx_bytes;
  }
};

}  // namespace Ramulator

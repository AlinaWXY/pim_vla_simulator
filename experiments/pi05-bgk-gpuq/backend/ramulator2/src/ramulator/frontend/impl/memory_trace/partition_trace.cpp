#include <fstream>
#include <sstream>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

// Independent, ordered request sources for static GPU/PIM resource partitions.
// One attempt per source per frontend tick. A full queue stalls only its source.
// Unlike LoadStoreTrace, there is no global normal/PIM phase fence: the existing
// controllers still enforce admission, bank conflicts and all device timings.
class PartitionTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PartitionTrace, "PartitionTrace")
  struct Entry { int type; Addr_t addr; };
  std::vector<std::vector<Entry>> m_streams;
  std::vector<size_t> m_next;
  std::ofstream m_completions;
  std::string m_path, m_completions_path;
  size_t m_total = 0, m_accepted = 0, m_completed = 0;
  uint64_t m_max_ticks = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_completions_path, std::string, "completions_path").required();
    RAMULATOR_PARSE_PARAM(m_max_ticks, uint64_t, "max_ticks").default_val(10000000);
    std::ifstream input(m_path);
    if (!input) throw std::runtime_error("PartitionTrace: cannot open input");
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream fields(line);
      int stream;
      std::string op, value, extra;
      if (!(fields >> stream >> op >> value) || fields >> extra || stream < 0 || stream > 4095) {
        throw std::runtime_error("PartitionTrace: expected stream(0..4095) OP ADDRESS");
      }
      const int type = op == "LD" ? Request::Type::Read :
          op == "PIM_MAC_AB" ? Request::Type::PIM_MAC_AB :
          op == "PIM_MAC_HB" ? Request::Type::PIM_MAC_HB : -1;
      size_t used = 0;
      const Addr_t addr = std::stoll(value, &used, 0);
      if (type < 0 || addr < 0 || used != value.size()) {
        throw std::runtime_error("PartitionTrace: invalid operation/address");
      }
      m_streams.resize(std::max(m_streams.size(), static_cast<size_t>(stream + 1)));
      m_streams[stream].push_back({type, addr});
      ++m_total;
    }
    if (!m_total) throw std::runtime_error("PartitionTrace: empty input");
    m_next.resize(m_streams.size(), 0);
    m_completions.open(m_completions_path);
    if (!m_completions) throw std::runtime_error("PartitionTrace: cannot open completions");
    m_completions << "stream,type,channel,address,arrive,depart\n";
    m_stats.add("requests_accepted", m_accepted);
    m_stats.add("requests_completed", m_completed);
  }

  void tick() override {
    if (++m_clk > static_cast<Clk_t>(m_max_ticks)) {
      throw std::runtime_error("PartitionTrace: tick limit exceeded");
    }
    for (size_t stream = 0; stream < m_streams.size(); ++stream) {
      if (m_next[stream] == m_streams[stream].size()) continue;
      const auto& entry = m_streams[stream][m_next[stream]];
      Request req(entry.addr, entry.type);
      req.size_bytes = m_memory_system->get_tx_bytes();
      if (entry.addr % req.size_bytes) throw std::runtime_error("PartitionTrace: unaligned address");
      req.callback = [this, stream](Request& done) {
        ++m_completed;
        m_completions << stream << ',' << done.type_id << ',' << done.addr_vec[0] << ','
                      << done.addr << ',' << done.arrive << ',' << done.depart << '\n';
      };
      if (m_memory_system->send(req)) {
        ++m_next[stream];
        ++m_accepted;
      }
    }
  }

  bool is_finished() override { return m_completed == m_total; }
  void finalize() override { m_completions.close(); }
};
}  // namespace Ramulator

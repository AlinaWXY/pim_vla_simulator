#ifndef RAMULATOR_BASE_REQUEST_H
#define RAMULATOR_BASE_REQUEST_H

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "ramulator/base/type.h"

namespace Ramulator {

struct Request {
  Addr_t addr = -1;
  Addr_t intra_channel_addr = -1;  // Flat address with channel bits stripped
  AddrVec_t addr_vec{};
  AddrVec_t pim_origin_addr_vec{};

  // Universal built-in external request types — always Read = 0, Write = 1.
  // Additional non-negative ids may exist as metadata for future extensions.
  struct Type {
    enum : int { Read = 0, Write = 1, PIM_MAC_AB = 2, PIM_WR_GB = 3, PIM_RD_RES = 4, PIM_MAC_HB = 5, PIM_LOCAL_GB = 6, PIM_WR_BG = 7, PIM_WR_Q = 8, PIM_MAC_K = 9, PIM_MAC_QK = 10, PIM_RD_BG = 11, PIM_LOCAL_WR = 12, PIM_RD_Q = 13, PIM_SPILL = 14, PIM_ROPE_COEF = 15, PIM_ROPE_LOAD = 16, PIM_ROPE_EXEC = 17, PIM_ROPE_STORE = 18, PIM_Q_EXPORT = 19, PIM_Q_REMOTE = 20, PIM_Z_STORE = 21, PIM_Z_LOAD = 22, PIM_GEMV_RELEASE = 23, PIM_STAGE_BEGIN = 24, PIM_STAGE_SEAL = 25, PIM_STAGE_LOAD = 26, PIM_STAGE_FRAGMENT = 27, PIM_FFN_PASS = 28, PIM_FFN_RDRES = 29 };
  };

  int type_id = -1;        // Request type. -1 is the convention for internal maintenance/direct-command requests.
  int source_id = -1;      // Source identifier (e.g., which core)
  int ingress_id = -1;     // External ingress identifier (e.g., trace line index).
  int channel_count = 1;   // Channel mapper geometry for rank descriptor address expansion.
  bool pim_rank_micro = false; // Internal child: retains rank CA scope through ordinary RD scheduling.
  bool pim_rank_batch = false; // One rank descriptor expands into the original 16 bank requests.
  int pim_buffer = -1;     // Optional ping-pong input SRAM slot, 0/1; -1 = legacy trace.
  uint64_t pim_bank_mask = 0; // New bank-local input/result destinations.
  int pim_rank_mask = 0;   // Controller-resolved PIM targets; input broadcasts may select all ranks.
  int pim_q_stage = 0;     // RD_Q composite: queued=0, ordinary read pending=1, staged=2.
  int pim_stage_bytes = 0, pim_stage_rows = 0, pim_stage_cursor = 0;

  int size_bytes = -1;     // Request size in bytes. Must be set explicitly by the frontend.

  int command = -1;        // Current command to issue to progress the request
  int final_command = -1;  // Terminal command needed to complete the request
  bool is_stat_updated = false;

  Clk_t arrive = -1;  // Clock cycle when the request arrives at the memory controller
  Clk_t depart = -1;  // Clock cycle when the request departs the memory controller

  std::function<void(Request&)> callback;

  // Tag type to disambiguate the internal-command constructor from the type_id one.
  struct Cmd_t {};
  static constexpr Cmd_t Cmd{};

  Request() = default;
  Request(Addr_t addr, int type);
  Request(AddrVec_t addr_vec, int type);
  Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback);
  Request(AddrVec_t addr_vec, Cmd_t, int final_cmd);  // internal commands (refresh, row close, etc.)
};

}  // namespace Ramulator

#endif  // RAMULATOR_BASE_REQUEST_H

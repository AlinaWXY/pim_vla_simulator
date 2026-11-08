#pragma once
#include "ramulator/dram/node.h"
namespace Ramulator::Cmd {
// Actions/timing of the sixteen internal ACT2 pulses are spec expansions.
template<class T> struct ACTAB {
  static constexpr DRAMCommandMeta meta = {.is_opening = true};
  static constexpr BankTarget bank_target = BankTarget::All;
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t&, Clk_t) {
    if (bank->m_state == T::State::Closed) return cmd;
    return cmd == T::Command::ACTAB_LOCAL ? T::Command::PREab_LOCAL : T::Command::PREab;
  }
};
}

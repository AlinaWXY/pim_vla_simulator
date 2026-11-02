#ifndef RAMULATOR_DRAM_COMMANDS_ACTAB_H
#define RAMULATOR_DRAM_COMMANDS_ACTAB_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// Coarse-grained PIM activation: open the addressed row in every bank of the
// selected rank, or the entire channel when Rank is a wildcard. The address
// vector also uses wildcards for BankGroup and Bank.
template <class T>
struct ACTAB {
  static constexpr DRAMCommandMeta meta = {.is_opening = true};
  static constexpr BankTarget bank_target = BankTarget::All;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    if (bank->m_state == T::State::Closed ||
        (bank->m_state == T::State::Opened &&
         bank->m_row_state.find(addr_vec[T::Level::Row]) != bank->m_row_state.end())) {
      return cmd;
    }
    return T::Command::PREab;
  }

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    bank->m_state = T::State::Opened;
    bank->m_row_state.clear();
    bank->m_row_state[addr_vec[T::Level::Row]] = T::State::Opened;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_ACTAB_H

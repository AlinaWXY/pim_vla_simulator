#ifndef RAMULATOR_DRAM_COMMANDS_MACMASK_H
#define RAMULATOR_DRAM_COMMANDS_MACMASK_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// Coarse-grained all-bank PIM read/compute command.  Prerequisites are
// aggregated across every target bank by DRAMDevice::get_preq_command(): a
// conflicting open row requests PREab, while any closed bank requests ACTAB.
template <class T>
struct MACMASK {
  static constexpr DRAMCommandMeta meta = {.is_accessing = true};
  static constexpr BankTarget bank_target = BankTarget::All;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    if (bank->m_state == T::State::Closed) {
      return T::Command::ACTMASK;
    }
    if (bank->m_state == T::State::Opened &&
        bank->m_row_state.find(addr_vec[T::Level::Row]) != bank->m_row_state.end()) {
      return cmd;
    }
    return T::Command::PREMASK;
  }

  static bool rowhit(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    return bank->m_state == T::State::Opened &&
           bank->m_row_state.find(addr_vec[T::Level::Row]) != bank->m_row_state.end();
  }

  static bool rowopen(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    return bank->m_state == T::State::Opened;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_MACMASK_H

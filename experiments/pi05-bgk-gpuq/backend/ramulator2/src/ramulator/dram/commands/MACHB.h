#ifndef RAMULATOR_DRAM_COMMANDS_MACHB_H
#define RAMULATOR_DRAM_COMMANDS_MACHB_H

#include "ramulator/dram/commands/MACAB.h"

namespace Ramulator::Cmd {

// Compute Half Banks: one internal 32B read per bank, without external DQ use.
template <class T>
struct MACHB : MACAB<T> {
  static constexpr BankTarget bank_target = BankTarget::Half;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    const int preq = MACAB<T>::preq(bank, cmd, addr_vec, clk);
    if (preq == T::Command::ACTAB) return T::Command::ACTHB;
    if (preq == T::Command::PREab) return T::Command::PREHB;
    return preq;
  }
};

}  // namespace Ramulator::Cmd
#endif

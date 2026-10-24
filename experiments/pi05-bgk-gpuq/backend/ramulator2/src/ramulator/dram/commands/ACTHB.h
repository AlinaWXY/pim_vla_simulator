#ifndef RAMULATOR_DRAM_COMMANDS_ACTHB_H
#define RAMULATOR_DRAM_COMMANDS_ACTHB_H

#include "ramulator/dram/commands/ACTAB.h"

namespace Ramulator::Cmd {

// Open the addressed row in BG 0/1 or BG 2/3 of the selected rank.
template <class T>
struct ACTHB : ACTAB<T> {
  static constexpr BankTarget bank_target = BankTarget::Half;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    return ACTAB<T>::preq(bank, cmd, addr_vec, clk) == cmd ? cmd : T::Command::PREHB;
  }
};

}  // namespace Ramulator::Cmd
#endif

#ifndef RAMULATOR_DRAM_COMMANDS_PREHB_H
#define RAMULATOR_DRAM_COMMANDS_PREHB_H

#include "ramulator/dram/commands/PREab.h"

namespace Ramulator::Cmd {

template <class T>
struct PREHB : PREab<T> {
  static constexpr BankTarget bank_target = BankTarget::Half;
};

}  // namespace Ramulator::Cmd
#endif

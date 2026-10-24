#ifndef RAMULATOR_DRAM_COMMANDS_RDRES_H
#define RAMULATOR_DRAM_COMMANDS_RDRES_H
#include "ramulator/dram/node.h"
namespace Ramulator::Cmd {
// One 32-byte burst from result registers to the shared external logic.
template <class T> struct RDRES {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};
}
#endif

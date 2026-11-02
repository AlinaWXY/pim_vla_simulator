#ifndef RAMULATOR_DRAM_COMMANDS_WRGB_H
#define RAMULATOR_DRAM_COMMANDS_WRGB_H
#include "ramulator/dram/node.h"
namespace Ramulator::Cmd {
// One 32-byte x16 burst to the selected rank's shared input SRAM.
// No array activation or write recovery.
template <class T> struct WRGB {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};
}
#endif

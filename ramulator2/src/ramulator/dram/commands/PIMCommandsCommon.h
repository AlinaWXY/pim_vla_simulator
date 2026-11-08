#ifndef RAMULATOR_PIM_COMMANDS_COMMON_H
#define RAMULATOR_PIM_COMMANDS_COMMON_H
#include "ramulator/dram/commands/RD.h"
#include "ramulator/dram/commands/WR.h"
#include "ramulator/dram/commands/ACT1.h"
#include "ramulator/dram/commands/ACT2.h"
#include "ramulator/dram/commands/PREpb.h"
#include "ramulator/dram/commands/PREab.h"
namespace Ramulator::Cmd {
// SRF broadcast/pool IO does not require or alter any open row.
template<class T> struct PIMBufferIO {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};
// The controller must open each targeted bank before issuing a broadcast MAC.
// Native prereq traversal checks every participating bank.
template<class T> struct PIMArrayReadAll : RD<T> {
  static constexpr BankTarget bank_target = BankTarget::All;
};
}
#endif

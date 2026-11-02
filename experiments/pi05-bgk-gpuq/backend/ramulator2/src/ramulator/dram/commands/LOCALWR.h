#ifndef RAMULATOR_DRAM_COMMANDS_LOCALWR_H
#define RAMULATOR_DRAM_COMMANDS_LOCALWR_H
#include "ramulator/dram/commands/WR.h"
namespace Ramulator::Cmd {
template <class T> struct LOCALWR : WR<T> {};
}
#endif

#ifndef RAMULATOR_DRAM_COMMANDS_LOCALRD_H
#define RAMULATOR_DRAM_COMMANDS_LOCALRD_H
#include "ramulator/dram/commands/RD.h"
namespace Ramulator::Cmd {
template <class T> struct LOCALRD : RD<T> {};
}
#endif

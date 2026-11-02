#ifndef RAMULATOR_DRAM_COMMANDS_MACK_H
#define RAMULATOR_DRAM_COMMANDS_MACK_H
#include "ramulator/dram/commands/MACAB.h"
namespace Ramulator::Cmd {
template <class T> struct MACK : MACAB<T> {};
}
#endif

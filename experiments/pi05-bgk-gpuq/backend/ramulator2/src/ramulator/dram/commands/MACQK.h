#ifndef RAMULATOR_DRAM_COMMANDS_MACQK_H
#define RAMULATOR_DRAM_COMMANDS_MACQK_H
#include "ramulator/dram/commands/MACAB.h"
namespace Ramulator::Cmd {
template <class T> struct MACQK : MACAB<T> {};
}
#endif

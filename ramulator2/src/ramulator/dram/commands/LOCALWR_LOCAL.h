#pragma once
#include "ramulator/dram/commands/LOCALWR.h"
namespace Ramulator::Cmd {
template<class T> struct LOCALWR_LOCAL : LOCALWR<T> {};
}

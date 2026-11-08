#pragma once
#include "ramulator/dram/commands/LOCALRD.h"
namespace Ramulator::Cmd {
template<class T> struct LOCALRD_LOCAL : LOCALRD<T> {};
}

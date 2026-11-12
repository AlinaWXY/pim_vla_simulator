#pragma once
#include "ramulator/dram/commands/ROPE_EXEC.h"
namespace Ramulator::Cmd {
template<class T> struct ROPE_EXEC_LOCAL : ROPE_EXEC<T> {
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    if (bank->m_spec->command_expansions[T::Command::ACTAB].empty())
      return PIMArrayReadAll<T>::preq(bank, cmd, addr, clk);
    return cmd;
  }
};
}

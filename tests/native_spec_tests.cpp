#include <iostream>
#include <stdexcept>
#include <string>
#include "ramulator/base/config.h"
#include "ramulator/dram/device.h"

using namespace Ramulator;

static void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
static DRAMDevice device(const ConfigNode& spec) {
  DRAMDevice d;
  d.init(DRAMSpec::create("LPDDR5XPIM", spec));
  return d;
}
static bool buffer(const std::string& name) { return name=="WRGB" || name=="RDRES"; }
static AddrVec_t addr(const std::string& name, int rank, int bg, int bank) {
  return {0, rank, buffer(name)?-1:bg, buffer(name)?-1:bank, 0, 0};
}

int main(int argc, char** argv) {
  try {
    if (argc!=3) throw std::runtime_error("usage: native_spec_tests spec.json command-pair-matrix.json");
    auto spec=Config::parse_config_file(argv[1]);
    auto matrix=Config::parse_config_file(argv[2]);
    int checks=0;
    for (const auto& e:matrix.seq()) {
      auto d=device(spec);
      auto first=e["first"].as<std::string>(), second=e["second"].as<std::string>();
      auto rel=e["relation"].as<std::string>();
      int c1=d.m_spec->get_command_id(first), c2=d.m_spec->get_command_id(second);
      auto a1=addr(first,1,0,0);
      auto a2=addr(second,rel=="different_rank"?0:1,rel=="different_bg"?1:0,1);
      const auto gap=e["issue_gap_ck"].as<int>();
      d.issue_command(c1,a1,100);
      require(!d.check_timing(c2,a2,100+gap-1),"accepted early: "+first+" -> "+second+" "+rel);
      require(d.check_timing(c2,a2,100+gap),"rejected legal boundary: "+first+" -> "+second+" "+rel);
      checks+=2;
    }
    // Private row commands retain their own identity when checking preqs.
    {
      auto d=device(spec); AddrVec_t a={0,1,0,0,0,0};
      int act1=d.m_spec->get_command_id("ACT1_LOCAL"),act2=d.m_spec->get_command_id("ACT2_LOCAL");
      require(d.get_preq_command(act1,a,100)==act1,"private ACT1 prerequisite identity");
      d.issue_command(act1,a,100);
      require(d.get_preq_command(act2,a,101)==act2,"private ACT2 prerequisite identity");
      require(d.check_timing(act2,a,101),"private ACT2 earliest legal tick");
      d.issue_command(act2,a,101);
      int read=d.m_spec->get_command_id("LOCALRD");
      require(d.get_preq_command(read,a,121)==read,"private activation did not open shared row state");
      require(!d.check_timing(read,a,120) && d.check_timing(read,a,121),"private activation RCD boundary");
      checks+=5;
    }
    // A private precharge of an OPENED bank must not turn back into external CA.
    for (const std::string name:{"PREpb_LOCAL","PREab_LOCAL"}) {
      auto d=device(spec); AddrVec_t a={0,1,0,0,0,0};
      int act1=d.m_spec->get_command_id("ACT1_LOCAL"),act2=d.m_spec->get_command_id("ACT2_LOCAL");
      d.issue_command(act1,a,100);d.issue_command(act2,a,101);
      int pre=d.m_spec->get_command_id(name),read=d.m_spec->get_command_id("LOCALRD");
      AddrVec_t pa=name=="PREab_LOCAL"?AddrVec_t{0,1,-1,-1,0,0}:a;
      require(d.get_preq_command(pre,pa,200)==pre,"private opened-row precharge identity: "+name);
      require(d.check_timing(pre,pa,200),"private precharge rejected after RAS: "+name);
      d.issue_command(pre,pa,200);
      require(d.get_preq_command(read,a,223)==d.m_spec->get_command_id("ACT1"),"private precharge did not close shared row state: "+name);
      checks+=3;
    }
    // No open row is needed for SRAM/pool IO; array operations retain prerequisites.
    {
      auto d=device(spec); int rd=d.m_spec->get_command_id("LOCALRD");
      AddrVec_t a={0,1,0,0,0,0},other={0,1,0,1,0,0};
      d.issue_command(rd,a,100);
      require(!d.check_timing(rd,other,100),"private CA accepted simultaneous local issues");
      require(d.check_timing(rd,other,101),"different-bank local read inherited a rank data-port lock");
      require(!d.check_timing(rd,a,103)&&d.check_timing(rd,a,104),"local read lost bank CCD boundary");
      require(d.check_timing(d.m_spec->get_command_id("BCAST"),{0,1,-1,-1,0,0},101),"local read locked independent rank unit until completion");
      require(d.check_timing(d.m_spec->get_command_id("WRGB"),{0,1,-1,-1,0,0},100),"private issue occupied external CA");
      checks+=5;
    }
    // No open row is needed for SRAM/pool IO; array operations retain prerequisites.
    {
      auto d=device(spec);
      for (const std::string c:{"WRGB","RDRES"}) {
        auto id=d.m_spec->get_command_id(c); auto a=addr(c,1,0,0);
        require(d.get_preq_command(id,a,0)==id,"buffer requires array row: "+c); ++checks;
      }
      for (const std::string c:{"MACAB_M16","MACAB_M16_LOCAL","LOCALRD","LOCALWR"}) {
        auto id=d.m_spec->get_command_id(c);
        AddrVec_t a={0,1,c.find("MAC")==0?-1:0,c.find("MAC")==0?-1:0,0,0};
        auto pre=d.get_preq_command(id,a,0);
        require(pre==d.m_spec->get_command_id("ACT1"),"missing closed-bank ACT1 prereq: "+c);
        auto pa=d.preq_addr_vec(id,pre,a,0);
        require(pa[2]>=0 && pa[3]>=0,"broadcast prereq not resolved to physical bank"); checks+=2;
      }
    }
    // Native REFpb pair targeting must block both banks n and n+8, and full-bank PIM.
    {
      auto d=device(spec); int refresh=d.m_spec->get_command_id("REFpb");
      d.issue_command(refresh,{0,1,0,0,0,0},100);
      for (const std::string c:{"WRGB","RDRES","MACAB_M16","MACAB_M16_LOCAL"}) {
        int id=d.m_spec->get_command_id(c); AddrVec_t a={0,1,-1,-1,0,0};
        require(!d.check_timing(id,a,227),"REFpb did not block "+c);
        require(d.check_timing(id,a,228),"REFpb blocked beyond recovery "+c); checks+=2;
      }
      const auto rd=d.m_spec->get_command_id("RD");
      require(!d.check_timing(rd,{0,1,2,0,0,0},227),"REFpb paired bank not blocked");
      require(d.check_timing(rd,{0,1,1,0,0,0},101),"REFpb unrelated bank blocked"); checks+=2;
    }
    std::cout<<"{\"native_spec_checks\":"<<checks<<",\"status\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<"\n"; return 1;
  }
}

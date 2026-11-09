#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "ramulator/controller/impl/pim_model.h"

// Standalone observer test; production scheduling and timing are unchanged.
using namespace PimVla;

static Operation op(const std::string& name,int id) {
  Operation result; result.name=name; result.id=id; result.addr={0,1,0,0,0,0};
  return result;
}

static void warm(PIMController& c) {
  const auto opened=c.device.m_spec->states.at("Opened");
  for(auto* b:c.device.m_bank_nodes) {b->m_state=opened;b->m_row_state[0]=opened;}
  c.refresh_due.fill(NEVER); // Explicit fixture: isolate CA/DQ concurrency.
}

static void gio_boundaries(const ConfigNode& spec) {
  int checks=0;
  const auto timing=[&](const char* n){return spec["timings"][n].as<int>();};
  for(const std::string b:{"BCAST","BCAST_LOCAL"}) {
    std::vector<std::tuple<std::string,std::string,int>> pairs={
      {b,b,timing("nCCDL")},
      {"WR",b,timing("nCWL")+timing("nBL_max")},
      {"WRGB",b,timing("nCWL")+timing("nBL_max")},
      {b,"WR",std::max(timing("nBL_min"),timing("nCCDL")+2-timing("nCWL"))},
      {b,"WRGB",std::max(timing("nBL_min"),timing("nCCDL")+2-timing("nCWL"))},
      {b,"RD",timing("nCCDL")+timing("nWTRL")},
      {b,"RDRES",timing("nCCDL")+timing("nWTRL")},
      {"RD",b,timing("nCL")+timing("nBL_min")},
      {"RDRES",b,timing("nCL")+timing("nBL_min")}};
    for(const auto& [first,second,gap]:pairs) {
      PIMController c(0,spec,spec["metadata"]);warm(c);
      const auto address=[](const std::string& name,int rank){
        const bool all=name=="WRGB"||name=="RDRES"||name.rfind("BCAST",0)==0;
        return AddrVec_t{0,rank,all?-1:0,all?-1:0,0,0};
      };
      int a=c.device.m_spec->get_command_id(first),z=c.device.m_spec->get_command_id(second);
      c.device.issue_command(a,address(first,1),100);
      require(!c.device.check_timing(z,address(second,1),100+gap-1),"GIO accepted early: "+first+" -> "+second);
      require(c.device.check_timing(z,address(second,1),100+gap),"GIO rejected boundary: "+first+" -> "+second);
      checks+=2;
    }
    for(const std::string gpu:{"RD","WR"}) {
      PIMController c(0,spec,spec["metadata"]);warm(c);
      c.device.issue_command(c.device.m_spec->get_command_id(b),{0,1,-1,-1,0,0},100);
      require(c.device.check_timing(c.device.m_spec->get_command_id(gpu),{0,0,0,0,0,0},100),"rank1 broadcast blocked rank0 "+gpu);
      ++checks;
    }
  }
  std::cout<<"{\"test\":\"native_gio_boundaries\",\"checks\":"<<checks<<",\"passed\":true}\n";
}

static void staged_window(const ConfigNode& spec) {
  PIMController c(0,spec,spec["metadata"]); c.fused=true;warm(c);
  for(const std::string name:{"LOCALRD_LOCAL","BCAST_LOCAL","MACAB_M1_LOCAL"}) {
    const auto& m=c.info.at(c.device.m_spec->get_command_id(name));
    require(m.external_ca==0 && m.direction==0 && m.burst==0 && m.private_ca==1,
            "private compute uses external CA/DQ: "+name);
  }
  auto input=op("WR",0);input.epoch="input_staging";
  auto fence=op("BARRIER",1);fence.deps={0};
  auto descriptor=op("MAC_RUN",2);descriptor.addr={0,1,-1,-1,0,0};descriptor.deps={1};
  for(int f=0;f<64;++f) {
    auto load=op("LOCALRD",f*3);load.dest_slot=0;load.tensor_fragment=f;
    if(f)load.deps={f*3-1};
    auto bcast=op("BCAST",f*3+1);bcast.addr={0,1,-1,-1,0,0};
    bcast.source_bank=0;bcast.slot=0;bcast.tensor_fragment=f;bcast.deps={load.id};
    auto mac=op("MACAB",f*3+2);mac.addr={0,1,-1,-1,0,0};
    mac.rows=1;mac.slot=0;mac.fragment=f;mac.output_columns=1;mac.deps={bcast.id};
    descriptor.children.push_back(load);descriptor.children.push_back(bcast);descriptor.children.push_back(mac);
  }
  auto output=op("RD",3);output.epoch="result_export";output.deps={2};
  std::vector<Operation> program={input,fence,descriptor,output};
  c.set_stage(program,0,0,"synthetic_staged_mac");
  Kernel kernel;kernel.reads=16*1024;kernel.nominal=1;kernel.start=0;
  c.begin_kernel(&kernel,0);
  Clock now=0,stage_end=NEVER,export_issue=NEVER;
  size_t iterations=0;
  std::vector<Clock> gpu_issues;
  std::set<Clock> private_issues;
  while(!(c.stage_finished(now)&&c.kernel_finished()&&now>=kernel.nominal)) {
    require(++iterations<100000,"concurrency fixture did not terminate");
    const Clock at=now,private_before=c.private_free;
    const auto gpu_before=std::accumulate(c.gpu_command_counts.begin(),c.gpu_command_counts.end(),uint64_t{0});
    const bool exported=c.top.dispatched[3];
    now=c.step(at,true);
    require(now>=at && now<100000,"nonmonotonic or excessive fixture time");
    const auto gpu_after=std::accumulate(c.gpu_command_counts.begin(),c.gpu_command_counts.end(),uint64_t{0});
    if(gpu_after>gpu_before)gpu_issues.push_back(at);
    if(c.private_free>private_before)private_issues.insert(at);
    if(!exported&&c.top.dispatched[3])export_issue=at;
    if(stage_end==NEVER&&c.stage_finished(now))stage_end=now;
  }
  c.checker.advance(now);
  const Clock staging_end=c.top.done[0],compute_end=c.top.done[2],export_end=c.top.done[3];
  require(stage_end>=export_end && now>=stage_end && now>=c.gpu_dq_end && now>=c.pim_dq_end,
          "completion excluded a PIM or GPU tail");
  require(staging_end>0 && compute_end>staging_end && export_end>export_issue,
          "staging/compute/export collapsed to issue-only timing");
  size_t during_compute=0,same_ck=0;
  for(Clock at:gpu_issues) {
    require(at>=staging_end,"GPU issued during input staging or its DQ tail");
    require(at<compute_end||at>=export_end,"GPU issued during output export or its DQ tail");
    if(at>=staging_end && at<compute_end)++during_compute;
    if(private_issues.count(at))++same_ck;
  }
  require(during_compute>0,"GPU made no native progress inside private PIM compute");
  // This longer workload's issue phases may interleave without exact-CK
  // coincidence. Actual GPU progress inside the private window is required;
  // a separate two-ready-path fixture proves exact-CK issue capability.
  require(c.pim_write_bytes==32 && c.pim_read_bytes==32,"incorrect PIM external bytes");
  require(c.gpu_read_bytes==1024*32 && c.gpu_write_bytes==0,"lost GPU traffic");
  const auto& s=c.checker.stats.at(0);
  require(s.pim_read+s.pim_write==64 && s.gpu_read==1024*32,"checker byte accounting mismatch");
  const auto rd=c.info.at(c.device.m_spec->get_command_id("RD"));
  const auto wr=c.info.at(c.device.m_spec->get_command_id("WR"));
  const auto desc=c.info.at(c.device.m_spec->get_command_id("MAC_RUN"));
  require(s.dq==1025*rd.burst+wr.burst,"checker failed to count all actual DQ tails");
  require(s.ca==1025*rd.external_ca+wr.external_ca+desc.external_ca,
          "private compute added external CA occupancy");
  std::cout<<"{\"test\":\"staging_private_compute_export\",\"passed\":true,\"staging_end_ck\":"<<staging_end
           <<",\"compute_end_ck\":"<<compute_end<<",\"export_issue_ck\":"<<export_issue<<",\"export_end_ck\":"<<export_end
           <<",\"pim_end_ck\":"<<stage_end<<",\"e2e_end_ck\":"<<now<<",\"gpu_commands_during_compute\":"<<during_compute
           <<",\"private_external_same_ck\":"<<same_ck<<",\"checked_commands\":"<<c.checker.checked<<"}\n";
}

static void same_tick_paths(const ConfigNode& spec) {
  PIMController c(0,spec,spec["metadata"]);c.fused=true;warm(c);
  auto descriptor=op("MAC_RUN",0);descriptor.addr={0,1,-1,-1,0,0};
  descriptor.children.push_back(op("LOCALRD",0));
  std::vector<Operation> program={descriptor};
  c.set_stage(program,0,0,"two_ready_paths");
  require(c.step(0,false)==0 && c.descriptor_index==0,"descriptor fixture did not issue");
  const Clock ready=c.descriptor_ready;
  Kernel kernel;kernel.reads=16;kernel.nominal=1;kernel.start=ready;
  c.begin_kernel(&kernel,ready);
  require(c.step(ready,true)==ready,"private issue advanced the clock");
  require(c.inner.dispatched[0] && c.private_free>ready && c.gpu_read_bytes==0,
          "private path was not selected first");
  require(c.step(ready,true)==ready && c.gpu_read_bytes==32,
          "ready GPU path could not issue in the same CK as private PIM");
  Clock now=ready;
  size_t iterations=0;
  while(!(c.stage_finished(now)&&c.kernel_finished())) {
    require(++iterations<1000,"two-ready-path fixture did not drain");
    now=c.step(now,true);
  }
  c.checker.advance(now);
  require(now>=c.gpu_dq_end && now>=c.top.tail,"same-CK fixture dropped a completion tail");
  std::cout<<"{\"test\":\"two_ready_paths_same_ck\",\"passed\":true,\"issue_ck\":"<<ready
           <<",\"drained_ck\":"<<now<<",\"gpu_read_bytes\":"<<c.gpu_read_bytes<<"}\n";
}

int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: native_gio_concurrency spec.json");
    auto spec=Config::parse_config_file(argv[1]);gio_boundaries(spec);staged_window(spec);same_tick_paths(spec);
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

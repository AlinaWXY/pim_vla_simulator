#include "ramulator/controller/impl/pim_resources.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace PimVla;

struct Op {
  std::string name,dest,source;
  std::vector<int> addr{0,1,0,0,0,0};
  int slot=0,dest_slot=-1,source_bank=0,tensor_fragment=0,rows=1;
  int output_columns=1,output_column=0,fragment=0,output_tile=0,kv_token=0;
  int elements=16,result_slot=0,token=0;
  bool release_result=true;
  std::vector<ResourceRegion> source_regions;
};
template<class F> void rejects(F f,const std::string& text) {
  try {f();assert(false && "expected resource guard");}
  catch(const std::runtime_error& e) {assert(std::string(e.what()).find(text)!=std::string::npos);}
}
Op make(const std::string& name) {Op o;o.name=name;return o;}

int main(int argc,char** argv) {
  if(argc==2) {
    std::ifstream input(argv[1]);assert(input.good());
    PimResources resource;PimResources::Time now=0;std::string line;int count=0;
    while(std::getline(input,line)) {
      std::istringstream row(line);std::string label;row>>label;
      if(label=="stage") {row>>label;resource.start_stage(label,now);continue;}
      Op op;op.name=label;int regions=0;
      row>>op.dest>>op.source>>op.addr[2]>>op.addr[3]>>op.slot>>op.dest_slot
         >>op.source_bank>>op.tensor_fragment>>op.rows>>op.output_columns>>op.output_column
         >>op.fragment>>op.output_tile>>op.kv_token>>op.elements>>op.result_slot>>op.token
         >>op.release_result>>regions;
      if(op.dest=="-")op.dest.clear();
      if(op.source=="-")op.source.clear();
      for(int i=0;i<regions;++i) {ResourceRegion r;row>>r.bank>>r.byte_offset>>r.bytes;op.source_regions.push_back(r);}
      try {now=resource.ready(op,now);resource.issue(op,now,now+1);++now;++count;}
      catch(const std::exception& e) {std::cerr<<"request "<<count<<" "<<line<<"\n"<<e.what()<<"\n";return 1;}
    }
    std::cout<<"PimResources accepts "<<count<<" generated requests\n";return 0;
  }
  // A legal input broadcast does not become consumable until its completion.
  PimResources r;r.start_stage("qkv",0);
  auto mac=make("MACAB");
  rejects([&]{r.ready(mac,0);},"unfilled SRF");
  auto wr=make("WRGB");r.issue(wr,0,14);
  assert(r.ready(mac,1)==14);
  rejects([&]{r.issue(mac,13,17);},"before resource readiness");
  r.issue(mac,14,18);
  // Output word production and rank DQ tail are independent.
  auto rd=make("RDRES");rd.source_regions={{0,0,4}};
  assert(r.ready(rd,15)==18);
  auto missing=rd;missing.source_regions={{0,4,4}};
  rejects([&]{r.ready(missing,18);},"unproduced");
  r.issue(rd,18,43);
  assert(r.ready(mac,19)==43);
  rejects([&]{r.issue(mac,19,23);},"before resource readiness");
  rejects([&]{r.ready(rd,43);},"released");
  r.issue(mac,43,47); // reinitialize after the old DQ tail is gone

  // Fragment accumulation needs both the current input and its previous sum.
  auto f2=mac;f2.fragment=2;
  auto input2=wr;input2.tensor_fragment=2;
  r.issue(input2,47,61);
  rejects([&]{r.ready(f2,61);},"preceding fragment");
  auto f1=mac;f1.fragment=1;
  rejects([&]{r.ready(f1,61);},"different SRF fragment");
  auto input1=wr;input1.tensor_fragment=1;r.issue(input1,61,75);
  r.issue(f1,75,79);

  // A naked LOCALRD into a scratch must never silently fill an SRF slot.
  PimResources scratch;scratch.start_stage("qkv",0);
  auto local=make("LOCALRD");local.dest="rmw_scratch";scratch.issue(local,0,4);
  rejects([&]{scratch.ready(mac,4);},"unfilled SRF");
  local.dest_slot=0;scratch.issue(local,4,8);
  auto broadcast=make("BCAST");assert(scratch.ready(broadcast,5)==8);
  scratch.issue(broadcast,8,10);scratch.issue(mac,10,14);
  auto too_big=mac;too_big.rows=16;too_big.output_columns=33;
  rejects([&]{scratch.ready(too_big,14);},"2KiB");
  auto bad_slot=wr;bad_slot.slot=32;
  rejects([&]{scratch.ready(bad_slot,14);},"1KiB");

  // Resident Q is intentionally preserved across global stage boundaries.
  PimResources q;q.start_stage("q_exchange_export",0);
  for(int bank=0;bank<16;++bank) for(int slot=0;slot<16;++slot) {
    auto load=make("LOCALRD");load.addr[2]=bank/4;load.addr[3]=bank%4;load.dest_slot=slot;
    q.issue(load,0,1);
  }
  q.start_stage("q_exchange_import",1);q.start_stage("qkt",1);
  auto qk=make("MACQK");qk.rows=16;q.issue(qk,1,17);
  auto reduce=make("REDUCE");assert(q.ready(reduce,2)==17);q.issue(reduce,17,19);
  auto score=make("RDRES");score.source="reduction_output";
  q.issue(score,19,44);score.result_slot=1;q.issue(score,23,48);
  qk.kv_token=1;assert(q.ready(qk,24)==48);q.issue(qk,48,64);
  reduce.kv_token=1;q.issue(reduce,64,66);
  qk.kv_token=2;
  rejects([&]{q.ready(qk,66);},"unconsumed reduction");

  // RoPE loads remain separate from GEMV input and share one rank execution unit.
  PimResources rope;rope.start_stage("rope",0);
  auto execute=make("ROPE_EXEC");
  rejects([&]{rope.ready(execute,0);},"without its input");
  auto coefficient=make("LOCALRD");coefficient.dest="rope_coefficient";rope.issue(coefficient,0,2);
  auto input=make("LOCALRD");input.dest="rope_input";rope.issue(input,0,2);
  rope.issue(execute,2,12);
  auto store=make("LOCALWR");store.source="rope_output";rope.issue(store,12,16);
  rejects([&]{rope.ready(store,16);},"released output");
  rejects([&]{rope.ready(mac,16);},"unfilled SRF");
  std::cout<<"PimResources lifecycle tests passed\n";
}

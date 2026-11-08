#include "ramulator/controller/impl/pim_model.h"
namespace PimVla {
void BusChecker::advance(Clock now) {
  require(now>=account_clock,"bus checker clock went backward");
  auto overlap=[&](Clock start,Clock end){return std::max<Clock>(0,std::min(now,end)-std::max(account_clock,start));};
  auto& s=stats[account_stage];s.ca+=overlap(external_start,external_end);
  for(const auto& b:bursts)s.dq+=overlap(b.start,b.end);
  account_clock=now;
}
void BusChecker::retire(Clock now) {
  advance(now);
  // Retain the preceding burst for direction/rank switching checks.
  while(bursts.size()>1 && bursts[1].end<=now) bursts.pop_front();
}
void BusChecker::observe(int cmd,const AddrVec_t& addr,Clock now,int stage) {
  advance(now);
  const auto& m=info->at(cmd);
  auto& s=stats[stage]; ++s.commands; ++checked;
  if(m.external_ca) {
    if(now<external_end)throw std::runtime_error("CA overlap at CK "+std::to_string(now));
    external_start=now;external_end=now+m.external_ca;
  }
  if(m.private_ca) {
    require(now>=private_end,"rank private CA overlap");
    private_end=now+m.private_ca;
  }
  if(!m.direction) return;
  Burst b{now+m.latency,now+m.latency+m.burst,m.direction,addr[1],cmd,stage};
  // Assertion only: never consulted by the scheduler. DQ ordering is causal.
  if(!bursts.empty()) {
    const auto& old=bursts.back();
    if(b.start<old.end)throw std::runtime_error("DQ overlap or return reordering at CK "+std::to_string(now)+" "+spec->command_names[old.command]+" -> "+spec->command_names[cmd]);
    // Physical interface minima are properties of the direction, not a
    // second command timing table. Values come from the exported spec.
    Clock gap=0;
    if(old.direction=='R' && b.direction=='W') gap=spec->get_timing_value("nDQ_RTW");
    if(old.direction=='W' && b.direction=='R') gap=spec->get_timing_value("nDQ_WTR");
    if(old.direction==b.direction && old.rank!=b.rank)
      gap=old.direction=='R'?spec->get_timing_value("nCS"):spec->get_timing_value("nDQ_RANK_WTW");
    if(b.start-old.end<gap)throw std::runtime_error("DQ turnaround violation at CK "+std::to_string(now));
  }
  bursts.push_back(b);
  (m.direction=='R'?s.read:s.write)+=32;
  if(addr[1]==0)(m.direction=='R'?s.gpu_read:s.gpu_write)+=32;
  else (m.direction=='R'?s.pim_read:s.pim_write)+=32;
}
}

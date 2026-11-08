// Evidence-sized uniform request replay. No bandwidth-pressure, queue-depth,
// or read/write-ratio knobs. PIM reservations have priority. The GPU row-hit
// assumption is explicit: this is a bus contention model, not an address trace.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using I = int64_t;
constexpr I CK=15008, BL=2*CK, CL=23*CK, CWL=12*CK;
constexpr I RFC=224*CK, REFI=4164*CK;
struct Event { int32_t channel, kind, rank, bg; I a,b; };
// kind: CA=0, read DQ=1, write DQ=2, rank0 refresh=3, rank1 refresh=4.
struct Slot { I a,b; int rank=0,bg=-1; bool write=false; };
struct Channel { std::vector<Slot> ca,dq; std::array<std::vector<Slot>,2> ref; I phase=0; };
struct Calendar { I end; std::vector<Channel> channels; };
struct Request { I issue,done; bool write; int bg; };
struct Plan { I duration,span; std::vector<Request> q; };
struct Result { I finish,first_issue,requests,ca_delay,dq_delay,refresh_delay,
                backpressure,max_inflight,solo_minimum,calibrated_span,
                last_ca,last_dq_a,last_dq_b,last_write,last_bg; };
static std::vector<Calendar> calendars;
static std::vector<Plan> plans;
static std::string error;
static I edge(I t) { return ((t+CK-1)/CK)*CK; }
static I guard(const Slot& a,const Slot& b) {
  if(a.rank!=b.rank) return (a.write ? 1 : 2)*CK;
  const bool same=a.bg<0 || b.bg<0 || a.bg==b.bg;
  if(a.write && !b.write) return (23+(same?13:7)+(same?2:0))*CK;
  if(!a.write && b.write) return (same?4:2)*CK;
  return (same?2:0)*CK;
}
static I hit(const std::vector<Slot>& v,I a,I b) {
  auto it=std::lower_bound(v.begin(),v.end(),a,[](auto s,I x){return s.b<=x;});
  return it!=v.end() && it->a<b ? it->b : a;
}
static I next_ref(const Channel& c,int rank,I after,I end) {
  auto& v=c.ref[rank];
  I phase=v.empty()?REFI+rank*CK+c.phase:v.back().a;
  I first=phase+std::max<I>(0,(end-phase+REFI-1)/REFI)*REFI;
  if(first<after) first+=((after-first+REFI-1)/REFI)*REFI;
  return first;
}
// All clocks here are relative to a template's memory-edge origin.
static I reserve(const Channel& c,I end,I ready,bool write,int bg,
                 const Slot* prior,I prior_ca,Result& stats,bool solo=false) {
  I latency=write?CWL:CL;
  auto align=[&](I t){return edge(t-c.phase)+c.phase;};
  I t=align(std::max<I>(ready,prior_ca+CK));
  Slot candidate{0,0,0,bg,write};
  if(prior) t=align(std::max(t,prior->b+guard(*prior,candidate)-latency));
  for(;;) {
    I old=t;
    if(!solo) {
      I r=hit(c.ca,t,t+CK);
      for(int rank=0;rank<2;++rank) {
        I ref=next_ref(c,rank,t-CK+1,end);
        if(ref<t+CK && ref+CK>t) r=std::max(r,ref+CK);
      }
      if(r>t) { stats.ca_delay+=r-t; t=align(r); continue; }
      // Rank blackout plus completion/write recovery before a fixed REF.
      I stop=t+latency+BL+(write?37*CK:0);
      r=hit(c.ref[0],t,stop);
      I ref=next_ref(c,0,t-RFC+1,end);
      if(ref<stop && ref+RFC>t) r=std::max(r,ref+RFC);
      if(r>t) { stats.refresh_delay+=r-t; t=align(r); continue; }
      candidate.a=t+latency; candidate.b=candidate.a+BL;
      // Search the neighborhood expanded by the largest direction guard.
      auto it=std::lower_bound(c.dq.begin(),c.dq.end(),candidate.a-40*CK,
                              [](auto s,I x){return s.b<=x;});
      for(;it!=c.dq.end() && it->a<candidate.b+40*CK;++it) {
        const bool before=candidate.b+guard(candidate,*it)<=it->a;
        const bool after=it->b+guard(*it,candidate)<=candidate.a;
        if(!before && !after) {
          I n=align(it->b+guard(*it,candidate)-latency);
          if(n>t) {stats.dq_delay+=n-t;t=n;break;}
        }
      }
      if(t!=old) continue;
    }
    return t;
  }
}
// Two uniform arrival streams. Keep serving the current direction while a
// request of that direction is ready; otherwise serve the oldest ready stream.
// This introduces no fixed batching size or outstanding-request limit.
static Plan build(I duration,I reads,I writes,I span) {
  Plan p{duration,span,{}}; p.q.reserve(reads+writes);
  I ri=0,wi=0,ca=-CK,done=0; bool dir=false; Slot prior{}; bool have=false;
  Channel empty; Result stats{};
  auto arrival=[&](I i,I n){return n<=1?I(0):I((__int128(span)*i)/(n-1));};
  while(ri<reads || wi<writes) {
    I ar=ri<reads?arrival(ri,reads):std::numeric_limits<I>::max()/4;
    I aw=wi<writes?arrival(wi,writes):std::numeric_limits<I>::max()/4;
    I ready=std::max<I>(ca+CK, std::min(ar,aw));
    if((dir?aw:ar)>ready) dir=aw<ar;
    I release=dir?aw:ar;
    int bg=int((ri+wi)%4); // Uniform striping across the four native bank groups.
    // The isolated reference includes the native periodic rank refresh as well.
    ca=reserve(empty,0,release,dir,bg,have?&prior:nullptr,ca,stats);
    done=ca+(dir?CWL:CL)+BL;
    p.q.push_back({ca,done,dir,bg});
    prior={ca+(dir?CWL:CL),done,0,bg,dir};have=true;
    if(dir)++wi;else ++ri;
  }
  return p;
}
extern "C" const char* uniform_error(){return error.c_str();}
extern "C" int uniform_add(I end,int n,const Event* es,int count) {
  try {
    if(end<0||n<1||n>16)throw std::runtime_error("invalid calendar");
    Calendar c{end,std::vector<Channel>(n)};
    for(int i=0;i<count;++i) {
      const auto& e=es[i];
      if(e.channel<0||e.channel>=n||e.kind<0||e.kind>4||e.b<=e.a)
        throw std::runtime_error("invalid reservation");
      Slot s{e.a,e.b,e.rank,e.bg,e.kind==2};auto& ch=c.channels[e.channel];
      if(e.kind==0)ch.ca.push_back(s);
      else if(e.kind<=2)ch.dq.push_back(s);
      else ch.ref[e.kind-3].push_back(s);
    }
    for(auto& ch:c.channels) {
      auto sort=[](auto& v){std::sort(v.begin(),v.end(),[](auto a,auto b){return a.a<b.a;});};
      sort(ch.ca);sort(ch.dq);for(auto& v:ch.ref)sort(v);
      if(!ch.ca.empty())ch.phase=(ch.ca.front().a%CK+CK)%CK;
      for(auto s:ch.ca)if((s.a-ch.phase)%CK!=0)
        throw std::runtime_error("calendar CA events do not share a clock phase");
    }
    calendars.push_back(std::move(c));return int(calendars.size())-1;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int uniform_plan(I duration,I reads,I writes,Result* info) {
  try {
    if(duration<=0||reads<0||writes<0)throw std::runtime_error("invalid workload");
    Plan low=build(duration,reads,writes,0);
    I minimum=low.q.empty()?0:low.q.back().done;
    *info={};info->solo_minimum=minimum;
    if(minimum>duration)throw std::runtime_error("Measured bytes cannot fit retained duration even at zero arrival span");
    I a=0,b=duration;
    // Find a uniform arrival span that drains within the measured duration.
    // We keep all bytes; only remove the protocol tail from the arrival span.
    // Direction policy can cause small non-monotonic changes; verify the final
    // plan and fall back to the proven-feasible zero-span plan if needed.
    while(a<b) {
      I m=(a+b+1)/2; auto q=build(duration,reads,writes,m);
      if(q.q.empty()||q.q.back().done<=duration)a=m;else b=m-1;
    }
    Plan p=build(duration,reads,writes,a);
    if(!p.q.empty()&&p.q.back().done>duration)throw std::runtime_error("uniform span calibration failed");
    info->calibrated_span=a;info->requests=reads+writes;
    plans.push_back(std::move(p));return int(plans.size())-1;
  } catch(const std::exception& e){error=e.what();return -1;}
}
// Fixed uniform arrivals over the retained kernel span, with the hardware
// return tail removed. No request trimming or rate fitting. If synchronous
// write-through demand drains later than the measured kernel, retain that
// reference-model discrepancy explicitly rather than pretending it fits.
extern "C" int uniform_stream_plan(I measured,I reads,I writes,Result* info) {
  try {
    if(measured<=0||reads<0||writes<0)throw std::runtime_error("invalid stream workload");
    I span=std::max<I>(0,measured-CL-BL);
    auto p=build(measured,reads,writes,span);
    I finish=p.q.empty()?measured:std::max(measured,p.q.back().done);
    p.duration=finish;
    *info={};info->finish=finish;info->requests=reads+writes;info->calibrated_span=span;
    plans.push_back(std::move(p));return int(plans.size())-1;
  } catch(const std::exception& e){error=e.what();return -1;}
}
static Result run_channel(const Channel& c,I end,I start,const Plan& p,const Result* previous) {
  Result out{};out.first_issue=-1;out.calibrated_span=p.span;
  I shift=0,ca=start-CK;Slot prior{};bool have=false;
  if(previous&&previous->requests) {
    ca=previous->last_ca;
    prior={previous->last_dq_a,previous->last_dq_b,0,int(previous->last_bg),bool(previous->last_write)};
    have=true;
  }
  // A nominal completion is a dependency checkpoint. Delayed returns stop the
  // virtual progress clock and consequently postpone later request releases.
  using Pair=std::pair<I,I>;
  std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> due;
  std::priority_queue<I,std::vector<I>,std::greater<I>> flight;
  for(auto q:p.q) {
    while(!due.empty()&&due.top().first<=q.issue) {
      auto [deadline,actual]=due.top();due.pop();
      shift=std::max(shift,actual-start-deadline);
    }
    I t=reserve(c,end,start+q.issue+shift,q.write,q.bg,have?&prior:nullptr,ca,out);
    I done=t+(q.write?CWL:CL)+BL;
    if(out.first_issue<0)out.first_issue=t;
    while(!flight.empty()&&flight.top()<=t)flight.pop();
    flight.push(done);out.max_inflight=std::max<I>(out.max_inflight,flight.size());
    due.push({q.done,done});prior={t+(q.write?CWL:CL),done,0,q.bg,q.write};have=true;ca=t;
    ++out.requests;
  }
  while(!due.empty()) {
    auto [deadline,actual]=due.top();due.pop();
    shift=std::max(shift,actual-start-deadline);
  }
  out.backpressure=shift;out.finish=start+p.duration+shift;
  out.last_ca=ca;out.last_dq_a=prior.a;out.last_dq_b=prior.b;
  out.last_write=prior.write;out.last_bg=prior.bg;
  return out;
}
extern "C" int uniform_run(int cal,int plan,I start,int channel,const Result* previous,Result* out) {
  try {
    const auto& c=calendars.at(cal);const auto& p=plans.at(plan);
    *out=run_channel(c.channels.at(channel),c.end,start,p,previous);return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int uniform_probe(int cal,I ready,int write,int bg,int channel,Result* out) {
  try {
    const auto& c=calendars.at(cal);*out={};
    I t=reserve(c.channels.at(channel),c.end,ready,write,bg,nullptr,ready-CK,*out);
    out->first_issue=t;out->finish=t+(write?CWL:CL)+BL;return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}
// A resource-capacity diagnostic only; never used to generate GPU work.
extern "C" int uniform_capacity(int cal,int write,int channel,Result* out) {
  try {
    const auto& c=calendars.at(cal);*out={};out->first_issue=-1;
    I ca=-CK;Slot prior{};bool have=false;
    for(;;) {
      int bg=int(out->requests%4);
      I t=reserve(c.channels.at(channel),c.end,0,write,bg,have?&prior:nullptr,ca,*out);
      I done=t+(write?CWL:CL)+BL;
      if(done>c.end)break;
      if(out->first_issue<0)out->first_issue=t;
      ++out->requests;out->finish=done;
      ca=t;prior={t+(write?CWL:CL),done,0,bg,bool(write)};have=true;
    }
    return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}

// Parameterized, PIM-priority reservation replay. Not a full DRAM row simulator.
// Time units: 1/16 ps. GPU rank 0, PIM rank 1; each request is one 32-byte burst.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using I=int64_t;
constexpr I CK=15008, BL=2*CK, CL=23*CK, CWL=12*CK, RFC=224*CK, REFI=4164*CK;
struct Interval { I a,b; bool operator==(const Interval&) const = default; };
struct Event { int32_t channel,kind; I a,b; }; // 0 DQ, 1 shared CA, 2 rank0 refresh, 3 rank1 refresh
struct Result {
  I finish,requests,ca_wait,dq_wait,refresh_wait,queue_wait,calibration_trim,max_inflight;
  I channel_finish[16];
};
struct Channel { std::array<std::vector<Interval>,4> v; bool operator==(const Channel&) const = default; };
struct Template { I end; std::vector<Channel> channels; std::vector<int> equivalent; };
static std::vector<Template> templates;
static std::string last_error;
static void canonical(std::vector<Interval>& v) {
  std::sort(v.begin(),v.end(),[](auto x,auto y){return x.a<y.a;});
  std::vector<Interval> r;
  for(auto x:v) { if(x.b<=x.a)throw std::runtime_error("empty interval");
    if(!r.empty() && x.a<=r.back().b)r.back().b=std::max(r.back().b,x.b);else r.push_back(x); }
  v.swap(r);
}
static I collision(const std::vector<Interval>& v,I a,I b,I pad=0) {
  auto it=std::lower_bound(v.begin(),v.end(),a-pad,[](auto x,I t){return x.b<=t;});
  return it!=v.end() && it->a-pad<b ? it->b+pad : a;
}
static I next_refresh(const Channel& c,int rank,I after,I template_end) {
  const auto& v=c.v[rank+2];
  I phase=v.empty()?REFI+rank*CK:v.back().a;
  I n=std::max<I>(0,(template_end-phase+REFI-1)/REFI);
  I first=phase+n*REFI;
  if(first<after)first+=((after-first+REFI-1)/REFI)*REFI;
  return first;
}
static I refresh_collision(const Channel& c,I a,I b,I template_end) {
  I r=collision(c.v[2],a,b);
  if(r>a)return r;
  I next=next_refresh(c,0,a-RFC+1,template_end);
  return next<b && next+RFC>a?next+RFC:a;
}
static I ca_collision(const Channel& c,I t,I template_end) {
  I r=collision(c.v[1],t,t+CK);
  if(r>t)return r;
  // During the template, recorded shared CA includes both ranks' REFab pulses.
  for(int rank=0;rank<2;++rank) {
    I next=next_refresh(c,rank,t-CK+1,template_end);
    if(next<t+CK && next+CK>t)r=std::max(r,next+CK);
  }
  return r;
}
extern "C" const char* replay_error(){return last_error.c_str();}
extern "C" int replay_add(I end,int n_channels,const Event* events,int count) {
  try {
    if(end<0 || n_channels<1 || n_channels>16)throw std::runtime_error("invalid template shape");
    Template t{end,std::vector<Channel>(n_channels),{}};
    for(int i=0;i<count;++i) {
      const auto& e=events[i];if(e.channel<0||e.channel>=n_channels||e.kind<0||e.kind>3)throw std::runtime_error("invalid event");
      t.channels[e.channel].v[e.kind].push_back({e.a,e.b});
    }
    for(auto& c:t.channels)for(auto& v:c.v)canonical(v);
    for(int i=0;i<n_channels;++i) {
      int first=i;for(int j=0;j<i;++j)if(t.channels[i]==t.channels[j]){first=j;break;}
      t.equivalent.push_back(first);
    }
    templates.push_back(std::move(t));return templates.size()-1;
  } catch(const std::exception& e){last_error=e.what();return -1;}
}
// A continuous streaming injection model, with finite per-channel outstanding requests.
// Requests can span any number of PIM DQ/CA gaps. No whole-kernel window test exists.
static Result channel_run(const Channel& c,I template_end,I start,I duration,I n,int q,int write_percent,I pad,bool solo) {
  Result r{};r.finish=start+duration;
  // Reserve one RFC plus read/write tail latency in the synthetic injection span.
  I tail=RFC+CL+BL+16*CK,span=std::max<I>(0,duration-tail);
  std::vector<I> ring(q,0);I previous_ca=start-2*CK,previous_dq=start,active_start=0;bool previous_write=false;
  for(I j=0;j<n;++j) {
    const bool write=(j%80)>=80*(100-write_percent)/100;
    I latency=write?CWL:CL;
    I release=start+(n<=1?0:span*j/(n-1));
    I queue_ready=j>=q?ring[j%q]:start;
    I t=std::max({release,queue_ready,previous_ca+2*CK});
    r.queue_wait+=std::max<I>(0,queue_ready-release);
    // Explicit direction turnaround. Row activation/precharge and row locality are not modeled.
    I turn=j && write!=previous_write ? (previous_write?(7*CK+CL):2*CK) : 0;
    t=std::max(t,previous_dq+turn-latency);
    for(;;) {
      I old=t;
      I ready=refresh_collision(c,t,t+latency+BL+(write?37*CK:0),template_end);
      if(ready>t){r.refresh_wait+=ready-t;t=ready;continue;}
      ready=ca_collision(c,t,template_end);
      if(ready>t){r.ca_wait+=ready-t;t=ready;continue;}
      if(!solo) {
        ready=collision(c.v[0],t+latency,t+latency+BL,pad);
        if(ready>t+latency){I nt=ready-latency;r.dq_wait+=nt-t;t=nt;continue;}
      }
      if(t==old)break;
    }
    I done=t+latency+BL;
    if(!solo) {
      assert(collision(c.v[0],t+latency,done,pad)==t+latency);
      assert(collision(c.v[1],t,t+CK)==t);
    }
    assert(refresh_collision(c,t,done+(write?37*CK:0),template_end)==t);
    assert(done>=previous_dq && t>=previous_ca+CK);
    assert(j<q || t>=ring[j%q]);
    while(active_start<j && ring[active_start%q]<=t)++active_start;
    r.max_inflight=std::max(r.max_inflight,j-active_start+1);
    ring[j%q]=done;previous_ca=t;previous_dq=done;previous_write=write;
    r.finish=std::max(r.finish,done);++r.requests;
  }
  return r;
}
extern "C" int replay_kernel(int handle,I start,I duration,double pressure,int q,int write_percent,int pad_ck,Result* result) {
  try {
    if(handle<0||handle>=int(templates.size())||duration<=0||start<0||pressure<0||pressure>1||q<1||q>4096||write_percent<0||write_percent>100||pad_ck<0)
      throw std::runtime_error("invalid replay argument");
    const auto& t=templates[handle];Result out{};out.finish=start+duration;
    I span=std::max<I>(0,duration-(RFC+CL+BL+16*CK));
    I n=std::max<I>(0,std::floor(pressure*double(span)/BL));
    // Calibrate ONCE at a fixed solo refresh phase. Identical GPU kernels must
    // carry identical byte demand in every layout, regardless of their start time.
    static std::map<std::tuple<I,int,int,int>,I> calibrated;
    const auto key=std::make_tuple(duration,int(std::llround(pressure*1000000)),q,write_percent);
    if(!calibrated.count(key)) {
      Channel isolated;I count=n;
      if(channel_run(isolated,0,0,duration,count,q,write_percent,0,true).finish>duration) {
        I low=0,high=count;
        while(low<high){I mid=(low+high+1)/2;
          if(channel_run(isolated,0,0,duration,mid,q,write_percent,0,true).finish<=duration)low=mid;else high=mid-1;}
        count=low;
      }
      if(channel_run(isolated,0,0,duration,count,q,write_percent,0,true).finish>duration)
        throw std::runtime_error("solo calibration failed");
      calibrated[key]=count;
    }
    I count=calibrated.at(key);
    std::vector<Result> done(t.channels.size());
    for(int ch=0;ch<int(t.channels.size());++ch) {
      if(t.equivalent[ch]!=ch)done[ch]=done[t.equivalent[ch]];
      else {
        done[ch]=channel_run(t.channels[ch],t.end,start,duration,count,q,write_percent,pad_ck*CK,false);
        done[ch].calibration_trim=n-count;
      }
      const auto& r=done[ch];out.finish=std::max(out.finish,r.finish);out.channel_finish[ch]=r.finish;
      out.requests+=r.requests;out.ca_wait+=r.ca_wait;out.dq_wait+=r.dq_wait;out.refresh_wait+=r.refresh_wait;
      out.queue_wait+=r.queue_wait;out.calibration_trim+=r.calibration_trim;out.max_inflight=std::max(out.max_inflight,r.max_inflight);
    }
    *result=out;return 0;
  } catch(const std::exception& e){last_error=e.what();return -1;}
}
// Exact single-burst probe for resource tests; no demand calibration.
extern "C" int replay_probe(int handle,I start,I duration,int channel,Result* out) {
  try {const auto& t=templates.at(handle);*out=channel_run(t.channels.at(channel),t.end,start,duration,1,1,0,0,false);return 0;}
  catch(const std::exception& e){last_error=e.what();return -1;}
}

// Causal bus replay: only released PIM commands and committed GPU returns
// participate in arbitration. Native command gaps remain dependency lower
// bounds; this is not a replacement for the native bank/row state machine.
#include "uniform_transactions.cpp"
#include <deque>
struct OnlineJob {
  int32_t channel,rank,bg,write,array_access;
  I ready,ca_length,dq_latency,dq_length;
};
#include "sequential_timing.h"
struct OnlineProgram { I end; int mode; std::vector<std::vector<OnlineJob>> jobs; int refresh_calendar; };
struct CommittedDQ { Slot slot; bool gpu; };
struct OnlineState {
  int program,channel; size_t cursor=0;
  I clock=0,shift=0,ca_until=0,pim_wait=0,pim_requests=0;
  std::deque<CommittedDQ> dq;
  std::deque<CommittedTransfer> transfers;
  bool external_port_owned=false;
};
struct OnlineResult {
  I finish,first_issue,requests,memory_stall,pim_end,pim_wait,pim_requests;
};
static std::vector<OnlineProgram> online_programs;
static std::vector<OnlineState> online_states;
constexpr I INF=std::numeric_limits<I>::max()/8;

static I align_online(const Channel& c,I x) { return edge(x-c.phase)+c.phase; }
static bool array_burst(const OnlineJob& j) {return j.array_access && j.dq_length;}
static I legal_time(OnlineState& s,I ready,const OnlineJob& j,bool gpu) {
  const auto& p=online_programs[s.program];
  const auto& c=calendars[p.refresh_calendar].channels[s.channel];
  I t=align_online(c,std::max(ready,s.clock));
  if(j.rank==-2 || j.rank==-3) {
    // Whole external-I/O epochs: drain already committed returns before
    // transferring ownership; no fine-grained GPU/PIM race inside the epoch.
    for(const auto& q:s.dq)t=std::max(t,q.slot.b);
    return align_online(c,t);
  }
  if(j.ca_length)t=align_online(c,std::max(t,s.ca_until));
  if(j.dq_length)for(const auto& old:s.transfers)
    if(gpu || old.gpu)t=align_online(c,std::max(t,old.issue+transfer_issue_gap(old.job,j)));
  for(;;) {
    I before=t;
    if(j.ca_length) {
      I next=hit(c.ca,t,t+j.ca_length);
      for(int rank=0;rank<2;++rank) {
        I ref=next_ref(c,rank,t-CK+1,p.end);
        if(ref<t+j.ca_length && ref+CK>t)next=std::max(next,ref+CK);
      }
      if(next>t){t=align_online(c,next);continue;}
    }
    if(j.array_access && j.rank>=0) {
      I stop=t+std::max<I>(CK,j.ca_length);
      if(array_burst(j))stop=t+j.dq_latency+j.dq_length+(j.write?37*CK:0);
      I next=hit(c.ref[j.rank],t,stop);
      I ref=next_ref(c,j.rank,t-RFC+1,p.end);
      if(ref<stop && ref+RFC>t)next=std::max(next,ref+RFC);
      if(next>t){t=align_online(c,next);continue;}
    }
    if(j.dq_length) {
      Slot candidate{t+j.dq_latency,t+j.dq_latency+j.dq_length,j.rank,j.bg,bool(j.write)};
      for(const auto& q:s.dq) {
        // PIM/PIM spacing is already encoded by the native causal program.
        I forward=(gpu||q.gpu)?guard(q.slot,candidate):0;
        I backward=(gpu||q.gpu)?guard(candidate,q.slot):0;
        bool earlier=candidate.b+backward<=q.slot.a;
        bool later=q.slot.b+forward<=candidate.a;
        if(!earlier&&!later) {
          t=align_online(c,q.slot.b+forward-j.dq_latency);
          break;
        }
      }
      if(t!=before)continue;
    }
    return t;
  }
}
static void commit_online(OnlineState& s,const OnlineJob& j,I t,bool gpu) {
  if(t<s.clock || (j.ca_length && t<s.ca_until))
    throw std::runtime_error("Committed CA overlaps an existing command");
  s.clock=t;
  if(j.rank==-2)s.external_port_owned=true;
  if(j.rank==-3)s.external_port_owned=false;
  if(j.ca_length)s.ca_until=t+j.ca_length;
  while(!s.transfers.empty() && s.transfers.front().issue+40*CK<=t)s.transfers.pop_front();
  if(j.dq_length) {
    for(const auto& old:s.transfers)
      if((gpu||old.gpu) && t<old.issue+transfer_issue_gap(old.job,j))
        throw std::runtime_error("Committed native transfer issue-spacing violation");
    s.transfers.push_back({j,t,gpu});
  }
  while(!s.dq.empty() && s.dq.front().slot.b+40*CK<=t)s.dq.pop_front();
  if(j.dq_length) {
    CommittedDQ q{{t+j.dq_latency,t+j.dq_latency+j.dq_length,j.rank,j.bg,bool(j.write)},gpu};
    for(const auto& old:s.dq) {
      I forward=(gpu||old.gpu)?guard(old.slot,q.slot):0;
      I backward=(gpu||old.gpu)?guard(q.slot,old.slot):0;
      if(!(old.slot.b+forward<=q.slot.a || q.slot.b+backward<=old.slot.a))
        throw std::runtime_error("Committed DQ/turnaround conflict");
    }
    auto at=std::upper_bound(s.dq.begin(),s.dq.end(),q.slot.a,
                             [](I x,const auto& a){return x<a.slot.a;});
    s.dq.insert(at,q);
  }
}
static I next_pim_time(OnlineState& s) {
  const auto& p=online_programs[s.program];auto& jobs=p.jobs[s.channel];
  if(s.cursor==jobs.size())return INF;
  const auto& j=jobs[s.cursor];
  return legal_time(s,j.ready+s.shift,j,false);
}
static void issue_pim_online(OnlineState& s,I t) {
  const auto& j=online_programs[s.program].jobs[s.channel][s.cursor];
  I delta=t-(j.ready+s.shift);
  if(delta<0)throw std::runtime_error("PIM issue precedes causal release");
  s.shift+=delta;s.pim_wait+=delta;s.pim_requests+=bool(j.dq_length);
  commit_online(s,j,t,false);++s.cursor;
}
static void advance_online(OnlineState& s,I until) {
  for(;;){I t=next_pim_time(s);if(t>until)break;issue_pim_online(s,t);}
  s.clock=std::max(s.clock,until);
}
static I finish_pim_online(const OnlineState& s) {
  const auto& p=online_programs[s.program];
  return s.cursor==p.jobs[s.channel].size()?p.end+s.shift:INF;
}
extern "C" int online_add(I end,int mode,int refresh_calendar,const OnlineJob* jobs,int count) {
  try {
    OnlineProgram p{end,mode,std::vector<std::vector<OnlineJob>>(16),refresh_calendar};
    for(int i=0;i<count;++i){auto j=jobs[i];
      if(j.channel<0||j.channel>=16||j.ca_length<0||j.dq_length<0||j.ready<0)
        throw std::runtime_error("Invalid online PIM job");
      p.jobs[j.channel].push_back(j);
    }
    for(auto& v:p.jobs)std::stable_sort(v.begin(),v.end(),[](auto a,auto b){
      auto order=[](auto j){return j.rank==-2?0:j.rank==-3?2:1;};
      return std::make_pair(a.ready,order(a))<std::make_pair(b.ready,order(b));});
    online_programs.push_back(std::move(p));return int(online_programs.size())-1;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int online_begin(int program,int channel) {
  try {
    (void)online_programs.at(program).jobs.at(channel);
    online_states.push_back({program,channel});return int(online_states.size())-1;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int online_clone(int context) {
  try {auto copy=online_states.at(context);online_states.push_back(std::move(copy));
    return int(online_states.size())-1;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int online_advance(int context,I until,OnlineResult* out) {
  try {auto& s=online_states.at(context);advance_online(s,until);
    *out={until,-1,0,0,finish_pim_online(s),s.pim_wait,s.pim_requests};return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int online_drain(int context,OnlineResult* out) {
  try {auto& s=online_states.at(context);
    for(;;){I t=next_pim_time(s);if(t==INF)break;issue_pim_online(s,t);}
    I end=finish_pim_online(s);s.clock=std::max(s.clock,end);
    *out={end,-1,0,0,end,s.pim_wait,s.pim_requests};return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
static int online_run_impl(int context,int plan,I start,OnlineResult* out,Plan* recording=nullptr) {
  try {
    auto& s=online_states.at(context);const auto& p=plans.at(plan);
    if(start<s.clock)throw std::runtime_error("GPU kernel starts before context time");
    advance_online(s,start);I shift=0;*out={};out->first_issue=-1;
    using Pair=std::pair<I,I>;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> due;
    for(const auto& q:p.q) {
      while(!due.empty() && due.top().first<=q.issue) {
        auto [deadline,actual]=due.top();due.pop();shift=std::max(shift,actual-start-deadline);
      }
      I released=start+q.issue+shift;
      OnlineJob gpu{s.channel,0,q.bg,int(q.write),1,0,CK,q.write?CWL:CL,BL};
      for(;;) {
        if(s.external_port_owned) {
          I tp=next_pim_time(s);
          if(tp==INF)throw std::runtime_error("External I/O ownership was not released");
          issue_pim_online(s,tp);continue;
        }
        I tg=legal_time(s,released,gpu,true);
        I tp=next_pim_time(s);
        // Pending PIM commands win ties. Only committed returns constrain
        // future PIM commands; no unissued future PIM DQ is consulted.
        if(tp<=tg){issue_pim_online(s,tp);continue;}
        if(tp!=INF) {
          const auto& program=online_programs[s.program];
          const auto& pending=program.jobs[s.channel][s.cursor];
          if(pending.ready+s.shift<=tg) {
            // A released PIM command may wait for an earlier committed GPU
            // return, but a newly issued GPU request must not prolong that wait.
            bool blocks=program.mode==1 || (pending.ca_length && tg+CK>tp);
            if(pending.dq_length) {
              blocks=blocks || tg+transfer_issue_gap(gpu,pending)>tp;
              Slot a{tg+gpu.dq_latency,tg+gpu.dq_latency+BL,0,gpu.bg,bool(gpu.write)};
              Slot b{tp+pending.dq_latency,tp+pending.dq_latency+pending.dq_length,
                     pending.rank,pending.bg,bool(pending.write)};
              blocks=blocks || !(a.b+guard(a,b)<=b.a || b.b+guard(b,a)<=a.a);
            }
            // mode 1: after configurations request exclusive ownership at
            // external-I/O boundaries; their private fine commands are absent.
            if(blocks){issue_pim_online(s,tp);continue;}
          }
        }
        commit_online(s,gpu,tg,true);
        I done=tg+gpu.dq_latency+BL;
        if(recording)recording->q.push_back({tg-start,done-start,q.write,q.bg});
        due.push({q.done,done});++out->requests;
        if(out->first_issue<0)out->first_issue=tg;
        break;
      }
    }
    while(!due.empty()){auto [deadline,actual]=due.top();due.pop();shift=std::max(shift,actual-start-deadline);}
    I end=start+p.duration+shift;advance_online(s,end);
    out->finish=end;out->memory_stall=shift;out->pim_end=finish_pim_online(s);
    if(recording)recording->duration=end-start;
    out->pim_wait=s.pim_wait;out->pim_requests=s.pim_requests;return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int online_run(int context,int plan,I start,OnlineResult* out) {
  return online_run_impl(context,plan,start,out);
}
extern "C" int online_reference(int empty_program,int plan,OnlineResult* out) {
  // Build the solo reference with the SAME committed-return legality checks.
  // Record every actual issue/completion as the subsequent replay checkpoint.
  // This prevents GPU self-contention from being attributed to PIM.
  try {
    if(!online_programs.at(empty_program).jobs.at(0).empty())
      throw std::runtime_error("Reference program must contain no PIM commands");
    int context=online_begin(empty_program,0);
    if(context<0)return -1;
    Plan ref{0,plans.at(plan).span,{}};
    if(online_run_impl(context,plan,0,out,&ref)<0)return -1;
    plans.push_back(std::move(ref));return int(plans.size())-1;
  }catch(const std::exception& e){error=e.what();return -1;}
}

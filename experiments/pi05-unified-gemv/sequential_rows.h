// GPU rank-0 row state for the existing LPDDR5X_8533 configuration.
// Included after OnlineState/legal_time. This is an address approximation,
// not a GPU cache/warp simulator or a measured Thor physical address map.
constexpr I RCD_RD=20*CK, RCD_WR=9*CK, RP=20*CK, RPAB=23*CK;
constexpr I RAS=45*CK, RC=65*CK, RTP=8*CK, WR_PRE=52*CK;
constexpr I RRD=4*CK, FAW=16*CK, PPD=2*CK;
enum RowCommand { DATA=0, ACT_ONE=1, ACT_TWO=2, PRE_BANK=3, PRE_ALL=4 };

static I row_next_ref(const OnlineState& s,I after) {
  const auto& p=online_programs[s.program];
  const auto& c=calendars[p.refresh_calendar].channels[s.channel];
  I result=next_ref(c,0,after,p.end);
  auto it=std::lower_bound(c.ref[0].begin(),c.ref[0].end(),after,
                          [](auto x,I y){return x.a<y;});
  if(it!=c.ref[0].end())result=std::min(result,it->a);
  return result;
}
static I row_next_pre(const OnlineState& s,I after) {
  return row_next_ref(s,after+RPAB)-RPAB;
}
static I row_latest_pre(const OnlineState& s,I t) {
  const auto& p=online_programs[s.program];
  const auto& c=calendars[p.refresh_calendar].channels[s.channel];
  I result=-INF;
  auto it=std::upper_bound(c.ref[0].begin(),c.ref[0].end(),t+RPAB,
                          [](I y,auto x){return y<x.a;});
  if(it!=c.ref[0].begin())result=(--it)->a-RPAB;
  I periodic=next_ref(c,0,t+RPAB-REFI+1,p.end)-RPAB;
  if(periodic<=t)result=std::max(result,periodic);
  return result;
}
static void row_sync(OnlineState& s,I t) {
  if(!s.rows_enabled)return;
  I pre=row_next_pre(s,std::max<I>(0,s.rows.synced+1));
  while(pre<=t) {
    bool needed=s.rows.active;
    for(auto& b:s.rows.banks){b.open=-1;b.activating=false;b.pre=pre;}
    s.rows.last_pre=pre;s.rows.synced=pre;s.rows.active=false;
    if(needed)++s.rows.stats.preab;
    if(needed && s.rows.trace_on)s.rows.trace.push_back({pre,PRE_ALL,-1,-1,0});
    pre=row_next_pre(s,pre+1);
  }
  s.rows.synced=std::max(s.rows.synced,t);
}
static RowBank row_bank_at(const OnlineState& s,int bank,I t) {
  auto b=s.rows.banks[bank];
  I pre=row_latest_pre(s,t);
  if(pre>s.rows.synced){b.open=-1;b.activating=false;b.pre=pre;}
  return b;
}
struct RowCandidate { OnlineJob job; I time; int command,bank; I row; };

static RowCandidate row_candidate(OnlineState& s,I released,bool write,I sector,int bg) {
  const int bank=int((sector/64)%16); // BG is the low two flat-bank bits.
  const I row=sector/(64*16);
  I t=std::max(released,s.clock);
  for(;;) {
    RowBank b=row_bank_at(s,bank,t);
    int cmd=DATA;I earliest=t;
    if(b.activating) {
      cmd=ACT_TWO;
      earliest=std::max({earliest,b.act1+CK,b.pre+RP,b.act2+RC,s.rows.last_act2+RRD});
    } else if(b.open<0) {
      cmd=ACT_ONE;
      // Issue ACT1 at most one CK before the ACT2 timing constraints permit it.
      earliest=std::max({earliest,b.pre+RP-CK,b.act2+RC-CK,s.rows.last_act2+RRD-CK});
      if(s.rows.act1s.size()>=4)earliest=std::max(earliest,s.rows.act1s[s.rows.act1s.size()-4]+FAW);
    } else if(b.open!=row) {
      cmd=PRE_BANK;
      earliest=std::max({earliest,b.act2+RAS,b.rd+RTP,b.wr+WR_PRE,s.rows.last_pre+PPD});
    } else earliest=std::max(earliest,b.act2+(write?RCD_WR:RCD_RD));
    OnlineJob j{s.channel,0,bg,int(write),1,0,CK,0,0};
    // Reserve a consecutive ACT1/ACT2 pair when admitting ACT1. Once ACT1
    // is sent, ACT2 has an nAAD=8 CK deadline and cannot be starved by PIM.
    // Consecutive issue is legal; no guessed ACT interleaving policy is needed.
    if(cmd==ACT_ONE)j.ca_length=2*CK;
    if(cmd==DATA){j.array_access=1;j.dq_latency=write?CWL:CL;j.dq_length=BL;}
    I legal=legal_time(s,earliest,j,true);
    // A fixed all-bank refresh gets an explicit PREab at REF-nRPab.
    // Never admit an operation whose RAS/read/write recovery misses that PREab.
    I pre=row_next_pre(s,legal);
    I recovery=cmd==ACT_ONE?RAS+CK:cmd==ACT_TWO?RAS:
               cmd==PRE_BANK?PPD:write?WR_PRE:RTP;
    if(legal+recovery>pre){t=pre+RPAB+RFC;continue;}
    // A legality wait may have crossed PREab/refresh and invalidated the row.
    auto effective=row_bank_at(s,bank,legal);
    if(effective.open!=b.open || effective.activating!=b.activating || effective.pre!=b.pre) {
      t=legal;continue;
    }
    return {j,legal,cmd,bank,row};
  }
}
static void row_commit(OnlineState& s,const RowCandidate& x,bool touched) {
  row_sync(s,x.time);auto& r=s.rows;auto& b=r.banks[x.bank];
  switch(x.command) {
    case ACT_ONE:
      b.activating=true;b.act1=x.time;r.active=true;++r.stats.act1;r.act1s.push_back(x.time);
      if(r.act1s.size()>4)r.act1s.pop_front();break;
    case ACT_TWO:
      b.activating=false;b.open=x.row;b.act2=x.time;r.last_act2=x.time;++r.stats.act2;break;
    case PRE_BANK:
      b.open=-1;b.pre=x.time;r.last_pre=x.time;++r.stats.prepb;
      r.active=std::any_of(r.banks.begin(),r.banks.end(),[](auto b){return b.open>=0||b.activating;});break;
    case DATA:
      if(x.job.write){b.wr=x.time;++r.stats.writes;}else{b.rd=x.time;++r.stats.reads;}
      if(touched)++r.stats.row_misses;else ++r.stats.row_hits;break;
  }
  if(r.trace_on)r.trace.push_back({x.time,x.command,x.bank,x.row,
                                  x.job.dq_length?x.time+x.job.dq_latency+x.job.dq_length:0});
}
extern "C" int sequential_enable(int program,int rows) {
  try {
    auto& p=online_programs.at(program);p.rows=bool(rows);
    // The existing placement is rank0 GPU / rank1 PIM. Do not silently accept
    // shared-bank PIM traffic without a unified array-state implementation.
    if(rows)for(auto& channel:p.jobs)for(auto& j:channel)
      if(j.rank==0 && j.array_access)throw std::runtime_error("GPU row model requires disjoint PIM rank1");
    return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int sequential_plan(int plan) {
  try {
    auto& p=plans.at(plan);
    for(size_t i=0;i<p.q.size();++i)p.q[i].bg=int((i/64)%4);
    return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int sequential_stats(int context,RowStats* out) {
  try {*out=online_states.at(context).rows.stats;return 0;}
  catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int sequential_address(int plan,int index,I sector) {
  try {
    auto& p=plans.at(plan);auto& v=row_addresses[plan];
    if(v.empty()){v.resize(p.q.size());for(size_t i=0;i<v.size();++i)v[i]=i;}
    v.at(index)=sector;p.q.at(index).bg=int((sector/64)%4);return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int sequential_trace(int context,RowTrace* out,int capacity) {
  try {
    auto& r=online_states.at(context).rows;r.trace_on=true;
    int n=std::min<int>(capacity,r.trace.size());
    std::copy_n(r.trace.begin(),n,out);return int(r.trace.size());
  }catch(const std::exception& e){error=e.what();return -1;}
}

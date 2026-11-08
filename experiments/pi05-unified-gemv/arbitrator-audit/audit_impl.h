static std::map<std::pair<int,int>,AuditNative> audit_native;
static int audit_type(const OnlineJob& j) {
  if(!j.dq_length)return 4;
  return j.array_access?(j.write?1:0):(j.write?2:3); // RD WR WRGB RDRES control
}
static AuditNative& audit_template(const OnlineState& s) {
  const auto key=std::make_pair(s.program,s.channel);
  auto found=audit_native.find(key);if(found!=audit_native.end())return found->second;
  AuditNative v;const auto& jobs=online_programs[s.program].jobs[s.channel];
  v.job_to_index.resize(jobs.size(),-1);
  for(size_t i=0;i<jobs.size();++i) {
    const auto& j=jobs[i];if(!j.dq_length)continue;
    I a=j.ready+j.dq_latency,b=a+j.dq_length;
    v.dq.push_back({a,b,a,b,I(i),audit_type(j),false});
  }
  std::sort(v.dq.begin(),v.dq.end(),[](auto a,auto b){return a.a<b.a;});
  for(size_t i=0;i<v.dq.size();++i){v.job_to_index[v.dq[i].index]=i;v.dq[i].index=i;}
  return audit_native.emplace(key,std::move(v)).first->second;
}
static I audit_guard(I previous_type,I next_type) {
  (void)next_type;return (previous_type==1||previous_type==2)?CK:2*CK;
}
static void audit_close_gap(OnlineState& s,const AuditDQ* next) {
  auto& a=s.audit;const auto& n=audit_template(s);
  I native_left=a.have_previous?a.previous.original_b:0;
  I actual_left=a.have_previous?a.previous.b:0;
  I native_right=next?next->original_a:online_programs[s.program].end;
  I actual_right=next?std::min(next->a,a.end):a.end;
  I length=native_right-native_left;
  I prev_type=a.have_previous?a.previous.type:-1,next_type=next?next->type:-1;
  I kind=!a.have_previous?1:!next?2:0; // internal / prefix / tail
  I expected=a.have_previous?a.previous.index+1:0;
  if(next && next->index!=expected)kind=3; // actual/native DQ order changed
  if(kind==3){a.current.fit_reads=0;a.current.fit_writes=0;}
  AuditKey key{kind,length,prev_type,next_type};auto& g=a.gaps[key];
  g.kind=kind;g.length=length;g.previous=prev_type;g.next=next_type;
  ++g.windows;g.used_windows+=bool(a.current.reads+a.current.writes);
  g.reads+=a.current.reads;g.writes+=a.current.writes;
  g.fit_reads+=a.current.fit_reads;g.fit_writes+=a.current.fit_writes;
  g.native_ticks+=std::max<I>(0,length);
  g.actual_ticks+=std::max<I>(0,actual_right-actual_left);
  g.extension_ticks+=std::max<I>(0,actual_right-actual_left)-std::max<I>(0,length);
  a.current={};
}
static void audit_flush(OnlineState& s,I before) {
  auto& a=s.audit;const auto& native=audit_template(s);
  size_t used=0;
  for(const auto& q:a.pending) {
    if(q.a>=before)break;++used;
    if(q.gpu) {
      bool wr=q.type==1;
      if(q.a>=a.end){++a.extra[wr?1:0];continue;}
      if(q.b>a.end){++a.extra[wr?3:2];a.extra[4]+=a.end-q.a;a.extra[5]+=q.b-a.end;continue;}
      if(wr)++a.current.writes;else ++a.current.reads;
      I expected=a.have_previous?a.previous.index+1:0;
      const AuditDQ* next=expected<I(native.dq.size())?&native.dq[expected]:nullptr;
      I native_left=a.have_previous?a.previous.original_b:0;
      I actual_left=a.have_previous?a.previous.b:0;
      I right=actual_left+(next?next->original_a:online_programs[s.program].end)-native_left;
      I left=actual_left+(a.have_previous?audit_guard(a.previous.type,q.type):0);
      bool fits=q.a>=left && q.b+(next?audit_guard(q.type,next->type):0)<=right;
      if(fits){if(wr)++a.current.fit_writes;else ++a.current.fit_reads;}
    } else {
      audit_close_gap(s,&q);a.previous=q;a.have_previous=true;
    }
  }
  a.pending.erase(a.pending.begin(),a.pending.begin()+used);
}
static void audit_observe(OnlineState& s,const OnlineJob& j,I t,bool gpu) {
  auto& a=s.audit;if(!a.enabled)return;
  if(a.done)throw std::runtime_error("audit observed a finalized context");
  if(a.trace_on)a.trace.push_back({t,gpu?-1:j.ready,s.shift,j.ca_length,
      j.dq_length?t+j.dq_latency:-1,j.dq_length?t+j.dq_latency+j.dq_length:-1,
      j.rank,audit_type(j),gpu?1:0,I(s.cursor)});
  if(j.dq_length) {
    if(gpu) {
      int occupied=0;
      for(auto old:s.dq)if(old.slot.a<=t && t<old.slot.b){occupied=old.gpu?2:1;break;}
      ++a.extra[6+occupied*2+int(bool(j.write))];
      if(t<a.end)++a.extra[12+occupied*2+int(bool(j.write))];
    }
    I index=gpu?-1:audit_template(s).job_to_index.at(s.cursor);
    AuditDQ q{t+j.dq_latency,t+j.dq_latency+j.dq_length,
      gpu?-1:j.ready+j.dq_latency,gpu?-1:j.ready+j.dq_latency+j.dq_length,index,audit_type(j),gpu};
    auto at=std::upper_bound(a.pending.begin(),a.pending.end(),q.a,[](I x,const auto& y){return x<y.a;});
    a.pending.insert(at,q);
  }
  audit_flush(s,t);
}
extern "C" int audit_enable(int context,I end,int trace) {
  try {auto& s=online_states.at(context);s.audit.enabled=true;s.audit.end=end;s.audit.trace_on=trace;
    (void)audit_template(s);return 0;}catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int audit_finalize(int context) {
  try {auto& s=online_states.at(context);if(s.audit.done)return 0;
    audit_flush(s,INF);audit_close_gap(s,nullptr);s.audit.done=true;return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int audit_gaps(int context,AuditGap* out,int capacity) {
  try {const auto& g=online_states.at(context).audit.gaps;int i=0;
    for(auto [key,value]:g){if(i<capacity)out[i]=value;++i;}return i;
  }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int audit_extra(int context,I* out) {
  try {auto& a=online_states.at(context).audit;std::copy(a.extra.begin(),a.extra.end(),out);return 0;}
  catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int audit_trace(int context,AuditTrace* out,int capacity) {
  try {auto& t=online_states.at(context).audit.trace;std::copy_n(t.begin(),std::min<int>(capacity,t.size()),out);return int(t.size());}
  catch(const std::exception& e){error=e.what();return -1;}
}

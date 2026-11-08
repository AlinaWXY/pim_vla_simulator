// Observation only: all time is in the production arbiter's integer ticks.
struct AuditDQ {I a,b,original_a,original_b,index,type;bool gpu;};
struct AuditCount {I reads=0,writes=0,fit_reads=0,fit_writes=0;};
struct AuditKey {
  I kind,length,previous,next;
  bool operator<(const AuditKey& b)const{return std::tie(kind,length,previous,next)<std::tie(b.kind,b.length,b.previous,b.next);}
};
struct AuditGap {
  I kind,length,previous,next,windows,used_windows,reads,writes,fit_reads,fit_writes,
    native_ticks,actual_ticks,extension_ticks;
};
struct AuditTrace {I issue,native_issue,shift,ca,dq_a,dq_b,rank,type,gpu,cursor;};
struct AuditState {
  bool enabled=false,trace_on=false,done=false,have_previous=false;
  I end=0;AuditDQ previous{};AuditCount current{};
  std::vector<AuditDQ> pending;
  std::map<AuditKey,AuditGap> gaps;
  // Outside-PIM reads/writes; straddling reads/writes; straddling DQ ticks
  // inside/outside; CA issue while DQ idle/PIM busy/GPU busy, by read/write.
  std::array<I,18> extra{};
  std::vector<AuditTrace> trace;
};
struct AuditNative {std::vector<AuditDQ> dq;std::vector<I> job_to_index;};

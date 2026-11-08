// Test-only bridge: exercise the production legality path, with independent
// expectations exported from the native Python DRAM specification.
#include "../sequential_transactions.cpp"
extern "C" I probe_gap(const OnlineJob* a,const OnlineJob* b) {
  return transfer_issue_gap(*a,*b);
}
extern "C" I probe_history(const OnlineJob* jobs,int count,const OnlineJob* next,I ready) {
  Calendar cal{0,std::vector<Channel>(16)};
  calendars.push_back(cal);
  int p=online_add(0,0,int(calendars.size())-1,nullptr,0);
  int h=online_begin(p,0);auto& s=online_states[h];
  for(int i=0;i<count;++i)commit_online(s,jobs[i],jobs[i].ready,false);
  I result=legal_time(s,ready,*next,true);
  commit_online(s,*next,result,true);
  return result;
}
extern "C" I probe_constant(int index) {
  I values[]={BL,CL,CWL,RFC,REFI,RCD_RD,RCD_WR,RP,RPAB,RAS,RC,RTP,WR_PRE,RRD,FAW,PPD};
  return values[index]/CK;
}

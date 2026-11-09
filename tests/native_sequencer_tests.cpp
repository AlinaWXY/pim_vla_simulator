#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "ramulator/controller/impl/pim_model.h"

using namespace PimVla;

namespace {
Operation operation(const std::string& name, int id, int bg=0, int row=0) {
  Operation op;
  op.name=name; op.id=id; op.addr={0,1,bg,0,row,id%64};
  return op;
}

void warm_row_zero(PIMController& c) {
  // A declared warm starting state isolates the sequencer. Requests still
  // execute through real check_timing, completion events and the bus checker.
  const auto opened=c.device.m_spec->states.at("Opened");
  for(auto* bank:c.device.m_bank_nodes) {
    bank->m_state=opened;
    bank->m_row_state[0]=opened;
  }
  c.refresh_due.fill(NEVER);
}

struct Observation {
  std::vector<Clock> issue, complete;
  Clock first_prefetch=NEVER, end=0;
  uint64_t out_of_order=0, prefetched=0, read_bytes=0;
};
using Inspect=std::function<void(const PIMController&,Clock,const Observation&)>;
using Setup=std::function<void(PIMController&)>;

Observation execute(const ConfigNode& spec, std::vector<Operation> children, const Inspect& inspect={}, const Setup& setup={}) {
  PIMController c(0,spec,spec["metadata"]);
  c.fused=true;
  warm_row_zero(c);
  if(setup)setup(c);
  Operation descriptor=operation("MAC_RUN",0);
  descriptor.addr={0,1,-1,-1,0,0};
  descriptor.children=std::move(children);
  std::vector<Operation> program={std::move(descriptor)};
  Observation observed;
  observed.issue.assign(program[0].children.size(),NEVER);
  observed.complete.assign(program[0].children.size(),NEVER);
  c.set_stage(program,0,0,"microbench");
  Clock now=0;
  size_t iterations=0;
  while(!c.stage_finished(now)) {
    require(++iterations<100000,"descriptor sequencer test failed to terminate");
    const Clock issued_at=now;
    now=c.step(now,false);
    require(now>=issued_at && now<100000,"descriptor clock did not make bounded progress");
    if(c.inner.ops) {
      require(c.inner.decoded_until>=c.inner.next && c.inner.decoded_until-c.inner.next<=32,
              "resident descriptor exceeded FIFO32");
      for(size_t i=0;i<c.inner.dispatched.size();++i) {
        if(c.inner.dispatched[i] && observed.issue[i]==NEVER) {
          observed.issue[i]=issued_at;
          observed.complete[i]=c.inner.done[i];
        }
      }
    }
    if(c.prefetched_prerequisites && observed.first_prefetch==NEVER)
      observed.first_prefetch=issued_at;
    if(inspect)inspect(c,issued_at,observed);
  }
  c.checker.advance(now);
  for(size_t i=0;i<observed.issue.size();++i)
    require(observed.issue[i]<NEVER && observed.complete[i]>=observed.issue[i],"child missing issue/completion observation");
  observed.end=now;
  observed.out_of_order=c.out_of_order_issues;
  observed.prefetched=c.prefetched_prerequisites;
  observed.read_bytes=c.pim_read_bytes;
  require(c.gpu_read_bytes==0 && c.gpu_write_bytes==0,"sequencer-only test executed GPU memory work");
  return observed;
}

void report(const std::string& name,const Observation& observation) {
  std::cout<<"{\"sequencer\":\""<<name<<"\",\"status\":\"passed\",\"end_ck\":"<<observation.end
           <<",\"out_of_order_issues\":"<<observation.out_of_order
           <<",\"prefetched_prerequisites\":"<<observation.prefetched<<"}\n";
}

void gpu_owned_act_bypass(const ConfigNode& spec,bool alternative) {
  PIMController c(0,spec,spec["metadata"]);
  warm_row_zero(c);
  const int act1=c.device.m_spec->get_command_id("ACT1");
  const int act2=c.device.m_spec->get_command_id("ACT2");
  const int pre=c.device.m_spec->get_command_id("PREpb");
  const int rd=c.device.m_spec->get_command_id("RD");
  const auto closed=c.device.m_spec->states.at("Closed");
  auto close_bank=[&](int bank) {
    c.device.m_bank_nodes.at(bank)->m_state=closed;
    c.device.m_bank_nodes.at(bank)->m_row_state.clear();
  };
  close_bank(0);
  close_bank(16+4);
  // Native ready-clock fixture: rank1 ACT1 owns the external lane until its
  // ACT2 becomes legal at CK104, within the CK108 nAAD deadline. Rank0 has
  // an older closed-bank read (ACT1) and optionally a younger row conflict
  // (PREpb), which may legally use the otherwise idle external CA at CK101.
  constexpr Clock start=100,probe=101,owner_ready=104;
  AddrVec_t owner={0,1,1,0,0,0};
  c.device.m_bank_nodes.at(16+4)->m_cmd_ready_clk[act2]=owner_ready;
  c.issue(act1,owner,start,-1);
  Kernel kernel;
  kernel.reads=(alternative?2:1)*16;kernel.start=start;kernel.nominal=1;
  c.begin_kernel(&kernel,start);
  c.gpu_queues[0].push_back({{0,0,0,0,1,0},rd,0,start});
  if(alternative)c.gpu_queues[2].push_back({{0,0,0,1,1,0},rd,1,probe});
  c.queued_gpu=c.gpu_outstanding=alternative?2:1;
  c.kernel_next=c.kernel_total;c.issued_reads=c.kernel_reads;
  const Clock next=c.step(probe,true);
  if(alternative) {
    require(c.gpu_command_counts[pre]==1 && next==probe,
            "GPU owner-blocked ACT1 hid another bank's ready PREpb");
    require(c.pending_act[0].cmd==act2 && c.gpu_command_counts[act1]==0,
            "GPU bypass stole the occupied ACT2 lane");
  } else {
    require(next==owner_ready && c.gpu_command_counts[act1]==0,
            "ownership-only GPU stall lost its ACT2 wakeup");
  }
  Clock now=next;
  size_t iterations=0;
  while(!c.kernel_finished()) {
    require(++iterations<1000,"GPU ownership regression failed to drain");
    now=c.step(now,true);
    require(now>=probe && now<1000,"GPU ownership regression lost bounded progress");
  }
  c.end_kernel();
  c.checker.advance(now);
  require(c.pending_act[0].cmd<0,"GPU ownership regression left an ACT2 owner");
  require(c.gpu_read_bytes==(alternative?64u:32u) && c.gpu_write_bytes==0,
          "GPU ownership bypass lost or duplicated read traffic");
  require(c.gpu_row_misses==(alternative?2u:1u) && c.gpu_row_hits==0,
          "GPU prerequisite bypass corrupted row-hit accounting");
  std::cout<<"{\"sequencer\":\"gpu_owned_act_"<<(alternative?"ready_bank_bypass":"wakeup")
           <<"\",\"status\":\"passed\",\"end_ck\":"<<now<<"}\n";
}
}

int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: native_sequencer_tests spec.json");
    const auto spec=Config::parse_config_file(argv[1]);
    gpu_owned_act_bypass(spec,true);
    gpu_owned_act_bypass(spec,false);
    {
      auto producer=operation("RD",0,0),head=operation("RD",1,1),ready=operation("RD",2,2);
      head.deps={0};
      const auto result=execute(spec,{producer,head,ready});
      require(result.issue[2]<result.complete[0],"ready younger RD did not bypass a real completion-blocked head");
      require(result.issue[1]>=result.complete[0],"dependent head issued before its producer completed");
      require(result.out_of_order>0 && result.read_bytes==96,"out-of-order issue or read-byte accounting missing");
      report("completion_dependency_bypass",result);
    }
    {
      auto producer=operation("RD",0,0),head=operation("RD",1,1);
      auto barrier=operation("BARRIER",2),after=operation("RD",3,2);
      head.deps={0}; // The barrier itself has no explicit dependency list.
      const auto result=execute(spec,{producer,head,barrier,after});
      require(result.issue[3]>=result.complete[1],"younger request crossed a completion barrier");
      require(result.complete[2]>=result.complete[0] && result.complete[2]>=result.complete[1],
              "barrier failed to drain all preceding completion tails");
      report("barrier_completion_fence",result);
    }
    {
      auto producer=operation("RD",0,0),head=operation("RD",1,1),after=operation("RD",2,2);
      producer.epoch=head.epoch="first";
      after.epoch="second";
      head.deps={0};
      const auto result=execute(spec,{producer,head,after});
      require(result.issue[2]>=result.issue[1],"younger request crossed an undispatched epoch group");
      // Epoch labels fence dispatch order, not every completion. Explicit
      // BARRIER handles all-I/O-complete requirements; pim_dq_end separately
      // protects the final external return against GPU reentry.
      require(result.issue[2]<result.complete[1],"epoch unnecessarily serialized independent completion tails");
      report("epoch_dispatch_fence",result);
    }
    {
      std::vector<Operation> children;
      children.push_back(operation("RD",0,0));
      for(int i=1;i<32;++i) {
        auto blocked=operation("RD",i,1);
        blocked.deps={0};children.push_back(blocked);
      }
      children.push_back(operation("RD",32,2));
      children.push_back(operation("RD",33,3));
      const auto result=execute(spec,std::move(children),[](const PIMController& c,Clock,const Observation& observed) {
        if(c.inner.ops && observed.issue[1]==NEVER)
          require(!c.inner.dispatched[33],"request outside [head, head+32) dispatched early");
      });
      require(result.issue[32]<result.complete[0],"last slot inside the sliding FIFO32 was not usable");
      require(result.issue[33]>=result.issue[1],"request beyond FIFO32 crossed the blocked head");
      require(result.read_bytes==34*32,"FIFO32 test lost or duplicated requests");
      report("fifo32_window_boundary",result);
    }
    {
      auto producer=operation("RD",0,0),head=operation("RD",1,1),future=operation("LOCALRD",2,2,1);
      head.deps={0};
      const auto result=execute(spec,{producer,head,future});
      require(result.prefetched>0 && result.first_prefetch<result.complete[0],
              "independent next-row native prerequisites did not prefetch during dependency wait");
      require(result.issue[1]>=result.complete[0],"prefetch bypassed head data dependency");
      report("safe_next_bank_prefetch",result);
    }
    {
      auto producer=operation("RD",0,0),head=operation("RD",1,1),future=operation("LOCALRD",2,1,1);
      head.deps={0};
      const auto result=execute(spec,{producer,head,future},[](const PIMController& c,Clock,const Observation& observed) {
        if(c.inner.ops && observed.issue[1]==NEVER) {
          const auto* bank=c.device.m_bank_nodes.at(16+4); // rank1, BG1, bank0
          const auto opened=c.device.m_spec->states.at("Opened");
          require(bank->m_state==opened && bank->m_row_state.count(0),
                  "younger prefetch evicted the row required by an older undispatched request");
          require(observed.first_prefetch==NEVER,"conflicting same-bank prerequisite prefetched before the older request");
        }
      });
      require(result.issue[2]>result.issue[1],"conflicting younger row displaced the older access");
      require(result.read_bytes==64,"same-bank guard lost or duplicated external reads");
      report("protect_older_required_row",result);
    }
    {
      auto blocked=operation("LOCALRD",0,0),ready=operation("LOCALRD",1,0);
      ready.addr[3]=1; // Rank1 REFab reserves both this and every other bank.
      constexpr Clock precharge_ready=100;
      const auto result=execute(spec,{blocked,ready},{},[&](PIMController& c) {
        const int descriptor=c.device.m_spec->get_command_id("MAC_RUN");
        c.refresh_due[1]=c.info[descriptor].completion;
        c.refresh_bank[1]=0;
        // Fixture: rank1 has an outstanding native tRAS restriction before
        // PREab. No child may bypass the overdue all-bank refresh.
        for(int bank=16;bank<32;++bank) {
          c.device.m_bank_nodes.at(bank)->m_cmd_ready_clk[c.device.m_spec->get_command_id("PREab")]=precharge_ready;
          c.device.m_bank_nodes.at(bank)->m_cmd_ready_clk[c.device.m_spec->get_command_id("PREab_LOCAL")]=precharge_ready;
        }
      });
      const Clock earliest=precharge_ready+spec["timings"]["nRPab"].as<int>()+spec["timings"]["nRFC"].as<int>();
      require(result.issue[0]>=earliest && result.issue[1]>=earliest,
              "a PIM bank accessed the rank before PREab/REFab recovery");
      require(result.read_bytes==0,"LOCALRD unexpectedly used the external DQ bus");
      report("rank1_all_bank_refresh_fence",result);
    }
    {
      // The producer must change row0 -> row1 through real PRE/ACT/tRCD.
      // A warm read in another bank could otherwise issue while those native
      // prerequisites are still outstanding.
      auto producer=operation("LOCALRD",0,0,1),consumer=operation("RD",1,1,0);
      const auto independent=execute(spec,{producer,consumer});
      require(independent.issue[1]<independent.issue[0],"issue-dependency fixture did not expose an independent early read");
      consumer.issue_deps={0};
      const auto result=execute(spec,{producer,consumer});
      require(result.issue[1]>=result.issue[0],"issue dependency allowed consumer before producer dispatch");
      require(result.issue[1]<result.complete[0],"issue dependency incorrectly waited for producer completion");
      require(result.read_bytes==32,"issue dependency lost or duplicated external read traffic");
      report("issue_dependency_is_not_completion_dependency",result);
    }
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"native_sequencer_tests: "<<error.what()<<'\n';
    return 1;
  }
}

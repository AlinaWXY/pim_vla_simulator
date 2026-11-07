// Descriptor sequencer. The decoded descriptor and cursors are bounded state;
// micro-requests use the existing finite PIM FIFO and native timing machinery.
bool m_coarse_commands=false;
std::string m_run_path;
std::ofstream m_run_events;
std::array<bool,256> m_run_done{};
uint64_t m_run_sequence=0;
struct RunState {
  bool active=false;
  Request parent;
  PIMDescriptor d;
  uint64_t id=0;
  Clk_t ready=0;
  int phase=0,cursor=0,inflight=0,batch=0,tile=0,serial=0;
  int output_issued=0,output_done=0,output_expected=0;
} m_run;
enum RunPhase { RunLoad,RunMac,RunRelease,RunQKSpill,RunVSpill,RunPVBegin,
  RunPVSpill,RunPVSeal,RunOutput,RunRewind,RunScore,RunRope,RunFinish };

bool is_run(const Request& r) const {
  return r.type_id==Request::Type::PIM_MAC_RUN || r.type_id==Request::Type::PIM_ROPE_RUN;
}
void record_run(const char* event,int type=-1,int bank=-1,int offset=-1,int slot=-1) {
  if(m_run_events) m_run_events<<m_clk<<','<<m_channel_id<<','<<m_run.id<<','
    <<m_run.d.opcode<<','<<m_run.d.b_base<<','<<event<<','<<m_run.phase<<','<<m_run.tile<<','
    <<m_run.batch<<','<<m_run.inflight<<','<<type<<','<<bank<<','<<offset<<','<<slot<<'\n';
}
void setup_coarse() {
  if(!m_coarse_commands) return;
  if(!m_rank_controller || !m_whole_srf || !m_rank_staging || m_staging_banks!=2)
    throw std::runtime_error("Coarse commands require whole-SRF two-bank rank controller");
  if(!m_run_path.empty()) {
    m_run_events.open(m_run_path);
    if(!m_run_events) throw std::runtime_error("Cannot open RUN events");
    m_run_events<<"clock,Channel,run_id,opcode,B_base,event,phase,tile,batch,inflight,type,bank,offset,buffer\n";
  }
}
bool send_run(Request& req) {
  if(!m_coarse_commands) throw std::runtime_error("RUN requires coarse command mode");
  if(m_run.active || m_clk<m_global_ca_ready) return false;
  PIMDescriptor d;d.decode(req.pim_descriptor,req.pim_descriptor_crc);
  if(d.opcode!=(req.type_id==Request::Type::PIM_MAC_RUN?1:2)) throw std::runtime_error("RUN opcode differs from trace mnemonic");
  m_addr_mapper->apply(req);req.addr_vec[0]=m_channel_id;req.pim_origin_addr_vec=req.addr_vec;
  if(req.addr_vec[m_rank_level]!=1 || req.addr_vec[m_bg_level]!=0 || req.addr_vec[m_bank_level]!=0)
    throw std::runtime_error("RUN requires canonical rank1 address");
  if(d.opcode==1 && !d.reduce) {
    Request probe=req;probe.addr_vec[m_device.m_spec->get_level_id("Row")]=d.b_base/2048;
    probe.addr_vec[m_device.m_spec->get_level_id("Column")]=(d.b_base%2048)/32;
    probe.pim_origin_addr_vec=probe.addr_vec;
    if(gemv_rows(probe)!=d.m || gemv_columns(probe)!=d.tile || output_fragments(d.b_base)!=d.fragments ||
       d.b_f!=d.tile*32 || (d.n!=d.tile && !(d.b_base==786432 && d.n==64 && d.tile_major)))
      throw std::runtime_error("RUN geometry does not match current resident matrix layout");
  }
  m_run=RunState{};m_run.active=true;m_run.parent=req;m_run.d=d;m_run.id=++m_run_sequence;
  if(d.wait==d.done) throw std::runtime_error("RUN cannot wait for its own DONE event");
  m_run_done[d.done]=false;
  const Clk_t dispatch=std::max<Clk_t>(1,m_clk);
  m_global_ca_ready=dispatch+21;m_run.ready=dispatch+22;
  m_run.phase=d.opcode==2?RunRope:d.reduce?RunMac:RunLoad;
  record_ca(dispatch,dispatch+21,1,d.opcode==1?"MAC_RUN":"ROPE_RUN","global");
  record_run("ACCEPT");return true;
}
Request run_child(int type,int offset,int bank=0,int slot=-1) {
  const Addr_t address=(Addr_t(offset/2048)<<20)|(Addr_t(bank%4)<<18)|(Addr_t(bank/4)<<16)|
    (Addr_t{1}<<15)|(Addr_t((offset%2048)/32)<<9)|(Addr_t(m_channel_id)<<5);
  Request r(address,type);r.size_bytes=32;r.channel_count=m_run.parent.channel_count;
  r.intra_channel_addr=((address>>9)<<5)|(address&31);
  r.pim_rank_micro=true;r.pim_run_id=m_run.id;r.pim_buffer=slot;
  r.ingress_id=1000000+int(m_run.id)*20000+m_run.serial;
  return r;
}
int run_columns() const { return std::min(m_run.d.tile,m_run.d.n-m_run.tile*m_run.d.tile); }
int run_batch_count() const { return std::min(m_run.d.pack,m_run.d.fragments-m_run.batch); }
int run_phase_count() const {
  const auto& d=m_run.d;
  switch(m_run.phase) {
    case RunLoad:return run_batch_count();
    case RunMac:return (d.reduce?d.fragments:run_batch_count())*run_columns();
    case RunRelease:return d.reduce?1:d.pack;
    case RunQKSpill:case RunVSpill:case RunPVSpill:return 256;
    case RunPVBegin:case RunPVSeal:case RunRewind:case RunScore:return 1;
    case RunRope:return d.m*7*16;
    default:return 0;
  }
}
Request run_next_child() {
  const auto& d=m_run.d;const int i=m_run.cursor;
  if(m_run.phase==RunLoad) { const int f=m_run.batch+i;return run_child(Request::Type::PIM_STAGE_FRAGMENT,d.a_address(f),d.a_bank(f),f%d.pack); }
  if(m_run.phase==RunMac) {
    const int f=(d.reduce?0:m_run.batch)+i/run_columns(),n=m_run.tile*d.tile+i%run_columns();
    return run_child(d.reduce?Request::Type::PIM_MAC_QK:Request::Type::PIM_MAC_AB,d.b_address(f,n),0,d.reduce?0:f%d.pack);
  }
  if(m_run.phase==RunRelease) return run_child(Request::Type::PIM_GEMV_RELEASE,0,0,i);
  if(m_run.phase==RunQKSpill) return run_child(Request::Type::PIM_SPILL,81920+(8+(i/16)%2*8+i/32)*32,i%16);
  if(m_run.phase==RunVSpill) return run_child(Request::Type::PIM_SPILL,163840+30*512+(i/16)*32,i%16);
  if(m_run.phase==RunPVBegin) { auto r=run_child(Request::Type::PIM_STAGE_BEGIN,524288);r.pim_stage_bytes=8192;r.pim_stage_rows=16;return r; }
  if(m_run.phase==RunPVSpill) {
    const int pair=i/32,q=(i%32)/2,f=pair*2+i%2;
    return run_child(Request::Type::PIM_PV_SPILL,524288+((f/2)*16+q)*32,4*(f%2));
  }
  if(m_run.phase==RunPVSeal) return run_child(Request::Type::PIM_STAGE_SEAL,524288);
  if(m_run.phase==RunRewind) return run_child(Request::Type::PIM_O_REWIND,524288);
  if(m_run.phase==RunScore) return run_child(Request::Type::PIM_SCORE_SPILL,655360+m_run.tile*1024);
  if(m_run.phase==RunRope) {
    const int token=i/(7*16),op=(i/16)%7,bank=i%16;
    const int q=(d.w[2]&0xffffff)*32+token*(d.w[2]>>24)*32;
    const int k=(d.w[3]&0xffffff)*32+token*(d.w[3]>>24)*32;
    const int c=(d.w[4]&0xffffff)*32+token*(d.w[4]>>24)*32;
    if(op==0) return run_child(Request::Type::PIM_ROPE_COEF,c,bank);
    if(op==1) return run_child(Request::Type::PIM_ROPE_LOAD,q,bank,0);
    if(op==2) return run_child(Request::Type::PIM_ROPE_EXEC,q,bank,0);
    if(op==3) return run_child(Request::Type::PIM_ROPE_LOAD,k,bank,1);
    if(op==4) return run_child(Request::Type::PIM_ROPE_EXEC,k,bank,1);
    return run_child(Request::Type::PIM_ROPE_STORE,op==5?q:k,bank);
  }
  throw std::runtime_error("Invalid RUN micro-operation cursor");
}
void run_phase(int phase) {
  m_run.phase=phase;m_run.cursor=0;
  if(phase==RunOutput) {
    m_run.output_issued=m_run.output_done=0;
    m_run.output_expected=m_run.d.reduce?m_run.d.m*run_columns()*4/32:output_export_packets();
  }
  record_run(phase==RunOutput?"TILE_READY":"PHASE");
}
void run_next_tile() {
  if((m_run.tile+1)*m_run.d.tile<m_run.d.n) {
    ++m_run.tile;m_run.batch=0;
    run_phase(m_run.d.reduce?RunMac:RunRewind);
  } else run_phase(m_run.d.reduce?RunRelease:RunFinish);
}
void run_advance() {
  const auto& d=m_run.d;
  switch(m_run.phase) {
    case RunLoad:run_phase(RunMac);break;
    case RunMac:
      if(d.reduce) run_phase(m_score_dram_spill?RunScore:RunOutput);
      else if(m_run.batch+d.pack<d.fragments) {m_run.batch+=d.pack;run_phase(RunLoad);}
      else run_phase(RunRelease);
      break;
    case RunRelease:
      if(d.reduce) run_phase(RunFinish);
      else if(d.b_base==393216) run_phase(RunQKSpill);
      else if(d.b_base==163840 && m_o_projection) run_phase(RunPVBegin);
      else run_phase(RunOutput);
      break;
    case RunQKSpill:run_phase(RunVSpill);break;
    case RunVSpill:case RunPVSeal:case RunRope:run_phase(RunFinish);break;
    case RunPVBegin:run_phase(RunPVSpill);break;
    case RunPVSpill:run_phase(RunPVSeal);break;
    case RunRewind:run_phase(RunLoad);break;
    case RunScore:case RunOutput:run_next_tile();break;
    default:throw std::runtime_error("Invalid RUN phase transition");
  }
}
void pump_coarse() {
  if(!m_run.active || m_clk<m_run.ready || (m_run.d.wait && !m_run_done[m_run.d.wait])) return;
  // At most 32 lookahead entries, sharing the existing bounded request FIFO.
  for(int budget=0;budget<32 && m_run.active;++budget) {
    if(m_run.phase==RunFinish) {
      record_run("COMPLETE");auto done=m_run.parent;done.depart=m_clk;
      m_run_done[m_run.d.done]=true;m_run.active=false;if(done.callback) done.callback(done);return;
    }
    if(m_run.phase==RunOutput) {
      if(m_run.output_done<m_run.output_expected) return;
      record_run("TILE_RELEASE");run_advance();continue;
    }
    if(m_run.cursor==run_phase_count()) {
      // Keep the bounded native lookahead across input/compute batches. Row
      // preparation may overlap input reads; the native SRF ownership checks
      // still prevent any MAC/overwrite before its actual data dependency.
      const bool stream=m_run.phase==RunLoad || (m_run.phase==RunMac && !m_run.d.reduce &&
          m_run.batch+m_run.d.pack<m_run.d.fragments);
      if(m_run.inflight && !stream) return;
      run_advance();continue;
    }
    if(m_pim_buffer.size()>=m_pim_buffer.max_size) return;
    auto child=run_next_child();
    const auto id=m_run.id;
    child.callback=[this,id](Request& done) {
      if(!m_run.active || m_run.id!=id || m_run.inflight<=0) throw std::runtime_error("RUN child lifetime violation");
      --m_run.inflight;
      const auto& a=done.pim_origin_addr_vec;
      record_run("CHILD_COMPLETE",done.type_id,a[m_bg_level]*4+a[m_bank_level],gemv_offset(done),done.pim_buffer);
    };
    const int type=child.type_id;
    if(!send_impl(child)) return;
    ++m_run.inflight;++m_run.cursor;++m_run.serial;
    const auto& a=child.pim_origin_addr_vec;
    record_run("CHILD_ENQUEUE",type,a[m_bg_level]*4+a[m_bank_level],gemv_offset(child),child.pim_buffer);
  }
}
bool coarse_result_send(Request& req) {
  if(!m_run.active || m_run.phase!=RunOutput || m_run.output_issued>=m_run.output_expected) return false;
  if((req.type_id==Request::Type::PIM_RD_BG)!=m_run.d.reduce) throw std::runtime_error("RUN result route mismatch");
  auto callback=req.callback;const auto id=m_run.id;
  req.callback=[this,callback,id](Request& done) {
    if(!m_run.active || m_run.id!=id) throw std::runtime_error("RUN result outlived owner");
    ++m_run.output_done;if(callback) callback(done);
  };
  const bool accepted=send_impl(req);req.callback=callback;
  if(accepted) ++m_run.output_issued;
  return accepted;
}

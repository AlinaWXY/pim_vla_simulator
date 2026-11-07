#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace Ramulator {
struct PIMDescriptor {
  std::array<uint32_t,8> w{};
  int opcode=0,m=0,n=0,tile=0,fragments=0,pack=0,mode=0,split=0;
  int a_base=0,a_m=0,a_f=0,b_base=0,b_n=0,b_f=0,jump=0,wait=0,done=0;
  bool reduce=false,tile_major=false;
  static uint16_t crc(const std::array<uint32_t,8>& words) {
    uint16_t value=0xffff;
    for(auto word:words) for(int byte=0;byte<4;++byte) {
      value^=uint16_t((word>>(8*byte))&255)<<8;
      for(int bit=0;bit<8;++bit) value=(value&0x8000)?uint16_t((value<<1)^0x1021):uint16_t(value<<1);
    }
    return value;
  }
  void decode(const std::array<uint32_t,8>& words,uint16_t check) {
    w=words;opcode=w[0]&15;
    if(crc(w)!=check || ((w[0]>>4)&15)!=2 || (opcode!=1 && opcode!=2))
      throw std::runtime_error("RUN CRC/opcode/version mismatch");
    if(((w[0]>>12)&65535)!=65535)
      throw std::runtime_error("Current H16 backend requires all sixteen banks");
    if(opcode==2) {
      if((w[0]&0xf0000c00u) || ((w[0]>>8)&3)!=3 || w[7])
        throw std::runtime_error("Unsupported RoPE descriptor flags");
      m=(w[1]&65535)+1;wait=(w[1]>>16)&255;done=w[1]>>24;
      if(m!=8 || !done) throw std::runtime_error("RoPE requires eight local tokens and nonzero DONE");
      for(int i=2;i<=6;++i) if((w[i]>>24)!=1) throw std::runtime_error("RoPE requires contiguous token vectors");
      if((w[2]&0xffffff)!=0xa08 || (w[3]&0xffffff)!=0xa10 || (w[4]&0xffffff)!=0xa00 || w[5]!=w[2] || w[6]!=w[3])
        throw std::runtime_error("RoPE descriptor does not match resident working layout");
      return;
    }
    if((w[0]&0xe0000800u) || (w[7]&0xf80007ffu))
      throw std::runtime_error("Unsupported RUN reserved bits, retained SRF or buffer offsets");
    mode=(w[0]>>8)&3;reduce=(w[0]>>10)&1;tile_major=(w[0]>>28)&1;
    m=(w[1]&15)+1;n=((w[1]>>4)&1023)+1;fragments=((w[1]>>14)&4095)+1;
    pack=1<<((w[1]>>26)&3);tile=((w[6]>>24)&63)+1;
    a_base=(w[2]&0xffffff)*32;a_m=(w[3]&65535)*32;a_f=(w[3]>>16)*32;
    b_base=(w[4]&0xffffff)*32;b_n=(w[5]&65535)*32;b_f=(w[5]>>16)*32;
    split=(w[4]>>24)|((w[6]>>30)<<8);
    jump=int32_t((w[6]&0xffffff)<<8)>>8;jump*=32;
    wait=(w[7]>>11)&255;done=(w[7]>>19)&255;
    if((w[1]>>28)!=15 || !done || tile>n || pack*m*32>1024 || m*tile*4>(reduce?1024:2048) || (split && split>=n) || (!split && jump))
      throw std::runtime_error("Invalid RUN geometry, output capacity or address split");
    if(a_m!=32 || b_n!=32 || (mode!=1 && mode!=3)) throw std::runtime_error("Unsupported RUN input/vector layout");
    if(mode==1 && ((w[2]>>24)!=0x40 || a_base!=524288 || a_f!=m*32 || pack!=1024/(m*32)))
      throw std::runtime_error("RUN source must match whole-SRF two-bank staging");
    if(reduce && (mode!=3 || m!=16 || n!=496 || tile!=16 || fragments!=1 || a_base || b_base!=65536 || split!=484 || jump!=1408))
      throw std::runtime_error("RUN rank reduction must match Q/K token layout");
    if(!reduce && mode!=1) throw std::runtime_error("Resident A currently requires rank-reduced QKT");
  }
  int a_address(int f,int row=0) const { return a_base+(f/2)*a_f+row*a_m; }
  int a_bank(int f) const { return 4*(f%2); }
  int b_address(int f,int column) const {
    const int t=column/tile,j=column%tile;
    return b_base+(tile_major?(t*fragments+f)*b_f+j*b_n:f*b_f+column*b_n)+(split && column>=split?jump:0);
  }
};
}

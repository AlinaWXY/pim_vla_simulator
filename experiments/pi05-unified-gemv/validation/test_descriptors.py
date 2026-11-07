"""Cross-language descriptor decoding, CRC rejection and address-boundary tests."""
import subprocess,tempfile
from pathlib import Path
from simulator.coarse import example,crc16,parse,line,HEADER,CA_CYCLES


def main():
    root=Path(__file__).resolve().parents[2]/'pi05-bgk-gpuq/backend/ramulator2/src'
    names=('qkv','qkt','pv','gate_up','down','o','rope')
    assert HEADER==0x04ed and CA_CYCLES==(14+256+16+8)//14==21
    vectors=[]
    for name in names:
        w=example(name,wait=5,done=6)
        assert parse(line(0x8000,w).split())==w
        bad=line(0x8000,w).split();bad[3]=f'{int(bad[3],16)^1:08x}'
        try:parse(bad)
        except ValueError:pass
        else:raise AssertionError('Corrupt CRC accepted')
        vectors.append('{'+','.join(str(x)+'u' for x in w)+'}')
    program='''#include "ramulator/base/pim_descriptor.h"
#include <cassert>
using namespace Ramulator;
int main(){
std::array<std::array<uint32_t,8>,7> vectors={{VECTORS}};
for(auto w:vectors){PIMDescriptor d;d.decode(w,PIMDescriptor::crc(w));assert(d.wait==5&&d.done==6);
bool rejected=false;try{d.decode(w,PIMDescriptor::crc(w)^1);}catch(...){rejected=true;}assert(rejected);}
PIMDescriptor d;d.decode(vectors[1],PIMDescriptor::crc(vectors[1]));
assert(d.b_address(0,483)==0x13c60);assert(d.b_address(0,484)==0x14200);
assert(d.b_address(0,495)==0x14360);assert(d.n==496&&d.tile==16);
d.decode(vectors[5],PIMDescriptor::crc(vectors[5]));
assert(d.b_address(0,32)==0xc4000&&d.b_address(15,63)==0xc7fe0);
assert(d.a_bank(1)==4&&d.a_address(1)==0x80000&&d.a_address(2)==0x80200);
auto bad=vectors[5];bad[1]=(bad[1]&~15u)|14u;
bool rejected=false;try{d.decode(bad,PIMDescriptor::crc(bad));}catch(...){rejected=true;}assert(rejected);
bad=vectors[6];bad[7]=1;rejected=false;try{d.decode(bad,PIMDescriptor::crc(bad));}catch(...){rejected=true;}assert(rejected);
}
'''.replace('VECTORS',','.join(vectors))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);(p/'test.cpp').write_text(program)
        subprocess.run(['c++','-std=c++20','-I',str(root),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
        subprocess.run([str(p/'test')],check=True)
    print('PASS: seven cross-language descriptors, CRC/reserved/geometry rejection, QKT split and O tile boundaries')


if __name__=='__main__':main()

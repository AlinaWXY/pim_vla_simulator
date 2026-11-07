"""Version-2 256-bit MAC_RUN/ROPE_RUN wire descriptors (no tensor payload)."""
import binascii
import struct

HEADER = 0x04ED
CA_CYCLES = 21


def crc16(words):
    return binascii.crc_hqx(struct.pack('<8I', *words), 0xffff)


def mac(*, m, n, tile, fragments, pack, a_base, a_stride, b_base, b_stride,
        resident=False, reduce=False, tile_major=False, split=0, jump=0,
        wait=0, done=1):
    assert 1 <= m <= 16 and 1 <= tile <= min(n,64) and n <= 1024
    assert 1 <= fragments <= 4096 and pack in (1,2,4,8)
    assert pack*m*32 <= 1024 and m*tile*4 <= (1024 if reduce else 2048)
    assert all(x % 32 == 0 for x in (a_base,a_stride,b_base,b_stride,jump))
    assert 0 <= split < n and -(1<<23) <= jump//32 < 1<<23
    w = [1 | 2<<4 | (3 if resident else 1)<<8 | int(reduce)<<10 | 0xffff<<12 | int(tile_major)<<28,
         m-1 | (n-1)<<4 | (fragments-1)<<14 | (pack.bit_length()-1)<<26 | 15<<28,
         a_base//32 | (0 if resident else 4)<<28,
         1 | a_stride//32<<16,
         b_base//32 | (split&255)<<24,
         1 | b_stride//32<<16,
         (jump//32 & 0xffffff) | (tile-1)<<24 | (split>>8)<<30,
         wait<<11 | done<<19]
    return tuple(w)


def rope(wait=0,done=1):
    return (2 | 2<<4 | 3<<8 | 0xffff<<12, 7 | wait<<16 | done<<24,
            0xa08 | 1<<24, 0xa10 | 1<<24, 0xa00 | 1<<24,
            0xa08 | 1<<24, 0xa10 | 1<<24, 0)


def example(name,wait=0,done=1):
    shared=dict(a_base=0x80000,a_stride=512,wait=wait,done=done)
    if name=='qkv':
        shared['a_stride']=256
        return mac(m=8,n=48,tile=48,fragments=64,pack=4,b_base=0x60000,b_stride=1536,**shared)
    if name=='qkt':
        shared['a_base']=0
        return mac(m=16,n=496,tile=16,fragments=1,pack=1,b_base=0x10000,b_stride=0,
                   resident=True,reduce=True,split=484,jump=1408,**shared)
    if name=='pv':
        return mac(m=16,n=16,tile=16,fragments=31,pack=2,b_base=0x28000,b_stride=512,**shared)
    if name=='gate_up':
        return mac(m=16,n=32,tile=32,fragments=64,pack=2,b_base=0x40000,b_stride=1024,**shared)
    if name=='down':
        return mac(m=16,n=4,tile=4,fragments=256,pack=2,b_base=0x50000,b_stride=128,**shared)
    if name=='o':
        return mac(m=16,n=64,tile=32,fragments=16,pack=2,b_base=0xc0000,b_stride=1024,tile_major=True,**shared)
    if name=='rope': return rope(wait,done)
    raise ValueError(name)


def line(address,words):
    return f'{"MAC_RUN" if words[0]&15==1 else "ROPE_RUN"} 0x{address:x} '+ ' '.join(f'{w:08x}' for w in words)+f' {crc16(words):04x}'


def parse(fields):
    if len(fields)!=11: raise ValueError('RUN requires address, eight words and CRC16')
    words=tuple(int(x,16) for x in fields[2:10])
    if any(not 0<=x<2**32 for x in words) or crc16(words)!=int(fields[10],16):
        raise ValueError('Invalid RUN word or CRC16')
    if words[0]&15 != (1 if fields[0]=='MAC_RUN' else 2) or (words[0]>>4)&15!=2:
        raise ValueError('Invalid RUN opcode/version')
    return words

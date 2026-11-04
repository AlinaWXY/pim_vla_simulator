"""FP32 rank-complete QKT scores, buffered in two DRAM banks before export."""
import horizon_attention_layout as L

BASE = 655360
TILES = 31
TILE_BYTES = 1024
PACKETS_PER_TILE = TILE_BYTES // 32


def descriptor_offset(tile):
    L.check(tile, TILES, "score tile")
    return BASE + tile * TILE_BYTES


def packet_location(channel, tile, packet):
    L.check(channel, 16, "score channel")
    L.check(tile, TILES, "score tile")
    L.check(packet, PACKETS_PER_TILE, "score packet")
    return (channel, 1, packet % 2, 0,
            BASE + tile * 512 + packet // 2 * 32)


def packet_elements(channel, tile, packet):
    """Eight complete scores in token-major/query-minor order."""
    L.check(channel, 16, "score channel")
    L.check(tile, TILES, "score tile")
    L.check(packet, PACKETS_PER_TILE, "score packet")
    return [dict(head=channel//2, token=2*(16*tile+(packet*8+i)//16)+channel%2,
                 query=(packet*8+i)%16, byte_in_packet=i*4, dtype="FP32")
            for i in range(8)]


def contract():
    return dict(dtype="FP32", logical_shape=[8,16,992], tile_tokens=16,
        tiles_per_rank=TILES, tile_bytes_per_rank=TILE_BYTES,
        packets_per_tile=PACKETS_PER_TILE, packet_bytes=32,
        bytes_per_rank=TILES*TILE_BYTES, total_bytes=16*TILES*TILE_BYTES,
        source="rank-reduced complete scores; no cross-rank sum",
        banks=[0,4], base_bytes=BASE, used_bytes_per_source_bank=TILES*512,
        reserved_bytes_per_source_bank=16384,
        descriptor_offset_formula="655360+tile*1024 (logical route tag)",
        physical_offset_formula="655360+tile*512+(packet//2)*32",
        physical_bank_formula="4*(packet%2)",
        score_order="[local_K_token][query16] FP32",
        local_write_command="PIM_SCORE_SPILL", export_command="LD",
        lifetime="Each 1KiB score tile spills completely before reuse; all 31 tiles complete before any GPU read",
        numerical_compute=False)

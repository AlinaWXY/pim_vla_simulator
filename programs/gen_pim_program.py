"""Build one layer's request program from tensor geometry, never native traces.

The frontend reuses it for 18 layers x 10 denoising steps. Indices in `deps`
refer to completion, not issue; nested descriptor children have their own index
space. No CK, DQ latency or DRAM timing constant is emitted here.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from workloads import layout as L
from programs import descriptors as D

VARIANTS = {
    "feature_gpu_before": ("feature", "gpu", "streaming"),
    "token_gpu_before": ("token", "gpu", "streaming"),
    "token_local_before": ("token", "local", "streaming"),
    "token_gpu_after": ("token", "gpu", "fused"),
    "token_local_after": ("token", "local", "fused"),
}
EXPECTED_DQ = dict(zip(VARIANTS, (6096, 5352, 5160, 4968, 4776)))
EXTERNAL = {"RD", "WR", "WRGB", "RDRES"}


class Builder:
    def __init__(self, channel, group):
        self.channel, self.group, self.ops = channel, group, []

    def emit(self, op, offset=0, bank=0, deps=(), **fields):
        i = len(self.ops)
        self.ops.append(dict(id=i, op=op, addr=L.address(self.channel, offset, bank),
                             deps=list(deps), **fields))
        return i

    def fence(self, deps):
        return self.emit("BARRIER", deps=deps)

    def descriptor(self, name, children, deps=()):
        words = D.for_operator(name)
        return self.emit("ROPE_RUN" if name == "rope" else "MAC_RUN", bank=-1,
                         deps=deps, descriptor=words, crc16=D.crc16(words),
                         operator=name, children=children.ops)


def staged_input(b, rows, fragments):
    writes = []
    for f0 in range(0, fragments, 2):
        for row in range(rows):
            for f in range(f0, min(f0 + 2, fragments)):
                bank, offset = L.staging(f, row, rows)
                writes.append(b.emit("WR", offset, bank, epoch=b.group + "_input",
                                     tensor_fragment=f, tensor_row=row))
    return b.fence(writes)


def gemm(b, *, rows, fragments, columns, base, fused, views, output_tile=0,
         tile_stride=0, initial=()):
    """Whole 1KiB input batches; weight columns pipeline through bank PEs."""
    previous = list(initial)
    last_mac = None
    for first in range(0, fragments, views):
        loads = []
        for f in range(first, min(first + views, fragments)):
            slot = (f % views) * rows
            for row in range(rows):
                if fused:
                    bank, offset = L.staging(f, row, rows)
                    rd = b.emit("LOCALRD", offset, bank, deps=previous,
                                dest_slot=slot + row, tensor_fragment=f, tensor_row=row)
                    loads.append(b.emit("BCAST", deps=[rd], bank=-1, slot=slot + row,
                                        source_bank=bank, tensor_fragment=f))
                else:
                    loads.append(b.emit("WRGB", row * 32, -1, deps=previous,
                                        slot=slot + row, rows=1, tensor_fragment=f,
                                        tensor_row=row, epoch=b.group + "_stream"))
        first_mac = True
        for f in range(first, min(first + views, fragments)):
            for col in range(columns):
                last_mac = b.emit("MACAB", base + output_tile * tile_stride + (f * columns + col) * 32,
                                  -1, deps=loads if first_mac else (), slot=(f % views) * rows,
                                  rows=rows, output_columns=columns, output_column=col,
                                  fragment=f, output_tile=output_tile,
                                  output_bytes_per_bank=rows * columns * 4)
                first_mac = False
        # Reuse is gated by the end of the entire current batch, not its issue.
        previous = [last_mac]
    return last_mac


def local_projection_spills(b, last, feature):
    writes = []
    for bank in range(16):
        for col in range(8, 24):
            kind, local = divmod(col - 8, 8)
            cols = 24 if feature else 48
            regions = ([dict(bank=bank, byte_offset=((2 * local + q) * cols + kind * 8) * 4, bytes=32)
                        for q in range(2)] if feature else
                       [dict(bank=bank, byte_offset=(local * cols + kind * 16) * 4, bytes=64)])
            gather = b.emit("GATHER", deps=[last], bank=bank, bytes=64, elements=16,
                            source_regions=regions, source="qkv_output")
            cast = b.emit("CAST", deps=[gather], bank=bank, elements=16,
                          source="qkv_output", output_bytes=32)
            writes.append(b.emit("LOCALWR", L.WORK_BASE + col * 32, bank, deps=[cast],
                                 source="qkv_output", output_bytes=32))
        for lane in range(16):
            offset = L.V_BASE + (30 * 16 + lane) * 32
            rd = b.emit("LOCALRD", offset, bank, deps=[last], dest="rmw_scratch")
            preserve = list(range(8)) if feature and lane < 8 else ([] if feature else list(range(4)))
            zero = list(range(8, 16)) if feature and lane >= 8 else ([] if feature else list(range(12, 16)))
            rows = range((lane // 8) * 8, (lane // 8 + 1) * 8) if feature else range(8)
            regions = [dict(bank=bank, byte_offset=(q * (24 if feature else 48) +
                            (16 + lane % 8 if feature else 32 + lane)) * 4, bytes=4) for q in rows]
            gather = b.emit("GATHER", bank=bank, deps=[rd], bytes=32, elements=8,
                            source="qkv_v_output", source_regions=regions, preserve_lanes=preserve)
            cast = b.emit("CAST", bank=bank, deps=[gather], elements=8,
                          zero_lanes=zero, output_bytes=32)
            writes.append(b.emit("LOCALWR", offset, bank, deps=[cast], source="rmw_scratch"))
    return b.fence(writes)


def qkv(channel, feature, fused):
    b = Builder(channel, "qkv")
    rows, cols, views = (16, 24, 2) if feature else (8, 48, 4)
    if fused:
        seal = staged_input(b, rows, 64)
        inner = Builder(channel, "qkv")
        last = gemm(inner, rows=rows, fragments=64, columns=cols, base=L.QKV_BASE,
                    fused=True, views=views)
        local_projection_spills(inner, last, feature)
        b.descriptor("qkv", inner, [seal])
    else:
        last = gemm(b, rows=rows, fragments=64, columns=cols, base=L.QKV_BASE,
                    fused=False, views=views)
        local_projection_spills(b, last, feature)
    return b.ops


def rope(channel, fused):
    outer, b = Builder(channel, "rope"), Builder(channel, "rope")
    last_exec = []
    writes = []
    for token in range(8):
        for bank in range(16):
            coef = b.emit("LOCALRD", L.COEF_BASE + token * 32, bank, dest="rope_coefficient", token=token)
            for which, base in enumerate((L.Q_BASE, L.K_SUFFIX_BASE)):
                read = b.emit("LOCALRD", base + token * 32, bank, dest="rope_input", slot=which, token=token)
                execute = b.emit("ROPE_EXEC", base + token * 32, bank,
                                 deps=[coef, read] + last_exec, slot=which, pairs=8,
                                 shared_unit="rank_rope", token=token)
                last_exec = [execute]
                writes.append(b.emit("LOCALWR", base + token * 32, bank, deps=[execute],
                                     source="rope_output", slot=which, token=token))
    b.fence(writes)
    if fused:
        outer.descriptor("rope", b)
        return outer.ops
    return b.ops


def rope_gpu(channel, export):
    b = Builder(channel, "rope_export" if export else "rope_import")
    for col in range(8, 24):
        for bank in range(16):
            b.emit("RD" if export else "WR", L.WORK_BASE + col * 32, bank,
                   epoch=b.group, tensor="QK")
    return b.ops


def q_exchange(channel, gpu_rope, feature, export):
    if feature:
        # Each feature shard already owns all sixteen Q rows.
        b = Builder(channel, "q_exchange_export" if export else "q_exchange_import")
        if not export:
            return b.ops
        for token in range(16):
            for bank in range(16):
                b.emit("LOCALRD", L.Q_BASE + (token // 2) * 32, bank,
                       dest_slot=token, useful_lanes=8, packed_token=token % 2,
                       source_lane_offset=8 * (token % 2), dest_lane_offset=0)
        return b.ops
    b = Builder(channel, "q_exchange_export" if export else "q_exchange_import")
    if export:
        for token in range(8):
            for bank in range(16):
                if not gpu_rope:
                    b.emit("RD", L.Q_BASE + token * 32, bank, epoch="q_export", tensor="Q")
                b.emit("LOCALRD", L.Q_BASE + token * 32, bank,
                       dest_slot=2 * token + channel % 2, tensor="local_Q")
        return b.ops
    # The stage boundary is a global barrier across all sixteen channels.
    for token in range(8):
        for bank in range(16):
            wr = b.emit("WR", L.Q_IMPORT_BASE + token * 32, bank,
                        epoch="q_import", tensor="remote_Q", source_channel=channel ^ 1,
                        source_offset=L.Q_BASE + token * 32)
            b.emit("LOCALRD", L.Q_IMPORT_BASE + token * 32, bank, deps=[wr],
                   dest_slot=2 * token + 1 - channel % 2, tensor="remote_Q")
    return b.ops


def qkt(channel, feature, fused):
    outer, b = Builder(channel, "qkt"), Builder(channel, "qkt")
    previous = []
    n = 992 if feature else 496
    for token in range(n):
        mac = b.emit("MACQK", L.key_offset(token, feature), -1, deps=previous,
                     slot=0, rows=16, lanes=8 if feature else 16,
                     weight_lane_offset=8 * (token % 2) if feature else 0,
                     kv_token=token, output_bytes_per_bank=64, output_columns=1)
        reduce = b.emit("REDUCE", bank=-1, deps=[mac], elements=16,
                        levels=["bank_group", "rank"], kv_token=token,
                        cache_result=False)
        outputs = []
        for packet in range(2):
            if fused:
                bank, offset = L.score_location(token, packet)
                outputs.append(b.emit("LOCALWR", offset, bank, deps=[reduce],
                                      source="reduction_output", result_slot=packet,
                                      release_result=True, kv_token=token))
            else:
                outputs.append(b.emit("RDRES", packet * 32, -1, deps=[reduce],
                                      source="reduction_output", result_slot=packet,
                                      release_result=True, kv_token=token, epoch="score_stream"))
        previous = outputs
    if fused:
        run = outer.descriptor("qkt", b)
        for token in range(n):
            for packet in range(2):
                bank, offset = L.score_location(token, packet)
                outer.emit("RD", offset, bank, deps=[run], epoch="score_export", tensor="scores")
        return outer.ops
    return b.ops


def export_results(b, count, deps, *, features_per_bank, bf16, epoch, tile=0):
    last = []
    elements = 16 if bf16 else 8
    for packet in range(count):
        # Export order [query][bank][feature] keeps ownership explicit.
        scalar = packet * elements
        query, rem = divmod(scalar, 16 * features_per_bank)
        bank, lane = divmod(rem, features_per_bank)
        remaining, cursor, regions = elements, scalar, []
        while remaining:
            q, rem = divmod(cursor, 16 * features_per_bank)
            owner, first = divmod(rem, features_per_bank)
            take = min(remaining, features_per_bank - first)
            regions.append(dict(bank=owner, byte_offset=(q * features_per_bank + first) * 4,
                                bytes=take * 4))
            cursor += take
            remaining -= take
        gather = b.emit("GATHER", bank=bank, deps=deps, bytes=elements * 4,
                        elements=elements, result_slot=(query * features_per_bank + lane) // 8,
                        result_bank=bank, source_regions=regions, source="output_pool", output_tile=tile)
        ready = gather
        if bf16:
            ready = b.emit("CAST", bank=bank, deps=[gather], elements=elements, output_bytes=32,
                           output_tile=tile)
        last.append(b.emit("RDRES", packet * 32, -1, deps=[ready], result_bank=bank,
                           result_slot=(query * features_per_bank + lane) // 8,
                           source_regions=regions, output_tile=tile, release_result=True, epoch=epoch))
    return b.fence(last)


def pv(channel, feature, fused):
    outer, b = Builder(channel, "pv"), Builder(channel, "pv")
    fragments, columns = (62, 8) if feature else (31, 16)
    seal = staged_input(outer, 16, fragments) if fused else None
    last = gemm(b, rows=16, fragments=fragments, columns=columns, base=L.V_BASE,
                fused=fused, views=2)
    if fused:
        writes = []
        for fragment in range(16):
            for query in range(16):
                gather = b.emit("GATHER", bank=fragment, deps=[last], bytes=64, elements=16,
                                source="pv_output", result_slot=query * 2, result_bank=fragment,
                                source_regions=[dict(bank=fragment, byte_offset=query * 64, bytes=64)])
                cast = b.emit("CAST", bank=fragment, deps=[gather], elements=16, output_bytes=32)
                bank, offset = L.staging(fragment, query, 16)
                writes.append(b.emit("LOCALWR", offset, bank, deps=[cast], source="pv_bf16",
                                     release_result=True))
        b.fence(writes)
        outer.descriptor("pv", b, [seal])
        return outer.ops
    export_results(b, 128 if feature else 256, [last], features_per_bank=columns,
                   bf16=True, epoch="pv_export")
    return b.ops


def output_projection(channel, feature, fused):
    outer, b = Builder(channel, "o"), Builder(channel, "o")
    fragments = 8 if feature else 16
    previous = []
    for tile in range(2):
        last = gemm(b, rows=16, fragments=fragments, columns=32, base=L.O_BASE,
                    fused=fused, views=2, output_tile=tile, tile_stride=fragments * 32 * 32,
                    initial=previous)
        done = export_results(b, 512, [last], features_per_bank=32, bf16=True,
                              epoch=f"o_tile{tile}_export", tile=tile)
        previous = [done]
    if fused:
        outer.descriptor("o", b)
        return outer.ops
    return b.ops


def ffn(channel, name, fused):
    outer, b = Builder(channel, name), Builder(channel, name)
    fragments, columns, base, packets = ((64, 32, L.GU_BASE, 1024) if name == "gate_up"
                                        else (256, 4, L.DOWN_BASE, 128))
    seal = staged_input(outer, 16, fragments) if fused else None
    last = gemm(b, rows=16, fragments=fragments, columns=columns, base=base,
                fused=fused, views=2)
    if fused:
        run = outer.descriptor(name, b, [seal])
        export_results(outer, packets, [run], features_per_bank=columns,
                       bf16=False, epoch=name + "_export")
        return outer.ops
    export_results(b, packets, [last], features_per_bank=columns,
                   bf16=False, epoch=name + "_export")
    return b.ops


def walk(ops):
    for op in ops:
        yield op
        yield from walk(op.get("children", []))


def validate_ops(ops, channel):
    for i, op in enumerate(ops):
        if op["id"] != i or any(type(d) is not int or not 0 <= d < i for d in op["deps"]):
            raise ValueError("dependency must reference an earlier operation in its scope")
        ch, rank, bg, bank, row, col = op["addr"]
        if ch != channel or rank != 1 or row < 0 or not 0 <= col < 64:
            raise ValueError("invalid physical address")
        if op["op"] in ("MACAB", "MACQK", "WRGB", "RDRES") and (bg, bank) != (-1, -1):
            raise ValueError("all-bank/buffer command must target the whole rank")
        if not 0 <= op.get("slot", 0) < 32 or not 0 <= op.get("dest_slot", 0) < 32:
            raise ValueError("SRF capacity exceeded")
        if op["op"] in ("MACAB", "MACQK") and op.get("slot", 0) + op["rows"] > 32:
            raise ValueError("MAC input exceeds SRF capacity")
        if not 0 <= op.get("result_slot", 0) < 64 or op.get("output_bytes_per_bank", 0) > 2048:
            raise ValueError("output pool capacity exceeded")
        for region in op.get("source_regions", []):
            if not (0 <= region["bank"] < 16 and region["byte_offset"] >= 0 and
                    region["bytes"] > 0 and region["byte_offset"] + region["bytes"] <= 2048):
                raise ValueError("source region exceeds the per-bank output pool")
        if "children" in op:
            if len(op["descriptor"]) != 8 or D.crc16(op["descriptor"]) != op["crc16"]:
                raise ValueError("invalid descriptor")
            validate_ops(op["children"], channel)


def generate(variant, channel_count=16):
    if variant not in VARIANTS or channel_count not in (1, 16):
        raise ValueError("unsupported variant/channel count")
    layout, rope_location, flow = VARIANTS[variant]
    feature, fused, gpu = layout == "feature", flow == "fused", rope_location == "gpu"
    factories = [("qkv", lambda c: qkv(c, feature, fused))]
    if gpu:
        factories += [("rope_export", lambda c: rope_gpu(c, True)),
                      ("rope_import", lambda c: rope_gpu(c, False))]
    else:
        factories += [("rope", lambda c: rope(c, fused))]
    factories += [("q_exchange_export", lambda c: q_exchange(c, gpu, feature, True)),
                  ("q_exchange_import", lambda c: q_exchange(c, gpu, feature, False)),
                  ("qkt", lambda c: qkt(c, feature, fused)),
                  ("pv", lambda c: pv(c, feature, fused)),
                  ("o", lambda c: output_projection(c, feature, fused)),
                  ("gate_up", lambda c: ffn(c, "gate_up", fused)),
                  ("down", lambda c: ffn(c, "down", fused))]
    stages, counts = [], {}
    for name, factory in factories:
        channels = [factory(c) for c in range(channel_count)]
        for c, ops in enumerate(channels):
            validate_ops(ops, c)
            for op in walk(ops):
                counts[op["op"]] = counts.get(op["op"], 0) + 1
        stage_dq = sum(op["op"] in EXTERNAL for ops in channels for op in walk(ops)) * 32
        stages.append(dict(name=name, gpu_before_us=0, expected_dq_bytes=stage_dq, channels=channels))
    dq = sum(counts.get(x, 0) for x in EXTERNAL) * 32
    expected = EXPECTED_DQ[variant] * 1024 * channel_count // 16
    if dq != expected:
        raise ValueError(f"DQ mismatch: {dq} != {expected}")
    sources = [Path(__file__), ROOT / "programs/descriptors.py", ROOT / "workloads/layout.py"]
    return dict(schema_version=1, variant=variant, channels=channel_count,
                layout=layout, rope=rope_location, dataflow=flow, repetitions=dict(layers=18, steps=10),
                address_format=["channel", "rank", "bank_group", "bank", "row", "column_32B"],
                slot_unit_bytes=32, dependency_semantics="completion; indices local to stage/channel or descriptor children",
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                tensor_contract=L.contract(layout), counts=counts,
                expected_dq_bytes=dq, expected_dq_bytes_per_layer=dq, stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true", help="omit indentation in persisted JSON")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in ([args.variant] if args.variant else VARIANTS):
        data = generate(variant)
        path = args.output / (variant + ".json")
        path.write_text(json.dumps(data, separators=(",", ":") if args.compact else None,
                                   indent=None if args.compact else 2) + "\n", encoding="utf-8")
        print(json.dumps(dict(variant=variant, path=str(path), dq_bytes=data["expected_dq_bytes_per_layer"],
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())), flush=True)


if __name__ == "__main__":
    main()

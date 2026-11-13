"""Check rank mode transitions, wire opcodes, and DQ against native CA."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, write_json
from programs.gen_pim_program import walk
from tools.capture_gap_traces import SLICES
from workloads import layout


def compare_fused_export(before, after):
    ignored = {"id", "deps", "issue_deps", "children"}
    control = {"BARRIER", "MAC_RUN"}
    count = 0
    for old, new in zip(before["channels"], after["channels"], strict=True):
        original = list(walk(old))
        current = list(walk(new))
        old_exports = [op for op in original if op["op"] == "RDRES"]
        stores = [op for op in current if op["op"] == "LOCALWR" and op.get("source") == "fused_result"]
        reads = [op for op in current if op["op"] == "RD" and op.get("tensor") == "fused_result"]
        if not (len(old_exports) == len(stores) == len(reads)):
            raise ValueError(f"fused export packet count differs in {before['name']}")
        for index, (old_op, store, read) in enumerate(zip(old_exports, stores, reads, strict=True)):
            bank, offset = layout.fused_result_location(index)
            address = layout.address(old_op["addr"][0], offset, bank)
            if store["addr"] != address or read["addr"] != address:
                raise ValueError(f"fused DRAM address mismatch in {before['name']}, packet {index}")
            if any(old_op[key] != store[key] for key in ("source_regions", "result_bank", "result_slot",
                                                       "output_tile", "release_result")):
                raise ValueError(f"fused output source differs in {before['name']}, packet {index}")
        old_other = [{k:v for k,v in op.items() if k not in ignored} for op in original
                     if op["op"] not in control | {"RDRES"}]
        exports = {id(op) for op in stores + reads}
        new_other = [{k:v for k,v in op.items() if k not in ignored} for op in current
                     if op["op"] not in control and id(op) not in exports]
        if old_other != new_other:
            raise ValueError(f"fused non-export operands changed in {before['name']}")
        count += len(stores)
    return count


def check_programs(baseline, current):
    paths = sorted((baseline / "programs").glob("*.json"))
    if len(paths) != len(SLICES):
        raise ValueError("expected five matched baseline programs")
    result = {}
    for path in paths:
        old = read_json(path)
        new = read_json(current / "programs" / path.name)
        spills = 0
        for stage in new["stages"]:
            for channel in stage["channels"]:
                for op in walk(channel):
                    if op["op"] == "RDRES":
                        if len(op.pop("spill_addr")) != 6:
                            raise ValueError(f"invalid accumulator spill in {path.name}")
                        spills += 1
        for program in (old, new):
            program.pop("source_sha256")
        old_stages = {stage["name"]:stage for stage in old["stages"]}
        new_stages = {stage["name"]:stage for stage in new["stages"]}
        rope_reads = rope_stores = 0
        if "rope" in old_stages:
            old_rope = old_stages.pop("rope")
            new_rope = new_stages.pop("rope")
            def rope_operands(stage):
                return Counter(json.dumps({k:v for k,v in op.items()
                                           if k not in {"id", "deps", "issue_deps", "children"}},
                                          sort_keys=True)
                               for channel in stage["channels"] for op in walk(channel))
            if rope_operands(old_rope) != rope_operands(new_rope):
                raise ValueError(f"RoPE coefficients, vectors or destinations changed: {path.name}")
            rope_reads = sum(op.get("dest") == "rope_coefficient" for channel in new_rope["channels"]
                             for op in walk(channel))
            rope_stores = sum(op.get("source") == "rope_output" for channel in new_rope["channels"]
                              for op in walk(channel))
        if old["dataflow"] == "fused":
            for name in ("o", "gate_up", "down"):
                spills += compare_fused_export(old_stages.pop(name), new_stages.pop(name))
        old["stages"] = [stage for name, stage in old_stages.items() if name != "qkv"]
        new["stages"] = [stage for name, stage in new_stages.items() if name != "qkv"]
        for before, after in zip(old_stages["qkv"]["channels"], new_stages["qkv"]["channels"], strict=True):
            def canonical(channel):
                return sorted(json.dumps({k:v for k,v in op.items()
                                          if k not in {"id", "deps", "issue_deps", "children"}},
                                         sort_keys=True)
                              for op in walk(channel))
            if canonical(before) != canonical(after):
                raise ValueError(f"Q/K/V physical operands changed: {path.name}")
        for program in (old, new):
            program.pop("counts")
        if old != new:
            raise ValueError(f"tensor contract or logical operation changed: {path.name}")
        result[path.stem] = dict(tensor_layout_and_other_stages_match=True,
                                 accumulator_spill_packets=spills,
                                 rope_coefficients_read=rope_reads, rope_vectors_written=rope_stores)
    return result


def audit(directory, spec):
    clock = spec["timings"]
    rows = spec["organization"]["row"]
    counters = Counter()
    modes = [False] * 16
    open_rows = [[-1] * 16 for _ in modes]
    register = [False] * len(modes)
    qk_halves = [None] * len(modes)
    coefficient = [None] * len(modes)
    allowed = {"ACT1", "ACT2", "ACTAB", "PREpb", "PREab", "REFpb", "REFab", "RD", "WR"}
    with (directory / "ca-commands.csv").open(newline="", encoding="utf-8") as ca, \
         (directory / "dq-bursts.csv").open(newline="", encoding="utf-8") as dq:
        bursts = csv.DictReader(dq)
        for record in csv.DictReader(ca):
            cmd = record["command"]
            ch, start, end, rank, bg, bank, row = (
                int(record[k]) for k in ("channel", "start_ck", "end_ck", "rank",
                                         "bank_group", "bank", "row"))
            if cmd not in allowed or end - start != (2 if cmd == "ACTAB" else 1):
                raise ValueError(f"unexpected CA opcode or duration: {record}")
            if rank == 0:
                if cmd not in {"PREpb", "REFpb"}:
                    raise ValueError(f"unexpected rank 0 command in AE solo: {record}")
                continue
            counters[cmd] += 1
            if cmd == "ACT2":
                open_rows[ch][4 * bg + bank] = row
            elif cmd == "ACTAB":
                if not modes[ch] or any(r >= 0 for r in open_rows[ch]):
                    raise ValueError(f"ACTAB without PIM mode or a closed rank: {record}")
                open_rows[ch] = [row] * 16
                register[ch] = row == rows - 3
            elif cmd == "PREab":
                open_rows[ch] = [-1] * 16
                register[ch] = False
            elif cmd == "PREpb":
                index = 4 * bg + bank
                active = open_rows[ch][index]
                open_rows[ch][index] = -1
                if index in (0, 1) and active == rows - 1 - index:
                    target = index == 0
                    if target == modes[ch]:
                        raise ValueError(f"redundant mode switch: {record}")
                    modes[ch] = target
                    counters["to_pim" if target else "to_dram"] += 1
            elif cmd in ("RD", "WR"):
                opcode, mask = int(record["opcode"]), int(record["bank_mask"])
                if opcode >= 0:
                    valid_opcodes = range(9) if cmd == "RD" else {0, 1, 3}
                    if (not modes[ch] or register[ch] or (bg, bank) != (-1, -1)
                            or open_rows[ch] != [row] * 16 or opcode not in valid_opcodes):
                        raise ValueError(f"invalid all-bank column opcode: {record}")
                    masks = {0x11, 0xffff} if cmd == "WR" and opcode == 3 else {0xffff}
                    if mask not in masks:
                        raise ValueError(f"invalid bank mask for opcode: {record}")
                    counters[f"{cmd}_opcode_{opcode}"] += 1
                    col = int(record["column"])
                    if cmd == "RD" and opcode in (4, 5):
                        address = (row, col)
                        if opcode == 4 and qk_halves[ch] is None:
                            qk_halves[ch] = address
                        elif opcode == 5 and qk_halves[ch] == address:
                            qk_halves[ch] = None
                        else:
                            raise ValueError(f"feature QK must read low then high at the same address: {record}")
                    if cmd == "RD" and opcode == 6:
                        coefficient[ch] = (row, col)
                    if cmd == "RD" and opcode in (7, 8):
                        expected = (row, col - 8 * (opcode - 6))
                        if coefficient[ch] != expected:
                            raise ValueError(f"RoPE read lacks its coefficient column: {record}")
                register_write = cmd == "WR" and register[ch] and row == rows - 3
                if not modes[ch] or register_write:
                    direction = cmd[0]
                    dq_start = start + clock["nCL" if direction == "R" else "nCWL"]
                    expected = (ch, dq_start, dq_start + clock["nBL_min"], direction, cmd)
                    burst = next(bursts, None)
                    if burst is None:
                        raise ValueError(f"missing DQ burst for {record}")
                    actual = (int(burst["channel"]), int(burst["start_ck"]), int(burst["end_ck"]),
                              burst["direction"], burst["command"])
                    if actual != expected:
                        raise ValueError(f"DQ burst does not match CA and rank mode: {actual} != {expected}")
                    counters["host_reads" if direction == "R" else "host_writes"] += 1
                    if register_write:
                        counters["host_srf_writes"] += 1
                else:
                    counters["internal_reads" if cmd == "RD" else "internal_writes"] += 1
        if next(bursts, None) is not None:
            raise ValueError(f"unexpected DQ burst after all CA commands: {directory}")
    if any(half is not None for half in qk_halves):
        raise ValueError(f"feature QK has an unmatched low-half read: {directory}")
    metrics = read_json(directory / "raw.json")["metrics"]
    if (32 * counters["host_reads"] != metrics["pim_read_bytes"] or
            32 * counters["host_writes"] != metrics["pim_write_bytes"]):
        raise ValueError(f"DQ and native bytes differ: {directory}")
    return dict(counters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    if not spec["metadata"].get("mc_command_control") or spec["timings"]["nMAC16"] != 32:
        parser.error("expected memory-controller protocol and 2× MAC")
    programs = check_programs(args.baseline, args.traces)
    results = {}
    for case, slices in SLICES.items():
        for step, layer in slices:
            label = f"{case}-s{step}-l{layer}"
            counts = audit(args.traces / label / "on", spec)
            expected_qk = {4: 16 * 496, 5: 16 * 496} if case == "c" else {3: 16 * 496}
            expected_rope = {6: 16 * 8, 7: 16 * 8, 8: 16 * 8} if case in "eg" else {}
            for opcode in range(3, 9):
                if counts.get(f"RD_opcode_{opcode}", 0) != (expected_qk | expected_rope).get(opcode, 0):
                    raise ValueError(f"wrong QK or RoPE opcode counts: {label}")
            results[label] = counts
    write_json(args.output, dict(programs=programs, slices=results))


if __name__ == "__main__":
    main()

"""Native timeline attribution for GPU-Q / BG-K attention and rank-local FFN.

Every reported elapsed column is mutually exclusive. Inclusive service and
overlap credit are also retained, and must not be added as elapsed latency.
GPU compute and unspecified nonlinear stages are explicit nulls.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


RESOURCES = ("gpu_input", "local_input", "mac_array", "array_column", "bg_reduce", "local_write",
             "dram_read", "dram_write", "gpu_output", "local_gather")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def partition(start, stop, windows):
    """Exact integer-tick union, retaining all simultaneous resource masks."""
    assert 0 <= start <= stop
    changes = defaultdict(Counter)
    changes[start]
    changes[stop]
    for resource, spans in windows.items():
        assert resource in RESOURCES, resource
        for begin, end in spans:
            assert start <= begin < end <= stop, (resource, begin, end, start, stop)
            changes[begin][resource] += 1
            changes[end][resource] -= 1
    active, masks, previous = Counter(), Counter(), start
    for now in sorted(changes):
        key = tuple(r for r in RESOURCES if active[r])
        masks[key] += now - previous
        active.update(changes[now])
        assert all(v >= 0 for v in active.values())
        previous = now
    assert not any(active.values()) and sum(masks.values()) == stop - start
    inclusive = {r: sum(n for mask, n in masks.items() if r in mask) for r in RESOURCES}
    exclusive = {r: masks[(r,)] for r in RESOURCES}
    exclusive["overlap"] = sum(n for mask, n in masks.items() if len(mask) >= 2)
    exclusive["other"] = masks[()]
    credit = sum((len(mask) - 1) * n for mask, n in masks.items() if len(mask) >= 2)
    assert sum(exclusive.values()) == stop - start
    assert sum(inclusive.values()) - credit + masks[()] == stop - start
    return dict(exclusive=exclusive, inclusive=inclusive, overlap_credit=credit,
                masks={"+".join(mask) or "other": n for mask, n in masks.items()})


def category(operation):
    op = operation.upper()
    if "MAC" in op:
        return "mac_array"
    if any(word in op for word in ("REDUCE", "BGACC", "BG_ACC")):
        return "bg_reduce"
    if "LOCAL" in op and any(word in op for word in ("WR", "WRITE")):
        return "local_write"
    if "LOCAL" in op and any(word in op for word in ("GB", "INPUT", "FEED")):
        return "local_input"
    if "WRGB" in op or "WR_GB" in op or op in ("PIM_WR_BG", "PIM_WR_Q"):
        return "gpu_input"
    if "RDRES" in op or "RD_RES" in op or op == "PIM_RD_BG":
        return "gpu_output"
    if op in ("RD", "RDA"):
        return "dram_read"
    if op in ("WR", "WRA"):
        return "dram_write"
    return None


def phase_kind(segment):
    explicit = segment.get("kind")
    aliases = dict(pipeline="pipeline", export="gpu_output", dram_read="dram_read",
        dram_write="dram_write", local_write="local_write", gather="local_gather")
    if explicit in aliases:
        return aliases[explicit]
    name = segment["name"].lower()
    if "gather" in name:
        return "local_gather"
    if "export" in name or "result_output" in name:
        return "gpu_output"
    if "local" in name and "write" in name:
        return "local_write"
    if "write" in name and "input" not in name and "compute" not in name:
        return "dram_write"
    if "read" in name and "input" not in name and "compute" not in name:
        return "dram_read"
    return "pipeline"


def inspect(directory):
    directory = Path(directory)
    manifest = load(directory / "manifest.json")
    result = load(directory / "simulation/result.json")
    summary = load(directory / "summary.json")
    cfg, timing = result["config"], result["resolved_timings_ck"]
    ratio = int(cfg["frontend_ratio"])
    assert cfg["memory_ratio"] == 1
    tick = lambda ck: (int(ck) - 1) * ratio + 1
    total_ticks = result["memory_cycles"] * ratio
    tick_us = result["tCK_ns"] / ratio / 1000
    assert abs(total_ticks * tick_us - summary["latency_us"]) < 1e-8
    trace_hash = digest(directory / "workload.trace")
    assert trace_hash == result["requests"]["sha256"] == digest(directory / "simulation/input.trace")
    barriers = {int(e["marker"]): int(e["frontend_tick"])
                for e in read_csv(directory / "simulation/barriers.csv")}
    phases = []
    for segment in manifest["segments"]:
        begin, end = barriers[segment["start"]], barriers[segment["end"]]
        assert 0 <= begin < end <= total_ticks
        assert not phases or phases[-1]["end"] == begin, "Noncontiguous phases"
        phases.append(dict(name=segment["name"], start=begin, end=end,
                           kind=phase_kind(segment), duration_us=(end-begin)*tick_us))
    assert phases
    all_native_counts, channels, resources, hashes, events_evidence = Counter(), [], [], {}, []
    for channel in range(cfg["channels"]):
        native_path = directory / "simulation" / f"timeline.csv.ch{channel}"
        rows = read_csv(native_path)
        hashes[native_path.name] = digest(native_path)
        assert all(int(e["Channel"]) == channel for e in rows)
        all_native_counts.update(e["command"] for e in rows)
        windows = defaultdict(list)
        completion_tails = defaultdict(list)
        work = Counter()
        rich_path = directory / "simulation" / f"bgk_events.csv.ch{channel}"
        rich = read_csv(rich_path) if rich_path.exists() else []
        rich_keys = set()
        if rich_path.exists():
            hashes[rich_path.name] = digest(rich_path)
        for e in rich:
            kind = category(e["operation"])
            if kind is None:
                raise AssertionError(f"Unclassified native BG-K event: {e}")
            start = tick(e["clock"])
            service = int(e["service_cycles"])
            assert service > 0
            end = start + service * ratio
            assert end <= tick(e["ready_at"]), "Completion precedes service"
            assert end <= total_ticks, "Run ended before modeled service"
            ready = tick(e["ready_at"])
            assert ready <= total_ticks, "Run ended before rich-event completion"
            if end < ready:
                completion_tails[kind].append((end, ready))
            windows[kind].append((start, end))
            if e["operation"] == "PIM_MAC_K":
                # MACK uses 2 CK service but its same-array column issue
                # guard is 4 CK. Attribute the uncovered guard window
                # separately; it is not an additional read or ALU latency.
                guard_end = start + max(service, timing["nCCDL"]) * ratio
                if guard_end > end:
                    windows["array_column"].append((end, min(guard_end, total_ticks)))
                    work["array_column_service_ticks"] += min(guard_end, total_ticks) - end
            rich_keys.add((int(e["clock"]), kind))
            work[kind + "_service_ticks"] += service * ratio
            work[kind + "_events"] += 1
            work[kind + "_bytes"] += int(e.get("bytes", 0))
        # Rich events are authoritative for their command. Legacy command
        # windows use the explicitly recorded configured service parameters.
        for e in rows:
            kind = category(e["command"])
            if kind is None or (int(e["clock"]), kind) in rich_keys:
                continue
            if kind == "mac_array":
                service = timing["nCCDAB"]
            elif kind in ("gpu_input", "gpu_output"):
                # DQ data occupancy, consistent with new rich WR_BG/WR_Q
                # events. nCCDL is an issue constraint, not burst duration.
                service = timing["nBL_min"]
            elif kind == "dram_read":
                service = timing["nCL"] + timing["nBL_min"]
            elif kind == "dram_write":
                service = timing["nCWL"] + timing["nBL_min"]
            else:
                raise AssertionError(f"Missing service model for native event {e}")
            start, end = tick(e["clock"]), tick(e["clock"]) + service * ratio
            assert end <= total_ticks, (directory.name, e, end, total_ticks)
            windows[kind].append((start, end))
            work[kind + "_service_ticks"] += service * ratio
            work[kind + "_events"] += 1
        local_path = directory / "simulation" / f"local_input.csv.ch{channel}"
        if local_path.exists():
            hashes[local_path.name] = digest(local_path)
            for e in read_csv(local_path):
                start, end = tick(e["clock"]), tick(e["ready_at"])
                assert end - start == int(e["port_cycles"]) * ratio
                windows["local_input"].append((start, end))
                work["local_input_service_ticks"] += end - start
                work["local_input_events"] += 1
                work["local_input_bytes"] += int(e["bytes"])
        # Fenced transfer/gather phases include scheduling, tails and row
        # management. Preserve these measured phases as exclusive categories.
        # Pipeline phases use native service windows and exact overlap unions.
        excl, incl, overlap_credit, partitions = Counter(), Counter(), 0, []
        for phase in phases:
            start, end, kind = phase["start"], phase["end"], phase["kind"]
            if kind != "pipeline":
                excl[kind] += end - start
                incl[kind] += end - start
                partitions.append(dict(phase=phase["name"], attribution=kind,
                                       ticks=end-start))
                continue
            selected = defaultdict(list)
            for resource, spans in windows.items():
                for a, b in spans:
                    if a < end and b > start:
                        # A resource may remain active across a fence.  This
                        # intersection conserves its elapsed contribution.
                        selected[resource].append((max(a, start), min(b, end)))
            part = partition(start, end, selected)
            excl.update(part["exclusive"])
            incl.update(part["inclusive"])
            overlap_credit += part["overlap_credit"]
            partitions.append(dict(phase=phase["name"], attribution="native_union", **part))
        boundary = total_ticks - sum(p["end"]-p["start"] for p in phases)
        excl["boundary"] = boundary
        assert sum(excl.values()) == total_ticks
        assert sum(incl.values()) - overlap_credit + excl["other"] + boundary == total_ticks
        channel_row = dict(operator=directory.name, channel=channel, total_us=total_ticks*tick_us,
            identity_error_ticks=0, last_service_tick=max((b for spans in windows.values() for _, b in spans), default=0))
        channel_row.update({k + "_only_us": excl[k]*tick_us for k in RESOURCES})
        channel_row.update(overlap_us=excl["overlap"]*tick_us, other_us=excl["other"]*tick_us,
                           boundary_us=boundary*tick_us)
        channels.append(channel_row)
        resources.append(dict(operator=directory.name, channel=channel,
            inclusive_union_us={k: incl[k]*tick_us for k in RESOURCES},
            overlap_subtraction_credit_us=overlap_credit*tick_us,
            summed_service_work_us={k[:-6]+"us":v*tick_us for k,v in work.items() if k.endswith("_ticks")},
            counts_and_bytes={k:v for k,v in work.items() if not k.endswith("_ticks")},
            completion_tail_union_us={kind: partition(0,total_ticks,{kind:spans})["inclusive"][kind]*tick_us
                                      for kind,spans in completion_tails.items()},
            completion_tail_note="Nonadditive completion/recovery windows after data service; local write nWR is not data-port occupancy."))
        events_evidence.append(dict(channel=channel, partitions=partitions))
    assert all_native_counts == Counter({k:v for k,v in result["issued_commands"].items() if v})
    critical = max(channels, key=lambda r:(r["last_service_tick"], -r["channel"]))
    return dict(operator=directory.name, row=critical, channel_rows=channels,
        phases=phases, resource_service=resources, traffic_bytes=result["traffic_bytes"],
        native_commands=dict(all_native_counts), native_file_sha256=hashes,
        trace_sha256=trace_hash, fresh_native_run=True, partitions=events_evidence,
        definitions=dict(mac_array="Native service including DRAM array operand reads; not isolated ALU time.",
            array_column="Native same-array column-spacing guard after MACK service: [issue+2 CK, issue+4 CK), intersected with pipeline phases. An attribution window, not an additional independently summed transfer latency.",
            bg_reduction="Four-bank local reduction is provisioned to keep pace with MAC_QK; a zero separate event column is not a measured zero physical delay.",
            multicast="WR_BG/WR_Q use command-level input traffic and readiness; no added fanout-dependent network delay.",
            input_window="Native data-service windows (2 CK per 32 B at the baseline); command-spacing gaps remain other unless another resource is active.",
            fenced_transfer="Measured phase includes its scheduling, row handling, and explicit tails.",
            pipeline="Native service-window union; simultaneous resources are a separate overlap column.",
            other="Unattributed elapsed scheduling/ACT/PRE/refresh/recovery/fence gaps; not isolated DRAM stall. K MAC service is 2 CK; its additional array column guard is separately attributed and is not arithmetic.",
            service_work="Summed work across potentially parallel resources, not additive wall time."))


def inventory():
    rows = []
    for name, x, w, y, place, measurement in (
        ("Q projection", [20,1024], [1024,2048], [20,8,256], "GPU", None),
        ("K projection", [20,1024], [1024,2048], [20,8,256], "PIM", "k"),
        ("V projection", [20,1024], [1024,2048], [20,8,256], "PIM", "v"),
        ("QK^T", [8,20,256], [8,256,988], [8,20,988], "PIM", "qk"),
        ("PV", [8,20,988], [8,988,256], [8,20,256], "PIM", "pv"),
        ("O projection", [20,2048], [2048,1024], [20,1024], "GPU", None),
        ("FFN Gate", [20,1024], [1024,4096], [20,4096], "PIM", "gate_up (joint)"),
        ("FFN Up", [20,1024], [1024,4096], [20,4096], "PIM", "gate_up (joint)"),
        ("FFN Down", [20,4096], [4096,1024], [20,1024], "PIM", "down_tiled"),
        ("Action input", [20,32], [32,1024], [20,1024], "not modeled", None),
        ("Action output", [20,1024], [1024,32], [20,32], "not modeled", None),
        ("Time MLP 1", [1,1024], [1024,1024], [1,1024], "precompute", None),
        ("Time MLP 2", [1,1024], [1024,1024], [1,1024], "precompute", None),
        ("AdaRMS conditioning (37 projections)", [1,1024], [1024,3072], [1,3072], "precompute", None),
    ):
        rows.append(dict(operator=name, input_shape=x, matrix_shape=w, output_shape=y,
                         execution=place, measured_case=measurement,
                         latency_us=None if measurement is None else "see measured case"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--ffn-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ffn = args.ffn_root or args.run_root / "ffn"
    output = args.output or args.run_root / "all-linear-breakdown"
    output.mkdir(parents=True, exist_ok=True)
    cases = [(op,args.run_root/op) for op in ("k","v","qk","pv")]
    cases += [(op,ffn/op) for op in ("gate_up","down_tiled")]
    evidence = {op:inspect(path) for op,path in cases}
    rows = [dict(evidence[op]["row"], operator=op) for op,_ in cases]
    subtotal = sum(row["total_us"] for row in rows)
    for row in rows:
        row["layers18_steps10_ms"] = row["total_us"] * 180 / 1000
    write_csv(output / "all-linear-breakdown.csv", rows)
    write_csv(output / "system-traffic.csv", [dict(operator=op,**evidence[op]["traffic_bytes"]) for op,_ in cases])
    write_csv(output / "channel-partitions.csv", [r for op,_ in cases for r in evidence[op]["channel_rows"]])
    resident = dict(attention_bytes_per_bank=155648,
        regions=[dict(name="W_K",start=0,end_exclusive=131072),
                 dict(name="W_V",start=131072,end_exclusive=139264),
                 dict(name="K cache",start=139264,end_exclusive=147456),
                 dict(name="V cache",start=147456,end_exclusive=155648),
                 dict(name="W_Gate + W_Up",start=155648,end_exclusive=188416),
                 dict(name="W_Down",start=188416,end_exclusive=204800)],
        total_bytes_per_bank=204800,total_bytes_per_layer=204800*512,
        no_overlap=True, excludes="GPU Q/O weights, action I/O and conditioning")
    assert all(a["end_exclusive"] == b["start"] for a,b in zip(resident["regions"],resident["regions"][1:]))
    payload = dict(primary_rows=rows, measured_pim_subset_us_per_layer_step=subtotal,
        analysis_source_sha256=digest(__file__),
        measured_pim_subset_ms_18layers_10steps=subtotal*180/1000,
        complete_ae_latency_ms=None, timing_scope="Separately timed native PIM stages; not an integrated AE chain.",
        unknown_latency_us=dict(GPU_Q=None,GPU_O=None,GPU_softmax=None,GPU_FFN_partial_reduce=None,
            GELU_multiply_cast=None,K_RoPE=None,normalization_residual=None,action_input_output=None,
            conditioning_precompute=None,GPU_dispatch_arrival_processing=None),
        linear_operator_inventory=inventory(), resident_layout=resident, operators=evidence,
        verification=dict(exclusive_tick_identities_exact=True, native_commands_match=True,
                          input_trace_hashes_match=True, analyzed_channels=len(rows)*16))
    dump(output / "breakdown.json", payload)
    dump(output / "linear-operator-inventory.json", inventory())
    dump(output / "resident-layout.json", resident)
    text = ["GPU-Q / BG-K native timing results", "", "case: native PIM service us per layer per denoise step"]
    text.extend(f"{r['operator']}: {r['total_us']:.6f}" for r in rows)
    text += ["", f"Measured PIM subset: {subtotal:.6f} us/layer/step; {subtotal*180/1000:.9f} ms x18x10.",
        "This is not complete AE latency. GPU Q/O, softmax, reduction, nonlinear, normalization and action I/O are unmeasured.",
        "Gate and Up share one measured fused schedule; do not divide its latency into two independent measurements.",
        "DRAM array operand reads are included in MAC service; no separate ALU-only latency is claimed.",
        "Exclusive CSV columns sum to wall time; resource work and inclusive unions must not be added again."]
    (output / "REPORT.txt").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps(dict(primary_rows=rows, measured_pim_subset_us=subtotal,
                         measured_pim_subset_ms_18x10=subtotal*180/1000), indent=2))


if __name__ == "__main__":
    main()

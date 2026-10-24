"""Independent native-event audit and attribution for rank-local FFN traces."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


NATIVE_TO_REQUEST = {"WRGB": "PIM_WR_GB", "MACAB": "PIM_MAC_AB", "RDRES": "PIM_RD_RES",
                     "RD": "LD", "RDA": "LD", "WR": "ST", "WRA": "ST"}
RESOURCES = ("wrgb", "local", "mac", "read")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tick(clock, ratio):
    return (int(clock)-1)*ratio+1


def decode(address):
    return (address >> 5 & 15, address >> 15 & 1, address >> 20,
            address >> 9 & 63, (address >> 20)*2048+(address >> 9 & 63)*32+(address & 31))


def read_native(path, channel, ratio):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{k: v if k == "command" else int(v) for k,v in raw.items()} for raw in csv.DictReader(stream)]
    check(all(row["Channel"] == channel for row in rows), "Native channel mismatch")
    check(all(a["clock"] <= b["clock"] for a,b in zip(rows, rows[1:])), "Nonmonotonic native clock")
    for row in rows:
        row["tick"] = tick(row["clock"], ratio)
    return rows


def read_local(path, channel, ratio, port_cycles):
    check(path.is_file(), f"Missing local-input event file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {"clock", "ready_at", "Channel", "Rank", "pim_buffer", "offset_bytes", "bytes",
                    "port_cycles", "buffer_in_use_until", "port_ready_before"}
        check(expected <= set(reader.fieldnames or ()), "Unexpected local-event CSV schema")
        rows = [{key: int(value) for key,value in row.items()} for row in reader]
    previous_ready = {0: 0, 1: 0}
    for row in rows:
        check(row["Channel"] == channel and row["Rank"] in (0,1), "Local event channel/rank mismatch")
        check(row["pim_buffer"] in (0,1), "Local input buffer index")
        check(row["bytes"] == 32 and row["port_cycles"] == port_cycles, "Local input service size/cycles")
        check(row["ready_at"] == row["clock"]+port_cycles, "Local completion latency mismatch")
        check(row["clock"] >= row["port_ready_before"], "Local port overlap within rank")
        check(row["clock"] >= row["buffer_in_use_until"], "Local write overwrites a consumed buffer")
        check(row["port_ready_before"] == previous_ready[row["Rank"]], "Local per-rank port predecessor mismatch")
        previous_ready[row["Rank"]] = row["ready_at"]
        row["tick"], row["ready_tick"] = tick(row["clock"], ratio), tick(row["ready_at"], ratio)
    return rows


def partition(start, stop, windows):
    """Four-resource exact union; bit0 WRGB, bit1 local, bit2 MAC, bit3 RD."""
    changes = defaultdict(lambda: [0]*4)
    changes[start]; changes[stop]
    for kind,name in enumerate(RESOURCES):
        for first,last in windows.get(name, ()):
            check(start <= first < last <= stop, f"{name} window outside pipeline")
            changes[first][kind] += 1
            changes[last][kind] -= 1
    counts, active, previous = Counter(), [0]*4, start
    for now in sorted(changes):
        mask = sum((1<<i) for i,value in enumerate(active) if value)
        counts[mask] += now-previous
        active = [a+b for a,b in zip(active, changes[now])]
        check(all(value >= 0 for value in active), "Negative interval count")
        previous = now
    check(active == [0]*4 and sum(counts.values()) == stop-start, "Unclosed pipeline partition")
    return {mask: counts[mask] for mask in range(16)}


def _kind(segment):
    name = segment["name"]
    if "gather" in name:
        return "gather"
    if "input_and_compute" in name or "pipeline" in name:
        return "pipeline"
    if "export" in name or "output" in name:
        return "export"
    raise AssertionError(f"Unknown FFN phase: {name}")


def _load(directory, manifest, result):
    directory = Path(directory)
    simulation = directory / "simulation"
    cfg, timings = result["config"], result["resolved_timings_ck"]
    ratio, channels, ranks = cfg["frontend_ratio"], cfg["channels"], cfg["ranks"]
    check((ratio, channels, ranks, cfg["memory_ratio"]) == (32,16,2,1), "Unexpected FFN topology")
    check(cfg["pim_input_broadcast"] and cfg["pim_mac_scope"] == "channel" and cfg["pim_data_path"] == "attacc", "Wrong FFN PIM mode")
    check(timings["nCCDAB"] == 20, "Expected twenty-query MAC service")
    local_ck = int(cfg["pim_local_transfer_cycles"])
    check(local_ck in (1,2,4), "Unexpected local-port model setting")
    check(manifest["pim_local_transfer_cycles"] == local_ck and manifest["local_input_external_dq_bytes"] == 0,
          "Manifest local-port policy differs from native configuration")
    check(manifest["integrated_chain"] is False and manifest["nonlinear_latency_ck"] is None,
          "Unmeasured nonlinear work must remain explicit")
    source_hash = sha256(directory / "workload.trace")
    check(source_hash == result["requests"]["sha256"] == sha256(simulation / "input.trace"), "Trace hash mismatch")
    trace, request_counts, expected_addresses = [], Counter(), defaultdict(Counter)
    for line in (directory / "workload.trace").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        op = parts[0]
        request_counts[op] += 1
        trace.append((op, *[int(value, 0) for value in parts[1:]]))
        if op in (*set(NATIVE_TO_REQUEST.values()), "PIM_LOCAL_GB"):
            ch,rank,row,col,offset = decode(int(parts[1],0))
            buf = int(parts[2]) if len(parts)>2 else -1
            key = (ch,row,col,buf) if op == "PIM_MAC_AB" else ((ch,rank,offset,buf) if op == "PIM_LOCAL_GB" else (ch,rank,row,col,buf))
            expected_addresses[op][key] += 1
    check(request_counts == Counter(manifest["counts"]), "Manifest/trace request counters differ")
    native, local, native_counts, actual_addresses = [], [], Counter(), defaultdict(Counter)
    native_hashes, local_hashes = {}, {}
    for channel in range(channels):
        np = simulation / f"timeline.csv.ch{channel}"
        lp = simulation / f"local_input.csv.ch{channel}"
        nr, lr = read_native(np, channel, ratio), read_local(lp, channel, ratio, local_ck)
        native.append(nr); local.append(lr)
        native_hashes[str(channel)], local_hashes[str(channel)] = sha256(np), sha256(lp)
        native_counts.update(row["command"] for row in nr)
        for event in nr:
            op = NATIVE_TO_REQUEST.get(event["command"])
            if op:
                key = ((channel,event["Row"],event["Column"],event["pim_buffer"]) if op == "PIM_MAC_AB"
                       else (channel,event["Rank"],event["Row"],event["Column"],event["pim_buffer"]))
                actual_addresses[op][key] += 1
        for event in lr:
            actual_addresses["PIM_LOCAL_GB"][(channel,event["Rank"],event["offset_bytes"],event["pim_buffer"])] += 1
    check(native_counts == Counter({key:value for key,value in result["issued_commands"].items() if value}), "Native command count mismatch")
    for op in set(NATIVE_TO_REQUEST.values()) | {"PIM_LOCAL_GB"}:
        check(actual_addresses[op] == expected_addresses[op], f"Native address multiset mismatch: {op}")
    nlocal = sum(len(rows) for rows in local)
    check(nlocal == request_counts["PIM_LOCAL_GB"] == result["requests"].get("pim_local_gb",0), "Local event/request count mismatch")
    traffic = result["traffic_bytes"]
    expected_bytes = dict(pim_input_write=request_counts["PIM_WR_GB"]*32,
                          pim_input_delivered=request_counts["PIM_WR_GB"]*64,
                          pim_result_read=request_counts["PIM_RD_RES"]*32,
                          pim_local_input_write=nlocal*32,
                          dram_read=0, dram_write=0,
                          pim_internal=request_counts["PIM_MAC_AB"]*32*32)
    for key,value in expected_bytes.items():
        check(traffic.get(key,0) == value, f"Traffic byte mismatch: {key}")
    check(request_counts["LD"] == request_counts["ST"] == 0, "FFN should not use external DRAM LD/ST")
    with (simulation / "barriers.csv").open(newline="") as stream:
        barriers = {int(row["marker"]): int(row["frontend_tick"]) for row in csv.DictReader(stream)}
    check(list(barriers) == list(range(len(barriers))), "Barrier marker sequence mismatch")
    phases, previous = [], None
    barrier_trace_indices = {item[1]: i for i,item in enumerate(trace) if item[0] == "BARRIER"}
    for segment in manifest["segments"]:
        start,stop = barriers[segment["start"]],barriers[segment["end"]]
        check(start < stop and (previous is None or start == previous), "Noncontiguous FFN phase")
        previous = stop
        actual = Counter()
        for rows in native:
            actual.update(NATIVE_TO_REQUEST[e["command"]] for e in rows if start<=e["tick"]<stop and e["command"] in NATIVE_TO_REQUEST)
        actual["PIM_LOCAL_GB"] = sum(start<=e["tick"]<stop for rows in local for e in rows)
        actual = +actual
        expected = Counter({key:value for key,value in segment["counts"].items()
                            if key in set(NATIVE_TO_REQUEST.values()) | {"PIM_LOCAL_GB"} and value})
        check(actual == expected, f"Phase request mismatch: {segment['name']}")
        kind = _kind(segment)
        if kind == "gather":
            body = trace[barrier_trace_indices[segment["start"]]+1:barrier_trace_indices[segment["end"]]]
            waits = [item[1] for item in body if item[0] == "WAIT"]
            check(waits == [640*local_ck], "Wrong assumed G/U gather port budget")
            check(stop-start >= 640*local_ck*ratio, "Gather phase shorter than declared budget")
            check(not actual, "Local gather must not generate external DQ or MAC requests")
        phases.append(dict(segment=segment, name=segment["name"], kind=kind, start=start, stop=stop, ticks=stop-start))
    for phase in phases:
        if phase["kind"] not in ("export", "gather"):
            continue
        for channel, rows in enumerate(native):
            previous_mac = [e for e in rows if e["command"] == "MACAB" and e["tick"] < phase["start"]]
            check(previous_mac, f"{phase['name']}/ch{channel}: no completed producer")
            producer_done = previous_mac[-1]["tick"]+20*ratio
            check(phase["start"] >= producer_done,
                  f"{phase['name']}/ch{channel}: producer MAC not drained")
            if phase["kind"] == "export":
                exports = [e for e in rows if e["command"] == "RDRES" and phase["start"]<=e["tick"]<phase["stop"]]
                check(exports and exports[0]["tick"] >= producer_done,
                      f"{phase['name']}/ch{channel}: RDRES before final MAC+20 CK")
                check(phase["stop"] >= exports[-1]["tick"]+4*ratio,
                      f"{phase['name']}/ch{channel}: missing final RDRES+4 CK tail")
    total_ticks = result["memory_cycles"]*ratio
    boundary = total_ticks-sum(item["ticks"] for item in phases)
    check(0 <= boundary <= 2*ratio, "Unexpected outer boundary duration")
    return dict(directory=directory, result=result, manifest=manifest, native=native, local=local, phases=phases,
                ratio=ratio, local_ck=local_ck, total_ticks=total_ticks, boundary_ticks=boundary,
                tick_us=result["tCK_ns"]/(ratio*1000), trace_hash=source_hash,
                native_hashes=native_hashes, local_hashes=local_hashes,
                request_counts=dict(request_counts), native_counts=dict(native_counts), traffic=expected_bytes)


def _partitions(data):
    ratio, tick_us = data["ratio"], data["tick_us"]
    channel_rows, details = [], []
    export_ticks = sum(p["ticks"] for p in data["phases"] if p["kind"] == "export")
    gather_ticks = sum(p["ticks"] for p in data["phases"] if p["kind"] == "gather")
    pipelines = [p for p in data["phases"] if p["kind"] == "pipeline"]
    check(pipelines, "Missing FFN pipeline phase")
    for channel in range(16):
        masks, fragments_evidence, last_mac = Counter(), [], 0
        for phase in pipelines:
            selected = [e for e in data["native"][channel] if phase["start"]<=e["tick"]<phase["stop"]]
            writes = [e for e in selected if e["command"] == "WRGB"]
            macs = [e for e in selected if e["command"] == "MACAB"]
            locals_ = [e for e in data["local"][channel] if phase["start"]<=e["tick"]<phase["stop"]]
            segment = phase["segment"]
            fragments = int(segment["fragment_count"])
            slots = int(segment["slots_per_fragment"])
            check(len(macs) == fragments*slots, "MAC fragment/slot coverage mismatch")
            check(bool(writes) != bool(locals_), "Pipeline must have exactly one input source")
            windows = {name: [] for name in RESOURCES}
            if writes:
                check(len(writes) == fragments*20, "WRGB fragment count mismatch")
            else:
                check(len(locals_) == fragments*20*2, "Local fragment/rank coverage mismatch")
                windows["local"] = [(e["tick"],e["ready_tick"]) for e in locals_]
            for fragment in range(fragments):
                buffer = fragment%2
                group_mac = macs[fragment*slots:(fragment+1)*slots]
                check(all(e["pim_buffer"] == buffer for e in group_mac), "MAC double-buffer order mismatch")
                ready = []
                if writes:
                    group = writes[fragment*20:(fragment+1)*20]
                    check(all(e["pim_buffer"] == buffer and e["Rank"] == 0 and e["pim_rank_mask"] == 3 for e in group), "WRGB rank broadcast or buffer mismatch")
                    check({e["Column"]*32 for e in group} == set(range(0,640,32)), "WRGB query-vector coverage")
                    check(group[-1]["tick"] <= group_mac[0]["tick"], "MAC started before complete WRGB fragment")
                    windows["wrgb"].append((group[0]["tick"],group[-1]["tick"]))
                    ready.append(group[-1]["tick"])
                else:
                    for rank in (0,1):
                        rank_events = [e for e in locals_ if e["Rank"] == rank]
                        group = rank_events[fragment*20:(fragment+1)*20]
                        check(all(e["pim_buffer"] == buffer for e in group), "LOCAL buffer order mismatch")
                        check({e["offset_bytes"] for e in group} == set(range(0,640,32)), "LOCAL query-vector coverage")
                        check(max(e["ready_tick"] for e in group) <= group_mac[0]["tick"], "MAC before complete rank-local fragment")
                        if fragment >= 2:
                            previous_mac = macs[(fragment-1)*slots-1]
                            check(group[0]["tick"] >= previous_mac["tick"]+20*ratio,
                                  "Local double-buffer reuse before previous consumer finished")
                        ready.append(max(e["ready_tick"] for e in group))
                fragments_evidence.append(dict(phase=phase["name"], fragment=fragment, buffer=buffer,
                                               input_ready_ticks=ready, first_mac_tick=group_mac[0]["tick"],
                                               last_mac_tick=group_mac[-1]["tick"], mac_slots=slots))
            windows["mac"] = [(e["tick"],e["tick"]+20*ratio) for e in macs]
            local_masks = partition(phase["start"], phase["stop"], windows)
            check(sum(value for mask,value in local_masks.items() if mask&4) == len(macs)*20*ratio,
                  "MAC service windows overlap")
            masks.update(local_masks)
            last_mac = max(last_mac, macs[-1]["clock"])
        incl = {name: sum(value for mask,value in masks.items() if mask&(1<<i)) for i,name in enumerate(RESOURCES)}
        overlap_union = sum(value for mask,value in masks.items() if mask.bit_count()>=2)
        credit = sum((mask.bit_count()-1)*value for mask,value in masks.items() if mask.bit_count()>=2)
        parts = dict(wrgb_input_window=incl["wrgb"], local_input_window=incl["local"], mac_service=incl["mac"],
                     overlap=credit, buffer_output_phase=export_ticks, local_gather_phase=gather_ticks,
                     dram_read_phase=incl["read"], dram_write_phase=0, other_pipeline=masks[0],
                     boundary=data["boundary_ticks"], total=data["total_ticks"],
                     wrgb_input_only=masks[1], local_input_only=masks[2], compute_only=masks[4],
                     dram_read_only=masks[8], overlap_union=overlap_union)
        elapsed = sum(incl.values())-credit+export_ticks+gather_ticks+masks[0]+data["boundary_ticks"]
        check(elapsed == data["total_ticks"], "FFN channel wall-time identity failed")
        row = dict(operator=data["directory"].name, channel=channel, critical_channel=channel)
        row.update({key+"_us":value*tick_us for key,value in parts.items()})
        row["identity_error_ticks"] = 0
        channel_rows.append(row)
        details.append(dict(channel=channel, last_mac_issue_ck=last_mac, partition_ticks=parts,
                            exclusive_masks_ticks=dict(masks), fragment_dependencies=fragments_evidence))
    return channel_rows, details


def audit_run(directory, manifest, result):
    data = _load(Path(directory), manifest, result)
    rows, details = _partitions(data)
    return dict(native_timeline_hashes=data["native_hashes"], local_event_hashes=data["local_hashes"],
                native_timelines_checked=16, local_event_files_checked=16,
                request_counts=data["request_counts"], native_counts=data["native_counts"],
                local_event_count=sum(len(rows) for rows in data["local"]),
                local_transfer_cycles=data["local_ck"], all_rank_input_dependencies_verified=True,
                no_local_dq_traffic=True, all_integer_tick_partitions_exact=True,
                channel_fragment_dependencies_checked=sum(len(item["fragment_dependencies"]) for item in details),
                modeled_gelu_multiply_cast_latency_us=None, modeled_gpu_reduce_latency_us=None)


def inspect_operator(root, operator):
    directory = Path(root) / operator
    manifest = load_json(directory / "manifest.json")
    result = load_json(directory / "simulation/result.json")
    summary = load_json(directory / "summary.json")
    check(summary["audit_status"] == "passed", "Prior FFN run audit did not pass")
    data = _load(directory, manifest, result)
    check(data["native_hashes"] == summary["audit"]["native_timeline_hashes"], "Recorded native timeline hashes changed")
    check(data["local_hashes"] == summary["audit"]["local_event_hashes"], "Recorded local event hashes changed")
    rows, details = _partitions(data)
    critical = max(details, key=lambda item: (item["last_mac_issue_ck"],-item["channel"]))["channel"]
    chosen = dict(rows[critical]); chosen.pop("channel")
    check(abs(chosen["total_us"]-summary["latency_us"])<1e-9, "FFN summary wall time mismatch")
    resources = []
    for channel in range(16):
        native = Counter(e["command"] for e in data["native"][channel])
        for rank in (0,1):
            nlocal = sum(e["Rank"]==rank for e in data["local"][channel])
            resources.append(dict(operator=operator, channel=channel, rank=rank,
                                  local_input_commands=nlocal, local_input_bytes=nlocal*32,
                                  local_port_work_us=nlocal*data["local_ck"]*result["tCK_ns"]/1000,
                                  channel_wrgb_commands=native["WRGB"], channel_macab_commands=native["MACAB"],
                                  observed_operator_wall_us=chosen["total_us"]))
    count = data["request_counts"]
    traffic = dict(operator=operator, channels=16, wrgb_commands=count.get("PIM_WR_GB",0),
                   rdres_commands=count.get("PIM_RD_RES",0), macab_commands=count.get("PIM_MAC_AB",0),
                   local_input_commands=count.get("PIM_LOCAL_GB",0), ordinary_rd_commands=0, ordinary_wr_commands=0,
                   buffer_input_bytes=data["traffic"]["pim_input_write"],
                   buffer_input_rank_delivered_bytes=data["traffic"]["pim_input_delivered"],
                   local_input_bytes=data["traffic"]["pim_local_input_write"],
                   buffer_output_bytes=data["traffic"]["pim_result_read"], ordinary_dram_read_bytes=0,
                   ordinary_dram_write_bytes=0, mac_internal_array_read_bytes=data["traffic"]["pim_internal"])
    evidence = dict(wall_time_breakdown=chosen, channels=details,
                    phases=[{key:value for key,value in phase.items() if key!="segment"} for phase in data["phases"]],
                    local_transfer_cycles=data["local_ck"], local_event_count=sum(len(rows) for rows in data["local"]),
                    native_timeline_hashes=data["native_hashes"], local_event_hashes=data["local_hashes"],
                    trace_sha256=data["trace_hash"], request_counts=count, traffic_bytes=data["traffic"],
                    source_files={str(path.relative_to(directory)):sha256(path) for path in
                                  (directory/"manifest.json", directory/"summary.json", directory/"simulation/result.json",
                                   directory/"simulation/barriers.csv")},
                    tCK_ns=result["tCK_ns"], frontend_ratio=data["ratio"],
                    tick_us=data["tick_us"], total_ticks=data["total_ticks"],
                    explicit_unknown_latency_us={"GELU":None, "multiply_cast":None,"GPU_reduce":None})
    return chosen, rows, resources, traffic, evidence

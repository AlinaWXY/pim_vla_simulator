"""Independent H16 address, issued-command and shared-bank-slot audit.

This module deliberately imports no workload or layout generator. It checks
the simulation's issued events against fixed matrix geometry and reconstructs
each physical bank's two slot lifetimes from native completion events.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

GEOMETRY = ("Channel", "Rank", "BankGroup", "Bank", "Row", "Column")
NATIVE = dict(PIM_WR_GB="WRGB", PIM_MAC_AB="MACAB", PIM_RD_RES="RDRES",
    PIM_SPILL="LOCALWR", PIM_ROPE_COEF="LOCALRD", PIM_ROPE_LOAD="LOCALRD",
    PIM_ROPE_STORE="LOCALWR", PIM_Q_EXPORT="RD", PIM_Q_REMOTE="WRGB",
    PIM_MAC_QK="MACQK", PIM_RD_BG="RDRES", PIM_Z_STORE="LOCALWR",
    PIM_Z_LOAD="LOCALRD", LD="RD", ST="WR")
EXTERNAL = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}
RANK_COMMANDS = {"WRGB", "MACAB", "MACQK", "RDRES"}
FILL_OPS = {"GEMV_MULTICAST", "GEMV_Z_SCATTER", "GEMV_Q_LOCAL", "GEMV_Q_REMOTE"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        result = []
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = int(value)
                except (ValueError, TypeError):
                    parsed[key] = value
            result.append(parsed)
        return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(address):
    return address >> 5 & 15, address >> 15 & 1, address >> 16 & 3, address >> 18 & 3, address >> 20, address >> 9 & 63


def trace_records(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        words = line.split()
        if words and not words[0].startswith("#"):
            records.append(dict(line=number, operation=words[0], address=int(words[1], 0),
                buffer=int(words[2]) if len(words) > 2 else -1))
    return records


def trace_key(record):
    return record["operation"], *decode(record["address"]), record["buffer"]


def expected_addresses(case):
    """Derive every request independently from matrix dimensions and ownership."""
    if case == "serial_layer_control":
        return sum((expected_addresses(stage) for stage in
                    ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram")), Counter())
    expected = Counter()
    def add(op, ch, offset, slot=-1, bank=0):
        expected[op, ch, 1, bank//4, bank%4, offset//2048, offset%2048//32, slot] += 1
    def release(slots=(0, 1)):
        for ch in range(16):
            for slot in slots:
                add("PIM_GEMV_RELEASE", ch, 0, slot)
    def ordinary(base, fragments, columns, rows):
        for ch in range(16):
            for f in range(fragments):
                for q in range(rows):
                    add("PIM_WR_GB", ch, q*32, f%2)
                for column in range(columns):
                    add("PIM_MAC_AB", ch, base+(f*columns+column)*32, f%2)
    if case == "chain":
        ordinary(0, 64, 32, 8)
        release()
        for ch in range(16):
            for bank in range(16):
                for q in range(8):
                    add("PIM_ROPE_COEF", ch, 81920+q*32, bank=bank)
                    for p in range(2):
                        offset = 81920+(8+8*p+q)*32
                        add("PIM_SPILL", ch, offset, bank=bank)
                        add("PIM_ROPE_LOAD", ch, offset, p, bank)
                        add("PIM_ROPE_EXEC", ch, offset, p, bank)
                        add("PIM_ROPE_STORE", ch, offset, bank=bank)
                    add("PIM_Q_EXPORT", ch, 81920+(8+q)*32, bank=bank)
                    add("PIM_Q_REMOTE", ch, 81920+(8+q)*32, 0, bank)
            for k in range(496):
                offset = 65536+k*32 if k < 484 else 81920+(16+k-484)*32
                add("PIM_MAC_QK", ch, offset, 0)
            for offset in range(0, 496*16*4, 32):
                add("PIM_RD_BG", ch, offset)
        release((0,))
    elif case in ("v", "pv"):
        ordinary(83968 if case == "v" else 100352, 64 if case == "v" else 62, 8, 16)
        release()
        for ch in range(16):
            for offset in range(0, 16*16*8*4, 32):
                add("PIM_RD_RES", ch, offset)
        if case == "v":
            for head in range(8):
                for d in range(256):
                    feature_slot, owner = divmod(d, 32)
                    parity, bank = divmod(owner, 16)
                    ch = head*2+parity
                    add("LD", ch, 100352+(60*8+feature_slot)*32, bank=bank)
                    for block in (60, 61):
                        add("ST", ch, 100352+(block*8+feature_slot)*32, bank=bank)
    else:
        require(case in ("gate_up_z_store", "down_tiled_dram", "combined_timing_chain"), "Unknown case")
        if case != "down_tiled_dram":
            ordinary(262144, 64, 32, 16)
            release()
            for ch in range(16):
                for bank in range(16):
                    for f in range(16):
                        add("PIM_Z_STORE", ch, 360448+f*32, bank=bank)
        if case != "gate_up_z_store":
            for tile in range(4):
                for ch in range(16):
                    for f in range(16):
                        for bank in range(16):
                            add("PIM_Z_LOAD", ch, 360448+f*32, f%2, bank)
                        for column in range(16):
                            add("PIM_MAC_AB", ch, 327680+((tile*16+f)*16+column)*32, f%2)
                    for offset in range(0, 16*256*4, 32):
                        add("PIM_RD_RES", ch, offset)
                release()
    return expected


def native_key(command, geometry):
    ch, rank, bg, bank, row, column = geometry
    if command in RANK_COMMANDS:
        bg = bank = -1
    return command, ch, rank, bg, bank, row, column


def audit_native_commands(records, native):
    expected = Counter(native_key(NATIVE[e["operation"]], decode(e["address"]))
                       for e in records if e["operation"] in NATIVE)
    actual = Counter(native_key({"RDA": "RD", "WRA": "WR"}.get(e["command"], e["command"]),
                                tuple(e[k] for k in GEOMETRY))
                     for e in native if e["command"] in set(NATIVE.values()) | {"RDA", "WRA"})
    require(expected == actual, f"Native issued command/address coverage differs: missing={expected-actual}, extra={actual-expected}")
    for event in native:
        if event["command"] in set(NATIVE.values()) | {"RDA", "WRA"}:
            require(event["Rank"] == 1, "PIM work touched physical rank0")
    return sum(actual.values())


def audit_array_rows(native):
    opened, checks = {}, 0
    for event in sorted(native, key=lambda e: (e["Channel"], e["clock"])):
        command = event["command"]
        ranks = range(2) if event["Rank"] < 0 else (event["Rank"],)
        groups = range(4) if event["BankGroup"] < 0 else (event["BankGroup"],)
        banks = range(4) if event["Bank"] < 0 else (event["Bank"],)
        if command in ("ACTAB", "PREab", "REFab"):
            groups, banks = range(4), range(4)
        targets = [(event["Channel"], r, g, b) for r in ranks for g in groups for b in banks]
        if command in ("ACT2", "ACTAB", "ACTHB"):
            for target in targets:
                opened[target] = event["Row"]
        elif command in ("PREpb", "PREab", "PREHB", "REFab", "REFpb"):
            for target in targets:
                opened.pop(target, None)
        elif command in ("MACAB", "MACQK", "LOCALRD", "LOCALWR", "RD", "RDA", "WR", "WRA"):
            for target in targets:
                require(opened.get(target) == event["Row"], f"Array command uses wrong/closed row: {event}")
                checks += 1
            if command in ("RDA", "WRA"):
                for target in targets:
                    opened.pop(target, None)
    return checks


def audit_slot_events(events, case, require_final_release=True, mac_context=None):
    """Replay masks per physical bank, never trusting a reported 'ready' bit."""
    states, port_until, counts = {}, defaultdict(int), Counter()
    generations, released, max_fanout = 0, 0, 0
    def new_state(generation):
        return dict(generation=generation, lanes=[0]*16, ready=0, busy=0,
                    kind=None, macs=0, limit=None, held=False, released=False)
    for event in sorted(enumerate(events), key=lambda p: (p[1]["Channel"], p[1]["clock"], p[0])):
        event = event[1]
        op, ch, slot = event["operation"], event["Channel"], event["buffer"]
        require(op in FILL_OPS | {"GEMV_MAC", "GEMV_QK", "GEMV_RELEASE"}, f"Unknown GEMV event: {op}")
        require(event["Rank"] == 1 and 0 <= ch < 16 and slot in (0, 1), "Invalid GEMV target")
        start, end = event["clock"], event["ready_at"]
        require(end >= start and event["service_cycles"] == end-start, "GEMV duration differs from completion")
        bank_mask = event["bank_mask"]
        require(0 < bank_mask <= 65535, "Invalid physical-bank mask")
        banks = [b for b in range(16) if bank_mask >> b & 1]
        counts[op] += 1
        max_fanout = max(max_fanout, len(banks))
        if op in FILL_OPS:
            require(event["bytes"] == 32, "GEMV input transaction is not one32B source column")
            require(start >= port_until[ch], "Overlapping writes on rank GEMV distribution port")
            port_until[ch] = end
            require(event["service_cycles"] >= 2, "GEMV multicast/scatter cannot be zero latency")
            if op == "GEMV_MULTICAST":
                require(bank_mask == 65535 and event["token_mask"].bit_count() == 1 and event["lane_mask"] == 65535,
                        "Ordinary multicast must write one complete token row to all16 banks")
            elif op == "GEMV_Z_SCATTER":
                require(bank_mask == 65535 and event["token_mask"] == 65535 and event["lane_mask"].bit_count() == 1,
                        "Down scatter must route one feature column across all16 tokens to every bank")
            else:
                require(len(banks) == 1 and event["token_mask"].bit_count() == 1 and event["lane_mask"] == 65535,
                        "Q tap/remote must fill exactly one query row in its feature-owning bank")
        else:
            require(bank_mask == 65535, "Rank consume/release does not cover all16 physical banks")
        for bank in banks:
            key = ch, slot, bank
            state = states.get(key)
            if op in FILL_OPS and (state is None or event["generation"] != state["generation"]):
                if state is not None:
                    require(event["generation"] == state["generation"]+1, "Slot generation skipped/repeated")
                    require(start >= state["busy"], "Input overwrote consuming slot")
                    require(state["released"] or (not state["held"] and state["macs"] == state["limit"]),
                            "New fill replaced incomplete or Q-held slot")
                states[key] = state = new_state(event["generation"])
                generations += 1
            require(state is not None, "Consume/release has no physical slot allocation")
            require(event["generation"] == state["generation"], "Event references wrong slot generation")
            if op in FILL_OPS:
                require(not state["released"] and start >= state["busy"] and state["macs"] == 0,
                        "Input overwrote consuming/released slot")
                kind = "Q" if op.startswith("GEMV_Q_") else "Z" if op == "GEMV_Z_SCATTER" else "X"
                require(state["kind"] in (None, kind), "Different input kinds alias one physical slot")
                state["kind"] = kind
                state["held"] = kind == "Q"
                for token in range(16):
                    if event["token_mask"] >> token & 1:
                        require(not (state["lanes"][token] & event["lane_mask"]), "Duplicate input lane overwrites unconsumed data")
                        state["lanes"][token] |= event["lane_mask"]
                state["ready"] = max(state["ready"], end)
            elif op in ("GEMV_MAC", "GEMV_QK"):
                rows = 8 if case == "chain" and op == "GEMV_MAC" else 16
                columns = 16 if state["kind"] == "Z" else 8 if case in ("v", "pv") else 32
                if op == "GEMV_MAC" and mac_context is not None:
                    row = mac_context[ch, start]
                    if 0 <= row < 32:
                        rows, columns, kind = 8, 32, "X"
                    elif 41 <= row < 57:
                        rows, columns, kind = 16, 8, "X"
                    elif 128 <= row < 160:
                        rows, columns, kind = 16, 32, "X"
                    elif 160 <= row < 176:
                        rows, columns, kind = 16, 16, "Z"
                    else:
                        raise AssertionError("Native MAC address outside independently defined H16 operators")
                    require(state["kind"] == kind, "Native operator consumes wrong input producer kind")
                require(event["token_mask"] == (1 << rows)-1 and event["lane_mask"] == 65535, "MAC token/lane footprint wrong")
                require(not state["released"] and all(x == 65535 for x in state["lanes"][:rows]),
                        "MAC consumes missing bank/token/lane input")
                require(start >= max(state["ready"], state["busy"]), "MAC begins before fill/prior consume completes")
                require((op == "GEMV_QK") == state["held"], "QKT and ordinary MAC use incompatible slot owner")
                limit = 496 if op == "GEMV_QK" else columns
                state["limit"] = limit
                state["macs"] += 1
                require(state["macs"] <= limit, "Too many weight columns consumed from input fragment")
                require(event["mac_columns"] == state["macs"], "Native MAC column ordinal differs from reconstructed lifetime")
                require(event["service_cycles"] == rows, "Native GEMV compute service does not match token count")
                state["busy"] = end
            else:
                require(start >= max(state["ready"], state["busy"]), "Release precedes input/consume completion")
                require(not state["released"] and state["macs"] == state["limit"], "Release discards incomplete input/weight work")
                state["released"] = True
                state["held"] = False
                released += 1
    if require_final_release:
        require(states and all(s["released"] for s in states.values()), "Live slot ownership remains after the stage")
    return dict(events_checked=sum(counts.values()), operation_counts=dict(counts),
                bank_generations_checked=generations, physical_bank_releases= released,
                maximum_destination_fanout=max_fanout, input_bytes_per_bank=1024,
                rank_full_input_SRAM_bytes=0, independent_mask_reconstruction=True)


def audit_ffn_streams(simulation, gemv_events, case):
    """Join local array reads/writes, bounded nonlinear staging and shared port."""
    paths = [simulation / f"ffn_dram_events.csv.ch{ch}" for ch in range(16)]
    require(all(p.is_file() for p in paths), "Missing native FFN movement evidence")
    events = [e for p in paths for e in read_csv(p)]
    by_op = defaultdict(list)
    for e in events:
        by_op[e["operation"]].append(e)
    def key(e):
        return tuple(e[k] for k in GEOMETRY)
    gu = case != "down_tiled_dram"
    down = case != "gate_up_z_store"
    for op in ("PIM_Z_GATHER", "PIM_Z_RETURN", "PIM_Z_STORE"):
        require(len(by_op[op]) == (4096 if gu else 0), f"Incorrect {op} physical-column count")
    for op in ("PIM_Z_LOAD", "PIM_Z_LOAD_PORT"):
        require(len(by_op[op]) == (16384 if down else 0), f"Incorrect {op} physical-column count")
    producer = defaultdict(int)
    for e in by_op["PIM_MAC_AB"]:
        if 128 <= e["Row"] < 160:
            producer[e["Channel"]] = max(producer[e["Channel"]], e["ready_at"])
    gathers = {key(e): e for e in by_op["PIM_Z_GATHER"]}
    returns = {key(e): e for e in by_op["PIM_Z_RETURN"]}
    stores = {key(e): e for e in by_op["PIM_Z_STORE"]}
    require(len(gathers) == len(by_op["PIM_Z_GATHER"]), "Repeated or missing GU feature gather")
    require(gathers.keys() == returns.keys() == stores.keys(), "GU/Z producer and local-store addresses differ")
    for location, gather in gathers.items():
        returned, stored = returns[location], stores[location]
        require(gather["bytes"] == 128 and gather["ready_at"]-gather["clock"] == 8,
                "GU per-feature bounded gather must transfer128B over4x2CK port service")
        require(gather["clock"] >= producer[gather["Channel"]], "Nonlinear input gathered before GU accumulation completes")
        require(returned["bytes"] == 32 and returned["clock"] >= gather["ready_at"] and
                returned["ready_at"]-returned["clock"] == 2, "Z return precedes its FP32 operands or has wrong port service")
        require(stored["clock"] >= returned["ready_at"] and stored["ready_at"] > stored["clock"],
                "Z LOCALWR starts before bounded return or lacks native completion")
    loads_by_location, ports_by_location = defaultdict(list), defaultdict(list)
    for e in by_op["PIM_Z_LOAD"]:
        loads_by_location[key(e)].append(e)
        if case in ("combined_timing_chain", "serial_layer_control"):
            require(key(e) in stores and e["clock"] >= stores[key(e)]["ready_at"],
                    "Combined Down reads a Z column before its native store completes")
    for e in by_op["PIM_Z_LOAD_PORT"]:
        ports_by_location[key(e)].append(e)
    require(loads_by_location.keys() == ports_by_location.keys(), "Down array and transpose source identities differ")
    for location, loads in loads_by_location.items():
        loads.sort(key=lambda e: e["clock"])
        ports = sorted(ports_by_location[location], key=lambda e: e["clock"])
        require(len(loads) == len(ports) == 4, "Four Down output tiles must reread each Z column four times")
        for read, port in zip(loads, ports):
            require(read["bytes"] == port["bytes"] == 32 and port["clock"] >= read["ready_at"],
                    "Down scatter begins before local DRAM read completion")
            require(read["buffer"] == port["buffer"] and port["generation"] == read["generation"],
                    "Down transpose source switches buffer/generation")
    expected_scatter = Counter((e["Channel"], e["clock"], e["ready_at"], e["buffer"],
                              1 << (4*e["BankGroup"]+e["Bank"])) for e in by_op["PIM_Z_LOAD_PORT"])
    actual_scatter = Counter((e["Channel"], e["clock"], e["ready_at"], e["buffer"], e["lane_mask"])
                            for e in gemv_events if e["operation"] == "GEMV_Z_SCATTER")
    require(expected_scatter == actual_scatter, "Native source-column transpose is not the shared-bank scatter")
    # LOAD_PORT is deliberately omitted: it is the same interval as SCATTER.
    port_events = [e for e in gemv_events if e["operation"] in FILL_OPS]
    port_events += by_op["PIM_Z_GATHER"]+by_op["PIM_Z_RETURN"]
    until = defaultdict(int)
    for e in sorted(port_events, key=lambda e: (e["Channel"], e["clock"])):
        require(e["clock"] >= until[e["Channel"]], "GU/Z/GEMV overlap the same physical rank distribution port")
        until[e["Channel"]] = e["ready_at"]
    return dict(native_local_events_checked=len(events), unified_port_intervals_checked=len(port_events),
        GU_gather_bytes=len(gathers)*128, Z_local_write_bytes=len(stores)*32,
        Z_local_read_bytes=len(by_op["PIM_Z_LOAD"])*32,
        Down_multicast_destination_bytes=len(by_op["PIM_Z_LOAD_PORT"])*32*16,
        combined_store_to_read_dependencies=case in ("combined_timing_chain", "serial_layer_control"),
        full_rank_Z_SRAM_bytes=0, full_rank_GU_SRAM_bytes=0,
        native_sha256={p.name: digest(p) for p in paths})


def audit_trace(records, case):
    actual = Counter(trace_key(e) for e in records if e["operation"] not in ("BARRIER", "WAIT"))
    expected = expected_addresses(case)
    require(actual == expected, f"Trace differs from independent matrix/address oracle: missing={expected-actual}, extra={actual-expected}")
    require(not any(e["operation"] == "WAIT" for e in records), "Artificial legacy completion WAIT retained")
    return len(records)


def audit_mac_order(native, case):
    """A shared producer/consumer permutation must not evade multiset checks."""
    stages = ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram") if case == "serial_layer_control" else (case,)
    ab, qk = [], []
    for stage in stages:
        if stage == "chain":
            ab.extend(range(0, 65536, 32))
            qk.extend(65536+k*32 if k < 484 else 81920+(16+k-484)*32 for k in range(496))
        elif stage in ("v", "pv"):
            base, columns = (83968, 512) if stage == "v" else (100352, 496)
            ab.extend(base+32*i for i in range(columns))
        elif stage in ("gate_up_z_store", "down_tiled_dram", "combined_timing_chain"):
            if stage != "down_tiled_dram":
                ab.extend(range(262144, 327680, 32))
            if stage != "gate_up_z_store":
                ab.extend(range(327680, 360448, 32))
    for ch in range(16):
        selected = sorted((e for e in native if e["Channel"] == ch), key=lambda e: e["clock"])
        for command, expected in (("MACAB", ab), ("MACQK", qk)):
            actual = [e["Row"]*2048+e["Column"]*32 for e in selected if e["command"] == command]
            require(actual == expected, f"Native {command} order differs from independent operator/reduction order on channel{ch}")
    return 16*(len(ab)+len(qk))


def audit_serial_continuity(manifest, result, simulation, native, events):
    require(result["config"]["refresh"] is True, "Serial resource control must retain enabled refresh")
    required = ("chain", "v", "pv", "gate_up_z_store", "down_tiled_dram")
    require(tuple(p["stage"] for p in manifest["parts"]) == required, "Serial control stage order differs")
    for first, second in zip(manifest["parts"], manifest["parts"][1:]):
        require(first["end_marker"] == second["start_marker"] and first["last_line"]+1 == second["first_line"],
                "Serial stages do not share a continuous trace boundary")
    barriers = {e["marker"]: e["frontend_tick"] for e in read_csv(simulation / "barriers.csv")}
    require(sorted(barriers) == list(range(len(barriers))), "Serial barrier IDs reset or skip across stages")
    ratio = result["config"]["frontend_ratio"]
    q_end = barriers[manifest["parts"][0]["end_marker"]]
    refresh = [e for e in native if e["command"].startswith("REF")]
    for ch in range(16):
        clocks = [(e["clock"]-1)*ratio+1 for e in refresh if e["Channel"] == ch]
        require(any(c < q_end for c in clocks) and any(c >= q_end for c in clocks),
                "No native refresh evidence before and after QKT/V resource transition")
    return dict(single_native_simulation=True, continuous_refresh_enabled=True,
        native_refresh_commands=len(refresh), refresh_observed_before_and_after_QKT=True,
        global_barriers_without_reset=True, all_five_stages_share_physical_bank_slots=True,
        excluded_producer_latencies_are_unknown=True, full_AE_end_to_end_latency_us=None)


def audit_run(directory):
    directory = Path(directory)
    manifest, native_result = read_json(directory / "manifest.json"), read_json(directory / "simulation/result.json")
    case = manifest["case"]
    records = trace_records(directory / "workload.trace")
    audit_trace(records, case)
    simulation = directory / "simulation"
    timelines = [simulation / f"timeline.csv.ch{ch}" for ch in range(16)]
    gemv_files = [simulation / f"unified_gemv.csv.ch{ch}" for ch in range(16)]
    require(all(p.is_file() for p in timelines+gemv_files), "Missing channel native evidence")
    native = [e for p in timelines for e in read_csv(p)]
    events = [e for p in gemv_files for e in read_csv(p)]
    issued_count = audit_native_commands(records, native)
    row_checks = audit_array_rows(native)
    order_checks = audit_mac_order(native, case)
    mac_context = {(e["Channel"], e["clock"]): e["Row"] for e in native if e["command"] == "MACAB"}
    slots = audit_slot_events(events, case, mac_context=mac_context)
    counts = Counter(e["operation"] for e in records)
    native_counts = Counter(e["command"] for e in native)
    require(slots["operation_counts"].get("GEMV_MAC", 0) == counts["PIM_MAC_AB"], "Missing native bank GEMV consume events")
    require(slots["operation_counts"].get("GEMV_QK", 0) == counts["PIM_MAC_QK"], "Missing shared-slot QKT consumes")
    require(slots["operation_counts"].get("GEMV_MULTICAST", 0) == counts["PIM_WR_GB"], "Missing ordinary bank multicast writes")
    require(slots["operation_counts"].get("GEMV_Z_SCATTER", 0) == counts["PIM_Z_LOAD"], "Missing Down column transpose/multicast")
    require(slots["operation_counts"].get("GEMV_RELEASE", 0) == counts["PIM_GEMV_RELEASE"], "Release event count mismatch")
    require(slots["operation_counts"].get("GEMV_Q_LOCAL", 0) == counts["PIM_Q_EXPORT"], "Missing local Q tap writes to shared slots")
    require(slots["operation_counts"].get("GEMV_Q_REMOTE", 0) == counts["PIM_Q_REMOTE"], "Missing peer Q writes to shared slots")
    for operation, command in (("GEMV_MAC", "MACAB"), ("GEMV_QK", "MACQK")):
        require(Counter((e["Channel"], e["clock"]) for e in events if e["operation"] == operation) ==
                Counter((e["Channel"], e["clock"]) for e in native if e["command"] == command),
                "Shared-bank consume events do not correspond to actual native MAC issue")
    dq = sum(native_counts[c]*32 for c in EXTERNAL)
    require(dq == manifest["expected_dq_bytes"]["total"], "External DQ differs from source contract")
    require(Counter(native_result["issued_commands"]) == native_counts, "Native summary differs from command files")
    ffn = audit_ffn_streams(simulation, events, case) if case not in ("chain", "v", "pv") else None
    continuity = audit_serial_continuity(manifest, native_result, simulation, native, events) if case == "serial_layer_control" else None
    return dict(status="passed", case=case, horizon=16,
        trace_and_native_address_coverage=True, native_data_commands_checked=issued_count,
        array_open_row_checks=row_checks, independent_MAC_order_checks=order_checks,
        unified_gemv=slots, ffn_streams=ffn, serial_continuity=continuity,
        native_external_DQ_bytes=dq, native_command_counts=dict(native_counts),
        trace_sha256=digest(directory / "workload.trace"),
        native_sha256={p.name: digest(p) for p in timelines+gemv_files},
        numerical_compute=False, unmeasured="GPU/nonlinear arithmetic and GPU contention")

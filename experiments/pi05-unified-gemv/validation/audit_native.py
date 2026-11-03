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
    PIM_MAC_QK="MACQK", PIM_RD_BG="RDRES", LD="RD", ST="WR", PIM_STAGE_LOAD="LOCALRD")
EXTERNAL = {"RD", "RDA", "WR", "WRA", "WRGB", "RDRES"}
RANK_COMMANDS = {"WRGB", "MACAB", "MACQK", "RDRES"}
FILL_OPS = {"GEMV_MULTICAST", "GEMV_Q_LOCAL", "GEMV_Q_REMOTE"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def counter_difference(expected, actual):
    missing, extra = expected-actual, actual-expected
    return (f"missing_count={sum(missing.values())}, extra_count={sum(extra.values())}; "
            f"first_missing={list(missing.items())[:5]}, first_extra={list(extra.items())[:5]}")


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
    stage_rows = {}
    for number, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        words = line.split()
        if words and not words[0].startswith("#"):
            record = dict(line=number, operation=words[0], address=int(words[1], 0),
                buffer=int(words[2]) if len(words) > 2 and words[0] != "PIM_STAGE_BEGIN" else -1)
            if words[0] == "PIM_STAGE_BEGIN":
                require(len(words) == 4, "Staging BEGIN requires byte count and token rows")
                record.update(stage_bytes=int(words[2]), stage_rows=int(words[3]))
                stage_rows[decode(record["address"])[0]] = record["stage_rows"]
            elif words[0] == "PIM_STAGE_FRAGMENT":
                channel = decode(record["address"])[0]
                require(channel in stage_rows, "Fragment descriptor has no declared staging geometry")
                record["fragment_rows"] = stage_rows[channel]
            records.append(record)
    return records


def trace_key(record):
    key = record["operation"], *decode(record["address"]), record["buffer"]
    return (*key, record["stage_bytes"], record["stage_rows"]) if record["operation"] == "PIM_STAGE_BEGIN" else key


def expected_addresses(case, rank_staging=True):
    """Derive every request independently from matrix dimensions and ownership."""
    if case == "serial_layer_control":
        return sum((expected_addresses(stage, rank_staging=rank_staging) for stage in
                    ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")), Counter())
    expected = Counter()
    def add(op, ch, offset, slot=-1, bank=0):
        expected[op, ch, 1, bank//4, bank%4, offset//2048, offset%2048//32, slot] += 1
    def release(slots=(0, 1)):
        for ch in range(16):
            for slot in slots:
                add("PIM_GEMV_RELEASE", ch, 0, slot)
    def ordinary(base, fragments, columns, rows):
        for ch in range(16):
            if rank_staging:
                expected["PIM_STAGE_BEGIN", ch, 1, 0, 0, 256, 0, -1, fragments*rows*32, rows] += 1
                for packet in range(fragments*rows):
                    add("ST", ch, 524288+packet*32)
                add("PIM_STAGE_SEAL", ch, 524288)
            for f in range(fragments):
                if rank_staging:
                    add("PIM_STAGE_FRAGMENT", ch, 524288+f*rows*32, f%2)
                else:
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
        require(case in ("gate_up_gpu", "down_gpu_input", "combined_timing_chain"), "Unknown case")
        if case != "down_gpu_input":
            ordinary(262144, 64, 32, 16)
            release()
            for ch in range(16):
                for offset in range(0, 16*256*2*4, 32):
                    add("PIM_RD_RES", ch, offset)
        if case != "gate_up_gpu":
            ordinary(327680, 256, 4, 16)
            release()
            for ch in range(16):
                for offset in range(0, 16*64*4, 32):
                    add("PIM_RD_RES", ch, offset)
    return expected


def native_key(command, geometry):
    ch, rank, bg, bank, row, column = geometry
    if command in RANK_COMMANDS:
        bg = bank = -1
    return command, ch, rank, bg, bank, row, column


def expected_native_commands(records):
    expected = Counter()
    for e in records:
        if e["operation"] == "PIM_STAGE_FRAGMENT":
            ch, rank, bg, bank, row, column = decode(e["address"])
            first = row*2048+column*32
            for token in range(e["fragment_rows"]):
                offset = first+token*32
                expected[native_key("LOCALRD", (ch, rank, bg, bank, offset//2048, offset%2048//32))] += 1
        elif e["operation"] in NATIVE:
            expected[native_key(NATIVE[e["operation"]], decode(e["address"]))] += 1
    return expected


def audit_native_commands(records, native):
    expected = expected_native_commands(records)
    actual = Counter(native_key({"RDA": "RD", "WRA": "WR"}.get(e["command"], e["command"]),
                                tuple(e[k] for k in GEOMETRY))
                     for e in native if e["command"] in set(NATIVE.values()) | {"RDA", "WRA"})
    require(expected == actual, "Native issued command/address coverage differs: "+counter_difference(expected, actual))
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
            require(event["bytes"] == 32, "GEMV input transaction is not one 32B source burst")
            if op == "GEMV_MULTICAST":
                require(event["service_cycles"] == 0, "External multicast must have zero added internal latency")
                require(bank_mask == 65535 and event["token_mask"].bit_count() == 1 and event["lane_mask"] == 65535,
                        "Ordinary multicast must write one complete token row to all16 banks")
            else:
                require(start >= port_until[ch], "Overlapping writes on rank Q distribution port")
                port_until[ch] = end
                require(event["service_cycles"] == 2, "Q fill must retain its 2CK internal port service")
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
                kind = "Q" if op.startswith("GEMV_Q_") else "X"
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
                columns = 4 if case == "down_gpu_input" else 8 if case in ("v", "pv") else 32
                if op == "GEMV_MAC" and mac_context is not None:
                    row = mac_context[ch, start]
                    if 0 <= row < 32:
                        rows, columns, kind = 8, 32, "X"
                    elif 41 <= row < 57:
                        rows, columns, kind = 16, 8, "X"
                    elif 128 <= row < 160:
                        rows, columns, kind = 16, 32, "X"
                    elif 160 <= row < 176:
                        rows, columns, kind = 16, 4, "X"
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


def audit_external_fills(native, gemv_events, burst_cycles=2, rank_staging=False):
    """Match physical WRGB DQ completion to every ordinary/Q-remote fill."""
    expected = Counter((e["Channel"], e["clock"]+burst_cycles)
                       for e in native if e["command"] == "WRGB")
    operations = {"GEMV_Q_REMOTE"} if rank_staging else {"GEMV_MULTICAST", "GEMV_Q_REMOTE"}
    actual = Counter((e["Channel"], e["clock"])
                     for e in gemv_events if e["operation"] in operations)
    require(actual == expected, "Bank fill must match external WRGB DQ completion exactly")
    return sum(actual.values())


def audit_ffn_streams(native, gemv_events, case, staging_events=None):
    """Reconstruct GPU-fed FFN traffic independently from native events."""
    require(not any(e["operation"] == "GEMV_Z_SCATTER" for e in gemv_events),
            "GPU-fed FFN must not use a local Z scatter path")
    mac_address = {(e["Channel"], e["clock"]): e["Row"]
                   for e in native if e["command"] == "MACAB"}
    groups = {"gu": set(), "down": set()}
    compute_done = {"gu": defaultdict(int), "down": defaultdict(int)}
    for e in gemv_events:
        if e["operation"] != "GEMV_MAC":
            continue
        row = mac_address[e["Channel"], e["clock"]]
        stage = "gu" if 128 <= row < 160 else "down" if 160 <= row < 176 else None
        if stage:
            groups[stage].add((e["Channel"], e["buffer"], e["generation"]))
            compute_done[stage][e["Channel"]] = max(compute_done[stage][e["Channel"]], e["ready_at"])
    gu = case != "down_gpu_input"
    down = case != "gate_up_gpu"
    fill_sets = {}
    input_begin = {}
    if staging_events is not None:
        stage_fills = {(e["Channel"], e["ready_at"], e["buffer"]): e["stage_generation"]
                       for e in staging_events if e["operation"] == "BROADCAST"}
        for e in staging_events:
            if e["operation"] == "WR":
                key = e["Channel"], e["stage_generation"]
                input_begin[key] = min(input_begin.get(key, e["clock"]), e["clock"])
    def input_issue(fill):
        if staging_events is None:
            return fill["clock"]-2
        generation = stage_fills[fill["Channel"], fill["clock"], fill["buffer"]]
        return input_begin[fill["Channel"], generation]
    for stage, enabled, fragments in (("gu", gu, 64), ("down", down, 256)):
        require(len(groups[stage]) == (16*fragments if enabled else 0),
                f"Incorrect {stage} GPU input fragment count")
        fills = [e for e in gemv_events if e["operation"] == "GEMV_MULTICAST" and
                 (e["Channel"], e["buffer"], e["generation"]) in groups[stage]]
        require(len(fills) == (16*fragments*16 if enabled else 0),
                f"Incorrect {stage} external GPU input row count")
        fill_sets[stage] = fills
    gu_reads, down_reads = [], []
    for ch in range(16):
        exports = sorted((e for e in native if e["Channel"] == ch and e["command"] == "RDRES"),
                         key=lambda e: e["clock"])
        down_fills = [e for e in fill_sets["down"] if e["Channel"] == ch]
        down_begin = min((input_issue(e) for e in down_fills), default=float("inf"))
        if gu:
            selected = [e for e in exports if compute_done["gu"][ch] <= e["clock"] < down_begin]
            require(len(selected) == 1024, "Gate/Up must export both full FP32 arrays before Down input")
            require([e["Row"]*2048+32*e["Column"] for e in selected] == list(range(0, 32768, 32)),
                    "Gate/Up external output address order differs")
            if down:
                require(max(e["clock"]+2 for e in selected) <= down_begin,
                        "Down GPU input begins before Gate/Up external DQ export completes")
            gu_reads += selected
        if down:
            selected = [e for e in exports if e["clock"] >= compute_done["down"][ch]]
            require(len(selected) == 128, "Down must export 64 complete FP32 output features per rank")
            require([e["Row"]*2048+32*e["Column"] for e in selected] == list(range(0, 4096, 32)),
                    "Down complete-output address order differs")
            down_reads += selected
    if gu and down:
        require(max(e["clock"]+2 for e in gu_reads) <=
                min(input_issue(e) for e in fill_sets["down"]),
                "Full GPU Z broadcast begins before every rank's Gate/Up export completes")
    return dict(GU_GPU_input_bytes=len(fill_sets["gu"])*32,
        GU_GPU_FP32_export_bytes=len(gu_reads)*32,
        Down_GPU_BF16_input_bytes=len(fill_sets["down"])*32,
        Down_GPU_FP32_output_bytes=len(down_reads)*32,
        GPU_input_multicast_destination_bytes=sum(map(len, fill_sets.values()))*32*16,
        Z_local_read_bytes=len(fill_sets["down"])*32 if staging_events is not None else 0,
        Z_local_write_bytes=0,
        GPU_Z_DRAM_store_bytes=len(fill_sets["down"])*32 if staging_events is not None else 0,
        separate_Z_staging_bytes=131072 if staging_events is not None and down else 0,
        direct_GPU_input=staging_events is None, GPU_input_via_single_DRAM_bank=staging_events is not None,
        extra_internal_broadcast_cycles=0,
        combined_GPU_export_before_Down_input=gu and down,
        GPU_GeLU_multiply_cast_latency_us=None,
        Down_output_features_per_rank=64, Down_outputs_are_complete=True,
        Down_cross_rank_reduction_required=False)


def audit_trace(records, case):
    actual = Counter(trace_key(e) for e in records if e["operation"] not in ("BARRIER", "WAIT"))
    expected = expected_addresses(case, rank_staging=any(e["operation"] == "PIM_STAGE_BEGIN" for e in records))
    require(actual == expected, "Trace differs from independent matrix/address oracle: "+counter_difference(expected, actual))
    require(not any(e["operation"] == "WAIT" for e in records), "Artificial legacy completion WAIT retained")
    require(not any(e["operation"].startswith("PIM_Z_") for e in records), "GPU-fed FFN must not use local Z commands")
    return len(records)


def audit_mac_order(native, case):
    """A shared producer/consumer permutation must not evade multiset checks."""
    stages = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input") if case == "serial_layer_control" else (case,)
    ab, qk = [], []
    for stage in stages:
        if stage == "chain":
            ab.extend(range(0, 65536, 32))
            qk.extend(65536+k*32 if k < 484 else 81920+(16+k-484)*32 for k in range(496))
        elif stage in ("v", "pv"):
            base, columns = (83968, 512) if stage == "v" else (100352, 496)
            ab.extend(base+32*i for i in range(columns))
        elif stage in ("gate_up_gpu", "down_gpu_input", "combined_timing_chain"):
            if stage != "down_gpu_input":
                ab.extend(range(262144, 327680, 32))
            if stage != "gate_up_gpu":
                ab.extend(range(327680, 360448, 32))
    for ch in range(16):
        selected = sorted((e for e in native if e["Channel"] == ch), key=lambda e: e["clock"])
        for command, expected in (("MACAB", ab), ("MACQK", qk)):
            actual = [e["Row"]*2048+e["Column"]*32 for e in selected if e["command"] == command]
            require(actual == expected, f"Native {command} order differs from independent operator/reduction order on channel{ch}")
    return 16*(len(ab)+len(qk))


def audit_serial_continuity(manifest, result, simulation, native, events):
    require(result["config"]["refresh"] is True, "Serial resource control must retain enabled refresh")
    required = ("chain", "v", "pv", "gate_up_gpu", "down_gpu_input")
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
    require(native_result["config"]["pim_ffn_dram"] is False,
            "GPU-fed experiment must disable the old local Z/nonlinear path")
    records = trace_records(directory / "workload.trace")
    audit_trace(records, case)
    simulation = directory / "simulation"
    timelines = [simulation / f"timeline.csv.ch{ch}" for ch in range(16)]
    gemv_files = [simulation / f"unified_gemv.csv.ch{ch}" for ch in range(16)]
    require(all(p.is_file() for p in timelines+gemv_files), "Missing channel native evidence")
    native = [e for p in timelines for e in read_csv(p)]
    events = [e for p in gemv_files for e in read_csv(p)]
    rank_staging = native_result["config"].get("pim_rank_staging", False)
    stage_files = [simulation / f"staging_events.csv.ch{ch}" for ch in range(16)] if rank_staging else []
    ca_files = [simulation / f"rank_ca_events.csv.ch{ch}" for ch in range(16)] if rank_staging else []
    controller_files = [simulation / f"bgk_events.csv.ch{ch}" for ch in range(16)] if rank_staging else []
    require(all(p.is_file() for p in stage_files+ca_files+controller_files), "Missing staging/rank-controller evidence")
    stage_events = [e for p in stage_files for e in read_csv(p)] if rank_staging else None
    ca_events = [e for p in ca_files for e in read_csv(p)]
    controller_events = [e for p in controller_files for e in read_csv(p)]
    issued_count = audit_native_commands(records, native)
    row_checks = audit_array_rows(native)
    order_checks = audit_mac_order(native, case)
    mac_context = {(e["Channel"], e["clock"]): e["Row"] for e in native if e["command"] == "MACAB"}
    slots = audit_slot_events(events, case, mac_context=mac_context)
    counts = Counter(e["operation"] for e in records)
    native_counts = Counter(e["command"] for e in native)
    require(slots["operation_counts"].get("GEMV_MAC", 0) == counts["PIM_MAC_AB"], "Missing native bank GEMV consume events")
    require(slots["operation_counts"].get("GEMV_QK", 0) == counts["PIM_MAC_QK"], "Missing shared-slot QKT consumes")
    expected_fills = (counts["PIM_STAGE_LOAD"] + sum(e["fragment_rows"] for e in records
                      if e["operation"] == "PIM_STAGE_FRAGMENT")) if rank_staging else counts["PIM_WR_GB"]
    require(slots["operation_counts"].get("GEMV_MULTICAST", 0) == expected_fills,
            "Missing ordinary bank multicast writes")
    require(not slots["operation_counts"].get("GEMV_Z_SCATTER", 0), "Local Z scatter retained in GPU-fed FFN")
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
    fill_checks = audit_external_fills(native, events, rank_staging=rank_staging)
    staging, ca = None, None
    if rank_staging:
        from staging_audit import audit_staging, audit_rank_ca
        staging = audit_staging(stage_events, native, events, case, native_result["resolved_timings_ck"])
        require(Counter(e["operation"] for e in controller_events)["PIM_ROPE_EXEC"] == counts["PIM_ROPE_EXEC"],
                "RoPE execute event count differs from trace")
        ca = audit_rank_ca(ca_events, native, stage_events, controller_events, records, events)
    ffn = audit_ffn_streams(native, events, case, stage_events) if case not in ("chain", "v", "pv") else None
    continuity = audit_serial_continuity(manifest, native_result, simulation, native, events) if case == "serial_layer_control" else None
    return dict(status="passed", case=case, horizon=16,
        trace_and_native_address_coverage=True, native_data_commands_checked=issued_count,
        array_open_row_checks=row_checks, independent_MAC_order_checks=order_checks,
        external_DQ_to_bank_fill_checks=fill_checks,
        unified_gemv=slots, staging=staging, rank_controller=ca, ffn_streams=ffn, serial_continuity=continuity,
        native_external_DQ_bytes=dq, native_command_counts=dict(native_counts),
        trace_sha256=digest(directory / "workload.trace"),
        native_sha256={p.name: digest(p) for p in timelines+gemv_files+stage_files+ca_files+controller_files},
        numerical_compute=False, unmeasured="GPU/nonlinear arithmetic and GPU contention")

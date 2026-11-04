"""Independent bounded-ingress, peer-dependency and rank-expansion audit."""
from collections import Counter, defaultdict, deque

from staging_audit import require, counter_difference

RANK_OPERATIONS = {"PIM_"+name for name in
    ("SPILL", "ROPE_COEF", "ROPE_LOAD", "ROPE_EXEC", "ROPE_STORE", "Q_EXPORT", "Q_REMOTE")}
TYPE_IDS = {name: number for number, name in enumerate((
    "LD", "ST", "PIM_MAC_AB", "PIM_WR_GB", "PIM_RD_RES", "PIM_MAC_HB", "PIM_LOCAL_GB",
    "PIM_WR_BG", "PIM_WR_Q", "PIM_MAC_K", "PIM_MAC_QK", "PIM_RD_BG", "PIM_LOCAL_WR",
    "PIM_RD_Q", "PIM_SPILL", "PIM_ROPE_COEF", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC",
    "PIM_ROPE_STORE", "PIM_Q_EXPORT", "PIM_Q_REMOTE", "PIM_Z_STORE", "PIM_Z_LOAD",
    "PIM_GEMV_RELEASE", "PIM_STAGE_BEGIN", "PIM_STAGE_SEAL", "PIM_STAGE_LOAD", "PIM_STAGE_FRAGMENT",
    "PIM_FFN_PASS", "PIM_FFN_RDRES"))}


def fine_operation(operation):
    return operation.replace("PIM_RANK_", "PIM_", 1)


def expand_rank_records(records):
    """Expand the fixed physical rank geometry without producer metadata."""
    expanded = []
    for record in records:
        if not record["operation"].startswith("PIM_RANK_"):
            expanded.append(record)
            continue
        operation, address = fine_operation(record["operation"]), record["address"]
        require(operation in RANK_OPERATIONS, "Unknown rank descriptor operation")
        require(address >> 15 & 1 == 1 and address >> 16 & 15 == 0,
                "Rank descriptor must use canonical rank1/BG0/bank0 address")
        for bank in range(16):
            expanded.append(dict(record, operation=operation,
                address=address | bank//4 << 16 | bank%4 << 18))
    return expanded


def request_epochs(records):
    """Reconstruct the old completion fences from trace operations alone."""
    epochs, current = [], []
    last_pim, phase_has_write = -1, False
    for record in records:
        op = fine_operation(record["operation"])
        if op in ("BARRIER", "WAIT"):
            if current:
                epochs.append(current)
                current = []
            continue
        pim = 2 if op == "PIM_MAC_HB" or op == "LD" and last_pim == 2 else int(op.startswith("PIM_"))
        reads_to_half = pim == 2 and last_pim == 0 and not phase_has_write
        if last_pim >= 0 and pim != last_pim and not reads_to_half and current:
            epochs.append(current)
            current = []
        if pim != last_pim:
            phase_has_write = False
        phase_has_write |= op == "ST"
        last_pim = pim
        current.append(record["line"]-1)
    if current:
        epochs.append(current)
    return epochs


def audit_ingress(records, dispatch, ingress, barriers, capacity=32):
    requests = {e["line"]-1: e for e in records if e["operation"] not in ("BARRIER", "WAIT")}
    accepted, completed, waits = {}, {}, []
    accept_order = defaultdict(list)
    accept_ticks = Counter()
    for e in dispatch:
        require(e["operation"] in ("ACCEPT", "COMPLETE", "PEER_WAIT"), "Unknown frontend dispatch event")
        index = e["trace_index"]
        require(index in requests, "Frontend event refers to absent trace request")
        request = requests[index]
        require((e["channel"], e["address"], e["type_id"], e["rank_batch"]) ==
                (request["address"] >> 5 & 15, request["address"], TYPE_IDS[fine_operation(request["operation"])],
                 int(request["operation"].startswith("PIM_RANK_"))), "Frontend dispatch identity differs from trace")
        if e["operation"] == "PEER_WAIT":
            require(fine_operation(request["operation"]) == "PIM_Q_REMOTE", "Only peer Q input may emit PEER_WAIT")
            waits.append(e)
            continue
        target = accepted if e["operation"] == "ACCEPT" else completed
        require(index not in target, "Frontend request accepted/completed more than once")
        target[index] = e["frontend_tick"]
        if e["operation"] == "ACCEPT":
            accept_order[e["channel"]].append(index)
            accept_ticks[e["channel"], e["frontend_tick"]] += 1
    require(set(accepted) == set(completed) == set(requests), "Frontend request coverage is incomplete")
    require(all(count == 1 for count in accept_ticks.values()), "Multiple ingress accepts on one channel/frontend tick")
    require(all(completed[i] >= accepted[i] for i in requests), "Request completes before acceptance")
    for channel, indices in accept_order.items():
        expected = [i for i, e in requests.items() if e["address"] >> 5 & 15 == channel]
        require(indices == expected, "Independent channel stream changes per-channel trace order")
    queues, enqueued, dispatched = defaultdict(deque), {}, {}
    dispatch_ticks = Counter()
    high_water = defaultdict(int)
    for e in ingress:
        index, channel = e["request_id"], e["channel"]
        require(index in requests, "Ingress event refers to absent trace request")
        request = requests[index]
        require((channel, e["address"], e["type_id"], e["rank_batch"]) ==
                (request["address"] >> 5 & 15, request["address"], TYPE_IDS[fine_operation(request["operation"])],
                 int(request["operation"].startswith("PIM_RANK_"))), "Ingress request identity differs from trace")
        require(e["capacity"] == capacity and capacity > 0, "Ingress capacity differs from configured finite queue")
        queue = queues[channel]
        if e["operation"] == "ENQUEUE":
            require(index not in enqueued, "Request enqueued twice")
            queue.append(index)
            enqueued[index] = e["clock"]
        else:
            require(e["operation"] == "DISPATCH", "Unknown memory ingress event")
            require(queue and queue.popleft() == index, "Memory ingress dispatch violates channel FIFO order")
            require(index not in dispatched and e["clock"] >= enqueued[index], "Invalid memory ingress dispatch lifetime")
            dispatched[index] = e["clock"]
            dispatch_ticks[channel, e["clock"]] += 1
        require(len(queue) == e["queue_depth"] and len(queue) <= capacity,
                "Memory ingress queue depth is inaccurate or exceeds capacity")
        high_water[channel] = max(high_water[channel], len(queue))
    require(set(enqueued) == set(dispatched) == set(requests) and all(not q for q in queues.values()),
            "Memory ingress did not dispatch every accepted request")
    require(all(count == 1 for count in dispatch_ticks.values()), "Multiple dispatches on one channel/memory CK")
    epochs = request_epochs(records)
    for first, second in zip(epochs, epochs[1:]):
        require(max(completed[i] for i in first) <= min(accepted[i] for i in second),
                "Independent ingress crosses an implicit/global completion fence")
    barrier_map = {e["marker"]: e for e in barriers}
    expected_barriers = [e for e in records if e["operation"] == "BARRIER"]
    require(len(barrier_map) == len(barriers) == len(expected_barriers), "Barrier log coverage differs from trace")
    for e in expected_barriers:
        require(e["address"] in barrier_map, "Missing global barrier marker")
        marker = barrier_map[e["address"]]
        before = [i for i in requests if i < e["line"]-1]
        after = [i for i in requests if i > e["line"]-1]
        require(marker["requests_completed"] == len(before), "Barrier completion count omits pending requests")
        require(not before or max(completed[i] for i in before) <= marker["frontend_tick"], "Barrier completes before pending callback")
        require(not after or marker["frontend_tick"] <= min(accepted[i] for i in after), "Request enters ingress before prior barrier")
    latest_exports, dependencies = {}, {}
    for index, request in requests.items():
        op = fine_operation(request["operation"])
        key = request["address"], request["operation"].startswith("PIM_RANK_")
        if op == "PIM_Q_EXPORT":
            latest_exports[key] = index
        elif op == "PIM_Q_REMOTE":
            peer = request["address"] ^ 32, key[1]
            require(peer in latest_exports, "Q_REMOTE has no preceding peer Q_EXPORT instance")
            source = latest_exports[peer]
            require(accepted[index] >= completed[source], "Q_REMOTE accepted before its latest peer export completes")
            dependencies[index] = source
    for e in waits:
        require(e["trace_index"] in dependencies and e["frontend_tick"] <= accepted[e["trace_index"]],
                "PEER_WAIT does not refer to a live peer dependency")
    return dict(requests_checked=len(requests), channel_count=len(queues), capacity_per_channel=capacity,
        queue_high_water_by_channel=dict(sorted(high_water.items())), no_channel_FIFO_reordering=True,
        one_dispatch_per_channel_per_CK=True, completion_fences_checked=max(0, len(epochs)-1),
        global_barriers_checked=len(barriers), peer_export_dependencies_checked=len(dependencies),
        peer_wait_events=len(waits), all_accepted_requests_complete=True)


def geometry_key(e, operation=None):
    return (operation or e["operation"], *(e[k] for k in
            ("Channel", "Rank", "BankGroup", "Bank", "Row", "Column")), e["buffer"])


def audit_rank_descriptors(records, events, controller, ca, dispatch=(), ratio=32):
    from audit_native import decode
    parents = {e["line"]-1: e for e in records if e["operation"].startswith("PIM_RANK_")}
    groups = defaultdict(list)
    for e in events:
        require(e["trace_index"] in parents, "Native descriptor has no trace parent")
        groups[e["trace_index"]].append(e)
    require(set(groups) == set(parents), "Native rank descriptor coverage is incomplete")
    fine_events = defaultdict(list)
    for e in controller:
        if e["operation"] in RANK_OPERATIONS:
            fine_events[geometry_key(e)].append(e)
    for values in fine_events.values():
        values.sort(key=lambda e: (e["ready_at"], e["clock"]))
    fine_events = {key: deque(values) for key, values in fine_events.items()}
    callbacks = {e["trace_index"]: e["frontend_tick"] for e in dispatch if e["operation"] == "COMPLETE"}
    expected_ca = Counter()
    global_commands = {(e["Channel"], e["clock"], e["command"]) for e in ca if e["scope"] == "global"}
    q_export_micro_reads = 0
    children_checked = 0
    for index, entries in sorted(groups.items()):
        parent = parents[index]
        phases = {op: [e for e in entries if e["event"] == op] for op in ("DISPATCH", "CHILD_COMPLETE", "COMPLETE")}
        require(sum(map(len, phases.values())) == len(entries), "Unknown rank descriptor event")
        require(len(phases["DISPATCH"]) == len(phases["COMPLETE"]) == 1 and len(phases["CHILD_COMPLETE"]) == 16,
                "Rank descriptor must dispatch once and complete exactly sixteen bank children")
        start, done = phases["DISPATCH"][0], phases["COMPLETE"][0]
        parent_key = parent["operation"], *decode(parent["address"]), parent["buffer"]
        require(geometry_key(start) == geometry_key(done) == parent_key, "Descriptor parent identity changed")
        require(len({e["descriptor_id"] for e in entries}) == 1, "Descriptor children have mixed parent IDs")
        expected = Counter((e["operation"], *decode(e["address"]), e["buffer"]) for e in expand_rank_records([parent]))
        actual = Counter(geometry_key(e) for e in phases["CHILD_COMPLETE"])
        require(expected == actual, "Rank descriptor fine bank/address coverage differs: "+counter_difference(expected, actual))
        for child in phases["CHILD_COMPLETE"]:
            key = geometry_key(child)
            require(key in fine_events and fine_events[key], "Descriptor callback lacks fine controller evidence")
            fine = fine_events[key].popleft()
            require(start["clock"] <= fine["clock"] and child["ready_at"] >= fine["ready_at"] and
                    child["clock"] >= child["ready_at"], "Descriptor child completes before its fine service")
            if child["operation"] == "PIM_Q_EXPORT":
                require(not any((child["Channel"], fine["clock"], command) in global_commands for command in ("RD", "RDA")),
                        "Rank descriptor Q_EXPORT child RD incorrectly occupies global CA")
                q_export_micro_reads += 1
            children_checked += 1
        require(done["clock"] == max(e["clock"] for e in phases["CHILD_COMPLETE"]) and
                done["ready_at"] == max(e["ready_at"] for e in phases["CHILD_COMPLETE"]),
                "Rank descriptor parent completion does not join all sixteen children")
        if dispatch:
            require(index in callbacks and callbacks[index] >= (done["clock"]-1)*ratio,
                    "Frontend parent callback precedes native rank descriptor completion")
        expected_ca[start["Channel"], start["clock"], parent["operation"]] += 1
    actual_ca = Counter((e["Channel"], e["clock"], e["command"]) for e in ca
                        if e["scope"] == "global" and e["command"].startswith("PIM_RANK_"))
    require(actual_ca == expected_ca, "Global CA must contain one admission per rank descriptor")
    # Full cases exclusively use descriptors for these operations; mixed guards
    # may additionally include explicit fine requests.
    expanded = expand_rank_records(records)
    expected_fine = Counter((e["operation"], *decode(e["address"]), e["buffer"]) for e in expanded
                            if e["operation"] in RANK_OPERATIONS)
    actual_fine = Counter(geometry_key(e) for e in controller if e["operation"] in RANK_OPERATIONS)
    require(expected_fine == actual_fine, "Rank expansion changes fine operation/address count")
    return dict(parent_descriptors_checked=len(parents), fine_children_checked=children_checked,
        children_per_descriptor=16, parent_waits_for_all_children=True,
        one_global_CA_per_descriptor=True, fine_bank_operations_preserved=True,
        Q_EXPORT_internal_RD_global_CA_exclusion_checks=q_export_micro_reads)

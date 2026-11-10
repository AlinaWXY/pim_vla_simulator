"""Check native AE traces and report offline DQ-gap interference for c/d/e."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, sha256, write_json
from tools.audit_fused_kernels import audit_timeline
from tools.capture_gap_traces import REPEATS, SLICES

N = 16
POLICIES = ("same_first", "best", "random")
ALPHAS = (1.0, 0.5)
MATRIX = ROOT / "reference/spec/command-pair-matrix.json"
PROFILE = ROOT / "input-data/vlm.json"
FUSED = ROOT / "reference/fused-kernel-continuous-e2e/summary.json"
FUSED_COMPUTE_STAGES = {"qkv", "qkt", "pv", "o", "gate_up", "down", "rope"}
FUSED_TRANSFER_STAGES = {"rope_export", "rope_import", "q_exchange_export", "q_exchange_import"}
PAIR_REFERENCE = dict(RRR=(6, 4), RRW=(6, 4), RWR=(15, 13), RWW=(5, 3),
                      WRR=(15, 13), WRW=(15, 13), WWR=(14, 12), WWW=(4, 2))


def edge_costs(path):
    commands = {"RD": "R", "WR": "W"}
    edges = {}
    for row in read_json(path):
        a, b = row["first"], row["second"]
        if row["relation"] == "different_rank" and a in commands and b in commands:
            pair = commands[a] + commands[b]
            if pair in edges:
                raise ValueError(f"duplicate edge {pair} in {path}")
            edges[pair] = row["dq_gap_ck"]
    if edges != dict(RR=2, RW=2, WR=11, WW=1):
        raise ValueError(f"unexpected cross-rank DQ edges: {edges}")
    edges.update({a + b: 0 for a in "RWO" for b in "RWO" if "O" in a + b})
    for trio, (required, deduction) in PAIR_REFERENCE.items():
        measured = edges[trio[:2]] + edges[trio[1:]]
        if (measured + 2, measured) != (required, deduction):
            raise ValueError(f"gap lookup failed for {trio}")
    return edges


def units_for(alpha, clock_ps):
    source = read_json(PROFILE)
    rows = sorted(source["units"], key=lambda r: r["ordinal"])
    if [r["ordinal"] for r in rows] != list(range(2357)):
        raise ValueError("expected 2357 contiguous VLM ordinals")
    if abs(math.fsum(r["us"] for r in rows) - source["measured_duration_sum_us"]) > 1e-6:
        raise ValueError("measured VLM profile sum differs from its units")
    units = []
    for r in rows:
        reads = r["read_sectors"]
        writes = r["write_sectors"]
        if r["write_is_proxy"]:
            writes = math.ceil(alpha * writes)
        if min(reads, writes) < 0 or reads + writes == 0 or r["us"] <= 0:
            raise ValueError(f"invalid VLM unit {r['ordinal']}")
        units.append((reads / N, writes / N, r["us"] * 1e6 / clock_ps))
    return units


def interval_union_ck(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def load_trace(directory, step, layer):
    on, off = (directory / mode for mode in ("on", "off"))
    verification = read_json(directory / "verification.json")
    if ((on / "raw.json").read_bytes() != (off / "raw.json").read_bytes()
            or sha256(on / "raw.json") != verification["raw_sha256"]):
        raise ValueError(f"raw.json verification failed in {directory}")
    for name, digest in verification["trace_sha256"].items():
        if sha256(on / name) != digest:
            raise ValueError(f"trace checksum mismatch in {directory}: {name}")
    timeline = read_json(on / "dq-timeline.json")
    raw = read_json(on / "raw.json")
    events = timeline["events"]
    if timeline["channels"] != N or raw["clock_ps"] != timeline["clock_ps"]:
        raise ValueError(f"unexpected channel count or clock in {directory}")
    last = 0
    for event in events:
        if ((event["step"], event["layer"], event["start_ck"]) != (step, layer, last)
                or event["end_ck"] < last):
            raise ValueError(f"broken timeline in {directory}: {event}")
        last = event["end_ck"]
    if (timeline["ae_end_ck"] != last
            or abs(last * timeline["clock_ps"] / 1e6 - raw["metrics"]["AE_elapsed_us"]) > 1e-7):
        raise ValueError(f"AE elapsed time differs from trace in {directory}")
    pim = {e["stage_id"]: e for e in events if e["kind"] == "pim"}
    bus = {(r["stage_id"], r["channel"]): r for r in raw["bus"]}
    bursts, epochs = defaultdict(list), defaultdict(list)
    with (on / "dq-bursts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid, channel = int(row["stage_id"]), int(row["channel"])
            if (row["stage"] != pim[sid]["name"] or not 0 <= channel < N
                    or row["direction"] not in ("R", "W") or row["fused_io_epoch"] not in ("0", "1")):
                raise ValueError(f"invalid burst in {directory}: {row}")
            bursts[sid, channel].append((int(row["start_ck"]), int(row["end_ck"]),
                                         row["direction"], row["fused_io_epoch"] == "1"))
    with (on / "dq-epochs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid, channel = int(row["stage_id"]), int(row["channel"])
            if sid not in pim or not 0 <= channel < N:
                raise ValueError(f"invalid epoch in {directory}: {row}")
            epochs[sid, channel].append((int(row["start_ck"]), int(row["end_ck"])))
    geometry, stage_gaps = [], {}
    for index, event in enumerate(events):
        if event["kind"] != "pim":
            continue
        start, end, sid = (event[k] for k in ("start_ck", "end_ck", "stage_id"))
        duration = end - start
        pairs, lengths = Counter(), Counter()
        dq = 0
        blocked_count = blocked_full = blocked_ck = 0
        channels = []
        for channel in range(N):
            key = sid, channel
            intervals = epochs[key]
            last = start
            for a, b in intervals:
                if not last <= a < b <= end:
                    raise ValueError(f"overlapping or out-of-stage epochs: {directory}, {key}")
                last = b
            epoch_index = 0

            def overlap(a, b):
                nonlocal epoch_index
                while epoch_index < len(intervals) and intervals[epoch_index][1] <= a:
                    epoch_index += 1
                used = 0
                for i in range(epoch_index, len(intervals)):
                    left, right = intervals[i]
                    if left >= b:
                        break
                    used += min(right, b) - max(left, a)
                return used

            gaps = []

            def add_gap(a, b, left, right):
                nonlocal blocked_count, blocked_full, blocked_ck
                if b <= a:
                    return
                excluded = overlap(a, b)
                blocked_ck += excluded
                blocked_count += excluded > 0
                blocked_full += excluded == b - a
                pairs[left + right] += 1
                lengths[left + right] += b - a
                gaps.append((a - start, b - start, "RWO".index(left), "RWO".index(right)))

            last, left = start, "O"
            occupied = Counter()
            byte_counts = Counter()
            for a, b, direction, fused in bursts[key]:
                if not last <= a < b <= end:
                    raise ValueError(f"overlapping or out-of-stage bursts: {directory}, {key}")
                add_gap(last, a, left, direction)
                if overlap(a, b) != (b - a if fused else 0):
                    raise ValueError(f"burst and I/O epoch disagree: {directory}, {key}")
                occupied[direction] += b - a
                byte_counts[direction] += 32
                last, left = b, direction
            add_gap(last, end, left, "O")
            r = bus[key]
            if (occupied.total(), byte_counts["R"], byte_counts["W"]) != (
                    r["dq_cycles"], r["pim_read_bytes"], r["pim_write_bytes"]):
                raise ValueError(f"burst and native DQ counters disagree: {directory}, {key}")
            dq += occupied.total()
            channels.append(gaps)
        if dq + lengths.total() != N * duration:
            raise ValueError(f"stage DQ plus idle differs from duration: {directory}, {sid}")
        stage_gaps[index] = channels
        epoch_union_ck = interval_union_ck(
            interval for channel in range(N) for interval in epochs[sid, channel])
        geometry.append(dict(event_index=index, name=event["name"], t_ck=duration,
                             dq_channel_ck=dq, dq_over_t=dq / (N * duration) if duration else 0,
                             gap_count=pairs.total(), gap_channel_ck=lengths.total(),
                             pairs={k: dict(count=pairs[k], length_ck=lengths[k]) for k in pairs},
                             epoch_union_ck=epoch_union_ck,
                             epoch_touching_gap_count=blocked_count, epoch_internal_gap_count=blocked_full,
                             epoch_blocked_idle_channel_ck=blocked_ck,
                             idle_outside_epoch_channel_ck=lengths.total() - blocked_ck))
    return timeline, geometry, stage_gaps, verification


def write_replay_input(path, clock_ps, edges, units, events=(), gaps=None):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{clock_ps}\n")
        handle.write(" ".join(str(edges[a + b]) for a in "RWO" for b in "RWO") + "\n")
        handle.write(f"{len(units)} {len(events)}\n")
        handle.writelines(f"{r:.17g} {w:.17g} {d:.17g}\n" for r, w, d in units)
        for index, event in enumerate(events):
            duration = event["end_ck"] - event["start_ck"]
            handle.write(f"{json.dumps(event['name'])} {duration} {int(index in gaps)}\n")
            for channel in gaps.get(index, ()):
                handle.write(f"{len(channel)}\n")
                handle.writelines(f"{a} {b} {left} {right}\n" for a, b, left, right in channel)


def replay(engine, path, policy, seed, detail=None, detail_repeats=1):
    command = [str(engine), str(path), str(REPEATS), str(POLICIES.index(policy)), str(seed)]
    if detail:
        command.extend((str(detail), str(detail_repeats)))
    return json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))


def expectation(values):
    present = [v for v in values if v is not None]
    sample = present[0]
    if isinstance(sample, dict):
        return {k: expectation([v.get(k) if v is not None else None for v in values])
                for k in sorted(set().union(*present))}
    if isinstance(sample, list):
        return [expectation([v[i] if v is not None else None for v in values]) for i in range(len(sample))]
    if isinstance(sample, str):
        if any(v != sample for v in present):
            raise ValueError("random samples have different stage identities")
        return sample
    return statistics.fmean(0 if v is None else v for v in values)


def add_units(result, ck_ms, b_ms):
    for name in ("ae", "tail", "total", "vlm_loss"):
        result[f"{name}_ms"] = result[f"{name}_ck"] * ck_ms
    result["speedup_vs_b"] = b_ms / result["total_ms"]
    return result


def validate_replay(result, geometry, ae_slice_ck):
    if result["ae_ck"] != ae_slice_ck * REPEATS:
        raise ValueError("replay AE duration differs from the fixed solo timeline")
    loss = 0
    for row, trace in zip(result["stages"], geometry, strict=True):
        active = row["active_repetitions"]
        if (row["event_index"] != trace["event_index"]
                or row["count"] != trace["gap_count"] * active
                or row["length_ck"] != trace["gap_channel_ck"] * active
                or not 0 <= row["sum_u_channel_ck"] <= row["length_ck"]):
            raise ValueError(f"replay stage coverage/capacity differs for {trace['name']}")
        for key in ("count", "length_ck", "infeasible_count", "infeasible_ck", "sum_u_channel_ck"):
            if sum(group[key] for group in row["combinations"].values()) != row[key]:
                raise ValueError(f"replay combination accounting differs for {trace['name']}: {key}")
        loss += trace["t_ck"] * active - row["sum_u_ck"]
    if loss != result["vlm_loss_ck"] or result["total_ck"] != result["ae_ck"] + result["tail_ck"]:
        raise ValueError("replay loss/latency accounting differs")


@lru_cache(maxsize=1)
def load_fused_program(path):
    return read_json(path)


def fused_result(path, case, ck_ms, expected_stages, b_ms):
    raw = read_json(path)
    metrics = raw["metrics"]
    if (raw["window_policy"] != "continuous_pim_descriptors" or len(raw["kernels"]) != 2357
            or len(raw["stages"]) != expected_stages
            or metrics["PIM_descriptor_continuations"] != {"f": 180, "g": 360}[case]
            or metrics["gpu_read_bytes"] or metrics["gpu_write_bytes"]):
        raise ValueError(f"unexpected fused-window results: {path}")
    cfg = read_json(path.parent / "input-config.json")
    ledger = read_json(cfg["calibrated_vlm_path"])
    audit = audit_timeline(raw, read_json(cfg["ae_schedule_path"]), ledger,
                           load_fused_program(cfg["program_path"]))
    total = metrics["total_us"] / 1000
    return dict(total_ms=total, AE_elapsed_ms=metrics["AE_elapsed_us"] / 1000,
                AE_VLM_join_ms=metrics["AE_VLM_join_us"] / 1000,
                PIM_native_ms=audit["native_PIM_cycles"] * ck_ms,
                AE_GPU_compute_ms=audit["AE_GPU_cycles"] * ck_ms,
                VLM_tail_ms=metrics["VLM_remaining_us"] / 1000,
                VLM_calibrated_sum_ms=math.fsum(k["duration_cycles"] for k in raw["kernels"]) * ck_ms,
                PIM_descriptor_continuations=metrics["PIM_descriptor_continuations"],
                raw_sha256=sha256(path), calibrated_vlm_sha256=sha256(cfg["calibrated_vlm_path"]),
                timeline_audit_passed=True, speedup_vs_b=b_ms / total,
                source="本轮原生融合窗口")


def fused_summary(traces, ck_ms, b_ms):
    archive = read_json(FUSED)
    if not archive["passed"]:
        raise ValueError("archived fused-window audit failed")
    rows = {}
    for case, reference in zip("fg", archive["results"], strict=True):
        path = traces / f"{case}-fused/raw.json"
        if path.is_file():
            row = fused_result(path, case, ck_ms, reference["PIM_stages"], b_ms)
            row["archived_total_ms"] = reference["total_ms"]
            row["delta_from_archive_ms"] = row["total_ms"] - reference["total_ms"]
        else:
            row = dict(reference, source=f"已验证归档 `{archive['source_commit'][:7]}`")
        row["speedup_vs_b"] = b_ms / row["total_ms"]
        rows[case] = row
    return rows


def fused_ae_solo_overlap(timeline, geometry, profile_ms, b_ms):
    stages = []
    for row in geometry:
        name, duration, epoch = (row[key] for key in ("name", "t_ck", "epoch_union_ck"))
        if not 0 <= epoch <= duration or name not in FUSED_COMPUTE_STAGES | FUSED_TRANSFER_STAGES:
            raise ValueError(f"unexpected fused AE solo stage: {name}")
        overlap = duration - epoch if name in FUSED_COMPUTE_STAGES else 0
        stages.append(dict(name=name, t_ck=duration, io_epoch_union_ck=epoch, overlap_ck=overlap))

    ae_ck = timeline["ae_end_ck"]
    ae_gpu_ck = sum(event["end_ck"] - event["start_ck"]
                    for event in timeline["events"] if event["kind"] == "gpu")
    if ae_ck != ae_gpu_ck + sum(row["t_ck"] for row in stages):
        raise ValueError("fused AE solo stages and GPU blocks do not cover the timeline")
    scale = timeline["clock_ps"] * REPEATS / 1e9
    ae_ms = ae_ck * scale
    compute_window_ms = sum(row["overlap_ck"] for row in stages) * scale
    overlap_ms = min(profile_ms, compute_window_ms)
    total_ms = ae_ms + profile_ms - overlap_ms
    return dict(ae_solo_ms=ae_ms, ae_gpu_ms=ae_gpu_ck * scale, vlm_profile_ms=profile_ms,
                pim_compute_window_ms=compute_window_ms, overlap_ms=overlap_ms,
                vlm_remaining_ms=profile_ms - overlap_ms, total_ms=total_ms,
                repetitions=REPEATS, speedup_vs_b=b_ms / total_ms, stages=stages)


def report_md(data):
    baseline = data["b_ms"]
    lines = [
        "# 基于 PIM DQ 空隙的 VLM 干扰估算", "",
        f"基准 b 为 c 的 AE solo 切片重复 180 次加 profile VLM，合计 **{baseline:.6f} ms**。"
        f"16 通道，{data['clock_ps']} ps/CK。七组 trace 开关对照的 raw.json 均逐字节相同。", "",
        "| α | VLM 单元 | solo (ms) | profile (ms) | 单元最大绝对误差 (%) | 总时长误差 (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha, row in data["solo"].items():
        lines.append(f"| {alpha} | {row['units']} | {row['total_ms']:.9f} | {row['profile_ms']:.9f} | "
                     f"{row['max_unit_error_percent']:.3g} | {row['total_error_percent']:.3g} |")
    lines += ["", "## c、d、e 的估算", "",
              "| 切片 × 180 | α | policy | AE (ms) | VLM 尾部 (ms) | total (ms) | VLM 损失 (ms) | b / total |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for label, case in data["replay"].items():
        for alpha, policies in case.items():
            for policy, row in policies.items():
                lines.append(f"| {label} | {alpha} | {policy} | {row['ae_ms']:.6f} | {row['tail_ms']:.6f} | "
                             f"{row['total_ms']:.6f} | {row['vlm_loss_ms']:.6f} | {row['speedup_vs_b']:.6f} |")
    lines += ["", "## f、g 的 AE solo 重叠估算", "",
              "端到端时间等于 AE solo 加 VLM profile 总长，再减去 VLM 可在 AE 期间完成的时间。"
              "使用回退版本实跑的 AE solo 切片，step 0 / layer 0 重复 180 次。AE 总长已经包含搬入、"
              "读出、Q 交换及 AE GPU 块。", "",
              "每个含私有计算的 PIM stage，扣除 16 通道融合 I/O epoch 的区间并集，剩余时间按理想重叠计入。"
              "RoPE 导入导出、Q 交换等纯传输 stage 全程保留在 AE 时间内。", "",
              "| 切片 × 180 | AE solo (ms) | VLM profile (ms) | PIM 计算期重叠 (ms) | AE 后 VLM 剩余 (ms) | 端到端 (ms) | b / 端到端 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, r in data["fused_ae_solo_overlap"].items():
        lines.append(f"| {label} | {r['ae_solo_ms']:.6f} | {r['vlm_profile_ms']:.6f} | {r['overlap_ms']:.6f} | "
                     f"{r['vlm_remaining_ms']:.6f} | {r['total_ms']:.6f} | {r['speedup_vs_b']:.6f} |")
    lines += ["", "JSON 中的 `fused_windows` 和 `fused_g_repeated` 保留整 kernel 调度重跑数据，供对照。", "",
              "## 切片代表性", "",
              "| 变体 | 指标 | α | policy | step 0 / layer 0 | step 9 / layer 17 | 差值 | 相对偏差 (%) |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for case, rows in data["slice_comparison"].items():
        for row in rows:
            lines.append(f"| {case} | {row['metric']} | {row.get('alpha', '—')} | {row.get('policy', '—')} | "
                         f"{row['base']:.6f} | {row['alternate']:.6f} | {row['difference']:+.6f} | {row['percent']:+.4f} |")
    lines += ["", "## 分阶段数据", "",
              "T 和 DQ 占用来自单个原生切片。空隙数量、长度及 ΣU 来自 α=1.0、same_first 的 180 次重放，"
              "f、g 列为单切片几何值。DQ、空隙长度是 16 通道之和，ΣU 是通道平均后累加的 CK。"
              "JSON 含各 policy、α 和 stage 的组合明细，O 表示 open，四方向字符表示中间切换，"
              "省略号表示多次切换，短横线表示未尝试填充。", "",
              "| 切片 | stage | T (CK) | PIM DQ (通道 CK) | DQ/T | 空隙数 | 空隙 (通道 CK) | 不可行数 | 不可行 (通道 CK) | ΣU (CK) | U/T |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, geometry in data["geometry"].items():
        rows = {r["event_index"]: r for r in data["replay"].get(label, {}).get("1.0", {})
                .get("same_first", {}).get("stages", [])}
        for r in geometry:
            p = rows.get(r["event_index"])
            suffix = (f"{p['infeasible_count']} | {p['infeasible_ck']} | {p['sum_u_ck']:.2f} | {p['u_over_t']:.4%}"
                      if p else "— | — | — | —")
            lines.append(f"| {label} | {r['name']} | {r['t_ck']} | {r['dq_channel_ck']} | {r['dq_over_t']:.4%} | "
                         f"{p['count'] if p else r['gap_count']} | {p['length_ck'] if p else r['gap_channel_ck']} | {suffix} |")
    lines += ["", "### g 的融合 I/O epoch", "",
              "| 切片 | stage | 接触 epoch 空隙数 | 完全在 epoch 内空隙数 | 排除 (通道 CK) | 保留 (通道 CK) |",
              "|---|---|---:|---:|---:|---:|"]
    for label in ("g-s0-l0", "g-s9-l17"):
        for r in data["geometry"][label]:
            lines.append(f"| {label} | {r['name']} | {r['epoch_touching_gap_count']} | "
                         f"{r['epoch_internal_gap_count']} | {r['epoch_blocked_idle_channel_ck']} | "
                         f"{r['idle_outside_epoch_channel_ck']} |")
        r = data["g_epoch_filter"][label]
        lines.append(f"| {label} | 合计 | {r['epoch_touching_gap_count']} | {r['epoch_internal_gap_count']} | "
                     f"{r['epoch_blocked_idle_channel_ck']} | {r['idle_outside_epoch_channel_ck']} |")
    lines += ["", "## 口径与局限", "",
              "- 只计 DQ，忽略 CA 争用和 rank0 行冲突。GPU 仅使用空隙，PIM 时间线固定，相当于理想的 pim_head。",
              "- 每单元均采用 memory-bound 流体速率。PIM 占用时间按 1:1 计入损失。VLM 损失在活跃 PIM stage "
              "内按 T 减 16 通道平均 ΣU 计算，AE GPU 块暂停 VLM，等待时间为零。",
              "- 写不能超前于读。实际 kernel 的读写交替形状未建模。单元在最慢通道完成时结束，"
              "其它通道在同一空隙内可以等到该时刻继续下一单元。",
              "- α 只缩放代理写请求，按原生输入约定向上取整到 sector，随后将请求平均分给 16 通道。"
              "每个 α 都从 profile 时长重新计算每条请求的 CK。",
              "- same_first 无历史时优先读，best 的入口成本相同时沿用 same_first。random 按剩余读写比例抽样，"
              "条件是两个方向当前均可发。表中 random 为固定种子样本均值，JSON 记录种子、样本和 total 的标准误。",
              "- 每个空隙先假设尾方向等于入口方向，最多重新计算两次右边开销。仍不收敛时预留两方向中较大的右边开销，"
              "JSON 按阶段记录次数。ΣU 是可用容量，逐空隙表另列实际发射的读写数，起止 CK 相对各自 stage 的起点。",
              "- DQ 估算把独立启动的单层 AE solo 时间线重复 180 次。层外 AE GPU 块和跨层 DRAM 状态延续未进入该模型。"
              "g 的代表性对照用两个实跑 AE solo 切片分别重复 180 次。",
              ""]
    return "\n".join(lines)


def compare(base, alternate, metric, **labels):
    return dict(metric=metric, base=base, alternate=alternate, difference=alternate - base,
                percent=100 * (alternate / base - 1), **labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--random-samples", type=int, default=8)
    parser.add_argument("--details", nargs="?", const="first", choices=("first", "all"),
                        help="write per-gap CSV for the first or all same_first repetitions")
    args = parser.parse_args()
    if args.random_samples < 2:
        parser.error("--random-samples requires at least two samples")
    output, engine = args.output.resolve(), args.engine.resolve()
    output.mkdir(parents=True, exist_ok=True)
    working = output / "replay-input"
    working.mkdir(exist_ok=True)
    edges = edge_costs(MATRIX)
    clock_ps = read_json(args.traces / "c-s0-l0/on/dq-timeline.json")["clock_ps"]
    ck_ms = clock_ps / 1e9
    units = {a: units_for(a, clock_ps) for a in ALPHAS}
    data = dict(schema_version=1, clock_ps=clock_ps, channels=N, repetitions=REPEATS,
                matrix_sha256=sha256(MATRIX), vlm_sha256=sha256(PROFILE), edge_ck=edges,
                replay_engine_sha256=sha256(engine),
                single_direction_reference=PAIR_REFERENCE, solo={}, geometry={}, provenance={},
                replay={}, slice_comparison={}, hand_checks_passed=True)
    for alpha in ALPHAS:
        path = working / f"solo-{alpha}.txt"
        write_replay_input(path, clock_ps, edges, units[alpha])
        data["solo"][str(alpha)] = replay(engine, path, "same_first", 0)["solo"]
    seeds = [20261110 + n for n in range(args.random_samples)]
    profile_ms = data["solo"]["1.0"]["profile_ms"]
    jobs, fused_slices = {}, {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for case, slices in SLICES.items():
            for step, layer in slices:
                label = f"{case}-s{step}-l{layer}"
                timeline, geometry, gaps, verification = load_trace(args.traces / label, step, layer)
                if timeline["clock_ps"] != clock_ps:
                    raise ValueError(f"trace clock differs for {label}")
                data["geometry"][label] = geometry
                data["provenance"][label] = verification
                verification["ae_slice_ck"] = timeline["ae_end_ck"]
                if case in "fg":
                    fused_slices[label] = timeline, geometry
                    continue
                for alpha in ALPHAS:
                    path = working / f"{label}-{alpha}.txt"
                    write_replay_input(path, clock_ps, edges, units[alpha], timeline["events"], gaps)
                    for policy in POLICIES:
                        detail = (output / f"{label}-a{alpha}-gaps.csv"
                                  if args.details and policy == "same_first" else None)
                        jobs[label, str(alpha), policy] = [
                            pool.submit(replay, engine, path, policy, seed, detail,
                                        REPEATS if args.details == "all" else 1)
                            for seed in (seeds if policy == "random" else [0])]
        baseline = data["provenance"]["c-s0-l0"]["ae_slice_ck"] * REPEATS * ck_ms + data["solo"]["1.0"]["profile_ms"]
        data["b_ms"] = baseline
        for (label, alpha, policy), futures in jobs.items():
            samples = [future.result() for future in futures]
            for row in samples:
                if row.pop("solo") != data["solo"][alpha]:
                    raise ValueError("replay solo validation differs")
                validate_replay(row, data["geometry"][label], data["provenance"][label]["ae_slice_ck"])
            result = expectation(samples) if policy == "random" else samples[0]
            if policy == "random":
                totals = [sample["total_ck"] * ck_ms for sample in samples]
                result.update(seeds=seeds, total_samples_ms=totals,
                              total_standard_error_ms=statistics.stdev(totals) / math.sqrt(len(totals)))
            data["replay"].setdefault(label, {}).setdefault(alpha, {})[policy] = add_units(result, ck_ms, baseline)
            print(f"{label} alpha={alpha} {policy}: {result['total_ms']:.6f} ms", flush=True)
    data["fused_windows"] = fused_summary(args.traces, ck_ms, baseline)
    data["fused_ae_solo_overlap"] = {
        label: fused_ae_solo_overlap(timeline, geometry, profile_ms, baseline)
        for label, (timeline, geometry) in fused_slices.items()
    }
    data["slice_comparison"]["d"] = [
        compare(row[key], data["replay"]["d-s9-l17"][alpha][policy][key],
                metric, alpha=alpha, policy=policy)
        for alpha, policies in data["replay"]["d-s0-l0"].items() for policy, row in policies.items()
        for key, metric in (("total_ms", "180× total (ms)"), ("vlm_loss_ms", "180× VLM 损失 (ms)"))]
    g_labels = ("g-s0-l0", "g-s9-l17")
    data["g_epoch_filter"] = {
        label: {key: sum(row[key] for row in data["geometry"][label]) for key in
                ("gap_count", "gap_channel_ck", "epoch_touching_gap_count", "epoch_internal_gap_count",
                 "epoch_blocked_idle_channel_ck", "idle_outside_epoch_channel_ck")}
        for label in g_labels
    }
    data["slice_comparison"]["g"] = [
        compare(*(data["provenance"][label]["ae_slice_ck"] * REPEATS * ck_ms
                  for label in g_labels), "180× AE solo (ms)"),
        compare(*(sum(r["idle_outside_epoch_channel_ck"] for r in data["geometry"][label])
                  for label in g_labels), "epoch 外 idle (通道 CK)"),
    ]
    g_paths = {label: args.traces / f"{label}-fused-repeated/raw.json" for label in g_labels}
    if all(path.is_file() for path in g_paths.values()):
        repeated = {label: fused_result(path, "g", ck_ms, len(data["geometry"][label]) * REPEATS, baseline)
                    for label, path in g_paths.items()}
        data["fused_g_repeated"] = repeated
    data["slice_comparison"]["g"] += [
        compare(*(data["fused_ae_solo_overlap"][label][key] for label in g_labels), metric)
        for key, metric in (("overlap_ms", "180× PIM 计算期重叠 (ms)"),
                            ("total_ms", "180× 端到端 (ms)"))]
    write_json(output / "gap-estimate.json", data)
    (output / "gap-estimate.md").write_text(report_md(data), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""Small native regressions for calibrated VLM kernels and fused PIM windows.

PIM requests, prerequisites, refresh and bus completion remain native. VLM
durations come from an integer-cycle ledger; their sector counts are metadata.
This test does not generate or run the measured full-workload programs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ramulator2/python"))
from ramulator.dram.lpddr5x_pim import build_spec

MODEL = "fused_kernel_windows"
CHANNELS = 16


def operation(channel, name, index, *, deps=(), **fields):
    return dict(id=index, op=name, addr=[channel, 1, 0, 0, 0, index % 64],
                deps=list(deps), **fields)


def fused_stage(name="work", *, upload_count=2, compute_count=24,
                export_count=3, slow_upload_channel=None, slow_compute_channel=None):
    channels = []
    for channel in range(CHANNELS):
        uploads = upload_count + (12 if channel == slow_upload_channel else 0)
        computations = compute_count + (48 if channel == slow_compute_channel else 0)
        outer = [operation(channel, "WR", i, epoch="input") for i in range(uploads)]
        seal = len(outer)
        outer.append(operation(channel, "BARRIER", seal, deps=range(uploads)))
        children = [operation(channel, "LOCALRD", i, deps=([i - 1] if i else []))
                    for i in range(computations)]
        children.append(operation(channel, "LOCALWR", len(children), deps=[len(children) - 1]))
        descriptor = len(outer)
        run = operation(channel, "MAC_RUN", descriptor, deps=[seal], children=children)
        run["addr"] = [channel, 1, -1, -1, 0, 0]
        outer.append(run)
        for _ in range(export_count):
            outer.append(operation(channel, "RD", len(outer), deps=[descriptor], epoch="output"))
        channels.append(outer)
    return dict(name=name, channels=channels)


def transfer_stage(name="transfer"):
    channels = []
    for channel in range(CHANNELS):
        channels.append([
            operation(channel, "WR", 0, epoch="transfer"),
            operation(channel, "RD", 1, deps=[0], epoch="transfer"),
        ])
    return dict(name=name, channels=channels)


def program(stages):
    return dict(schema_version=1, execution_model=MODEL, stages=stages)


def schedule(stages, *, nonlinear_us=0.2):
    events = []
    for stage in stages:
        events.append(dict(kind="pim", stage=stage["name"], step=0, layer=0))
    if nonlinear_us:
        events.append(dict(kind="gpu", name="nonlinear", us=nonlinear_us, step=0, layer=0))
    return dict(events=events)


class Runner:
    def __init__(self, binary, directory):
        self.binary = binary
        self.directory = Path(directory)
        self.spec = build_spec()
        self.clock_ps = self.spec["clock_ps"]
        self.tick_us = self.clock_ps / 1_000_000
        self.count = 0

    def run(self, name, stages, durations, *, events=None, sectors=256,
            change_program=None, change_ledger=None, rejected=False):
        self.count += 1
        directory = self.directory / f"{self.count:02d}-{name}"
        directory.mkdir()

        def save(filename, value):
            path = directory / filename
            path.write_text(json.dumps(value), encoding="utf-8")
            return str(path)

        source_units = [dict(ordinal=i, name=f"vlm-{i}", kind="kernel",
                             # Deliberately differ from the calibrated duration.
                             us=duration * self.tick_us * 2,
                             read_sectors=sectors, write_sectors=sectors,
                             write_is_proxy=False, base_sector=i * 4096)
                        for i, duration in enumerate(durations)]
        ledger = dict(kind="calibrated_vlm_durations", clock_ps=self.clock_ps,
                      total_cycles=sum(durations), units=[
                          dict(ordinal=i, name=unit["name"], kind=unit["kind"],
                               duration_cycles=duration, us=duration * self.tick_us,
                               measured_us=unit["us"])
                          for i, (unit, duration) in enumerate(zip(source_units, durations))])
        workload = program(deepcopy(stages))
        if change_program:
            change_program(workload)
        if change_ledger:
            change_ledger(ledger)
        config = dict(execution_model=MODEL, spec=self.spec, run_kind="combined",
                      layout="token", rope="gpu", dataflow="fused", mode="concurrent",
                      program_path=save("program.json", workload),
                      vlm_path=save("vlm.json", dict(units=source_units)),
                      calibrated_vlm_path=save("calibrated-vlm.json", ledger),
                      ae_schedule_path=save("ae.json", events or schedule(stages)),
                      calibration=dict(alpha=1, window=64, write_high=48, write_low=16))
        result = subprocess.run([str(self.binary), "--config", save("config.json", config),
                                 "--output", str(directory / "native")],
                                capture_output=True, text=True, timeout=30)
        if rejected:
            assert result.returncode != 0, f"{name}: malformed coarse input was accepted"
            print(f"fused-kernel regression passed: {name} (rejected)", flush=True)
            return None
        assert result.returncode == 0, f"{name}: native run failed\n{result.stdout}\n{result.stderr}"
        raw = json.loads((directory / "native/raw.json").read_text())
        self.common(raw, stages, durations, sectors)
        return raw

    def common(self, raw, stages, durations, sectors):
        metrics = raw["metrics"]
        assert raw["execution_model"] == MODEL
        assert metrics["AE_GPU_wait_us"] == 0
        assert metrics["vlm_accounted_read_bytes"] == len(durations) * sectors * 32
        assert metrics["vlm_accounted_write_bytes"] == len(durations) * sectors * 32
        assert len(raw["kernels"]) == len(durations)
        for i, (kernel, duration) in enumerate(zip(raw["kernels"], durations)):
            assert kernel["ordinal"] == i
            assert kernel["duration_cycles"] == duration
            assert kernel["end_ck"] - kernel["start_ck"] == duration
            if i:
                assert kernel["start_ck"] >= raw["kernels"][i - 1]["end_ck"]
        for key in ("GPU_reads", "GPU_writes", "GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab",
                    "GPU_row_hits", "GPU_row_misses", "gpu_read_bytes", "gpu_write_bytes",
                    "PIM_out_of_order_issues", "PIM_prefetched_prerequisites"):
            assert metrics[key] == 0, (key, metrics[key])
        expected_reads = expected_writes = 0
        for stage in stages:
            for channel in stage["channels"]:
                expected_reads += sum(op["op"] == "RD" for op in channel) * 32
                expected_writes += sum(op["op"] == "WR" for op in channel) * 32
        assert metrics["pim_read_bytes"] == expected_reads
        assert metrics["pim_write_bytes"] == expected_writes
        assert raw["refresh_policy"] == {"rank0": "REFpb", "rank1": "REFab"}
        assert raw["refresh_commands"]["rank0"]["REFab"] == 0
        assert raw["refresh_commands"]["rank1"]["REFpb"] == 0
        for bus in raw["bus"]:
            assert bus["ca_cycles"] <= bus["duration_cycles"]
            assert bus["dq_cycles"] <= bus["duration_cycles"]
            assert bus["gpu_read_bytes"] == bus["gpu_write_bytes"] == 0
        assert sum(bus["pim_read_bytes"] for bus in raw["bus"]) == expected_reads
        assert sum(bus["pim_write_bytes"] for bus in raw["bus"]) == expected_writes
        for phase in raw["phases"]:
            assert phase["start_ck"] <= phase["end_ck"]
            if phase["phase"] != "join":
                ready = phase["channel_ready_ck"]
                assert len(ready) == CHANNELS
                assert max(ready) == phase["end_ck"], phase
            if phase["phase"] in {"upload", "join", "download", "transfer"}:
                assert all(not phase["start_ck"] <= k["start_ck"] < phase["end_ck"]
                           for k in raw["kernels"]), ("kernel started in exclusive phase", phase)
        for ae in raw["ae_gpu"]:
            for kernel in raw["kernels"]:
                assert kernel["end_ck"] <= ae["start_ck"] or kernel["start_ck"] >= ae["end_ck"], \
                    ("VLM kernel overlapped AE GPU nonlinear work", kernel, ae)
        last_phase = max(p["end_ck"] for p in raw["phases"])
        last_ae = max((a["end_ck"] for a in raw["ae_gpu"]), default=last_phase)
        ae_end = max(last_phase, last_ae)
        last_kernel = max((k["end_ck"] for k in raw["kernels"]), default=ae_end)
        join_cycles = sum(p["end_ck"] - p["start_ck"] for p in raw["phases"] if p["phase"] == "join")
        assert abs(metrics["AE_VLM_join_us"] - join_cycles * self.tick_us) < 1e-7
        assert abs(metrics["AE_elapsed_us"] - ae_end * self.tick_us) < 1e-7
        native_stage_us = sum(stage["duration_us"] for stage in raw["stages"])
        nonlinear_us = sum(a["end_ck"] - a["start_ck"] for a in raw["ae_gpu"]) * self.tick_us
        assert abs(metrics["AE_elapsed_us"] - native_stage_us - nonlinear_us) < 1e-7
        assert abs(metrics["total_us"] - max(ae_end, last_kernel) * self.tick_us) < 1e-7
        assert abs(metrics["VLM_remaining_us"] - max(0, last_kernel - ae_end) * self.tick_us) < 1e-7


def phase(raw, kind, stage_id=0):
    matches = [p for p in raw["phases"] if p["stage_id"] == stage_id and p["phase"] == kind]
    assert len(matches) == 1, (kind, stage_id, matches)
    return matches[0]


def assert_private_start(raw, stage_id=0):
    upload, compute = phase(raw, "upload", stage_id), phase(raw, "compute", stage_id)
    assert compute["start_ck"] == upload["end_ck"]
    assert raw["kernels"][0]["start_ck"] == compute["start_ck"]
    assert raw["kernels"][0]["start_ck"] >= max(upload["channel_ready_ck"])
    return compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pim-fused-kernels-") as directory:
        runner = Runner(args.binary.resolve(), directory)
        stages = [fused_stage()]

        # The probe uses the native phase length only to place an exact CK tie;
        # the scheduler assertions below are independent ordering requirements.
        probe = runner.run("calibrated_duration_and_private_window", stages, [1])
        compute = assert_private_start(probe)
        span = compute["end_ck"] - compute["start_ck"]
        assert span > 8, "fixture has no meaningful private-compute interval"
        assert probe["kernels"][0]["end_ck"] < compute["end_ck"]
        print("fused-kernel regression passed: calibrated_duration_and_private_window", flush=True)

        long_duration = max(10_000, span * 4)
        long = runner.run("long_kernel_join_export_nonlinear", stages, [long_duration, 7])
        compute = assert_private_start(long)
        join, download = phase(long, "join"), phase(long, "download")
        first, second = long["kernels"]
        assert compute["end_ck"] < first["end_ck"]
        assert join["start_ck"] == compute["end_ck"]
        assert join["end_ck"] == first["end_ck"]
        assert download["start_ck"] >= first["end_ck"]
        assert long["ae_gpu"][0]["start_ck"] >= download["end_ck"]
        assert second["start_ck"] >= long["ae_gpu"][0]["end_ck"]
        assert long["refresh_commands"]["rank0"]["REFpb"] > 0
        assert long["refresh_commands"]["rank1"]["REFab"] > 0
        print("fused-kernel regression passed: long_kernel_join_export_nonlinear", flush=True)

        short_duration = max(1, span // 4)
        short = runner.run("short_kernels_repeat_until_ready", stages, [short_duration] * 9)
        compute = assert_private_start(short)
        inside = [k for k in short["kernels"] if k["start_ck"] < compute["end_ck"]]
        assert len(inside) >= 3
        assert all(b["start_ck"] == a["end_ck"] for a, b in zip(inside, inside[1:]))
        assert any(k["start_ck"] >= short["ae_gpu"][0]["end_ck"] for k in short["kernels"])
        print("fused-kernel regression passed: short_kernels_repeat_until_ready", flush=True)

        tie = runner.run("ready_kernel_end_same_ck", stages, [span, 5])
        compute = assert_private_start(tie)
        assert tie["kernels"][0]["end_ck"] == compute["end_ck"]
        assert tie["kernels"][1]["start_ck"] >= tie["ae_gpu"][0]["end_ck"]
        print("fused-kernel regression passed: ready_kernel_end_same_ck", flush=True)

        # Real generated descriptors commonly end with a zero-cost completion
        # barrier. Settle it before admitting a new kernel at the same CK as
        # the final LOCALWR completion; inspecting only inner.issued() would
        # leave Compute open until a later controller step and admit one more.
        barrier_stages = deepcopy(stages)
        for channel in barrier_stages[0]["channels"]:
            descriptor = next(op for op in channel if op["op"] == "MAC_RUN")
            children = descriptor["children"]
            children.append(operation(children[0]["addr"][0], "BARRIER", len(children),
                                      deps=[len(children) - 1]))
        barrier_tie = runner.run("trailing_barrier_kernel_end_same_ck", barrier_stages, [span, 5])
        compute = assert_private_start(barrier_tie)
        assert compute["end_ck"] - compute["start_ck"] == span, "zero-cost barrier delayed native completion"
        assert barrier_tie["kernels"][0]["end_ck"] == compute["end_ck"]
        assert barrier_tie["kernels"][1]["start_ck"] >= barrier_tie["ae_gpu"][0]["end_ck"], \
            "new kernel started while the terminal completion barrier was ready to retire"
        assert barrier_tie["metrics"]["AE_VLM_join_us"] == 0
        print("fused-kernel regression passed: trailing_barrier_kernel_end_same_ck", flush=True)

        skew_stages = [fused_stage(slow_upload_channel=15, slow_compute_channel=15)]
        skew = runner.run("all_channels_phase_barriers", skew_stages, [1] * 4)
        compute = assert_private_start(skew)
        upload = phase(skew, "upload")
        assert upload["channel_ready_ck"][15] > max(upload["channel_ready_ck"][:15])
        assert compute["channel_ready_ck"][15] > max(compute["channel_ready_ck"][:15])
        assert phase(skew, "download")["start_ck"] >= compute["channel_ready_ck"][15]
        print("fused-kernel regression passed: all_channels_phase_barriers", flush=True)

        transfers = [transfer_stage(), fused_stage()]
        transfer = runner.run("pure_transfer_cannot_open_kernel_window", transfers, [1])
        gate = phase(transfer, "transfer", 0)
        assert transfer["kernels"][0]["start_ck"] >= gate["end_ck"]
        assert_private_start(transfer, 1)
        print("fused-kernel regression passed: pure_transfer_cannot_open_kernel_window", flush=True)

        traffic = runner.run("sector_counts_do_not_generate_gpu_requests", stages, [long_duration, 7],
                             sectors=10_000_000)
        assert traffic["phases"] == long["phases"]
        assert traffic["ae_gpu"] == long["ae_gpu"]
        for key in ("AE_elapsed_us", "AE_GPU_wait_us", "VLM_remaining_us", "total_us",
                    "pim_read_bytes", "pim_write_bytes", "checked_commands"):
            assert traffic["metrics"][key] == long["metrics"][key]
        for left, right in zip(traffic["kernels"], long["kernels"]):
            for key in ("ordinal", "start_ck", "end_ck", "duration_cycles"):
                assert left[key] == right[key]
            assert left["read_bytes"] == left["write_bytes"] == 10_000_000 * 32
        print("fused-kernel regression passed: sector_counts_do_not_generate_gpu_requests", flush=True)

        def legacy_internal_export(p):
            for channel in p["stages"][0]["channels"]:
                descriptor = next(op for op in channel if op["op"] == "MAC_RUN")
                children = descriptor["children"]
                children.append(operation(children[0]["addr"][0], "RD", len(children),
                                          deps=[len(children) - 1], epoch="legacy-internal-export"))

        runner.run("reject_descriptor_internal_external_io", stages, [1],
                   change_program=legacy_internal_export, rejected=True)
        runner.run("reject_program_without_execution_model", stages, [1],
                   change_program=lambda p: p.pop("execution_model"), rejected=True)
        runner.run("reject_duration_clock_mismatch", stages, [1],
                   change_ledger=lambda ledger: ledger.update(clock_ps=ledger["clock_ps"] + 1), rejected=True)
        runner.run("reject_duration_ordinal_mismatch", stages, [1],
                   change_ledger=lambda ledger: ledger["units"][0].update(ordinal=7), rejected=True)
        runner.run("reject_duration_unit_count_mismatch", stages, [1],
                   change_ledger=lambda ledger: ledger["units"].clear(), rejected=True)
        print(f"fused-kernel frontend smoke: {runner.count} small native fixtures passed", flush=True)


if __name__ == "__main__":
    main()

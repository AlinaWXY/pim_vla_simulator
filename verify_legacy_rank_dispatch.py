"""Verify the compatibility mode against unmodified xinyao_1103 (786842a)."""
import argparse
import json
from pathlib import Path


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    reference = Path(__file__).resolve().parent / "reference"
    expected = read(reference / "baseline-xinyao_1103-before-rank-dispatch.json")
    actual = read(root / "summary.json")
    assert actual["status"] == "passed" and actual["staging_banks"] == 2
    assert not actual.get("channel_independent_dispatch", False)
    assert not actual.get("rank_descriptors", False)
    for case, ref in expected["cases"].items():
        run = actual["runs"][case]
        for key in ("memory_cycles", "external_DQ_bytes", "trace_sha256"):
            assert run[key] == ref[key], (case, key, run[key], ref[key])
        assert abs(run["latency_us"] - ref["latency_us"]) < 1e-9, case
        audit = read(root / case / "native-audit.json")
        assert audit["status"] == "passed" and audit["native_sha256"] == ref["native_sha256"], case
    channels = read(root / "channel-analysis.json")
    keys = ("stage_us", "modelled_dq_occupied_us", "modelled_dq_idle_us",
            "tail_clear_complete_zero_external_phase_us", "modelled_dq_idle_outside_tail_clear_work_us")
    for case, refs in expected["stage_channel_partitions"].items():
        for ref, row in zip(refs, channels["stages"][case]["channels"], strict=True):
            assert ref["channel"] == row["channel"]
            for key in keys:
                assert abs(ref[key] - row[key]) < 1e-9, (case, row["channel"], key)
            assert ref["tail_clear_windows"] == row["tail_clear_windows"]
    return dict(status="passed", baseline_commit="786842a", cases=len(expected["cases"]),
        native_cycles_bytes_traces_and_event_hashes_identical=True,
        all_channel_partitions_identical=True, stage_sum_us=channels["stage_sum_us"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run / "legacy-dispatch-verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

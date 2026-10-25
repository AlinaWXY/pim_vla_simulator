"""Freeze H=16 postprocessing provenance after native execution and independent audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def main(root):
    experiment = Path(__file__).resolve().parents[1]
    audits = {}
    for case in sorted(root.iterdir()):
        if not (case / "independent-native-audit.json").exists():
            continue
        audit = load(case / "independent-native-audit.json")
        assert audit["status"] == "passed"
        assert audit["horizon"] == 16 and audit["attention_heads"] == 8
        summary = load(case / "summary.json")
        summary["audit_status"] = "passed"
        summary["independent_native_audit"] = "independent-native-audit.json"
        dump(case / "summary.json", summary)
        audits[case.name] = dict(status="passed", native_event_count=audit["native_event_count"],
            local_store_row_checks=audit["local_store_row_checks"])
    top = load(root / "attention-summary.json")
    assert set(audits) == {r['name'] for r in top['runs']}, audits
    for row in top["runs"]:
        row["audit_status"] = "passed"
        row["independent_native_audit"] = row["name"]+"/independent-native-audit.json"
    assert top["private_backend_unchanged_during_run"]
    original_checked = top.get("original_backend_check_performed",
                               top.get("original_backend_unchanged") is not None)
    if original_checked:
        assert top["original_backend_unchanged"]
    else:
        assert top["original_backend_unchanged"] is None
    dump(root / "attention-summary.json", top)
    snapshot = root / "postprocessing-code"
    snapshot.mkdir(exist_ok=True)
    for name in ("analysis", "validation"):
        for source in (experiment / name).rglob("*.py"):
            target = snapshot / name / source.relative_to(experiment / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    faults = load(root / "audit-fault-detection.json")
    assert faults["status"] == "passed" and all(f["detected"] for f in faults["faults"])
    ffn = load(root / "ffn/ffn-regression.json")
    assert all(checks['same_trace_request_counts'] and checks['same_work_commands']
        and checks['comparison_horizon'] == 16
        and checks['control_attention_bytes_per_bank'] == 0
        and checks['primary_attention_bytes_per_bank'] == 43008 for checks in ffn.values())
    breakdown = load(root / "all-linear-breakdown/breakdown.json")
    assert breakdown['schema_version'] == 'xinyao_1025_h16_qkv_dram_q'
    assert breakdown['horizon'] == 16 and breakdown['attention_heads'] == 8
    assert breakdown["verification"]["exclusive_tick_identities_exact"]
    dump(root / "validation-summary.json", dict(horizon=16, attention_heads=8, attention_cases=audits,
        attention_independent_audits_passed=len(audits),
        mutation_detection_checks=len(faults["faults"]),
        ffn_H16_relocation_controls=ffn,
        shared_backend_directed_test_note="Shared native backend directed tests are configuration-level checks; the nine independently audited attention traces above use H=16.",
        exclusive_timeline_partition_checks=breakdown["verification"]["analyzed_channels"],
        native_backend_unchanged_during_final_run=True,
        original_backend_check_performed=original_checked,
        original_backend_unchanged=top["original_backend_unchanged"],
        postprocessing_source_note="These analysis/audit snapshots were frozen after execution; workload/native provenance was captured by the run driver."))
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact-sha256.json")
    dump(root / "artifact-sha256.json", {str(p.relative_to(root)): digest(p) for p in files})
    print(json.dumps(dict(files_hashed=len(files), independent_attention_cases=len(audits),
        native_PIM_subset_us=breakdown["measured_pim_subset_us_per_layer_step"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    main(parser.parse_args().run_root.resolve())

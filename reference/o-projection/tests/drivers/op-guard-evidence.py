"""Build portable validation-reader paths around preserved tests and one retry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    a = p.parse_args()
    root, original = a.root.resolve(strict=True), a.original.resolve(strict=True)
    corrected = root / "o-native-guards-corrected/summary.json"
    summary = json.loads(corrected.read_text(encoding="utf-8"))
    if not summary["passed"] or summary["case_count"] != 13 or not summary.get("correction"):
        raise AssertionError("Corrected O guard evidence is incomplete")
    if sha(root / "provenance.json") != sha(original / "provenance.json"):
        raise AssertionError("Original full-test provenance was changed")
    sources = {
        "runner-logs/unit-tests.log": original / "runner-logs/unit-tests.log",
        "score-native-guards/summary.json": original / "score-native-guards/summary.json",
        "o-native-guards/summary.json": corrected,
    }
    legacy = original / "legacy-native-guards/summary.json"
    if not legacy.is_file():
        legacy = original / "legacy-native-guards/native-guards.json"
    sources["legacy-native-guards/summary.json"] = legacy
    copied = {}
    for relative, source in sources.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            raise AssertionError("Refusing to replace differing evidence: " + relative)
        if not target.exists():
            shutil.copyfile(source, target)
        copied[relative] = dict(source=str(source), source_sha256=sha(source), copied_sha256=sha(target))
    evidence = dict(status="passed", original_full_test_run=str(original), retry_root=str(root),
        policy="Portable copies of original unit/legacy/score summaries plus a separately corrected O summary; not a claim of a single all-passing runner execution.",
        correction="assertion-correction.json", correction_sha256=sha(root / "assertion-correction.json"),
        original_provenance_preserved=True, provenance_sha256=sha(root / "provenance.json"),
        assembly_driver_sha256=sha(Path(__file__).resolve()), copied_artifacts=copied)
    (root / "evidence-assembly.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", validation_evidence=str(root), o_guard_count=summary["case_count"],
        separate_retry_case="O_missing_rewind", compatible_paths=list(sources))), flush=True)


if __name__ == "__main__":
    main()

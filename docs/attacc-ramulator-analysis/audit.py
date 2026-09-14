#!/usr/bin/env python3
"""Reconstruct and audit the pinned ATTACC delta, without changing ramulator2.

Only Python's standard library and GNU patch are required. Scratch is explicit;
the upstream install script (including its git reset) is never executed.
"""
import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import math
import operator
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sources/attacc"
BASELINE = HERE.parents[1] / "ramulator2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def integers(expr):
    """Evaluate integer literals and shifts in the pinned organization tables."""
    node = ast.parse(expr.strip(), mode="eval").body
    def walk(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.LShift):
            return operator.lshift(walk(n.left), walk(n.right))
        raise ValueError(expr)
    return walk(node)


def definition(text, name):
    match = re.search(r"ImplDef\s+" + name + r"\s*=\s*\{(.*?)\};", text, re.S)
    return re.findall(r'"([^"]+)"', re.sub(r"//[^\n]*", "", match[1]))


def extract_model(model, path, source):
    text = path.read_text()
    keys = definition(text, "m_timings")
    timings, orgs, constraints = [], [], []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'\{"(HBM3_[^"]*Gbps[^\"]*)",\s*\{([^}]+)\}', line)
        if match:
            values = [int(x.strip()) for x in match[2].split(",")]
            assert len(keys) == len(values), (model, match[1])
            for key, value in zip(keys, values):
                timings.append(dict(model=model, preset=match[1], parameter=key,
                                    raw_value=value, source=source, line=number))
        match = re.search(r'\{"(HBM3_[^"]+)",\s*\{([^{}]+),\s*\{([^{}]+)\}', line)
        if match:
            prefix = [integers(x) for x in match[2].split(",")]
            counts = [integers(x) for x in match[3].split(",")]
            levels = definition(text, "m_levels")
            assert len(prefix) == 2 and len(counts) == len(levels)
            for key, value in zip(["density_Mb", "DQ"] + levels, prefix + counts):
                orgs.append(dict(model=model, preset=match[1], parameter=key,
                                 value=value, source=source, line=number))
        if ".latency =" in line and not line.lstrip().startswith("//"):
            match = re.search(r'\.level = "([^"]+)".*?\.preceding = \{([^}]+)\}.*?\.following = \{([^}]+)\}.*?\.latency = (.*?)(?:, \.window = (\d+))?\},?\s*$', line)
            assert match, (path, number, line)
            constraints.append(dict(model=model, level=match[1], preceding=";".join(re.findall(r'"([^"]+)"', match[2])),
                                    following=";".join(re.findall(r'"([^"]+)"', match[3])), expression=match[4].strip(),
                                    window=match[5] or "1", source=source, line=number))
    return timings, orgs, constraints


def reconstruct(work, manifest):
    targets = {}
    for rel, expected in manifest["baseline_files"].items():
        original = BASELINE / rel
        assert sha(original) == expected, f"Baseline changed: {original}"
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
    # Read only the copy manifest; do not execute the installation script.
    for line in (SOURCE / "set_pim_ramulator.sh").read_text().splitlines():
        if not line.startswith("cp "):
            continue
        args = shlex.split(line)
        recursive = args[1] == "-r"
        src, dst = args[-2:]
        if src.endswith("/patches"):
            continue  # installation payload, counted separately from effective files
        src_path = SOURCE / src
        target = Path(dst).relative_to("ramulator2")
        if recursive:
            pairs = [(p, target / src_path.name / p.name) for p in sorted(src_path.iterdir()) if p.is_file()]
        else:
            pairs = [(src_path, target / src_path.name)]
        for p, rel in pairs:
            assert not (BASELINE / rel).exists(), f"Expected new file: {rel}"
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, work / rel)
            targets[str(rel)] = str(p.relative_to(SOURCE))
    patch_log = []
    for patch in sorted((SOURCE / "pim_ramulator_src/patches").glob("*.patch")):
        rels = re.findall(r"^--- a/(.+)$", patch.read_text(), re.M)
        assert len(rels) == 1 and rels[0] in manifest["baseline_files"]
        result = subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
                                cwd=work, capture_output=True, text=True, check=True)
        assert "offset" not in result.stdout.lower(), result.stdout
        targets[rels[0]] = str(patch.relative_to(SOURCE))
        patch_log.append(patch.name + ": " + result.stdout.strip())
    inventory, diff = [], []
    commit = manifest["attacc_commit"]
    for rel, origin in sorted(targets.items()):
        before_path, after_path = BASELINE / rel, work / rel
        before = before_path.read_text().splitlines(keepends=True) if before_path.exists() else []
        after = after_path.read_text().splitlines(keepends=True)
        delta = list(difflib.unified_diff(before, after, fromfile="a/" + rel, tofile="b/" + rel))
        assert delta, f"No delta: {rel}"
        inventory.append(dict(path=rel, change="modified" if before else "added", source=origin,
                              source_url=f"https://github.com/scale-snu/attacc_simulator/blob/{commit}/{origin}",
                              before_sha256=sha(before_path) if before_path.exists() else "",
                              after_sha256=sha(after_path),
                              added_lines=sum(x.startswith("+") and not x.startswith("+++") for x in delta),
                              removed_lines=sum(x.startswith("-") and not x.startswith("---") for x in delta)))
        diff.extend(delta)
    return inventory, "".join(diff), "\n".join(patch_log) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True, help="Existing disposable directory below /scratch")
    parser.add_argument("--refresh", action="store_true", help="Regenerate machine-readable audit tables")
    args = parser.parse_args()
    scratch = args.scratch_root.resolve()
    assert scratch.is_dir() and Path("/scratch") in scratch.parents
    manifest = json.loads((HERE / "sources/manifest.json").read_text())
    for item in manifest["attacc_files"]:
        assert sha(SOURCE / item["path"]) == item["sha256"], f"Snapshot changed: {item['path']}"
    for rel, expected in manifest["baseline_support_files"].items():
        assert sha(BASELINE / rel) == expected, f"Baseline support file changed: {rel}"
    generated = {}
    with tempfile.TemporaryDirectory(prefix="reconstruct-", dir=scratch) as directory:
        work = Path(directory)
        inventory, diff, patch_log = reconstruct(work, manifest)
        generated["file_delta.csv"] = csv_text(inventory)
        generated["effective.diff"] = diff
        generated["patch_application.txt"] = patch_log
        generated["attacc_HBM3.cpp"] = (work / "src/dram/impl/HBM3.cpp").read_text()
        timings, orgs, constraints = [], [], []
        models = [
            ("ramulator-original-HBM3", BASELINE / "src/dram/impl/HBM3.cpp", "ramulator2/src/dram/impl/HBM3.cpp"),
            ("attacc-HBM3", work / "src/dram/impl/HBM3.cpp", "generated/attacc_HBM3.cpp"),
            ("attacc-HBM3-PIM", SOURCE / "pim_ramulator_src/HBM3-PIM.cpp", "sources/attacc/pim_ramulator_src/HBM3-PIM.cpp")]
        for model, path, source in models:
            t, o, c = extract_model(model, path, source)
            timings.extend(t); orgs.extend(o); constraints.extend(c)
        generated["timing_presets.csv"] = csv_text(timings)
        generated["organization_presets.csv"] = csv_text(orgs)
        generated["timing_constraints.csv"] = csv_text(constraints)
        # Independent arithmetic cross-check, not a claimed author derivation.
        hbm2_ns = {"nCL": 14, "nRCDRD": 14, "nRCDWR": 14, "nRP": 14,
                   "nRAS": 34, "nWR": 16, "nRTPS": 4, "nRTPL": 6, "nCWL": 4,
                   "nWTRS": 6, "nWTRL": 8, "nFAW": 30, "nREFI": 3900, "nRC": 48}
        checked = []
        for row in timings:
            if row["model"] == "attacc-HBM3" and row["parameter"] in hbm2_ns:
                rate = int(round(float(row["preset"].split("_")[1].replace("Gbps", "")) * 1000))
                ns = hbm2_ns[row["parameter"]]
                result = math.ceil(ns * rate / 4000)
                assert result == row["raw_value"], row
                checked.append(dict(preset=row["preset"], parameter=row["parameter"], hbm2_ns=ns,
                                    rule="ceil(ns*rate/4000)", computed_cycles=result,
                                    attacc_cycles=row["raw_value"], status="DERIVED; independent cross-check; S2"))
        assert len(checked) == 70
        generated["hbm2_crosscheck.csv"] = csv_text(checked)
        default = {r["parameter"]: r["raw_value"] for r in timings
                   if r["model"] == "attacc-HBM3-PIM" and r["preset"] == "HBM3_5.2Gbps"}
        clock_ps = int(1e6 / (default["rate"] / 4))
        def jedec(ns):
            return ((ns * 1000 * 1000 // clock_ps) + 974) // 1000
        effective = dict(default, tCK_ps=clock_ps, nRFC=jedec(350), nREFISB=jedec(82))
        generated["timing_effective_default.csv"] = csv_text([
            dict(parameter=k, raw_value=v, effective_value=effective[k],
                 unit="MT/s" if k == "rate" else "ps" if k == "tCK_ps" else "cycles",
                 qualification="HBM3_8Gb_2R; 5.2Gbps PC; no YAML overrides; reproduces known rank-index issue")
            for k, v in default.items()])
        original_cmds = definition(models[0][1].read_text(), "m_commands")
        pim_cmds = definition(models[2][1].read_text(), "m_commands")
        added = [x for x in pim_cmds if x not in original_cmds]
        removed = [x for x in original_cmds if x not in pim_cmds]
        summary = dict(snapshot_files=len(manifest["attacc_files"]), patches=len(manifest["baseline_files"]),
                       effective_files=len(inventory), added_files=sum(x["change"] == "added" for x in inventory),
                       modified_files=sum(x["change"] == "modified" for x in inventory),
                       timing_values=len(timings), organization_values=len(orgs), constraint_rows=len(constraints),
                       hbm2_numeric_crosschecks=len(checked),
                       commands_added=added, commands_removed_from_pim=removed,
                       validation="source hashes + baseline hashes + zero-fuzz/zero-offset patch application + catalog coverage; no simulator build or silicon validation")
        generated["summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        # Human provenance tables must cover every changed file and named timing.
        file_reviews = list(csv.DictReader((HERE / "file_review.csv").open()))
        assert {x["path"] for x in file_reviews} == {x["path"] for x in inventory}
        assert len(file_reviews) == len(inventory)
        timing_reviews = list(csv.DictReader((HERE / "timing_sources.csv").open()))
        assert {x["parameter"] for x in timing_reviews} == {x["parameter"] for x in timings}
        assert len(timing_reviews) == len({x["parameter"] for x in timings})
        for row in file_reviews + timing_reviews:
            assert all(str(value).strip() for value in row.values()), row
    dest = HERE / "generated"
    if args.refresh:
        dest.mkdir(exist_ok=True)
    for name, text in generated.items():
        if args.refresh:
            (dest / name).write_text(text)
        else:
            assert (dest / name).read_text() == text, f"Generated artifact differs: {name}"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PASS: provenance coverage and all reproducible artifacts agree.")


if __name__ == "__main__":
    main()

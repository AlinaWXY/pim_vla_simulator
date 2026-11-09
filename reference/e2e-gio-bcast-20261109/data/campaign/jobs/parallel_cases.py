"""Run one frozen case/repetition, or validate and aggregate all completed runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("repeat", "finalize"), required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / "frozen/source"))
    from experiments.common import read_json, sha256
    from experiments.config import CASES
    from experiments.native import execute, runtime_libraries, library_hashes
    from experiments.run import run_matrix, aggregate

    cases = tuple(CASES)
    binary = run / "frozen/bin/pim_vla_sim"
    configs = {case: read_json(run / "configs" / f"{case}.json") for case in cases}
    calibration_path = run / "calibration/calibration.json"
    calibration = read_json(calibration_path)
    if not calibration.get("validation", {}).get("passed"):
        raise ValueError("VLM calibration has not passed")
    if calibration["binary_sha256"] != sha256(binary):
        raise ValueError("calibration binary differs")
    if library_hashes(calibration) != library_hashes({"runtime_libraries": runtime_libraries(binary)}):
        raise ValueError("calibration runtime library differs")
    if calibration["spec"] != configs["b"]["spec"]:
        raise ValueError("calibration hardware spec differs")
    if calibration["vlm_sha256"] != sha256(configs["b"]["vlm_path"]):
        raise ValueError("calibration VLM input differs")
    if calibration["raw_sha256"] != sha256(calibration["raw_path"]):
        raise ValueError("calibration raw result differs")

    def destination(case, repetition):
        return run / "cases" / case / "native" / case / f"repeat-{repetition}"

    if args.phase == "repeat":
        if not 0 <= args.index < 2 * len(cases):
            raise ValueError(f"repeat index must be in [0, {2 * len(cases) - 1}]")
        case_index, repetition = divmod(args.index, 2)
        case = cases[case_index]
        config = dict(configs[case], calibration=calibration["parameters"])
        print(f"CASE {case} REPEAT {repetition}", flush=True)
        execute(binary, config, destination(case, repetition))
        return

    # Fail before reporting if any repetition is incomplete. In particular,
    # never let run_matrix's resume mode start an absent native simulation.
    for case in cases:
        for repetition in range(2):
            for name in ("input-config.json", "raw.json", "provenance.json"):
                path = destination(case, repetition) / name
                if not path.is_file():
                    raise FileNotFoundError(f"incomplete case/repetition: {path}")
    for case in cases:
        run_matrix(binary, configs, run / "cases" / case, cases=[case], repeats=2,
                   resume=True, calibration_path=calibration_path,
                   reuse_solos=run / "solos")
    aggregate([run / "cases" / case for case in cases], run / "matrix")


if __name__ == "__main__":
    main()

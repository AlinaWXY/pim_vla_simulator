"""JESD209-5C LPDDR5X-8533: x16, BG, BL16, CKR4, RL0/WLA, ECC/DVFS off."""

import math

from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint


class LPDDR5X(LPDDR5):
    name = "LPDDR5X"
    timing_params = LPDDR5.timing_params + ["nRCDWR"]
    timing_constraints = [
        tc for tc in LPDDR5.timing_constraints
        if not (tc.level == "Bank" and tc.preceding == ["ACT2"]
                and tc.following == ["RD", "RDA", "WR", "WRA"])
    ] + [
        # Table 396 notes 1-2: ordinary writes may start earlier than reads.
        # Masked write is not a supported command in this model.
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["RD", "RDA"], latency="nRCD"),
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["WR", "WRA"], latency="nRCDWR"),
    ]

    @classmethod
    def validate_organization(cls, org):
        super().validate_organization(org)
        expected = {"dq": 16, "channel_width": 16, "bankgroup": 4, "bank": 4, "column": 1024}
        for key, value in expected.items():
            if org[key] != value:
                raise ValueError(f"LPDDR5X calibration supports only x16 BG BL16: {key}={value}")
        if org["rank"] not in (1, 2):
            raise ValueError("LPDDR5X supports one or two ranks")
        rows = {8192: 32768, 16384: 65536}.get(org["density"])
        if rows is None or org["row"] != rows:
            raise ValueError("LPDDR5X density and row count must describe the same organization")

    @classmethod
    def resolve_secondary_timings(cls, timings, org):
        # Table 240 refresh requirements apply to both LPDDR5 and LPDDR5X.
        super().resolve_secondary_timings(timings, org)
        ck = timings["tCK_ps"]
        # Table 396, not the LPDDR5 Table 381 inherited defaults.
        timings["nRRDS"] = timings["nRRDL"] = max(2, math.ceil(3750 / ck))
        timings["nFAW"] = math.ceil(15000 / ck)

    def resolve(self):
        # Base DSL applies clock overrides after secondary timing resolution.
        # Reject mismatched clocks rather than silently retaining stale timings.
        for key, expected in (("rate", 8533), ("tCK_ps", 938)):
            if key in self._overrides and self._overrides[key] != expected:
                raise ValueError(f"LPDDR5X_8533 fixes {key}={expected}")
        return super().resolve()


LPDDR5X.org_presets = {
    "LPDDR5X_8Gb_x16": dict(LPDDR5.org_presets["LPDDR5_8Gb_x16"]),
    "LPDDR5X_16Gb_x16": dict(LPDDR5.org_presets["LPDDR5_16Gb_x16"]),
}

# Integer ps clock rounds UP to 938ps. Effective rate is 8528.785 MT/s,
# within Table 225's (7500,8533] data-rate and CK-frequency bin.
LPDDR5X.timing_presets = {
    "LPDDR5X_8533": {
        "rate": 8533, "tCK_ps": 938,
        "nBL_min": 2, "nBL_max": 4,
        # Tables 225 (RL set 0), 229 (WL set A), 230 (x16/ECC/DVFS off).
        "nCL": 23, "nCWL": 12, "nWR": 37, "nRTP": 8,
        # Table 396: 18ns read RCD, 8ns write RCD, 18/21ns RP, 42ns RAS.
        "nRCD": 20, "nRCDWR": 9, "nRP": 20, "nRPab": 23,
        "nRAS": 45, "nRC": 65, "nPPD": 2,
        # Tables 339-342: fixed column cycles; Table 396 WTR 6.25/12ns.
        "nCCDS": 2, "nCCDL": 4, "nCCDS_WR": 2, "nCCDL_WR": 4,
        "nWTRS": 7, "nWTRL": 13,
        # MR10 WCKPST=4.5WCK, RPST=0.5WCK, RPRE=4WCK; Table 377.
        "nWCKPST": 1, "nCAS": 1, "nAAD": 8, "nCS": 2,
    },
}

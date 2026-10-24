"""Research-only all-bank and half-bank PIM on calibrated LPDDR5 timings."""

from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint


def _pim_constraints(base=LPDDR5):
    # Reuse activation/refresh constraints, but NOT RD's external DQ/CAS
    # constraints. MACAB reads the local array without driving the DQ bus.
    # ACTAB occupies two CA cycles; the DSL adjusts to command start clocks.
    constraints = []
    for tc in base.timing_constraints:
        preceding = list(tc.preceding)
        following = list(tc.following)
        for ordinary, pim in (("ACT2", "ACTAB"), ("ACT2", "ACTHB")):
            if ordinary in preceding:
                preceding.append(pim)
            if ordinary in following:
                following.append(pim)
        constraints.append(TimingConstraint(
            tc.level, preceding, following, tc.latency,
            window=tc.window, sibling=tc.sibling, shared_window=tc.shared_window,
        ))
        # Half-bank compute does not occupy DQ. Only the banks it reads must
        # obey ordinary column/row dependencies; the other half can issue RD.
        if tc.level == "Bank" or (tc.level == "Rank" and
                                 "PREab" in tc.preceding + tc.following):
            half_preceding = list(preceding)
            half_following = list(following)
            for ordinary, half in (("RD", "MACHB"), ("PREab", "PREHB")):
                if ordinary in half_preceding:
                    half_preceding.append(half)
                if ordinary in half_following:
                    half_following.append(half)
            constraints.append(TimingConstraint(
                "Bank", half_preceding, half_following, tc.latency,
                window=tc.window, sibling=tc.sibling, shared_window=tc.shared_window,
            ))
    constraints.extend([
        TimingConstraint("Rank", ["MACAB"], ["MACAB"], "nCCDAB"),
        # Explicit local-array dependencies. These are independent of the
        # external RD burst, read/write turnaround, rank switching, and CAS.
        TimingConstraint("Bank", ["ACT2", "ACTAB", "ACTHB"], ["MACAB"], "nRCD"),
        TimingConstraint("Bank", ["MACAB"], ["PREpb", "PREHB"], "nRTP"),
        TimingConstraint("Rank", ["MACAB"], ["PREab"], "nRTP"),
        TimingConstraint("Bank", ["MACAB"], ["MACAB", "RD", "RDA", "WR", "WRA"], "nCCDL"),
        TimingConstraint("Bank", ["RD", "RDA"], ["MACAB"], "nCCDL"),
        # A preceding normal write must reach the array and recover before
        # its bank supplies weights; this is not a cross-rank DQ constraint.
        TimingConstraint("Bank", ["WR", "WRA"], ["MACAB"], "nCWL+nBL_max+nWTRL"),
        # Buffer transfers share x16 DQ, but do not access the DRAM array.
        TimingConstraint("Channel", ["WRGB"], ["WRGB"], "nBL_min"),
        TimingConstraint("Channel", ["RDRES"], ["RDRES"], "nBL_min"),
        TimingConstraint("Channel", ["RDRES"], ["WRGB"], "nCL+nBL_min+2-nCWL"),
        TimingConstraint("Channel", ["WRGB"], ["RDRES"], "max(1,nCWL+nBL_min+1-nCL)"),
        # Selecting a different rank still incurs the calibrated DQ turnarounds.
        TimingConstraint("Rank", ["RDRES"], ["RDRES"], "nBL_min+nCS", sibling=True),
        TimingConstraint("Rank", ["WRGB"], ["WRGB"], "nBL_min+1", sibling=True),
        # Input readiness is a per-rank/per-buffer dependency in the controller.
        # A write to the inactive buffer must not delay MACs using the active one.
        TimingConstraint("Channel", ["RDRES"], ["MACAB"], "nPIMRdLatency"),
        TimingConstraint("Rank", ["MACHB"], ["MACHB"], "nCCDAB"),
        # Preserve column-array spacing when switching compute/normal modes.
        # Scope these to participating banks so the other half can still read.
        TimingConstraint("Bank", ["MACHB"],
                         ["RD", "RDA", "WR", "WRA", "MACAB"], "nCCDL"),
        TimingConstraint("Bank", ["MACAB"], ["MACHB"], "nCCDL"),
        TimingConstraint("Bank", ["RD", "RDA"], ["MACHB"], "nCL+nBL_min"),
        TimingConstraint("Channel", ["WRGB"], ["MACHB"], "nPIMWrLatency"),
        TimingConstraint("Channel", ["RDRES"], ["MACHB"], "nPIMRdLatency"),
    ])
    # The two new MAC commands use native local-array constraints but a
    # different input batch size. Do not inherit the legacy nCCDAB=20 service.
    original = list(constraints)
    for tc in original:
        if "MACAB" not in tc.preceding + tc.following:
            continue
        if tc.latency == "nCCDAB":
            continue
        for op in ("MACK", "MACQK"):
            constraints.append(TimingConstraint(tc.level,
                [op if c == "MACAB" else c for c in tc.preceding],
                [op if c == "MACAB" else c for c in tc.following],
                tc.latency, window=tc.window, sibling=tc.sibling,
                shared_window=tc.shared_window))
    for op, interval in (("MACK", "nCCDK"), ("MACQK", "nCCDQK")):
        constraints.append(TimingConstraint("Bank", [op], ["MACK", "MACQK"], interval))
        constraints.append(TimingConstraint("Bank", [op], ["LOCALWR"], "nCCDL"))
    # Bank-local 32B write path: no external DQ, CL/CWL, or rank turnaround.
    constraints += [
        TimingConstraint("Bank", ["MACAB", "MACK", "MACQK"], ["MACAB", "MACK", "MACQK"], "nCCDL"),
        TimingConstraint("Bank", ["ACT2", "ACTAB", "ACTHB"], ["LOCALWR"],
                         "nRCDWR" if "nRCDWR" in base.timing_params else "nRCD"),
        TimingConstraint("Bank", ["LOCALWR"], ["LOCALWR"], "nLOCALWR"),
        TimingConstraint("Bank", ["LOCALWR"], ["PREpb", "PREab", "PREHB"], "nLOCALWR+nWR"),
        TimingConstraint("Bank", ["LOCALWR"], ["MACK", "MACQK", "MACAB", "RD", "RDA"], "nLOCALWR+nWTRL"),
        TimingConstraint("Bank", ["PREpb", "PREab", "PREHB"], ["LOCALWR"], "nRP"),
        TimingConstraint("Bank", ["REFpb", "REFab"], ["LOCALWR"], "nRFC"),
    ]
    return constraints


class LPDDR5PIM(LPDDR5):
    name = "LPDDR5PIM"
    commands = LPDDR5.commands + ["ACTAB", "MACAB", "WRGB", "RDRES", "ACTHB", "MACHB", "PREHB", "MACK", "MACQK", "LOCALWR"]
    command_cycles = {**LPDDR5.command_cycles, "ACTAB": 2, "ACTHB": 2}
    timing_params = LPDDR5.timing_params + [
        "nCCDAB", "nPIMWrLatency", "nPIMRdLatency", "nCCDK", "nCCDQK", "nLOCALWR",
    ]
    supported_requests = {
        **LPDDR5.supported_requests, "PIM_MAC_AB": "MACAB",
        "PIM_WR_GB": "WRGB", "PIM_RD_RES": "RDRES",
        "PIM_MAC_HB": "MACHB",
        "PIM_LOCAL_GB": "WRGB",  # Controller-only SRAM event; placeholder request id 6.
        "PIM_WR_BG": "WRGB", "PIM_WR_Q": "WRGB", "PIM_MAC_K": "MACK",
        "PIM_MAC_QK": "MACQK", "PIM_RD_BG": "RDRES", "PIM_LOCAL_WR": "LOCALWR",
    }
    timing_constraints = _pim_constraints()

    def __init__(self, *, pim_data_path="lpddr", **kwargs):
        if pim_data_path not in ("lpddr", "attacc"):
            raise ValueError("pim_data_path must be lpddr or attacc")
        self.pim_data_path = pim_data_path
        super().__init__(**kwargs)
        if pim_data_path == "attacc":
            # AttAcc HBM3-PIM.cpp:420-427: transfers share one path; MAC
            # has a separate path. Keep the LPDDR array and CA constraints.
            # All constraints are instance-local; a legacy model is unchanged.
            self.timing_constraints = [
                tc for tc in type(self).timing_constraints
                if not {"WRGB", "RDRES"}.intersection(tc.preceding + tc.following)
            ]
            transfers = ["WRGB", "RDRES", "RD", "RDA", "WR", "WRA"]
            self.timing_constraints += [
                TimingConstraint("Channel", transfers, transfers, "nBL_min"),
                TimingConstraint("Rank", transfers, transfers, "nCCDS"),
                TimingConstraint("BankGroup", transfers, transfers, "nCCDL"),
            ]

    @classmethod
    def _generate_bus_constraints(cls, cmd_idx, cmd_cycles, tick_mult):
        # LOCALWR is an internal per-bank port event, not a CA transaction.
        local = cmd_idx["LOCALWR"]
        constraints = super()._generate_bus_constraints(cmd_idx, cmd_cycles, tick_mult)
        return [[level, [x for x in preceding if x != local],
                 [x for x in following if x != local], *rest]
                for level, preceding, following, *rest in constraints
                if any(x != local for x in preceding) and any(x != local for x in following)]

    def resolve(self):
        # Derive from the FINAL normal timings, including user overrides.
        # PE frequency and pipeline latency are not inferred from read latency.
        org, timings = super().resolve()
        overrides = self._overrides
        for key, default in (("nCCDK", 2), ("nCCDQK", 20), ("nLOCALWR", 4)):
            if key not in overrides:
                timings[key] = default
        if "nCCDAB" not in overrides:
            # Each broadcast revisits the same banks. Assume the PE accepts
            # one operand block per same-BG column interval, without adding
            # power throttling.
            timings["nCCDAB"] = timings["nCCDL"]
        if self.pim_data_path == "attacc":
            if {"nPIMWrLatency", "nPIMRdLatency"}.intersection(overrides):
                raise ValueError("AttAcc transfers retire at issue; transfer latency overrides are unsupported")
            timings["nPIMWrLatency"] = timings["nPIMRdLatency"] = 0
            if type(timings["nCCDAB"]) is not int or timings["nCCDAB"] <= 0:
                raise ValueError("nCCDAB must be a positive integer")
            return org, timings
        for name, formula in (
            ("nPIMWrLatency", timings["nCWL"] + timings["nBL_min"]),
            ("nPIMRdLatency", timings["nCL"] + timings["nBL_min"]),
        ):
            if name not in overrides:
                timings[name] = formula
        for name in ("nCCDAB", "nPIMWrLatency", "nPIMRdLatency"):
            if type(timings[name]) is not int or timings[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return org, timings

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        super().resolve_secondary_timings(timing_dict, org_dict)
        timing_dict["nCCDAB"] = 0
        timing_dict["nPIMWrLatency"] = 0
        timing_dict["nPIMRdLatency"] = 0
        timing_dict["nCCDK"] = 2
        timing_dict["nCCDQK"] = 20
        timing_dict["nLOCALWR"] = 4


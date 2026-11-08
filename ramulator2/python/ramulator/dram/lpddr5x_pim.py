"""LPDDR5X-8533 and PIM extensions, with a single native timing authority.

JEDEC JESD209-5C: Tables 225, 229, 230, 237, 240, 339-342, 377, 396.
PIM resource latencies are architectural assumptions, not JEDEC timings.
"""
import math

from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint as TC

ROWS = (1, 2, 4, 8, 16, 32)
IO_ALIASES = {"RD": "RDRES", "RDA": "RDRES", "WR": "WRGB", "WRA": "WRGB"}
ROW_ALIASES = {c: c + "_LOCAL" for c in ("ACT1", "ACT2", "PREpb", "PREab")}
MAC_ROWS = {f"{kind}_M{rows}{suffix}": rows
            for kind in ("MACAB", "MACQK") for rows in ROWS
            for suffix in ("", "_LOCAL")}
MAC_ROWS.update({"MACAB": 16, "MACQK": 16, "MACAB_LOCAL": 16, "MACQK_LOCAL": 16})
MACS = list(MAC_ROWS)
LOCAL_EVENTS = ["LOCALRD", "LOCALWR", "BCAST", "GATHER", "CAST", "REDUCE", "ROPE_EXEC"]
# A fixed one-command/CK rank-internal issue path serves autonomous local
# events in streaming and descriptor-expanded commands in fused programs.
# Issue occupancy is independent of each unit's execution latency.
PRIVATE = list(ROW_ALIASES.values()) + [c for c in MACS if c.endswith("_LOCAL")] + LOCAL_EVENTS
NO_EXTERNAL_CA = set(PRIVATE)


def _copy(tc, pre=None, fol=None, level=None):
    return TC(level or tc.level, pre or tc.preceding, fol or tc.following, tc.latency,
              window=tc.window, sibling=tc.sibling, shared_window=tc.shared_window)


def _expand(items, aliases):
    return list(dict.fromkeys(items + [aliases[c] for c in items if c in aliases]))


def _base_constraints():
    result = []
    for tc in LPDDR5.timing_constraints:
        # LPDDR5X permits ACT2 -> ordinary WR after 8 ns; RD needs 18 ns.
        if tc.level == "Bank" and tc.preceding == ["ACT2"] and tc.following == ["RD", "RDA", "WR", "WRA"]:
            result.extend([TC("Bank", ["ACT2"], ["RD", "RDA"], "nRCD"),
                           TC("Bank", ["ACT2"], ["WR", "WRA"], "nRCDWR")])
        else:
            result.append(_copy(tc))
    return result


def _constraints():
    result = []
    for tc in _base_constraints():
        # Only transfer-to-transfer interface constraints alias to SRF/pool IO.
        # Rank RD/WR -> PREab is an array recovery rule, not an IO rule.
        if tc.level in ("Channel", "Rank", "BankGroup") and all(c in IO_ALIASES for c in tc.preceding + tc.following):
            tc = _copy(tc, _expand(tc.preceding, IO_ALIASES), _expand(tc.following, IO_ALIASES))
        tc = _copy(tc, _expand(tc.preceding, ROW_ALIASES), _expand(tc.following, ROW_ALIASES))
        result.append(tc)
    acts = ["ACT2", "ACT2_LOCAL"]
    pres = ["PREpb", "PREab", "PREpb_LOCAL", "PREab_LOCAL"]
    reads = ["RD", "RDA"]
    writes = ["WR", "WRA"]
    result += [TC("Bank", acts, MACS + ["LOCALRD"], "nRCD"),
               TC("Bank", acts, ["LOCALWR"], "nRCDWR"),
               TC("Bank", MACS + ["LOCALRD"], pres, "nRTP"),
               TC("Bank", MACS, reads + writes + ["LOCALRD", "LOCALWR"], "nCCDL"),
               TC("Bank", reads, MACS + ["LOCALRD"], "nCCDL"),
               TC("Bank", writes, MACS + ["LOCALRD"], "nCWL+nBL_max+nWTRL"),
               TC("Bank", ["LOCALRD"], MACS + reads + ["LOCALRD"], "nCCDL"),
               TC("Bank", ["LOCALRD"], writes + ["LOCALWR"], "nCL+nBL_min"),
               TC("Bank", ["LOCALWR"], ["LOCALWR"], "nLOCALWR"),
               TC("Bank", ["LOCALWR"], pres, "nLOCALWR+nWR"),
               TC("Bank", ["LOCALWR"], MACS + reads + ["LOCALRD"], "nLOCALWR+nWTRL"),
               TC("Bank", pres, ["LOCALRD", "LOCALWR"], "nRP")]
    for cmd, rows in MAC_ROWS.items():
        # One column fetch feeds M rows; reads overlap PE processing.
        result.append(TC("Bank", [cmd], MACS, f"max(nCCDL,nMAC{rows})"))
    for cmd, timing in (("BCAST","nBCAST"),("GATHER","nGATHER"),("REDUCE","nREDUCE"),("ROPE_EXEC","nROPEII")):
        result.append(TC("Rank", [cmd], [cmd], timing))
    result.append(TC("Bank", ["CAST"], ["CAST"], "nCAST"))
    # Refresh applies even to buffer IO; this is the requested conservative
    # rank model. REFpb uses native paired-bank targeting (n and n+8).
    nonrefresh = [c for c in LPDDR5.commands + list(ROW_ALIASES.values()) + MACS + LOCAL_EVENTS + ["WRGB", "RDRES", "MAC_RUN", "ROPE_RUN"] if c not in ("REFab", "REFpb")]
    result += [TC("Rank", ["REFab"], nonrefresh, "nRFC"),
               TC("Bank", ["REFpb"], nonrefresh, "nRFCpb")]
    result += [TC("Rank", ["WRGB"], ["REFab"], "nCWL+nBL_min"),
               TC("Rank", ["RDRES"], ["REFab"], "nCL+nBL_min"),
               TC("Bank", ["WRGB"], ["REFpb"], "nCWL+nBL_min"),
               TC("Bank", ["RDRES"], ["REFpb"], "nCL+nBL_min")]
    return result


class LPDDR5XPIM(LPDDR5):
    name = "LPDDR5XPIM"
    commands = LPDDR5.commands + ["WRGB", "RDRES"] + MACS + list(ROW_ALIASES.values()) + LOCAL_EVENTS + ["MAC_RUN", "ROPE_RUN"]
    timing_params = LPDDR5.timing_params + ["nRCDWR", "nDQ_RTW", "nDQ_WTR", "nDQ_RANK_WTW"] + [f"nMAC{r}" for r in ROWS] + ["nLOCALRD", "nLOCALWR", "nBCAST", "nGATHER", "nCAST", "nREDUCE", "nROPE", "nROPEII", "nDESCDecode"]
    # Internal commands retain one logical receive tick: native shared FAW
    # histories record clk + command_cycles - 1. External CA is metadata-only.
    command_cycles = {"MAC_RUN": 21, "ROPE_RUN": 21}
    timing_constraints = _constraints()
    supported_requests = {"Read": "RD", "Write": "WR"}

    def __init__(self, *, mac_freq_mult=4, org_preset="LPDDR5X_8Gb_x16", timing_preset="LPDDR5X_8533", **kwargs):
        if type(mac_freq_mult) is not int or mac_freq_mult not in (1, 2, 4):
            raise ValueError("mac_freq_mult must be 1, 2, or 4")
        self.mac_freq_mult = mac_freq_mult
        super().__init__(org_preset=org_preset, timing_preset=timing_preset, **kwargs)

    @classmethod
    def validate_organization(cls, org):
        super().validate_organization(org)
        for key, value in {"dq":16,"channel_width":16,"bankgroup":4,"bank":4,"column":1024}.items():
            if org[key] != value:
                raise ValueError(f"LPDDR5X PIM requires {key}={value}")
        if org["rank"] != 2:
            raise ValueError("LPDDR5X PIM requires rank0 GPU and rank1 PIM")

    @classmethod
    def resolve_secondary_timings(cls, t, org):
        super().resolve_secondary_timings(t, org)
        t["nRRDS"] = t["nRRDL"] = max(2, math.ceil(3750/t["tCK_ps"]))
        t["nFAW"] = math.ceil(15000/t["tCK_ps"])
        for r in ROWS:
            t[f"nMAC{r}"] = r
        # Inherited microarchitecture assumptions: local 32 B port, SRF
        # multicast, cast pipeline, two-level rank reduction and RoPE PE.
        t.update(nDQ_RTW=2, nDQ_WTR=1, nDQ_RANK_WTW=1,
                 nLOCALRD=4, nLOCALWR=4, nBCAST=2, nGATHER=2,
                 nCAST=1, nREDUCE=4, nROPE=4, nROPEII=1, nDESCDecode=1)

    def resolve(self):
        for k, v in {"rate":8533,"tCK_ps":938}.items():
            if k in self._overrides and self._overrides[k] != v:
                raise ValueError(f"fixed LPDDR5X-8533 requires {k}={v}")
        org, t = super().resolve()
        for r in ROWS:
            t[f"nMAC{r}"] = 4*r//self.mac_freq_mult
        return org, t

    @classmethod
    def _generate_bus_constraints(cls, idx, cycles, mult):
        external = [idx[c] for c in cls.commands if c not in NO_EXTERNAL_CA]
        private = [idx[c] for c in PRIVATE]
        # Unlike upstream's controller-implicit one-command-per-tick rule,
        # this spec explicitly serializes both buses, including 1 CK commands.
        result = []
        for name in cls.commands:
            if name not in NO_EXTERNAL_CA:
                result.append([0, [idx[name]], external, cycles.get(name, 1)])
        for name in PRIVATE:
            result.append([1, [idx[name]], private, 1])
        return result

LPDDR5XPIM.org_presets = {name.replace("LPDDR5_", "LPDDR5X_"): {**value, "rank":2} for name, value in LPDDR5.org_presets.items()}
LPDDR5XPIM.timing_presets = {"LPDDR5X_8533": {
    "rate":8533,"tCK_ps":938,"nBL_min":2,"nBL_max":4,
    "nCL":23,"nCWL":12,"nWR":37,"nRTP":8,
    "nRCD":20,"nRCDWR":9,"nRP":20,"nRPab":23,"nRAS":45,"nRC":65,
    "nPPD":2,"nCCDS":2,"nCCDL":4,"nCCDS_WR":2,"nCCDL_WR":4,
    "nWTRS":7,"nWTRL":13,"nWCKPST":1,"nCAS":0,"nAAD":8,"nCS":2,
}}


def build_spec(mac_freq_mult=4):
    """JSON-serializable native config and spec-owned completion/bus metadata."""
    spec = LPDDR5XPIM(mac_freq_mult=mac_freq_mult)
    org, timings = spec.resolve()
    commands = {}
    event_cycles = {"LOCALRD":"nLOCALRD","LOCALWR":"nLOCALWR","BCAST":"nBCAST",
                    "GATHER":"nGATHER","CAST":"nCAST","REDUCE":"nREDUCE","ROPE_EXEC":"nROPE"}
    for i, c in enumerate(spec.commands):
        meta = {"id":i,"external_ca":0 if c in NO_EXTERNAL_CA else spec.command_cycles.get(c,1),
                "private_ca":1 if c in PRIVATE else 0,"completion":0,"dq_latency":0,"dq_cycles":0,"direction":"none"}
        if c in ("RD","RDA","RDRES","WR","WRA","WRGB"):
            read = c in ("RD","RDA","RDRES")
            meta.update(dq_latency=timings["nCL" if read else "nCWL"], dq_cycles=timings["nBL_min"], direction="read" if read else "write")
            meta["completion"] = meta["dq_latency"] + meta["dq_cycles"]
        elif c in MAC_ROWS:
            meta["completion"] = timings[f"nMAC{MAC_ROWS[c]}"]
            meta["rows"] = MAC_ROWS[c]
        elif c in event_cycles:
            meta["completion"] = timings[event_cycles[c]]
        elif c in ("MAC_RUN","ROPE_RUN"):
            meta["completion"] = meta["external_ca"] + timings["nDESCDecode"]
        elif c in ("REFab","REFpb"):
            meta["completion"] = timings["nRFC" if c == "REFab" else "nRFCpb"]
        commands[c] = meta
    native_commands = {name:{**{k:v for k,v in meta.items() if k not in ("dq_latency","dq_cycles","direction")},
                             "latency":meta["dq_latency"], "burst":meta["dq_cycles"],
                             "direction":{"none":"","read":"R","write":"W"}[meta["direction"]]}
                       for name,meta in commands.items()}
    metadata = {"commands":native_commands,
                "dq_turnaround":{"R_W":timings["nDQ_RTW"],"W_R":timings["nDQ_WTR"],
                                 "sibling_R_R":timings["nCS"],"sibling_W_W":timings["nDQ_RANK_WTW"]}}
    return {"dram":spec.to_config(),"timings":timings,"commands":commands,"metadata":metadata,"organization":org,
            "mac_freq_mult":mac_freq_mult,"clock_ps":timings["tCK_ps"],
            "source":"JESD209-5C + task-document 2026-11-09 PIM assumptions"}

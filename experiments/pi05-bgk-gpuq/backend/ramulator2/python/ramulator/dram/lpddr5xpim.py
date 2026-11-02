"""All-bank PIM on the existing, calibrated LPDDR5X-8533 model."""

from ramulator.dram.lpddr5pim import LPDDR5PIM, _pim_constraints
from ramulator.dram.lpddr5x import LPDDR5X


class LPDDR5XPIM(LPDDR5PIM, LPDDR5X):
    name = "LPDDR5XPIM"
    timing_params = LPDDR5X.timing_params + [
        "nCCDAB", "nPIMWrLatency", "nPIMRdLatency", "nCCDK", "nCCDQK", "nLOCALWR",
    ]
    timing_constraints = _pim_constraints(LPDDR5X)

"""Public API for whole-batch attention with buffers scaled to action horizon.

H changes token counts and storage; attention heads and weight copies stay fixed.
"""

from copy import deepcopy

import horizon_attention_layout as L
import horizon_attention_workload as W

physical_address = L.physical_address
decode_address = L.decode_address


def configure(horizon):
    L.configure(horizon)


def layout_for(horizon):
    previous = L.QUERIES
    try:
        configure(horizon)
        return deepcopy(L.manifest())
    finally:
        configure(previous)


def generate(path, horizon=None, case="chain", schedule="double"):
    if horizon is not None:
        configure(horizon)
    result = W.generate(path, case=case, schedule=schedule)
    result["layout"] = layout_for(L.QUERIES)
    return result

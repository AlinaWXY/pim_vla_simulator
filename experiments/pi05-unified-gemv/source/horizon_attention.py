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


def generate(path, horizon=None, case="chain", schedule="double", rank_staging=False, staging_banks=1, rank_descriptors=False):
    if horizon is not None:
        configure(horizon)
    result = W.generate(path, case=case, schedule=schedule, rank_staging=rank_staging, staging_banks=staging_banks, rank_descriptors=rank_descriptors)
    result["layout"] = layout_for(L.QUERIES)
    if rank_staging:
        from rank_staging import reservation
        result["layout"]["input_staging"] = reservation(staging_banks)
    return result

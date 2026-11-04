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


def layout_for(horizon, qkv_fusion=False):
    previous = L.QUERIES
    try:
        configure(horizon)
        result = deepcopy(L.manifest())
        if qkv_fusion:
            from qkv_token_layout import layout
            result = layout(result)
        return result
    finally:
        configure(previous)


def generate(path, horizon=None, case="chain", schedule="double", rank_staging=False, staging_banks=1, rank_descriptors=False, qkv_fusion=False, whole_srf=False, score_dram_spill=False):
    if horizon is not None:
        configure(horizon)
    result = W.generate(path, case=case, schedule=schedule, rank_staging=rank_staging, staging_banks=staging_banks, rank_descriptors=rank_descriptors, qkv_fusion=qkv_fusion, whole_srf=whole_srf,score_dram_spill=score_dram_spill)
    result["layout"] = layout_for(L.QUERIES, qkv_fusion=qkv_fusion)
    if rank_staging:
        from rank_staging import reservation
        result["layout"]["input_staging"] = reservation(staging_banks)
    return result

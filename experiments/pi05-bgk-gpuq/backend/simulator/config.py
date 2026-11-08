"""Typed configuration shared by the CLI and future trace generators."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

MODELS = {
    "lpddr5-6400": ("LPDDR5", "LPDDR5_6400"),
    "lpddr5x-8533": ("LPDDR5X", "LPDDR5X_8533"),
}


@dataclass(frozen=True)
class MemoryConfig:
    model: str = "lpddr5-6400"
    channels: int = 1
    ranks: int = 1
    density_gbit: int = 8
    refresh: bool = True
    pim: bool = False
    wck_mode: str = "always_on"
    frontend_ratio: int = 8
    memory_ratio: int = 1
    max_cycles: int = 1_000_000
    timing_overrides: dict[str, int] = field(default_factory=dict)
    pim_input_broadcast: bool = False
    pim_mac_scope: str = "rank"
    pim_data_path: str = "lpddr"
    pim_local_transfer_cycles: int = 2
    pim_unified_gemv: bool = False
    pim_rank_staging: bool = False
    pim_staging_banks: int = 1
    pim_rank_controller: bool = False
    pim_rank_descriptors: bool = False
    pim_coarse_commands: bool = False
    pim_stream_inputs: bool = False
    pim_gpu_rope: bool = False
    pim_feature_layout: bool = False
    channel_independent_dispatch: bool = False
    channel_queue_capacity: int = 32
    gemv_port_cycles: int = 2
    pim_token_pv: bool = False
    pim_qkv_fusion: bool = False
    pim_whole_srf: bool = False
    v_cache_preinstalled: bool = False
    pim_score_dram_spill: bool = False
    pim_o_projection: bool = False
    pim_pv_bf16_output: bool = False
    pim_rank_rope: bool = False
    pim_horizon: int = 16
    pim_ffn_dram: bool = False
    ffn_z_preinstalled: bool = False
    rope_ii_cycles: int = 1
    rope_latency_cycles: int = 4
    pim_cast_cycles: int = 1
    rank_reduce_cycles: int = 4

    def __post_init__(self):
        for name in ("pim_token_pv", "pim_qkv_fusion", "pim_whole_srf", "v_cache_preinstalled",
                     "pim_score_dram_spill", "pim_o_projection", "pim_pv_bf16_output"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.pim_whole_srf and (not self.pim_unified_gemv or not self.pim_rank_staging):
            raise ValueError("Whole SRF requires unified GEMV and rank staging")
        if self.pim_token_pv and not self.pim_whole_srf:
            raise ValueError("Token PV requires whole SRF")
        if self.pim_qkv_fusion and (not self.pim_token_pv or not self.pim_rank_rope):
            raise ValueError("QKV fusion requires token PV and rank RoPE")
        if self.v_cache_preinstalled and not self.pim_token_pv:
            raise ValueError("Preinstalled token V cache requires token PV")
        if self.pim_score_dram_spill and (not self.pim_rank_rope or not self.pim_rank_controller):
            raise ValueError("Score DRAM spill requires rank RoPE and rank controller")
        if self.pim_pv_bf16_output and not self.pim_token_pv:
            raise ValueError("BF16 PV output requires token PV")
        if self.pim_o_projection and (not self.pim_pv_bf16_output or not self.pim_rank_staging or self.pim_staging_banks != 2):
            raise ValueError("O projection requires BF16 PV output and two-bank rank staging")
        if type(self.pim_coarse_commands) is not bool or (self.pim_coarse_commands and
                (not self.pim_rank_controller or not self.pim_whole_srf or self.pim_staging_banks != 2 or self.pim_rank_descriptors)):
            raise ValueError("Coarse mode requires two-bank whole-SRF rank controller without legacy rank descriptors")
        if type(self.pim_rank_descriptors) is not bool or type(self.channel_independent_dispatch) is not bool:
            raise ValueError("Rank descriptor/channel dispatch flags must be boolean")
        if self.pim_rank_descriptors and (not self.pim_rank_controller or not self.pim_unified_gemv):
            raise ValueError("Rank descriptors require rank controller and unified GEMV")
        if type(self.channel_queue_capacity) is not int or self.channel_queue_capacity < 1:
            raise ValueError("channel_queue_capacity must be a positive integer")
        if type(self.pim_staging_banks) is not int or self.pim_staging_banks not in (1, 2):
            raise ValueError("pim_staging_banks must be 1 or 2")
        if self.pim_staging_banks != 1 and not self.pim_rank_staging:
            raise ValueError("Multiple staging banks require rank staging")
        if type(self.pim_rank_staging) is not bool or type(self.pim_rank_controller) is not bool:
            raise ValueError("Rank staging/controller flags must be boolean")
        if self.pim_rank_controller and not self.pim_unified_gemv:
            raise ValueError("Rank controller requires unified GEMV")
        if self.pim_rank_staging and (not self.pim_unified_gemv or not self.pim_rank_controller or self.pim_ffn_dram):
            raise ValueError("Rank staging requires unified GEMV and rank controller, without legacy FFN DRAM mode")
        if type(self.pim_unified_gemv) is not bool or type(self.gemv_port_cycles) is not int or self.gemv_port_cycles < 1:
            raise ValueError("Unified GEMV flag must be boolean and port cycles positive")
        if self.pim_unified_gemv and (not self.pim or self.pim_horizon != 16 or self.ranks != 2 or self.channels != 16 or self.pim_data_path != "attacc" or self.pim_input_broadcast or self.pim_mac_scope != "rank"):
            raise ValueError("Unified GEMV requires H16, 16 channels, 2 ranks and rank-scoped AttAcc")
        if type(self.pim_horizon) is not int or not 2 <= self.pim_horizon <= 64 or self.pim_horizon % 2:
            raise ValueError("pim_horizon must be an even integer from2 through64")
        if type(self.pim_ffn_dram) is not bool or type(self.ffn_z_preinstalled) is not bool:
            raise ValueError("FFN DRAM flags must be boolean")
        if self.ffn_z_preinstalled and not self.pim_ffn_dram:
            raise ValueError("ffn_z_preinstalled requires pim_ffn_dram")
        if self.pim_ffn_dram and (not self.pim or self.ranks != 2 or self.channels != 16 or self.pim_data_path != "attacc" or self.pim_input_broadcast or self.pim_mac_scope != "rank" or (self.pim_rank_rope and not self.pim_unified_gemv)):
            raise ValueError("FFN DRAM requires 16 channels, 2 ranks, rank-scoped AttAcc without broadcast or rank RoPE mode")
        if type(self.pim_rank_rope) is not bool:
            raise ValueError("pim_rank_rope must be boolean")
        for name in ("rope_ii_cycles", "rope_latency_cycles", "pim_cast_cycles", "rank_reduce_cycles"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.pim_rank_rope and (not self.pim or self.ranks != 2 or self.channels != 16 or self.pim_data_path != "attacc" or self.pim_input_broadcast or self.pim_mac_scope != "rank"):
            raise ValueError("rank RoPE requires 16 channels, 2 ranks, PIM AttAcc, rank scope, and no cross-rank input broadcast")
        if type(self.pim_local_transfer_cycles) is not int or self.pim_local_transfer_cycles < 1:
            raise ValueError("pim_local_transfer_cycles must be a positive integer")
        if not isinstance(self.model, str) or self.model not in MODELS:
            raise ValueError(f"Unsupported memory model: {self.model}")
        for name in (
            "channels",
            "ranks",
            "density_gbit",
            "frontend_ratio",
            "memory_ratio",
            "max_cycles",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels not in (1, 2, 4, 8, 16, 32, 64) or self.ranks not in (1, 2):
            raise ValueError("channels must be a power of two up to 64; ranks must be 1/2")
        if self.density_gbit not in (8, 16):
            raise ValueError("density_gbit must be 8 or 16 per channel per rank")
        if type(self.refresh) is not bool:
            raise ValueError("refresh must be a boolean")
        if type(self.pim) is not bool:
            raise ValueError("pim must be a boolean")
        if type(self.pim_input_broadcast) is not bool:
            raise ValueError("pim_input_broadcast must be a boolean")
        if self.pim_input_broadcast and not self.pim:
            raise ValueError("pim_input_broadcast requires pim: true")
        if self.pim_mac_scope not in ("rank", "channel"):
            raise ValueError("pim_mac_scope must be rank or channel")
        if self.pim_mac_scope == "channel" and not self.pim:
            raise ValueError("pim_mac_scope=channel requires pim: true")
        if self.pim_data_path not in ("lpddr", "attacc"):
            raise ValueError("pim_data_path must be lpddr or attacc")
        if self.pim_data_path == "attacc" and not self.pim:
            raise ValueError("pim_data_path=attacc requires pim: true")
        if self.wck_mode not in ("always_on", "need_sync"):
            raise ValueError("wck_mode must be always_on or need_sync")
        if self.ranks > 1 and self.wck_mode != "always_on":
            raise ValueError("Multiple ranks require pre-synchronized always_on WCK")
        if not isinstance(self.timing_overrides, dict):
            raise ValueError("timing_overrides must be a mapping")
        for key, value in self.timing_overrides.items():
            if (
                not isinstance(key, str)
                or key in ("rate", "tCK_ps")
                or key == "nPIMLatency"
                or not key.startswith("n")
                or type(value) is not int
            ):
                raise ValueError(f"Invalid timing override: {key}={value!r}")
            if value < 1:
                raise ValueError(f"Timing override must be >= 1: {key}")

    @classmethod
    def from_yaml(cls, path: str | Path):
        with Path(path).open() as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a YAML mapping")
        if not all(isinstance(key, str) for key in data):
            raise ValueError("Configuration field names must be strings")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**data)

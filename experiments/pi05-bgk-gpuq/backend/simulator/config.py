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

    def __post_init__(self):
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

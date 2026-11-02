"""Public experiment API for LPDDR and coarse all-bank PIM timing."""

from .backend import run_trace
from .config import MemoryConfig

__all__ = ["MemoryConfig", "run_trace"]

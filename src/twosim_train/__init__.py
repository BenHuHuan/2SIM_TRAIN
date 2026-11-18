"""2SIM_TRAIN: Zero-Lag Heterogeneous GPU+CPU Dual-Simulator Crossfire Plugin."""
from .autotune import autotune_crossfire_ratio
from .config import CrossfireConfig, resolve_hybrid_split
from .crossfire import HybridCrossfireBackend, attach_crossfire_to_env, enable_crossfire
from .ieee_fp32 import enforce_strict_ieee_fp32
from .warp_slice import slice_model_np, slice_wp_struct

__version__ = "0.1.0"
__all__ = [
    "CrossfireConfig",
    "HybridCrossfireBackend",
    "attach_crossfire_to_env",
    "autotune_crossfire_ratio",
    "enable_crossfire",
    "enforce_strict_ieee_fp32",
    "resolve_hybrid_split",
    "slice_model_np",
    "slice_wp_struct",
]

"""Zero-copy GPU struct slicing utilities for `mujoco_warp.Model` and `mujoco_warp.Data`."""
from __future__ import annotations

import copy
import numpy as np
import torch
import warp as wp


def slice_wp_struct(obj, start: int, end: int, old_nworld: int, wp_device):
    """Create a zero-copy Warp struct view covering world rows `[start:end]`."""
    new_nworld = end - start
    out = copy.copy(obj)
    for k, v in vars(obj).items():
        if k == "nworld":
            setattr(out, k, new_nworld)
        elif isinstance(v, wp.array):
            if v.ndim >= 1 and v.shape[0] == old_nworld:
                if any(dim == 0 for dim in v.shape[1:]):
                    with wp.ScopedDevice(wp_device):
                        setattr(out, k, wp.empty((new_nworld, *v.shape[1:]), dtype=v.dtype))
                else:
                    setattr(out, k, v[start:end])
        elif hasattr(v, "__dict__") and not isinstance(v, type):
            setattr(out, k, slice_wp_struct(v, start, end, old_nworld, wp_device))
    return out


def slice_model_np(t: torch.Tensor, start: int, end: int) -> np.ndarray:
    """Extract per-world model field slice `[start:end]` as contiguous float32 NumPy array."""
    n = end - start
    if t.shape[0] == 1:
        t = t.expand(n, *t.shape[1:])
    else:
        t = t[start:end]
    return t.detach().cpu().to(torch.float32).contiguous().numpy()

"""Lossless IEEE 754 single-precision (`allow_tf32=False`) enforcement for 2SIM_TRAIN."""
from __future__ import annotations

import torch


def enforce_strict_ieee_fp32() -> None:
    """Enforce full 23-bit mantissa IEEE 754 FP32 across PyTorch CUDA & cuDNN backends,
    and hook `mjlab.utils.torch.configure_torch_backends` if `mjlab` / `MLAB4_Fast` is installed."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends.cudnn, "fp32_precision"):
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cudnn.allow_tf32 = False

    try:
        import mjlab.utils.torch as _mj_torch

        if not getattr(_mj_torch, "_twosim_ieee_patched", False):
            _orig = _mj_torch.configure_torch_backends

            def _ieee_configure(allow_tf32: bool = False, deterministic: bool = False):
                _orig(allow_tf32=False, deterministic=deterministic)
                if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                    torch.backends.cuda.matmul.fp32_precision = "ieee"
                else:
                    torch.backends.cuda.matmul.allow_tf32 = False
                if hasattr(torch.backends.cudnn, "fp32_precision"):
                    torch.backends.cudnn.fp32_precision = "ieee"
                else:
                    torch.backends.cudnn.allow_tf32 = False

            _mj_torch.configure_torch_backends = _ieee_configure
            _mj_torch._twosim_ieee_patched = True
    except ImportError:
        pass

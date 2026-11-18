#!/usr/bin/env python3
"""Minimal 2-line Python API integration example for 2SIM_TRAIN on top of MLAB4_Fast + Raisim2_FAST."""
from twosim_train import enable_crossfire

# Call `enable_crossfire(...)` once before creating ANY MJLab ManagerBasedRlEnv!
# - Worlds [0 : 3072] run on GPU via MuJoCo 4.0 Warp CUDA Graph
# - Worlds [3072 : 4096] run SIMULTANEOUSLY on CPU via Raisim2_FAST C++20 OpenMP (FP64)
enable_crossfire(
    raisim_ratio=0.25,       # Dial freely in [0.0 .. 1.0]: 0.0=pure MJ4, 0.25=75/25 Crossfire, 1.0=pure RaiSim2
    rs_threads=64,           # CPU OpenMP threads for Raisim2_FAST
    strict_ieee_fp32=True,   # Enforce lossless 23-bit IEEE 754 FP32 (allow_tf32=False)
)

if __name__ == "__main__":
    from mjlab.scripts.train import main

    main()

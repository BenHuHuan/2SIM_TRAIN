#!/usr/bin/env bash
# Example: Zero-code-change Crossfire launch on any existing MLAB4_Fast / Xmimic training script!
set -euo pipefail

# 1. Activate Raisim2_FAST runtime paths
source "${RAISIM_DIR:-$HOME/raisim2Lib}/raisim_env.sh"

# 2. Optional: probe your GPU + CPU once to see optimal --raisim-ratio for your machine
# 2sim-autotune --num-envs 4096

# 3. Launch MLAB4_Fast built-in task OR your own train_*.py script with 2sim-train!
exec 2sim-train Mjlab-Velocity-Flat-Unitree-G1 \
  --raisim-ratio "${HYBRID_RAISIM_RATIO:-0.25}" \
  --rs-threads "${RAISIM_NUM_THREADS:-64}" \
  --num-envs 4096 \
  "$@"

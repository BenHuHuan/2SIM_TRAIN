"""Configuration & world-partitioning rules for 2SIM_TRAIN Dual-Simulator Crossfire."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrossfireConfig:
    """User-facing configuration for GPU MuJoCo 4.0 Warp + CPU Raisim2_FAST Crossfire."""

    raisim_ratio: float | None = None
    mj_ratio: float | None = None
    mj_envs: int | None = None
    rs_envs: int | None = None
    align: int = 32
    rs_threads: int = 64
    asset_dir: str | Path | None = None
    lib_path: str | Path | None = None
    entity_name: str = "robot"
    action_term_name: str = "joint_pos"
    self_collision_sensor_name: str = "self_collision"
    foot_sensor_name: str | None = None  # Auto-detects "sdk_feet", "feet_ground", etc. when None
    solver_max_iters: int = 25
    solver_tol: float = 1e-6
    profile_every_steps: int = 240
    strict_ieee_fp32: bool = True


def resolve_hybrid_split(
    total_envs: int,
    *,
    raisim_ratio: float | None = None,
    mj_ratio: float | None = None,
    mj_envs: int | None = None,
    rs_envs: int | None = None,
    align: int = 32,
) -> tuple[int, int]:
    """Resolve `(mj_envs, rs_envs)` from ratio or explicit count, aligned to `align` when fractional."""
    if total_envs <= 0:
        raise ValueError(f"total_envs must be positive, got {total_envs}")

    provided = sum(x is not None for x in (raisim_ratio, mj_ratio, mj_envs, rs_envs))
    if provided > 1:
        raise ValueError("Specify at most one of raisim_ratio, mj_ratio, mj_envs, or rs_envs.")

    if mj_envs is not None:
        n_mj = int(mj_envs)
    elif rs_envs is not None:
        n_mj = int(total_envs) - int(rs_envs)
    else:
        if raisim_ratio is not None:
            r_rs = float(raisim_ratio)
        elif mj_ratio is not None:
            r_rs = 1.0 - float(mj_ratio)
        else:
            r_rs = 0.5  # Default: 50% MuJoCo 4.0 Warp (GPU) + 50% Raisim2_FAST (CPU)
        if not (0.0 <= r_rs <= 1.0):
            raise ValueError(f"Mixing ratio must be in [0.0, 1.0], got raisim_ratio={r_rs}")
        if r_rs <= 0.0:
            return total_envs, 0
        if r_rs >= 1.0:
            return 0, total_envs
        raw_rs = int(round(total_envs * r_rs))
        if align > 1 and total_envs >= 2 * align:
            aligned_rs = int(round(raw_rs / align)) * align
            aligned_rs = max(align, min(total_envs - align, aligned_rs))
            n_rs = aligned_rs
        else:
            n_rs = max(1, min(total_envs - 1, raw_rs))
        n_mj = total_envs - n_rs

    n_mj = max(0, min(total_envs, n_mj))
    return n_mj, total_envs - n_mj

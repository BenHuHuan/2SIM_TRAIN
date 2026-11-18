"""Hardware Autotuner: probes GPU MuJoCo 4.0 Warp vs CPU Raisim2_FAST to find optimal `--raisim-ratio`."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

import numpy as np


_DEFAULT_HUMANOID_BENCH_MJCF = """
<mujoco model="humanoid29_bench">
  <option timestep="0.002" iterations="4" ls_iterations="8"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="robot/pelvis" pos="0 0 0.88">
      <freejoint name="robot/root"/>
      <inertial pos="0 0 0.02" mass="14.0" diaginertia="0.14 0.12 0.10"/>
      <geom type="capsule" size="0.10 0.04"/>
      <body name="robot/torso" pos="0 0 0.12">
        <joint name="robot/waist_yaw" type="hinge" axis="0 0 1" range="-1.0 1.0"/>
        <inertial pos="0 0 0.18" mass="18.0" diaginertia="0.30 0.26 0.16"/>
        <geom type="capsule" size="0.11 0.16" pos="0 0 0.18"/>
        <body name="robot/l_arm" pos="0 0.20 0.28">
          <joint name="robot/sh_l" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <inertial pos="0 0 -0.14" mass="2.5" diaginertia="0.02 0.02 0.004"/>
          <geom type="capsule" size="0.04 0.12" pos="0 0 -0.14"/>
        </body>
        <body name="robot/r_arm" pos="0 -0.20 0.28">
          <joint name="robot/sh_r" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <inertial pos="0 0 -0.14" mass="2.5" diaginertia="0.02 0.02 0.004"/>
          <geom type="capsule" size="0.04 0.12" pos="0 0 -0.14"/>
        </body>
      </body>
      <body name="robot/l_thigh" pos="0 0.11 -0.08">
        <joint name="robot/hip_l" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
        <inertial pos="0 0 -0.18" mass="6.5" diaginertia="0.06 0.06 0.01"/>
        <geom type="capsule" size="0.055 0.15" pos="0 0 -0.18"/>
        <body name="robot/l_shank" pos="0 0 -0.38">
          <joint name="robot/knee_l" type="hinge" axis="0 1 0" range="-0.1 2.4"/>
          <inertial pos="0 0 -0.17" mass="4.0" diaginertia="0.04 0.04 0.006"/>
          <geom type="capsule" size="0.045 0.15" pos="0 0 -0.17"/>
          <body name="robot/ankle_roll_l_link" pos="0 0 -0.36">
            <joint name="robot/ankle_l" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
            <inertial pos="0.03 0 -0.02" mass="1.1" diaginertia="0.003 0.006 0.006"/>
            <geom type="box" size="0.12 0.05 0.025" pos="0.03 0 -0.025"/>
          </body>
        </body>
      </body>
      <body name="robot/r_thigh" pos="0 -0.11 -0.08">
        <joint name="robot/hip_r" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
        <inertial pos="0 0 -0.18" mass="6.5" diaginertia="0.06 0.06 0.01"/>
        <geom type="capsule" size="0.055 0.15" pos="0 0 -0.18"/>
        <body name="robot/r_shank" pos="0 0 -0.38">
          <joint name="robot/knee_r" type="hinge" axis="0 1 0" range="-0.1 2.4"/>
          <inertial pos="0 0 -0.17" mass="4.0" diaginertia="0.04 0.04 0.006"/>
          <geom type="capsule" size="0.045 0.15" pos="0 0 -0.17"/>
          <body name="robot/ankle_roll_r_link" pos="0 0 -0.36">
            <joint name="robot/ankle_r" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
            <inertial pos="0.03 0 -0.02" mass="1.1" diaginertia="0.003 0.006 0.006"/>
            <geom type="box" size="0.12 0.05 0.025" pos="0.03 0 -0.025"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def autotune_crossfire_ratio(
    *,
    mjcf_path: str | Path | None = None,
    total_envs: int = 4096,
    decimation: int = 10,
    rs_threads: int | None = None,
    device: str = "cuda:0",
    steps: int = 20,
) -> dict:
    """Measure GPU MJWarp and CPU Raisim2_FAST substep latency across ratios and recommend `--raisim-ratio`."""
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    from raisim_fast import RaisimFastVecEngine, compile_mjcf_to_analytical_urdf

    from .config import resolve_hybrid_split

    if rs_threads is None:
        rs_threads = os.cpu_count() or 16

    if mjcf_path is not None:
        m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        m = mujoco.MjModel.from_xml_string(_DEFAULT_HUMANOID_BENCH_MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    candidates = [0.0, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.50]
    rows = []

    with tempfile.TemporaryDirectory(prefix="2sim_autotune_") as tmp:
        meta = compile_mjcf_to_analytical_urdf(m, tmp)
        nj = len(meta["joint_names"])

        for r in candidates:
            n_mj, n_rs = resolve_hybrid_split(total_envs, raisim_ratio=r)

            mj_ms = 0.0
            if n_mj > 0:
                with wp.ScopedDevice(device):
                    wpm = mjwarp.put_model(m)
                    wpd = mjwarp.put_data(m, d, nworld=n_mj, nconmax=64, njmax=256)
                    with wp.ScopedCapture() as cap:
                        for _ in range(decimation):
                            mjwarp.step(wpm, wpd)
                    for _ in range(3):
                        wp.capture_launch(cap.graph)
                    wp.synchronize_device(device)
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        wp.capture_launch(cap.graph)
                    wp.synchronize_device(device)
                    mj_ms = (time.perf_counter() - t0) * 1000.0 / steps

            rs_ms = 0.0
            if n_rs > 0:
                eng = RaisimFastVecEngine.from_asset_dir(tmp, num_envs=n_rs, dt=float(m.opt.timestep), num_threads=rs_threads)
                try:
                    qpos = np.zeros((n_rs, eng.nq), dtype=np.float32)
                    if eng.floating_base:
                        qpos[:, 2] = 0.88
                        qpos[:, 3] = 1.0
                    qvel = np.zeros((n_rs, eng.nv), dtype=np.float32)
                    eng.set_all_states(qpos, qvel)
                    tgt = np.zeros((n_rs, nj), dtype=np.float32)
                    kp = np.full((n_rs, nj), 120.0, dtype=np.float32)
                    kd = np.full((n_rs, nj), 5.0, dtype=np.float32)
                    lim = np.full((n_rs, nj), 180.0, dtype=np.float32)
                    qp_o = np.empty((n_rs, eng.nq), dtype=np.float32)
                    qv_o = np.empty((n_rs, eng.nv), dtype=np.float32)
                    qf_o = np.empty((n_rs, nj), dtype=np.float32)
                    import ctypes
                    f32p = ctypes.POINTER(ctypes.c_float)

                    def _step_c():
                        eng.lib.rfast_step_decimated(
                            ctypes.c_void_p(eng.handle),
                            int(decimation),
                            tgt.ctypes.data_as(f32p),
                            kp.ctypes.data_as(f32p),
                            kd.ctypes.data_as(f32p),
                            lim.ctypes.data_as(f32p),
                            None,
                            qp_o.ctypes.data_as(f32p),
                            qv_o.ctypes.data_as(f32p),
                            qf_o.ctypes.data_as(f32p),
                            None,
                            None,
                            0,
                            None,
                            None,
                        )

                    _step_c()
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        _step_c()
                    rs_ms = (time.perf_counter() - t0) * 1000.0 / steps
                finally:
                    eng.close()

            # Account for ~3.5 ms Python post-step kernel pre-dispatch headroom
            eff_ms = mj_ms if n_rs == 0 else max(mj_ms, rs_ms + 3.5)
            rows.append(
                {
                    "raisim_ratio": r,
                    "mj_envs": n_mj,
                    "rs_envs": n_rs,
                    "mj_gpu_ms": round(mj_ms, 2),
                    "rs_cpu_ms": round(rs_ms, 2),
                    "effective_step_ms": round(eff_ms, 2),
                }
            )

    base_ms = rows[0]["effective_step_ms"]
    faster_rows = [x for x in rows[1:] if x["effective_step_ms"] <= base_ms * 1.01]
    best_speed = min(rows[1:], key=lambda x: x["effective_step_ms"])
    best_max_raisim_fast = max(faster_rows, key=lambda x: x["raisim_ratio"]) if faster_rows else best_speed
    best_parity = min(rows[1:], key=lambda x: abs(x["mj_gpu_ms"] - x["rs_cpu_ms"]))

    return {
        "total_envs": total_envs,
        "rs_threads": rs_threads,
        "baseline_mj_only_ms": base_ms,
        "recommended_max_speed_ratio": best_speed["raisim_ratio"],
        "recommended_fast_high_mix_ratio": best_max_raisim_fast["raisim_ratio"],
        "recommended_gpu_cpu_parity_ratio": best_parity["raisim_ratio"],
        "sweep": rows,
    }

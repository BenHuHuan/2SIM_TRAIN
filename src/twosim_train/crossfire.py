"""Zero-Copy Unified Split-World Crossfire: GPU MuJoCo 4.0 Warp + CPU Raisim2_FAST OpenMP.

Splits a single `ManagerBasedRlEnv(num_envs=N)` in-place across hardware with:
  - ZERO policy lag (strictly synchronous On-Policy RL)
  - ZERO duplicate manager overhead (single N-env observation/reward/command/reset batch)
  - Simultaneous hardware execution: `[0 : mj_envs]` step via captured MuJoCo 4.0 Warp CUDA Graph
    on GPU while `[mj_envs : N]` step concurrently via `Raisim2_FAST` C++20 OpenMP on CPU!
"""
from __future__ import annotations

import ctypes
import gc
import time

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.sensor.contact_sensor import ContactSensor

from .asset_resolver import resolve_raisim_fast_assets
from .config import CrossfireConfig, resolve_hybrid_split
from .ieee_fp32 import enforce_strict_ieee_fp32
from .raisim_bridge import _detect_foot_sensor, attach_full_raisim
from .warp_slice import slice_model_np, slice_wp_struct


class HybridCrossfireBackend:
    """Simultaneous GPU MJWarp (`[0:mj_envs]`) + CPU Raisim2_FAST (`[mj_envs:N]`) in one unified Env."""

    def __init__(self, env, mj_envs: int, cfg: CrossfireConfig) -> None:
        self.env = env
        self.sim = env.sim
        self.device = env.device
        self.cfg = cfg
        self.total_envs = int(env.num_envs)
        self.mj_envs = int(mj_envs)
        self.rs_envs = self.total_envs - self.mj_envs
        if not (0 < self.mj_envs < self.total_envs):
            raise ValueError(f"Crossfire requires 0 < mj_envs < {self.total_envs}, got {self.mj_envs}")

        self.decimation = int(env.cfg.decimation)
        self.dt = float(env.physics_dt)
        self.num_threads = int(cfg.rs_threads)
        self.profile_every_steps = int(cfg.profile_every_steps)
        self._step_calls = 0
        self.last_mj_gpu_ms: float = 0.0
        self.last_rs_cpu_ms: float = 0.0

        self.lib, self.lib_path, self.meta, urdf_path = resolve_raisim_fast_assets(self.sim.mj_model, cfg)
        meta = self.meta

        self.floating_base = bool(meta.get("floating_base", True))
        self.nj = len(meta["joint_names"])
        self.q_off = 7 if self.floating_base else 0
        self.v_off = 6 if self.floating_base else 0
        self.nq = self.q_off + self.nj
        self.nv = self.v_off + self.nj

        self.robot = env.scene[cfg.entity_name]
        self.action_term = env.action_manager.get_term(cfg.action_term_name)

        s0, s1 = self.mj_envs, self.total_envs
        mj_m = self.sim.mj_model
        armature = slice_model_np(self.sim.model.dof_armature[:, self.v_off : self.v_off + self.nj], s0, s1)
        damping = slice_model_np(self.sim.model.dof_damping[:, self.v_off : self.v_off + self.nj], s0, s1)
        friction = slice_model_np(self.sim.model.dof_frictionloss[:, self.v_off : self.v_off + self.nj], s0, s1)

        prefix = meta.get("entity_prefix", f"{cfg.entity_name}/")

        def _mj_body_id(name: str) -> int:
            for cand in (f"{prefix}{name}", f"{cfg.entity_name}/{name}", name):
                try:
                    return mj_m.body(cand).id
                except KeyError:
                    continue
            raise KeyError(f"Cannot resolve body {name!r} in MjModel")

        foot_names = list(meta.get("foot_body_names", ()))
        foot_body_ids = [_mj_body_id(n) for n in foot_names]
        foot_geom_ids = [
            gid
            for gid in range(mj_m.ngeom)
            if int(mj_m.geom_bodyid[gid]) in foot_body_ids
            and (mj_m.geom_contype[gid] or mj_m.geom_conaffinity[gid])
        ]
        if foot_geom_ids:
            foot_mu = slice_model_np(self.sim.model.geom_friction[:, foot_geom_ids[0], 0], s0, s1)
        else:
            foot_mu = np.ones(self.rs_envs, dtype=np.float32)

        anchor_name = meta.get("anchor_body_name", "")
        if anchor_name:
            anchor_bid = _mj_body_id(anchor_name)
            base_mj_mass = float(mj_m.body_mass[anchor_bid])
            base_mj_ipos = np.asarray(mj_m.body_ipos[anchor_bid], dtype=np.float32)
            cur_mass = slice_model_np(self.sim.model.body_mass[:, anchor_bid], s0, s1)
            cur_ipos = slice_model_np(self.sim.model.body_ipos[:, anchor_bid, :], s0, s1)
            anchor_delta_mass = np.ascontiguousarray(cur_mass - base_mj_mass, dtype=np.float32)
            anchor_delta_com = np.ascontiguousarray(cur_ipos - base_mj_ipos[None, :], dtype=np.float32)
        else:
            anchor_delta_mass = None
            anchor_delta_com = None

        grav = np.asarray(self.sim.cfg.mujoco.gravity, dtype=np.float32)
        j_csv = ",".join(meta["joint_names"]).encode()
        ex_csv = ";".join(f"{a}:{b}" for a, b in meta.get("collision_excludes", ())).encode()
        feet_csv = ",".join(foot_names).encode()

        t0 = time.perf_counter()
        self.handle = self.lib.rfast_create(
            int(self.rs_envs),
            int(self.num_threads),
            int(1 if self.floating_base else 0),
            str(urdf_path).encode(),
            str(urdf_path.parent).encode(),
            ctypes.c_double(self.dt),
            grav.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            j_csv,
            ex_csv,
            feet_csv,
            anchor_name.encode(),
            armature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            damping.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            friction.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            foot_mu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            anchor_delta_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if anchor_delta_mass is not None else None,
            anchor_delta_com.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if anchor_delta_com is not None else None,
            int(cfg.solver_max_iters),
            ctypes.c_double(cfg.solver_tol),
        )
        if not self.handle:
            raise RuntimeError("Failed to initialize Raisim2_FAST worlds for 2SIM_TRAIN Crossfire")
        self.init_seconds = time.perf_counter() - t0
        self.num_feet = int(self.lib.rfast_get_num_feet(ctypes.c_void_p(self.handle)))
        mj_pct = 100.0 * self.mj_envs / self.total_envs
        rs_pct = 100.0 * self.rs_envs / self.total_envs
        print(
            f"[2SIM_TRAIN Crossfire] Unified {self.total_envs} envs = "
            f"{self.mj_envs} ({mj_pct:.1f}%) MuJoCo 4.0 Warp [GPU] + "
            f"{self.rs_envs} ({rs_pct:.1f}%) Raisim2_FAST [{self.num_threads} CPU threads] "
            f"initialized in {self.init_seconds:.2f}s",
            flush=True,
        )

        self.self_sensor: ContactSensor | None = env.scene.sensors.get(cfg.self_collision_sensor_name)
        self.feet_sensor: ContactSensor | None = _detect_foot_sensor(env, cfg)
        self.hist_len = int(self.self_sensor.cfg.history_length) if self.self_sensor is not None else 4
        self.root_body_id = int(self.robot.indexing.root_body_id)

        self.sim.data.nefc[self.mj_envs :].zero_()

        self.mj_stream = torch.cuda.Stream(device=self.device)
        self.mj_wp_stream = wp.stream_from_torch(self.mj_stream)
        self.rs_d2h_stream = torch.cuda.Stream(device=self.device)
        self.rs_h2d_stream = torch.cuda.Stream(device=self.device)
        self.rs_h2d_wp_stream = wp.stream_from_torch(self.rs_h2d_stream)

        self.rs_tgt_ready_event = torch.cuda.Event(enable_timing=False)
        self.rs_reset_ready_event = torch.cuda.Event(enable_timing=False)
        self._prof_mj_start = torch.cuda.Event(enable_timing=True)
        self._prof_mj_end = torch.cuda.Event(enable_timing=True)
        self._prof_pending = False
        self._prof_rs_ms = 0.0

        self._qpos_all_cpu = torch.empty((self.rs_envs, self.nq), dtype=torch.float32, pin_memory=True)
        self._qvel_all_cpu = torch.empty((self.rs_envs, self.nv), dtype=torch.float32, pin_memory=True)
        self._tgt_cpu = torch.empty((self.rs_envs, self.nj), dtype=torch.float32, pin_memory=True)
        self._kp_cpu = torch.empty((self.rs_envs, self.nj), dtype=torch.float32, pin_memory=True)
        self._kd_cpu = torch.empty((self.rs_envs, self.nj), dtype=torch.float32, pin_memory=True)
        self._lim_cpu = torch.empty((self.rs_envs, self.nj), dtype=torch.float32, pin_memory=True)
        self._wrench_cpu = torch.empty((self.rs_envs, 6), dtype=torch.float32, pin_memory=True)

        self._reset_ids_pinned = torch.empty((self.total_envs,), dtype=torch.int64, pin_memory=True)
        self._reset_qpos_pinned = torch.empty((self.total_envs, self.nq), dtype=torch.float32, pin_memory=True)
        self._reset_qvel_pinned = torch.empty((self.total_envs, self.nv), dtype=torch.float32, pin_memory=True)
        self._pending_reset_count: int = 0

        self._qpos_out_cpu = torch.empty((self.rs_envs, self.nq), dtype=torch.float32, pin_memory=True)
        self._qvel_out_cpu = torch.empty((self.rs_envs, self.nv), dtype=torch.float32, pin_memory=True)
        self._qfrc_out_cpu = torch.empty((self.rs_envs, self.nj), dtype=torch.float32, pin_memory=True)
        self._self_found_cpu = torch.zeros((self.rs_envs, max(self.hist_len, 1)), dtype=torch.float32, pin_memory=True)
        self._self_force_cpu = torch.zeros(
            (self.rs_envs, max(self.hist_len, 1), 3), dtype=torch.float32, pin_memory=True
        )
        self._foot_found_cpu = torch.zeros((self.rs_envs, max(self.num_feet, 1)), dtype=torch.float32, pin_memory=True)
        self._foot_force_cpu = torch.zeros(
            (self.rs_envs, max(self.num_feet, 1), 3), dtype=torch.float32, pin_memory=True
        )

        fused = self.robot._fused_actuator_group
        assert len(fused._groups) == 1, f"Expected 1 fused actuator group, got {len(fused._groups)}"
        self._fused_grp = fused._groups[0]
        self._kp_rs_gpu = self._fused_grp.params["stiffness"][s0:s1]
        self._kd_rs_gpu = self._fused_grp.params["damping"][s0:s1]
        self._lim_rs_gpu = self._fused_grp.params["force_limit"][s0:s1]
        self._kp_cpu.copy_(self._kp_rs_gpu, non_blocking=False)
        self._kd_cpu.copy_(self._kd_rs_gpu, non_blocking=False)
        self._lim_cpu.copy_(self._lim_rs_gpu, non_blocking=False)

        self._all_states_dirty: bool = True
        self._inside_reset_idx: bool = False
        self._has_active_wrench: bool = False

        self._signature = None
        self._wpm_mj = None
        self._wpd_mj = None
        self._wpm_rs = None
        self._wpd_rs = None
        self._mj_warm_steps = 0
        self._mj_graph: torch.cuda.CUDAGraph | None = None
        self._rs_fk_graph: torch.cuda.CUDAGraph | None = None

        self._install_entity_write_hooks()

    def _install_entity_write_hooks(self) -> None:
        edata = self.robot.data

        def _on_external_write(env_ids) -> None:
            if self._inside_reset_idx or self._all_states_dirty:
                return
            self._all_states_dirty = True

        for mname in ("write_root_pose", "write_root_velocity", "write_root_velocity_b"):
            if not hasattr(edata, mname):
                continue
            orig = getattr(edata, mname)

            def _wrap_root(val, env_ids=None, _orig=orig):
                _orig(val, env_ids)
                _on_external_write(env_ids)

            setattr(edata, mname, _wrap_root)

        for mname in ("write_joint_position", "write_joint_velocity"):
            if not hasattr(edata, mname):
                continue
            orig = getattr(edata, mname)

            def _wrap_jnt(val, joint_ids=None, env_ids=None, _orig=orig):
                _orig(val, joint_ids, env_ids)
                _on_external_write(env_ids)

            setattr(edata, mname, _wrap_jnt)

        if hasattr(edata, "write_external_wrench"):
            orig_wrench = edata.write_external_wrench

            def _wrap_wrench(force, torque, body_ids=None, env_ids=None):
                orig_wrench(force, torque, body_ids, env_ids)
                self._has_active_wrench = True

            edata.write_external_wrench = _wrap_wrench

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.rfast_destroy(ctypes.c_void_p(self.handle))
            self.handle = None

    def stage_async_reset_d2h(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            self._all_states_dirty = True
            self._pending_reset_count = 0
            return
        k = int(env_ids.numel())
        if k == 0 or self._all_states_dirty:
            return
        if k >= self.total_envs:
            self._all_states_dirty = True
            self._pending_reset_count = 0
            return

        default_stream = torch.cuda.current_stream(self.device)
        self.rs_d2h_stream.wait_stream(default_stream)
        with torch.cuda.stream(self.rs_d2h_stream):
            idx64 = env_ids if env_ids.dtype == torch.int64 else env_ids.to(dtype=torch.int64)
            qpos_sel = self.sim.data.qpos[:, : self.nq].index_select(0, idx64)
            qvel_sel = self.sim.data.qvel[:, : self.nv].index_select(0, idx64)
            base = self._pending_reset_count
            if base + k > self.total_envs:
                self._all_states_dirty = True
                self._pending_reset_count = 0
                return
            self._reset_ids_pinned[base : base + k].copy_(idx64, non_blocking=True)
            self._reset_qpos_pinned[base : base + k].copy_(qpos_sel, non_blocking=True)
            self._reset_qvel_pinned[base : base + k].copy_(qvel_sel, non_blocking=True)
            self._kp_cpu.copy_(self._kp_rs_gpu, non_blocking=True)
            self._kd_cpu.copy_(self._kd_rs_gpu, non_blocking=True)
            self._pending_reset_count = base + k
            self.rs_reset_ready_event.record(self.rs_d2h_stream)

    def _apply_staged_resets_cpu(self) -> None:
        s0, s1 = self.mj_envs, self.total_envs
        if self._all_states_dirty:
            self._qpos_all_cpu.copy_(self.sim.data.qpos[s0:s1, : self.nq], non_blocking=False)
            self._qvel_all_cpu.copy_(self.sim.data.qvel[s0:s1, : self.nv], non_blocking=False)
            self._kp_cpu.copy_(self._kp_rs_gpu, non_blocking=False)
            self._kd_cpu.copy_(self._kd_rs_gpu, non_blocking=False)
            self.lib.rfast_set_all_states(
                ctypes.c_void_p(self.handle),
                ctypes.c_void_p(self._qpos_all_cpu.data_ptr()),
                ctypes.c_void_p(self._qvel_all_cpu.data_ptr()),
            )
            self._all_states_dirty = False
            self._pending_reset_count = 0
            return

        k = self._pending_reset_count
        if k == 0:
            return
        self.rs_reset_ready_event.synchronize()
        ids_np = self._reset_ids_pinned[:k].numpy()
        rs_mask = ids_np >= s0
        if np.any(rs_mask):
            local_ids = np.ascontiguousarray(ids_np[rs_mask] - s0, dtype=np.int32)
            qp_pack = np.ascontiguousarray(self._reset_qpos_pinned[:k].numpy()[rs_mask], dtype=np.float32)
            qv_pack = np.ascontiguousarray(self._reset_qvel_pinned[:k].numpy()[rs_mask], dtype=np.float32)
            self.lib.rfast_set_states_packed(
                ctypes.c_void_p(self.handle),
                local_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(local_ids.shape[0]),
                qp_pack.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                qv_pack.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        self._pending_reset_count = 0

    def _ensure_sliced_views(self) -> None:
        model_ptrs = tuple(
            v.ptr
            for v in vars(self.sim.wp_model).values()
            if isinstance(v, wp.array) and v.ndim >= 1 and v.shape[0] == self.total_envs
        )
        post_check = getattr(self.sim, "_post_decimation_check", None)
        sig = (
            id(self.sim.wp_model),
            id(self.sim.wp_data),
            model_ptrs,
            getattr(post_check, "__func__", post_check),
            self.decimation,
        )
        if self._signature != sig:
            self._wpm_mj = slice_wp_struct(
                self.sim.wp_model, 0, self.mj_envs, self.total_envs, self.sim.wp_device
            )
            self._wpd_mj = slice_wp_struct(
                self.sim.wp_data, 0, self.mj_envs, self.total_envs, self.sim.wp_device
            )
            self._wpm_rs = slice_wp_struct(
                self.sim.wp_model, self.mj_envs, self.total_envs, self.total_envs, self.sim.wp_device
            )
            self._wpd_rs = slice_wp_struct(
                self.sim.wp_data, self.mj_envs, self.total_envs, self.total_envs, self.sim.wp_device
            )
            self._mj_graph = None
            self._rs_fk_graph = None
            self._mj_warm_steps = 0
            self._signature = sig

    def _mj_substep_body(self) -> None:
        n_mj = self.mj_envs
        edata = self.robot.data
        grp = self._fused_grp
        ids = grp.target_ids
        ctrl_ids = grp.ctrl_ids
        mj_params = {name: val[:n_mj] for name, val in grp.params.items()}
        ctrl_mj = self.sim.data.ctrl[:n_mj]

        pos_target = edata.joint_pos_target[:n_mj, ids]
        vel_target = edata.joint_vel_target[:n_mj, ids]
        eff_target = edata.joint_effort_target[:n_mj, ids]

        self_hist_buf = None
        self_force_view = None
        if self.self_sensor is not None and self.self_sensor._history_state is not None:
            self_hist_buf = self.self_sensor._history_state.get("force")
            if self.self_sensor._field_strided_views is not None:
                self_force_view = self.self_sensor._field_strided_views.get("force")

        with wp.ScopedDevice(self.sim.wp_device):
            for _ in range(self.decimation):
                pos = edata.joint_pos[:n_mj, ids]
                vel = edata.joint_vel[:n_mj, ids]
                cmd = ActuatorCmd(
                    position_target=pos_target,
                    velocity_target=vel_target,
                    effort_target=eff_target,
                    pos=pos,
                    vel=vel,
                )
                torques = grp.actuator_type.control_law(mj_params, cmd)
                ctrl_mj[:, ctrl_ids] = torques
                mjwarp.step(self._wpm_mj, self._wpd_mj)
                if self_hist_buf is not None and self_force_view is not None:
                    if self_hist_buf.size(2) > 1:
                        self_hist_buf[:n_mj, :, 1:, :] = self_hist_buf[:n_mj, :, :-1, :].clone()
                    self_hist_buf[:n_mj, :, 0, :] = self_force_view[:n_mj]

            post_check = getattr(self.sim, "_post_decimation_check", None)
            if post_check is not None:
                post_check()

    def _rs_pure_fk_body(self) -> None:
        with wp.ScopedDevice(self.sim.wp_device):
            mjwarp.fwd_kinematics(self._wpm_rs, self._wpd_rs)
            mjwarp.com_vel(self._wpm_rs, self._wpd_rs)

    def _write_rs_contacts_to_sensors(self) -> None:
        s0, s1 = self.mj_envs, self.total_envs
        if self.self_sensor is not None:
            for slot in self.self_sensor._slots:
                if slot.field_name == "found":
                    slot.data_view[s0:s1].copy_(self._self_found_cpu[:, 0:1], non_blocking=True)
                elif slot.field_name == "force":
                    slot.data_view[s0:s1].copy_(self._self_force_cpu[:, 0, :], non_blocking=True)
            if self.self_sensor._history_state is not None and "force" in self.self_sensor._history_state:
                hist_buf = self.self_sensor._history_state["force"]
                h = min(hist_buf.size(2), self.hist_len)
                hist_buf[s0:s1, 0, :h, :].copy_(self._self_force_cpu[:, :h, :], non_blocking=True)

        if self.feet_sensor is not None and self.num_feet > 0:
            f_idx, frc_idx = 0, 0
            for slot in self.feet_sensor._slots:
                if slot.field_name == "found" and f_idx < self.num_feet:
                    slot.data_view[s0:s1].copy_(self._foot_found_cpu[:, f_idx : f_idx + 1], non_blocking=True)
                    f_idx += 1
                elif slot.field_name == "force" and frc_idx < self.num_feet:
                    slot.data_view[s0:s1].copy_(self._foot_force_cpu[:, frc_idx, :], non_blocking=True)
                    frc_idx += 1

    def _collect_previous_profile_if_ready(self) -> None:
        if not self._prof_pending:
            return
        if not self._prof_mj_end.query():
            return
        self.last_mj_gpu_ms = float(self._prof_mj_start.elapsed_time(self._prof_mj_end))
        self.last_rs_cpu_ms = float(self._prof_rs_ms)
        self._prof_pending = False
        mj_pct = 100.0 * self.mj_envs / self.total_envs
        rs_pct = 100.0 * self.rs_envs / self.total_envs
        bottleneck = "MJ-GPU" if self.last_mj_gpu_ms > self.last_rs_cpu_ms else "RS-CPU"
        print(
            f"[2SIM Crossfire Balance @ step {self._step_calls}] "
            f"MJ({self.mj_envs}, {mj_pct:.1f}%): {self.last_mj_gpu_ms:.2f} ms/step | "
            f"RS({self.rs_envs}, {rs_pct:.1f}%): {self.last_rs_cpu_ms:.2f} ms/step | "
            f"Pacing side: {bottleneck}",
            flush=True,
        )

    def run_crossfire_decimation_step(self) -> None:
        self._collect_previous_profile_if_ready()
        self._step_calls += 1
        do_prof = (
            self.profile_every_steps > 0
            and self._mj_warm_steps >= 2
            and self._mj_graph is not None
            and (self._step_calls % self.profile_every_steps == 0)
        )

        self._ensure_sliced_views()
        s0, s1 = self.mj_envs, self.total_envs
        default_stream = torch.cuda.current_stream(self.device)

        self.env.action_manager.apply_action()

        self.rs_d2h_stream.wait_stream(default_stream)
        with torch.cuda.stream(self.rs_d2h_stream):
            self._tgt_cpu.copy_(self.robot.data.joint_pos_target[s0:s1, : self.nj], non_blocking=True)
            if self._has_active_wrench:
                self._wrench_cpu.copy_(self.sim.data.xfrc_applied[s0:s1, self.root_body_id, :], non_blocking=True)
            self.rs_tgt_ready_event.record(self.rs_d2h_stream)

        if self._mj_warm_steps < 2:
            with wp.ScopedStream(wp.stream_from_torch(default_stream)):
                self._mj_substep_body()
            self._mj_warm_steps += 1
        else:
            self.mj_stream.wait_stream(default_stream)
            ctrl = self.sim.data.ctrl
            orig_ctrl_stream = ctrl._torch_stream
            was_gc = gc.isenabled()
            try:
                with torch.cuda.stream(self.mj_stream), wp.ScopedStream(self.mj_wp_stream):
                    if self._mj_graph is None:
                        ctrl._torch_stream = self.mj_stream
                        gc.disable()
                        self._mj_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self._mj_graph, stream=self.mj_stream):
                            with wp.ScopedCapture(stream=self.mj_wp_stream, external=True):
                                self._mj_substep_body()
                    if do_prof:
                        self._prof_mj_start.record(self.mj_stream)
                    self._mj_graph.replay()
                    if do_prof:
                        self._prof_mj_end.record(self.mj_stream)
            finally:
                ctrl._torch_stream = orig_ctrl_stream
                if was_gc:
                    gc.enable()

        self._apply_staged_resets_cpu()
        self.rs_tgt_ready_event.synchronize()
        t_rs0 = time.perf_counter() if do_prof else 0.0
        wrench_ptr = ctypes.c_void_p(self._wrench_cpu.data_ptr()) if self._has_active_wrench else None

        self.lib.rfast_step_decimated(
            ctypes.c_void_p(self.handle),
            int(self.decimation),
            ctypes.c_void_p(self._tgt_cpu.data_ptr()),
            ctypes.c_void_p(self._kp_cpu.data_ptr()),
            ctypes.c_void_p(self._kd_cpu.data_ptr()),
            ctypes.c_void_p(self._lim_cpu.data_ptr()),
            wrench_ptr,
            ctypes.c_void_p(self._qpos_out_cpu.data_ptr()),
            ctypes.c_void_p(self._qvel_out_cpu.data_ptr()),
            ctypes.c_void_p(self._qfrc_out_cpu.data_ptr()),
            ctypes.c_void_p(self._self_found_cpu.data_ptr()),
            ctypes.c_void_p(self._self_force_cpu.data_ptr()),
            int(max(self.hist_len, 1)),
            ctypes.c_void_p(self._foot_found_cpu.data_ptr()) if self.num_feet > 0 else None,
            ctypes.c_void_p(self._foot_force_cpu.data_ptr()) if self.num_feet > 0 else None,
        )
        if do_prof:
            self._prof_rs_ms = (time.perf_counter() - t_rs0) * 1000.0
            self._prof_pending = True

        self.rs_h2d_stream.wait_stream(default_stream)
        was_gc = gc.isenabled()
        try:
            with torch.cuda.stream(self.rs_h2d_stream), wp.ScopedStream(self.rs_h2d_wp_stream):
                self.sim.data.qpos[s0:s1, : self.nq].copy_(self._qpos_out_cpu, non_blocking=True)
                self.sim.data.qvel[s0:s1, : self.nv].copy_(self._qvel_out_cpu, non_blocking=True)
                self.sim.data.qfrc_actuator[s0:s1, self.v_off : self.v_off + self.nj].copy_(
                    self._qfrc_out_cpu, non_blocking=True
                )
                self.sim.data.actuator_force[s0:s1, : self.nj].copy_(self._qfrc_out_cpu, non_blocking=True)
                self._write_rs_contacts_to_sensors()

                if self._mj_warm_steps < 2 or self._rs_fk_graph is None:
                    if self._mj_warm_steps >= 2:
                        gc.disable()
                        self._rs_fk_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self._rs_fk_graph, stream=self.rs_h2d_stream):
                            with wp.ScopedCapture(stream=self.rs_h2d_wp_stream, external=True):
                                self._rs_pure_fk_body()
                        self._rs_fk_graph.replay()
                    else:
                        self._rs_pure_fk_body()
                else:
                    self._rs_fk_graph.replay()
        finally:
            if was_gc:
                gc.enable()

        default_stream.wait_stream(self.mj_stream)
        default_stream.wait_stream(self.rs_h2d_stream)

        for sensor in self.env.scene.sensors.values():
            sensor._invalidate_cache()
        self.env._sim_step_counter += self.decimation


def attach_crossfire_to_env(env, cfg: CrossfireConfig | None = None) -> HybridCrossfireBackend | object | None:
    """Attach 2SIM_TRAIN Crossfire directly to an instantiated `ManagerBasedRlEnv` (or subclass)."""
    if getattr(env, "_twosim_attached", False):
        return getattr(env, "_twosim_backend", None)

    cfg = cfg or CrossfireConfig()
    if cfg.strict_ieee_fp32:
        enforce_strict_ieee_fp32()

    n_mj, n_rs = resolve_hybrid_split(
        env.num_envs,
        raisim_ratio=cfg.raisim_ratio,
        mj_ratio=cfg.mj_ratio,
        mj_envs=cfg.mj_envs,
        rs_envs=cfg.rs_envs,
        align=cfg.align,
    )
    env._twosim_attached = True

    if n_rs <= 0:
        print(
            f"[2SIM_TRAIN] raisim_ratio=0.0 ({n_mj}/{env.num_envs} MJWarp): "
            f"running 100% native GPU MuJoCo 4.0 Warp.",
            flush=True,
        )
        env._twosim_backend = None
        return None

    if n_mj <= 0:
        return attach_full_raisim(env, cfg)

    backend = HybridCrossfireBackend(env, mj_envs=n_mj, cfg=cfg)

    orig_reset_idx = env._reset_idx

    def _crossfire_reset_idx(env_ids: torch.Tensor | None = None) -> None:
        backend._inside_reset_idx = True
        try:
            orig_reset_idx(env_ids)
        finally:
            backend._inside_reset_idx = False
        backend.stage_async_reset_d2h(env_ids)
        log = env.extras.get("log")
        if isinstance(log, dict) and backend.last_mj_gpu_ms > 0.0:
            log["Crossfire/raisim_ratio"] = float(backend.rs_envs) / float(backend.total_envs)
            log["Crossfire/mj_gpu_substep_ms"] = backend.last_mj_gpu_ms
            log["Crossfire/rs_cpu_substep_ms"] = backend.last_rs_cpu_ms

    env._reset_idx = _crossfire_reset_idx

    class _CrossfireSubstepRunner:
        def run(self) -> None:
            backend.run_crossfire_decimation_step()

    stub = _CrossfireSubstepRunner()
    env._twosim_runner_stub = stub
    env._physics_substep_graph = stub
    env._decimation_substep_runner = stub

    # Preserve stub across MLAB4_Fast enable_fp32_step_optimizations() calls!
    if hasattr(env, "enable_fp32_step_optimizations"):
        orig_enable_fp32 = env.enable_fp32_step_optimizations

        def _wrapped_enable_fp32(*args, **kwargs):
            res = orig_enable_fp32(*args, **kwargs)
            env._decimation_substep_runner = stub
            env._physics_substep_graph = stub
            res["decimation_substep_graph"] = True
            return res

        env.enable_fp32_step_optimizations = _wrapped_enable_fp32

    orig_close = env.close

    def _crossfire_close() -> None:
        backend.close()
        orig_close()

    env.close = _crossfire_close
    env._twosim_backend = backend
    return backend


def enable_crossfire(cfg: CrossfireConfig | None = None, **kwargs) -> CrossfireConfig:
    """Non-invasively upgrade ALL `ManagerBasedRlEnv` (and subclasses like `SdkRecoveryEnv`)
    instantiated in the current Python process to run 2SIM_TRAIN GPU+CPU Crossfire."""
    if cfg is None:
        cfg = CrossfireConfig(**kwargs)
    elif kwargs:
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    if cfg.strict_ieee_fp32:
        enforce_strict_ieee_fp32()

    from mjlab.envs import ManagerBasedRlEnv

    if not getattr(ManagerBasedRlEnv, "_twosim_hooked", False):
        orig_init = ManagerBasedRlEnv.__init__

        def _get_psg(self):
            stub = self.__dict__.get("_twosim_runner_stub")
            if stub is not None:
                return stub
            return self.__dict__.get("_raw_physics_substep_graph")

        def _set_psg(self, val):
            self.__dict__["_raw_physics_substep_graph"] = val

        ManagerBasedRlEnv._physics_substep_graph = property(_get_psg, _set_psg)

        def _patched_init(self, *args, **kw):
            orig_init(self, *args, **kw)
            attach_crossfire_to_env(self, ManagerBasedRlEnv._twosim_cfg)

        ManagerBasedRlEnv.__init__ = _patched_init
        ManagerBasedRlEnv._twosim_hooked = True

    ManagerBasedRlEnv._twosim_cfg = cfg
    return cfg

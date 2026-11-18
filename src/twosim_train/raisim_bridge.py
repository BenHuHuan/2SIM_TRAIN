"""100%-RaiSim (`raisim_ratio == 1.0`) drop-in backend for MJLab `ManagerBasedRlEnv`."""
from __future__ import annotations

import ctypes
import time

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from mjlab.sensor.contact_sensor import ContactSensor

from .asset_resolver import resolve_raisim_fast_assets
from .config import CrossfireConfig
from .warp_slice import slice_model_np


def _detect_foot_sensor(env, cfg: CrossfireConfig) -> ContactSensor | None:
    if cfg.foot_sensor_name is not None:
        return env.scene.sensors.get(cfg.foot_sensor_name)
    for cand in ("sdk_feet", "feet_ground", "feet_contact", "contact_forces"):
        s = env.scene.sensors.get(cand)
        if isinstance(s, ContactSensor):
            return s
    return None


class RaisimFullBackend:
    """Replaces MJWarp physics stepping with 100% `Raisim2_FAST` C++20 OpenMP (`raisim_ratio == 1.0`)."""

    def __init__(self, env, cfg: CrossfireConfig) -> None:
        self.env = env
        self.sim = env.sim
        self.device = env.device
        self.num_envs = int(env.num_envs)
        self.decimation = int(env.cfg.decimation)
        self.dt = float(env.physics_dt)
        self.cfg = cfg

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

        mj_m = self.sim.mj_model
        armature = slice_model_np(self.sim.model.dof_armature[:, self.v_off : self.v_off + self.nj], 0, self.num_envs)
        damping = slice_model_np(self.sim.model.dof_damping[:, self.v_off : self.v_off + self.nj], 0, self.num_envs)
        friction = slice_model_np(
            self.sim.model.dof_frictionloss[:, self.v_off : self.v_off + self.nj], 0, self.num_envs
        )

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
            foot_mu = slice_model_np(self.sim.model.geom_friction[:, foot_geom_ids[0], 0], 0, self.num_envs)
        else:
            foot_mu = np.ones(self.num_envs, dtype=np.float32)

        anchor_name = meta.get("anchor_body_name", "")
        if anchor_name:
            anchor_bid = _mj_body_id(anchor_name)
            base_mj_mass = float(mj_m.body_mass[anchor_bid])
            base_mj_ipos = np.asarray(mj_m.body_ipos[anchor_bid], dtype=np.float32)
            cur_mass = slice_model_np(self.sim.model.body_mass[:, anchor_bid], 0, self.num_envs)
            cur_ipos = slice_model_np(self.sim.model.body_ipos[:, anchor_bid, :], 0, self.num_envs)
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
            int(self.num_envs),
            int(cfg.rs_threads),
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
            raise RuntimeError("Failed to create Raisim2_FAST worlds")
        self.num_feet = int(self.lib.rfast_get_num_feet(ctypes.c_void_p(self.handle)))
        print(
            f"[2SIM_TRAIN | 100% Raisim2_FAST] Initialized {self.num_envs} worlds "
            f"on {cfg.rs_threads} OpenMP threads in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

        self.self_sensor: ContactSensor | None = env.scene.sensors.get(cfg.self_collision_sensor_name)
        self.feet_sensor: ContactSensor | None = _detect_foot_sensor(env, cfg)
        self.hist_len = int(self.self_sensor.cfg.history_length) if self.self_sensor is not None else 4
        self.root_body_id = int(self.robot.indexing.root_body_id)

        pin = str(self.device).startswith("cuda")
        self._qpos_cpu = torch.empty((self.num_envs, self.nq), dtype=torch.float32, pin_memory=pin)
        self._qvel_cpu = torch.empty((self.num_envs, self.nv), dtype=torch.float32, pin_memory=pin)
        self._tgt_cpu = torch.empty((self.num_envs, self.nj), dtype=torch.float32, pin_memory=pin)
        self._kp_cpu = torch.empty((self.num_envs, self.nj), dtype=torch.float32, pin_memory=pin)
        self._kd_cpu = torch.empty((self.num_envs, self.nj), dtype=torch.float32, pin_memory=pin)
        self._lim_cpu = torch.empty((self.num_envs, self.nj), dtype=torch.float32, pin_memory=pin)
        self._wrench_cpu = torch.empty((self.num_envs, 6), dtype=torch.float32, pin_memory=pin)

        self._qpos_out_cpu = torch.empty((self.num_envs, self.nq), dtype=torch.float32, pin_memory=pin)
        self._qvel_out_cpu = torch.empty((self.num_envs, self.nv), dtype=torch.float32, pin_memory=pin)
        self._qfrc_out_cpu = torch.empty((self.num_envs, self.nj), dtype=torch.float32, pin_memory=pin)
        self._self_found_cpu = torch.zeros((self.num_envs, max(self.hist_len, 1)), dtype=torch.float32, pin_memory=pin)
        self._self_force_cpu = torch.zeros((self.num_envs, max(self.hist_len, 1), 3), dtype=torch.float32, pin_memory=pin)
        self._foot_found_cpu = torch.zeros((self.num_envs, max(self.num_feet, 1)), dtype=torch.float32, pin_memory=pin)
        self._foot_force_cpu = torch.zeros((self.num_envs, max(self.num_feet, 1), 3), dtype=torch.float32, pin_memory=pin)

        fused = self.robot._fused_actuator_group
        self._fused_grp = fused._groups[0]
        self._kp_cpu.copy_(self._fused_grp.params["stiffness"], non_blocking=False)
        self._kd_cpu.copy_(self._fused_grp.params["damping"], non_blocking=False)
        self._lim_cpu.copy_(self._fused_grp.params["force_limit"], non_blocking=False)

        self._all_states_dirty = True
        self._dirty_ids: list[torch.Tensor] = []
        self._gains_dirty = False
        self._has_active_wrench = False

        wp.to_torch(self.sim.wp_data.overflow).zero_()
        self.sim.data.nefc.zero_()

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.rfast_destroy(ctypes.c_void_p(self.handle))
            self.handle = None

    def refresh_pure_fk_gpu(self) -> None:
        t_stream = torch.cuda.current_stream(self.device)
        w_stream = wp.stream_from_torch(t_stream)
        with wp.ScopedDevice(self.sim.wp_device), wp.ScopedStream(w_stream):
            mjwarp.fwd_kinematics(self.sim.wp_model, self.sim.wp_data)
            mjwarp.com_vel(self.sim.wp_model, self.sim.wp_data)

    def sync_dirty_states(self) -> None:
        if self._gains_dirty:
            self._kp_cpu.copy_(self._fused_grp.params["stiffness"], non_blocking=False)
            self._kd_cpu.copy_(self._fused_grp.params["damping"], non_blocking=False)
            self._gains_dirty = False
        if not self._all_states_dirty and not self._dirty_ids:
            return
        if self._all_states_dirty:
            self._qpos_cpu.copy_(self.sim.data.qpos[:, : self.nq], non_blocking=False)
            self._qvel_cpu.copy_(self.sim.data.qvel[:, : self.nv], non_blocking=False)
            self.lib.rfast_set_all_states(
                ctypes.c_void_p(self.handle),
                ctypes.c_void_p(self._qpos_cpu.data_ptr()),
                ctypes.c_void_p(self._qvel_cpu.data_ptr()),
            )
            self._all_states_dirty = False
            self._dirty_ids.clear()
        else:
            ids = torch.unique(torch.cat(self._dirty_ids, dim=0))
            self._dirty_ids.clear()
            if ids.numel() == 0:
                return
            idx64 = ids.to(dtype=torch.int64)
            qp = self.sim.data.qpos[:, : self.nq].index_select(0, idx64).cpu().to(torch.float32).contiguous()
            qv = self.sim.data.qvel[:, : self.nv].index_select(0, idx64).cpu().to(torch.float32).contiguous()
            ids32 = idx64.to(dtype=torch.int32, device="cpu").contiguous()
            self.lib.rfast_set_states_packed(
                ctypes.c_void_p(self.handle),
                ctypes.c_void_p(ids32.data_ptr()),
                int(ids32.numel()),
                ctypes.c_void_p(qp.data_ptr()),
                ctypes.c_void_p(qv.data_ptr()),
            )

    def _write_contacts(self) -> None:
        if self.self_sensor is not None:
            for slot in self.self_sensor._slots:
                if slot.field_name == "found":
                    slot.data_view.copy_(self._self_found_cpu[:, 0:1], non_blocking=True)
                elif slot.field_name == "force":
                    slot.data_view.copy_(self._self_force_cpu[:, 0, :], non_blocking=True)
            if self.self_sensor._history_state is not None and "force" in self.self_sensor._history_state:
                hist_buf = self.self_sensor._history_state["force"]
                h = min(hist_buf.size(2), self.hist_len)
                hist_buf[:, 0, :h, :].copy_(self._self_force_cpu[:, :h, :], non_blocking=True)
            self.self_sensor._invalidate_cache()

        if self.feet_sensor is not None and self.num_feet > 0:
            f_idx, frc_idx = 0, 0
            for slot in self.feet_sensor._slots:
                if slot.field_name == "found" and f_idx < self.num_feet:
                    slot.data_view.copy_(self._foot_found_cpu[:, f_idx : f_idx + 1], non_blocking=True)
                    f_idx += 1
                elif slot.field_name == "force" and frc_idx < self.num_feet:
                    slot.data_view.copy_(self._foot_force_cpu[:, frc_idx, :], non_blocking=True)
                    frc_idx += 1
            self.feet_sensor._invalidate_cache()

    def run_decimation_step(self) -> None:
        self.sync_dirty_states()
        self.env.action_manager.apply_action()
        self._tgt_cpu.copy_(self.robot.data.joint_pos_target[:, : self.nj], non_blocking=False)
        wrench_ptr = None
        if self._has_active_wrench:
            self._wrench_cpu.copy_(self.sim.data.xfrc_applied[:, self.root_body_id, :], non_blocking=False)
            wrench_ptr = ctypes.c_void_p(self._wrench_cpu.data_ptr())

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

        self.sim.data.qpos[:, : self.nq].copy_(self._qpos_out_cpu, non_blocking=True)
        self.sim.data.qvel[:, : self.nv].copy_(self._qvel_out_cpu, non_blocking=True)
        self.sim.data.qfrc_actuator[:, self.v_off : self.v_off + self.nj].copy_(self._qfrc_out_cpu, non_blocking=True)
        self.sim.data.actuator_force[:, : self.nj].copy_(self._qfrc_out_cpu, non_blocking=True)
        self.refresh_pure_fk_gpu()
        self._write_contacts()
        self.env._sim_step_counter += self.decimation


def attach_full_raisim(env, cfg: CrossfireConfig) -> RaisimFullBackend:
    backend = RaisimFullBackend(env, cfg)
    orig_reset_idx = env._reset_idx

    def _reset_idx(env_ids: torch.Tensor | None = None) -> None:
        orig_reset_idx(env_ids)
        backend._gains_dirty = True
        if env_ids is None:
            backend._all_states_dirty = True
            backend._dirty_ids.clear()
        elif not backend._all_states_dirty and env_ids.numel() > 0:
            backend._dirty_ids.append(env_ids.reshape(-1))

    env._reset_idx = _reset_idx

    def _forward() -> None:
        backend.sync_dirty_states()
        backend.refresh_pure_fk_gpu()
        for sensor in env.scene.sensors.values():
            sensor._invalidate_cache()

    env.sim.forward = _forward
    env.sim._post_decimation_check = None
    env.sim.use_cuda_graph = True

    class _Runner:
        def run(self) -> None:
            backend.run_decimation_step()

    stub = _Runner()
    env._twosim_runner_stub = stub
    env._physics_substep_graph = stub
    env._decimation_substep_runner = stub
    env._checked_kinematic_post_step_forward = False

    orig_close = env.close

    def _close() -> None:
        backend.close()
        orig_close()

    env.close = _close
    env._twosim_backend = backend
    return backend

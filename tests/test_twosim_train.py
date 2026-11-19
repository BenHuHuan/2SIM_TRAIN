"""Unit and GPU/Warp integration tests for `twosim_train` (`2SIM_TRAIN`)."""
from __future__ import annotations

import pytest
import torch

from twosim_train import (
    CrossfireConfig,
    enforce_strict_ieee_fp32,
    resolve_hybrid_split,
    slice_model_np,
    slice_wp_struct,
)


def test_resolve_hybrid_split_ratios_and_alignment():
    assert resolve_hybrid_split(4096) == (2048, 2048)  # Default 50/50 Crossfire
    assert resolve_hybrid_split(4096, raisim_ratio=0.0) == (4096, 0)
    assert resolve_hybrid_split(4096, raisim_ratio=1.0) == (0, 4096)
    assert resolve_hybrid_split(4096, raisim_ratio=0.25) == (3072, 1024)
    assert resolve_hybrid_split(4096, mj_ratio=0.625) == (2560, 1536)
    assert resolve_hybrid_split(4096, mj_envs=3328) == (3328, 768)
    assert resolve_hybrid_split(4096, rs_envs=512) == (3584, 512)
    # Fractional ratio rounds cleanly to 32-world warp alignment
    n_mj, n_rs = resolve_hybrid_split(4096, raisim_ratio=0.20, align=32)
    assert n_mj + n_rs == 4096
    assert n_rs % 32 == 0


def test_resolve_hybrid_split_validation():
    with pytest.raises(ValueError):
        resolve_hybrid_split(4096, raisim_ratio=0.25, mj_envs=3072)
    with pytest.raises(ValueError):
        resolve_hybrid_split(4096, raisim_ratio=1.5)


def test_strict_ieee_fp32_guard():
    enforce_strict_ieee_fp32()
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    else:
        assert torch.backends.cuda.matmul.allow_tf32 is False


def test_slice_model_np_broadcast_and_slice():
    shared = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    out = slice_model_np(shared, 10, 14)
    assert out.shape == (4, 3)
    assert float(out[3, 2]) == 3.0

    per_world = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out2 = slice_model_np(per_world, 3, 7)
    assert out2.shape == (4, 2)
    assert float(out2[0, 0]) == 6.0


def test_zero_copy_warp_struct_slicing():
    wp = pytest.importorskip("warp")
    mujoco = pytest.importorskip("mujoco")
    mjwarp = pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mjwarp zero-copy slice test")

    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body pos="0 0 1.5"><freejoint/><geom type="sphere" size="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    with wp.ScopedDevice("cuda:0"):
        wpm = mjwarp.put_model(m)
        wpd = mjwarp.put_data(m, d, nworld=128, nconmax=32, njmax=64)

    wpd_mj = slice_wp_struct(wpd, 0, 96, 128, wp.get_device("cuda:0"))
    wpd_rs = slice_wp_struct(wpd, 96, 128, 128, wp.get_device("cuda:0"))

    qpos = wp.to_torch(wpd.qpos)
    qpos[:, 2] = 2.0
    with wp.ScopedDevice("cuda:0"):
        for _ in range(5):
            mjwarp.step(wpm, wpd_mj)
    wp.synchronize()

    # MJ slice [0:96] stepped under gravity; RS slice [96:128] remained untouched!
    assert float(qpos[0, 2]) < 2.0
    assert float(qpos[95, 2]) < 2.0
    assert float(qpos[96, 2]) == 2.0
    assert float(qpos[127, 2]) == 2.0

    # Sub-slice pure FK updates xpos[96:128] in-place without touching [0:96]
    qpos[96:, 2] = 4.25
    with wp.ScopedDevice("cuda:0"):
        mjwarp.fwd_kinematics(wpm, wpd_rs)
    wp.synchronize()
    xpos = wp.to_torch(wpd.xpos)
    assert abs(float(xpos[96, 1, 2]) - 4.25) < 1e-6


def test_config_defaults():
    cfg = CrossfireConfig()
    assert cfg.align == 32
    assert cfg.strict_ieee_fp32 is True

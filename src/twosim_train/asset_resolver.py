"""Automatic MJCF-to-Raisim2_FAST asset & library resolution for 2SIM_TRAIN."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import CrossfireConfig


def resolve_raisim_fast_assets(mj_model, cfg: CrossfireConfig) -> tuple[Any, Path, dict, Path]:
    """Return `(cdll, lib_path, meta_dict, urdf_template_path)` for `mj_model` and `cfg`."""
    from raisim_fast import compile_mjcf_to_analytical_urdf, load_raisim_fast_cdll

    cdll, lib_path = load_raisim_fast_cdll(cfg.lib_path)

    if cfg.asset_dir is not None:
        asset_p = Path(cfg.asset_dir).resolve()
        meta_file = asset_p / "raisim_fast_meta.json"
        if not meta_file.is_file():
            meta_file = asset_p / "raisim_tg3_meta.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"No raisim_fast_meta.json found in {asset_p}")
        meta = json.loads(meta_file.read_text())
        urdf_name = meta.get("template_urdf", "tiangong3_v2_29_template.urdf")
        return cdll, lib_path, meta, (asset_p / urdf_name).resolve()

    # Zero-config auto-compilation cached by MjModel signature
    sig_str = f"nb={mj_model.nbody}_nj={mj_model.njnt}_ng={mj_model.ngeom}_nm={mj_model.nmesh}_m0={float(mj_model.body_subtreemass[0]):.6f}"
    h = hashlib.sha256(sig_str.encode()).hexdigest()[:12]
    cache_root = Path(os.environ.get("TWOSIM_CACHE_DIR", Path.home() / ".cache" / "twosim_train" / "assets"))
    asset_p = cache_root / f"mjcf_{h}"
    meta_file = asset_p / "raisim_fast_meta.json"
    if not meta_file.is_file():
        meta = compile_mjcf_to_analytical_urdf(mj_model, asset_p)
    else:
        meta = json.loads(meta_file.read_text())
    urdf_name = meta.get("template_urdf", "robot_fast_template.urdf")
    return cdll, lib_path, meta, (asset_p / urdf_name).resolve()

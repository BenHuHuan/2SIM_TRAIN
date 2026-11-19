"""CLI entrypoints (`2sim-train` and `2sim-autotune`) for 2SIM_TRAIN."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from .autotune import autotune_crossfire_ratio
from .config import CrossfireConfig
from .crossfire import enable_crossfire


def _pop_opt(flags: tuple[str, ...], cast_fn):
    argv = sys.argv
    for flag in flags:
        for i in range(1, len(argv)):
            if argv[i] == flag and i + 1 < len(argv):
                val = cast_fn(argv[i + 1])
                del argv[i : i + 2]
                return val
            if argv[i].startswith(flag + "="):
                val = cast_fn(argv[i].split("=", 1)[1])
                del argv[i]
                return val
    return None


def _pop_bool_flag(flag: str) -> bool | None:
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i] == flag:
            del argv[i]
            return True
    return None


def main_train() -> None:
    """Universal 2SIM_TRAIN launcher wrapping either a `.py` training script or `MLAB4_Fast` task ID."""
    if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  1) Wrap any MLAB4_Fast task:\n"
            "       2sim-train Mjlab-Velocity-Flat-Unitree-G1 --raisim-ratio 0.25\n"
            "  2) Wrap any custom training script:\n"
            "       2sim-train path/to/train_script.py --raisim-ratio 0.25 [script args...]\n\n"
            "2SIM_TRAIN Options (stripped before forwarding remaining args):\n"
            "  --raisim-ratio / --rs-ratio <0.0..1.0>   Fraction of envs stepped on CPU Raisim2_FAST (default: 0.50)\n"
            "  --mj-ratio <0.0..1.0>                    Fraction of envs stepped on GPU MuJoCo 4.0 Warp\n"
            "  --mj-envs <int> / --rs-envs <int>        Explicit per-simulator world counts\n"
            "  --rs-threads <int>                       OpenMP CPU thread count for Raisim2_FAST (default: $RAISIM_NUM_THREADS or nproc)\n"
            "  --raisim-asset-dir <path>                Optional pre-compiled URDF+meta folder (auto-compiled from MjModel if omitted)\n"
            "  --raisim-lib <path>                      Optional explicit path to libraisim_fast_vec.so\n"
            "  --hybrid-profile-every <steps>           Print MJ(GPU) vs RS(CPU) ms/step balance every N steps (default: 240)\n"
            "  --no-strict-ieee-fp32                    Allow TF32 instead of strict 23-bit IEEE 754 FP32\n"
        )
        return

    raisim_ratio = _pop_opt(("--raisim-ratio", "--rs-ratio"), float)
    mj_ratio = _pop_opt(("--mj-ratio",), float)
    mj_envs = _pop_opt(("--mj-envs",), int)
    rs_envs = _pop_opt(("--rs-envs",), int)
    align = _pop_opt(("--hybrid-align",), int)
    rs_threads = _pop_opt(("--rs-threads",), int)
    asset_dir = _pop_opt(("--raisim-asset-dir",), str)
    lib_path = _pop_opt(("--raisim-lib",), str)
    profile_every = _pop_opt(("--hybrid-profile-every",), int)
    no_ieee = _pop_bool_flag("--no-strict-ieee-fp32")

    if raisim_ratio is None and mj_ratio is None and mj_envs is None and rs_envs is None:
        if "HYBRID_RAISIM_RATIO" in os.environ:
            raisim_ratio = float(os.environ["HYBRID_RAISIM_RATIO"])
        elif "HYBRID_MJ_RATIO" in os.environ:
            mj_ratio = float(os.environ["HYBRID_MJ_RATIO"])
        elif "HYBRID_MJ_ENVS" in os.environ:
            mj_envs = int(os.environ["HYBRID_MJ_ENVS"])
        elif "HYBRID_RS_ENVS" in os.environ:
            rs_envs = int(os.environ["HYBRID_RS_ENVS"])
        else:
            raisim_ratio = 0.50

    os.environ.setdefault("_MJLAB_BACKEND_REEXEC", "1")
    py_exe = sys.executable
    os.environ.setdefault("MJLAB_MUJOCO4_PYTHON", py_exe)
    os.environ.setdefault("MJLAB_MUJOCO4_GIT_PYTHON", py_exe)

    cfg = CrossfireConfig(
        raisim_ratio=raisim_ratio,
        mj_ratio=mj_ratio,
        mj_envs=mj_envs,
        rs_envs=rs_envs,
        align=32 if align is None else align,
        rs_threads=int(os.environ.get("RAISIM_NUM_THREADS", os.cpu_count() or 16)) if rs_threads is None else rs_threads,
        asset_dir=asset_dir,
        lib_path=lib_path,
        profile_every_steps=240 if profile_every is None else profile_every,
        strict_ieee_fp32=(not bool(no_ieee)),
    )
    enable_crossfire(cfg)

    first_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if first_arg.endswith(".py") or Path(first_arg).is_file():
        script_path = Path(first_arg).resolve()
        del sys.argv[1]
        sys.argv[0] = str(script_path)
        sys.path.insert(0, str(script_path.parent))
        sys.path.insert(0, str(script_path.parent.parent))
        runpy.run_path(str(script_path), run_name="__main__")
    else:
        from mjlab.scripts.train import main as mjlab_train_main

        mjlab_train_main()


def main_autotune() -> None:
    parser = argparse.ArgumentParser(
        description="Probe GPU MuJoCo 4.0 Warp vs CPU Raisim2_FAST and recommend optimal --raisim-ratio."
    )
    parser.add_argument("--mjcf", type=str, default=None, help="Optional scene MJCF XML path")
    parser.add_argument("--num-envs", type=int, default=4096, help="Total parallel environments (default: 4096)")
    parser.add_argument("--decimation", type=int, default=10, help="Physics substeps per policy step (default: 10)")
    parser.add_argument("--rs-threads", type=int, default=None, help="OpenMP threads for Raisim2_FAST")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--steps", type=int, default=20, help="Timed policy steps per ratio")
    parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    res = autotune_crossfire_ratio(
        mjcf_path=args.mjcf,
        total_envs=args.num_envs,
        decimation=args.decimation,
        rs_threads=args.rs_threads,
        device=args.device,
        steps=args.steps,
    )
    print("\n=== 2SIM_TRAIN Hardware Crossfire Autotune Report ===")
    print(f"Total envs: {res['total_envs']} | CPU threads: {res['rs_threads']} | Pure MJWarp: {res['baseline_mj_only_ms']:.2f} ms/step")
    print(f"{'Ratio':>8} | {'MJ Envs':>8} | {'RS Envs':>8} | {'MJ GPU ms':>10} | {'RS CPU ms':>10} | {'Est. Step ms':>12}")
    print("-" * 70)
    for row in res["sweep"]:
        print(
            f"{row['raisim_ratio']:8.4f} | {row['mj_envs']:8d} | {row['rs_envs']:8d} | "
            f"{row['mj_gpu_ms']:10.2f} | {row['rs_cpu_ms']:10.2f} | {row['effective_step_ms']:12.2f}"
        )
    print("-" * 70)
    print(f"Recommended Peak Speed Ratio         : --raisim-ratio {res['recommended_max_speed_ratio']}")
    print(f"Recommended Max-RaiSim Fast Ratio    : --raisim-ratio {res['recommended_fast_high_mix_ratio']}")
    print(f"Recommended GPU=CPU Parity Ratio     : --raisim-ratio {res['recommended_gpu_cpu_parity_ratio']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2) + "\n")

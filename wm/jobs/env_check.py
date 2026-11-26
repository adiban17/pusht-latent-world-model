"""Job env_check (Phase 1): is this machine ready to run the project?

Checks platform, Python/torch versions, device, a 64x64 render (saved as frame.png),
the reset_to_state round trip, a store push -> pull round trip, and a 2048x2048 matmul timing.
"""
from __future__ import annotations

import platform as pyplatform
import secrets
import sys
import time
from importlib.metadata import version

import imageio.v2 as imageio
import numpy as np
import torch

from wm.envs.pusht import get_state, make_env, pose_error, render64, reset_to_state
from wm.jobs import JobContext
from wm.report import Report

# Pusher well clear of the block, so gym-pusht's settling physics step doesn't move anything.
ROUNDTRIP_STATES = np.array([
    [100.0, 100.0, 300.0, 300.0, 0.0],
    [400.0, 120.0, 220.0, 330.0, 1.2],
    [80.0, 420.0, 330.0, 200.0, -2.5],
], dtype=np.float32)
MAX_POS_PX, MAX_ANG_DEG = 1.0, 0.5


def _pkg(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not installed"


def roundtrip_error(env) -> tuple[float, float]:
    """Worst (pos px over pusher+block, |angle| deg) error of reset_to_state -> get_state."""
    worst_pos, worst_ang = 0.0, 0.0
    for s in ROUNDTRIP_STATES:
        reset_to_state(env, s)
        got = get_state(env)
        block_pos, ang = pose_error(got, s)
        pusher_pos = float(np.linalg.norm(got[:2] - s[:2]))
        worst_pos = max(worst_pos, block_pos, pusher_pos)
        worst_ang = max(worst_ang, abs(ang))
    return worst_pos, worst_ang


def matmul_ms(device: torch.device, n: int = 2048, reps: int = 5) -> float:
    """Mean milliseconds for one n x n float32 matmul."""
    a, b = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
    (a @ b).sum().item()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    c.sum().item()
    return (time.perf_counter() - t0) / reps * 1000.0


def run(ctx: JobContext, report: Report) -> None:
    report.check_true("platform", ctx.platform in ("local", "kaggle", "colab"),
                      ctx.platform, "local|kaggle|colab")
    py = pyplatform.python_version()
    report.check_true("python_version", sys.version_info >= (3, 10), py, ">= 3.10")

    want = "cpu" if ctx.platform == "local" else "cuda"
    report.check_true("device", ctx.device.type == want and ctx.device.type != "mps",
                      ctx.device.type, f"{want} on {ctx.platform}; never mps")

    env = make_env()
    obs, _ = env.reset(seed=0)
    frame = render64(obs["pixels"])
    report.check_true("render64", frame.shape == (64, 64, 3) and frame.dtype == np.uint8,
                      f"{frame.shape} {frame.dtype}", "(64, 64, 3) uint8")
    frame_path = ctx.out_dir / "frame.png"
    imageio.imwrite(frame_path, frame)
    report.save(frame_path)

    pos_err, ang_err = roundtrip_error(env)
    env.close()
    report.check_true("reset_roundtrip", pos_err < MAX_POS_PX and ang_err < MAX_ANG_DEG,
                      f"{pos_err:.4f} px, {ang_err:.4f} deg",
                      f"< {MAX_POS_PX} px, < {MAX_ANG_DEG} deg")

    token = secrets.token_hex(16)
    payload = ctx.out_dir / "roundtrip.txt"
    payload.write_text(token)
    ok_so_far = not report.failed
    saved = ctx.store.push("env-check", payload, ctx.meta(complete=ok_so_far))
    pulled = ctx.store.pull("env-check")
    got = (pulled / "roundtrip.txt").read_text() if pulled else None
    report.check_true("store_roundtrip", got == token,
                      "bytes match" if got == token else f"mismatch ({got!r})", "push -> pull")
    if report.failed and ok_so_far:  # never leave a failed run marked complete
        saved = ctx.store.push("env-check", payload, ctx.meta(complete=False))
    report.save(saved)

    report.metric("python", py)
    report.metric("torch", torch.__version__)
    report.metric("cuda_available", torch.cuda.is_available())
    report.metric("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    for pkg in ("gym-pusht", "gymnasium", "pymunk", "pygame", "opencv-python-headless",
                "opencv-python", "numpy", "wandb"):
        report.metric(pkg, _pkg(pkg))
    report.metric("matmul_2048_ms", f"{matmul_ms(ctx.device):.1f}")

"""Angle audit (Phase 5b): is the block angle in the stored labels the angle drawn in the frames?

Tools, none of them learned:
  render_pose      renders a block pose given by its centre of gravity (CoG) and angle;
  block_mask       the block's pixels in a 64x64 frame (by colour);
  mask_pose        CoG and angle read from the mask with image moments: the T's long principal
                   axis is its body y axis (bar -> stem), and the skew along it gives the sign;
  template_angle   "perfect reader" with the true position known: renders candidate angles at
                   that position and picks the best pixel match (works on blurry decoded frames).

Image axes: column = x * 64/512, row = y * 64/512 (gym-pusht draws without a y flip and the
observation is transposed to [row, col]). A body's +y axis in world coordinates is
R(theta) (0, 1) = (-sin theta, cos theta), so theta = atan2(-u_x, u_y) for that unit vector u.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from typing import Any

import cv2
import numpy as np

from wm.envs.pusht import BLOCK_COG_LOCAL, block_cog, render64, reset_to_state, wrap_angles_rad

PX_PER_CELL = 512.0 / 64.0                 # world px per 64-px image pixel
BLOCK_COLOURS = np.array([[143, 163, 184],  # fill: 1.2 x LightSlateGray (gym-pusht light_color)
                          [119, 136, 153]],  # outline: LightSlateGray
                         np.float32)
FAR_PUSHER = (40.0, 40.0)                  # a pusher position clear of every grid pose


def origin_from_cog(cog: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Body origin (the stored state's block x, y) of a block whose CoG is at `cog`."""
    cog, theta = np.asarray(cog, np.float64), np.asarray(theta, np.float64)
    c, s = np.cos(theta), np.sin(theta)
    ox, oy = BLOCK_COG_LOCAL
    return np.stack([cog[..., 0] - (c * ox - s * oy), cog[..., 1] - (s * ox + c * oy)], -1)


def pose_state(cog: tuple[float, float], theta: float,
               pusher: tuple[float, float] = FAR_PUSHER) -> np.ndarray:
    """5-vector state with the block's CoG at `cog` and angle theta (rad)."""
    ox, oy = origin_from_cog(np.array(cog), np.array(theta))
    return np.array([pusher[0], pusher[1], ox, oy, theta], np.float64)


def render_state(env, state: np.ndarray) -> np.ndarray:
    """64x64x3 uint8 frame of an exact state."""
    return render64(reset_to_state(env, state)["pixels"])


def render_pose(env, cog: tuple[float, float], theta: float,
                pusher: tuple[float, float] = FAR_PUSHER) -> tuple[np.ndarray, np.ndarray]:
    """(frame, state) for a block at CoG `cog`, angle theta, pusher at `pusher`."""
    state = pose_state(cog, theta, pusher)
    return render_state(env, state), state


def grid_poses(angles_deg: list[float], positions: list[tuple[float, float]]) -> list[tuple[tuple[float, float], float]]:
    """Every (CoG, theta rad) pair of the scoring grid."""
    return [(tuple(p), float(np.radians(a))) for p in positions for a in angles_deg]


def scoring_grid(env, step_deg: float, positions: list) -> tuple[np.ndarray, np.ndarray]:
    """(frames uint8 [N, 64, 64, 3], states [N, 5]) of the scoring grid: every `step_deg` at each
    CoG in `positions`, pusher far away (the judges' "grid" angle score, before/after)."""
    angles = list(np.arange(0.0, 360.0, float(step_deg)))
    out = [render_pose(env, cog, th) for cog, th in grid_poses(angles, [tuple(p) for p in positions])]
    return np.stack([f for f, _ in out]), np.stack([s for _, s in out])


# ---- reading the pose from pixels ------------------------------------------------------------
def block_mask(frame: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """Boolean [64, 64]: pixels within `tol` (max abs per channel) of a block colour."""
    f = frame.astype(np.float32)[..., None, :]                       # [64, 64, 1, 3]
    return (np.abs(f - BLOCK_COLOURS).max(-1) < tol).any(-1)


def mask_pose(frame: np.ndarray) -> dict[str, float] | None:
    """Block centroid (world px) and angle (rad) from the mask's moments; None if no mask."""
    rows, cols = np.nonzero(block_mask(frame))
    if len(rows) < 5:
        return None
    xy = np.stack([cols + 0.5, rows + 0.5], -1) * PX_PER_CELL       # pixel centres -> world px
    centre = xy.mean(0)
    d = xy - centre
    evals, evecs = np.linalg.eigh(d.T @ d / len(d))
    u = evecs[:, np.argmax(evals)]                                   # long axis = body y axis
    if np.mean((d @ u) ** 3) < 0:                                    # stem side = positive skew
        u = -u
    return {"x": float(centre[0]), "y": float(centre[1]), "angle": float(np.arctan2(-u[0], u[1])),
            "pixels": int(len(rows))}


def template_angle(env, frame: np.ndarray, state: np.ndarray, step_deg: float = 5.0) -> float:
    """Angle (rad) whose render, with the block's CoG and the pusher at their true positions,
    best matches `frame` (uint8 or float in [0, 255]) in pixel MSE."""
    cog = block_cog(state)
    target = np.asarray(frame, np.float32)
    best, best_err = 0.0, np.inf
    for a in np.radians(np.arange(-180.0, 180.0, step_deg)):
        cand = render_state(env, pose_state(tuple(cog), a, (state[0], state[1]))).astype(np.float32)
        err = float(((cand - target) ** 2).mean())
        if err < best_err:
            best, best_err = float(a), err
    return best


def angle_err_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in degrees, wrapped to [0, 180]."""
    return np.abs(np.degrees(wrap_angles_rad(np.asarray(a) - np.asarray(b))))


# ---- audit pieces ----------------------------------------------------------------------------
def rerender_consistency(env, frames: np.ndarray, states: np.ndarray) -> dict[str, float]:
    """Re-render each stored state and compare with its stored frame (mean abs uint8 diff)."""
    diffs = np.array([np.abs(render_state(env, s.astype(np.float64)).astype(np.float32)
                             - f.astype(np.float32)).mean() for f, s in zip(frames, states)])
    return {"frames": int(len(diffs)), "mean_abs_diff": float(diffs.mean()),
            "max_mean_abs_diff": float(diffs.max()), "exact_share": float((diffs == 0).mean())}


def orientation_check(env, angles_deg: list[float], positions: list[tuple[float, float]]) -> dict[str, Any]:
    """On rendered grid poses: mask-moment angle vs the state angle, and the mask centroid vs
    block_cog and vs the body origin (the CoG must be the closer one)."""
    ang_err, d_cog, d_origin = [], [], []
    for cog, theta in grid_poses(angles_deg, positions):
        frame, state = render_pose(env, cog, theta)
        mp = mask_pose(frame)
        if mp is None:
            ang_err.append(180.0); d_cog.append(np.inf); d_origin.append(0.0)
            continue
        ang_err.append(float(angle_err_deg(mp["angle"], theta)))
        c = np.array([mp["x"], mp["y"]])
        d_cog.append(float(np.linalg.norm(c - block_cog(state))))
        d_origin.append(float(np.linalg.norm(c - state[2:4])))
    ang_err, d_cog, d_origin = map(np.array, (ang_err, d_cog, d_origin))
    return {"poses": int(len(ang_err)), "angle_err_mean_deg": float(ang_err.mean()),
            "angle_err_max_deg": float(ang_err.max()), "centroid_to_cog_mean_px": float(d_cog.mean()),
            "centroid_to_origin_mean_px": float(d_origin.mean()),
            "cog_closer_share": float((d_cog < d_origin).mean())}


def target_roundtrip(states: np.ndarray) -> dict[str, float]:
    """state_targets -> decode_targets -> probe_errors must give zero error; plus wrap cases."""
    from wm.data.windows import state_targets
    from wm.models.probe import decode_targets, probe_errors

    errs = probe_errors(decode_targets(state_targets(states)), states)
    # wrap edge cases: 359 deg vs 1 deg -> 2 deg; -179 vs 179 -> 2; 3*pi vs pi -> 0
    s = np.zeros((3, 5)); s[:, 4] = np.radians([1.0, 179.0, 180.0])
    pred = {"pusher": s[:, :2], "block": block_cog(s), "angle": np.radians([359.0, -179.0, 540.0])}
    wrap = probe_errors(pred, s)["block_deg"]
    return {"max_block_px": float(errs["block_px"].max()), "max_block_deg": float(errs["block_deg"].max()),
            "max_pusher_px": float(errs["pusher_px"].max()), "wrap_cases_deg": [float(x) for x in wrap]}


# ---- the labelled grid image -----------------------------------------------------------------
def grid_image(env, angles_deg: list[float], cog: tuple[float, float] = (256.0, 256.0),
               scale: int = 4, arrows: bool = True) -> np.ndarray:
    """One labelled tile per angle: the rendered frame (upscaled), the state angle, the mask-moment
    angle, and (if `arrows`) a red arrow from the CoG along the body +y axis computed from the
    state (it must point from the bar into the stem)."""
    tiles = []
    for a in angles_deg:
        frame, state = render_pose(env, cog, float(np.radians(a)))
        tile = np.ascontiguousarray(frame.repeat(scale, 0).repeat(scale, 1))
        if arrows:
            c = block_cog(state)
            u = np.array([-np.sin(state[4]), np.cos(state[4])])
            p0 = tuple(int(v) for v in (c / PX_PER_CELL * scale))
            p1 = tuple(int(v) for v in ((c + 60.0 * u) / PX_PER_CELL * scale))
            cv2.arrowedLine(tile, p0, p1, (220, 30, 30), 2, cv2.LINE_AA, tipLength=0.3)
        mp = mask_pose(frame)
        label = np.full((34, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(label, f"state {a:g} deg", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1,
                    cv2.LINE_AA)
        read = "none" if mp is None else f"{np.degrees(mp['angle']) % 360:.0f}"
        cv2.putText(label, f"mask reads {read} deg", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([label, tile], 0))
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    out = [tiles[0]]
    for t in tiles[1:]:
        out += [gap, t]
    return np.concatenate(out, 1)

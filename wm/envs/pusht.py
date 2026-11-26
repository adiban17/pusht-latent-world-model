"""PushT environment helpers (CLAUDE.md §1).

State vector used everywhere in this repo (float32[5]):
    [pusher_x, pusher_y, block_x, block_y, block_angle]   (pixels, pixels, radians)
read from the unwrapped gym-pusht env (verified against the installed gym-pusht):
    env.unwrapped.agent.position   -> pymunk Vec2d (x, y) of the pusher
    env.unwrapped.block.position   -> pymunk Vec2d (x, y) of the T block's body
    env.unwrapped.block.angle      -> float, radians (unbounded; we wrap to [-pi, pi))
gym-pusht's own `reset(options={"reset_to_state": s})` takes the same 5-vector.

`block.position` is the T's body origin (the middle of the top edge of the bar), not its centre.
For pose errors, success and data statistics we use the T's centre of gravity instead
(pre-registered decision, Phase 2): cog = (block_x, block_y) + R(angle) @ BLOCK_COG_LOCAL.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import cv2
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import gymnasium as gym
import numpy as np

ENV_ID = "gym_pusht/PushT-v0"
IMG_SIZE = 64
SUCCESS_POS_PX = 20.0   # fixed; never tune
SUCCESS_ANG_DEG = 20.0  # fixed; never tune
# T centre of gravity in body coordinates: mean of the bar (0, 15) and stem (0, 75) centroids,
# as gym-pusht's add_tee sets it (checked against block.center_of_gravity in tests).
BLOCK_COG_LOCAL = np.array([0.0, 45.0])


def make_env() -> gym.Env:
    """PushT with 96x96 pixel + agent_pos observations; action = absolute (x, y) in [0, 512]^2."""
    return gym.make(ENV_ID, obs_type="pixels_agent_pos", render_mode="rgb_array")


def render64(pixels: np.ndarray) -> np.ndarray:
    """Resize a 96x96x3 uint8 observation to 64x64x3 uint8 (INTER_AREA)."""
    return cv2.resize(pixels, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def wrap_angle_rad(a: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def wrap_angles_rad(a: np.ndarray) -> np.ndarray:
    """Vectorised wrap of radians to [-pi, pi) (float64)."""
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def circular_std_deg(angles_rad: np.ndarray) -> tuple[float, float]:
    """(circular std in degrees, mean resultant length R) of angles in radians.

    std = sqrt(-2 ln R). It is unbounded: R -> 0 (orientations spread evenly around the circle)
    gives values well above 180 deg, e.g. R = 0.027 -> 154 deg. It is NOT bounded by 180.
    """
    a = wrap_angles_rad(angles_rad)
    r = float(np.hypot(np.sin(a).mean(), np.cos(a).mean()))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-12))))), r


def wrap_angle_deg(a: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return float((a + 180.0) % 360.0 - 180.0)


def get_state(env: gym.Env) -> np.ndarray:
    """[pusher x, pusher y, block x, block y, block angle (rad, wrapped)] as float32[5]."""
    u = env.unwrapped
    return np.array(
        [u.agent.position[0], u.agent.position[1],
         u.block.position[0], u.block.position[1],
         wrap_angle_rad(u.block.angle)],
        dtype=np.float32,
    )


def reset_to_state(env: gym.Env, state: np.ndarray) -> dict:
    """Reset the env to an exact state; returns the observation.

    gym-pusht supports `reset(options={"reset_to_state": state})` natively, but its `_set_state`
    sets the block position *before* the angle, and pymunk rotates the body about its centre of
    gravity c (body-local (0, 45) for the T), which moves `block.position` by c - R(theta) c
    (up to 90 px). We pre-compensate the position (p_in = p - c + R(theta) c) so that
    get_state(env) == state afterwards. It also runs one physics step after placing the bodies,
    so a pusher placed *inside* the block gets pushed out.
    """
    u = env.unwrapped
    if getattr(u, "block", None) is None:  # bodies only exist after the first reset
        env.reset()
    target = np.asarray(state, dtype=np.float64).copy()
    c = np.array(u.block.center_of_gravity, dtype=np.float64)
    th = target[4]
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    target[2:4] = target[2:4] - c + rot @ c
    obs, _ = env.reset(options={"reset_to_state": target})
    return obs


def block_cog(state: np.ndarray) -> np.ndarray:
    """World (x, y) of the T's centre of gravity from state(s) [..., 5] -> [..., 2] (float64)."""
    s = np.asarray(state, dtype=np.float64)
    th = s[..., 4]
    c, sn = np.cos(th), np.sin(th)
    ox, oy = BLOCK_COG_LOCAL
    return np.stack([s[..., 2] + c * ox - sn * oy, s[..., 3] + sn * ox + c * oy], axis=-1)


def pose_error(state: np.ndarray, goal_state: np.ndarray) -> tuple[float, float]:
    """(distance between block centres of gravity in px, angle difference in deg in [-180, 180))."""
    pos_px = float(np.linalg.norm(block_cog(state) - block_cog(goal_state)))
    ang_deg = wrap_angle_deg(np.degrees(float(state[4]) - float(goal_state[4])))
    return pos_px, ang_deg


def is_success(state: np.ndarray, goal_state: np.ndarray) -> bool:
    """Block within 20 px and 20 degrees of the goal pose."""
    pos_px, ang_deg = pose_error(state, goal_state)
    return pos_px < SUCCESS_POS_PX and abs(ang_deg) < SUCCESS_ANG_DEG


# ---- normalisation -------------------------------------------------------------------------

def normalize_image(img: np.ndarray) -> np.ndarray:
    """uint8 [..., H, W, 3] -> float32 [..., 3, H, W] in [-0.5, 0.5]."""
    x = img.astype(np.float32) / 255.0 - 0.5
    return np.moveaxis(x, -1, -3)


def denormalize_image(x: np.ndarray) -> np.ndarray:
    """Inverse of normalize_image (clipped to valid uint8)."""
    img = np.moveaxis(x, -3, -1)
    return np.clip(np.round((img + 0.5) * 255.0), 0, 255).astype(np.uint8)


def normalize_pos(xy: np.ndarray) -> np.ndarray:
    """Positions / actions in [0, 512] px -> [-1, 1]."""
    return np.asarray(xy, np.float32) / 256.0 - 1.0


def denormalize_pos(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> [0, 512] px."""
    return (np.asarray(x, np.float32) + 1.0) * 256.0


def angle_features(theta: np.ndarray) -> np.ndarray:
    """Angle (rad, wrapped or not) -> [..., 2] = (sin, cos). Never feed raw angles to a model."""
    theta = np.asarray(theta, np.float32)
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)

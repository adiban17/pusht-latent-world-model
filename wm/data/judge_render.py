"""Rendered frames for the judges only (Phase 5b, pre-registered judge decision).

Micro-episodes: a random valid start state (block CoG uniform in [lo, hi]^2, angle uniform in
[-pi, pi), pusher uniform on the table and not overlapping the block), then `steps - 1` random
actions (target = p + N(0, sigma^2), clipped to the table). Frames, states and actions follow the
pusht_*.npz layout of collect.py (frame i is before action i; the last frame has zero action),
so FrameData / WindowData read them unchanged. Only the judges (probe_mix, latent MLP) train on
these; wm, wm_1step, pixel and bc never see them.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wm.envs.pusht import get_state, render64, reset_to_state, wrap_angle_rad
from wm.eval.angle_audit import pose_state

TABLE = 512.0
PUSHER_MARGIN = 20.0       # pusher centre stays this far from the table edge (radius 15 + wall)
PUSHER_RADIUS = 15.0
CLEARANCE = 2.0            # min gap (px) between the pusher's edge and the block


def pusher_clearance(env) -> float:
    """Gap (px) between the pusher's circle and the T (negative = overlap), from pymunk's
    signed point-to-shape distance of the pusher centre to each of the T's two polygons."""
    u = env.unwrapped
    return min(sh.point_query(u.agent.position).distance for sh in u.block.shapes) - PUSHER_RADIUS


def sample_valid_state(env, rng: np.random.Generator, cog_lo: float, cog_hi: float,
                       max_tries: int = 100) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """(state, first observation, tries) for a random pose with the pusher clear of the block
    (>= CLEARANCE px; checked geometrically, since one physics step does not separate an
    overlapping pair) whose reset round-trips (pusher and block < 1 px, angle < 0.5 deg)."""
    for tries in range(1, max_tries + 1):
        cog = rng.uniform(cog_lo, cog_hi, 2)
        theta = rng.uniform(-np.pi, np.pi)
        pusher = rng.uniform(PUSHER_MARGIN, TABLE - PUSHER_MARGIN, 2)
        state = pose_state((cog[0], cog[1]), theta, (pusher[0], pusher[1]))
        obs = reset_to_state(env, state)
        got = get_state(env).astype(np.float64)
        d_ang = abs(np.degrees(wrap_angle_rad(got[4] - state[4])))
        if pusher_clearance(env) < CLEARANCE:
            continue
        if (np.abs(got[:2] - state[:2]).max() < 1.0 and np.abs(got[2:4] - state[2:4]).max() < 1.0
                and d_ang < 0.5):
            return got, obs, tries
    raise RuntimeError(f"no valid state in {max_tries} tries")


def micro_episode(env, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """One micro-episode of cfg['steps'] frames (cfg['steps'] - 1 random actions)."""
    rng = np.random.default_rng(seed)
    n = int(cfg["steps"])
    state, obs, tries = sample_valid_state(env, rng, float(cfg["cog_low"]), float(cfg["cog_high"]))
    frames = np.zeros((n, 64, 64, 3), np.uint8)
    states = np.zeros((n, 5), np.float32)
    actions = np.zeros((n, 2), np.float32)
    for i in range(n):
        frames[i] = render64(obs["pixels"])
        states[i] = get_state(env)
        if i == n - 1:
            break
        p = states[i, :2].astype(np.float64)
        actions[i] = np.clip(p + rng.normal(0.0, float(cfg["action_sigma"]), 2), 0.0, TABLE)
        obs, *_ = env.step(actions[i])
    return {"frames": frames, "states": states, "actions": actions, "tries": tries}


def render_split(env, seeds: list[int], cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Micro-episodes for `seeds` packed into the pusht_*.npz arrays (+ rejection stats)."""
    eps = [micro_episode(env, s, cfg) for s in seeds]
    lengths = np.array([len(e["frames"]) for e in eps], np.int64)
    arrays = {"frames": np.concatenate([e["frames"] for e in eps]),
              "states": np.concatenate([e["states"] for e in eps]),
              "actions": np.concatenate([e["actions"] for e in eps]),
              "episode_starts": np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64),
              "episode_lengths": lengths,
              "episode_seeds": np.array(seeds, np.int64)}
    arrays["clean_actions"] = arrays["actions"].copy()
    tries = np.array([e["tries"] for e in eps])
    return arrays, {"episodes": len(eps), "frames": int(lengths.sum()),
                    "mean_tries": float(tries.mean()), "max_tries": int(tries.max())}

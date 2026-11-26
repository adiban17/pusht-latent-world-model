"""Roll the scripted expert in PushT and pack episodes into arrays (CLAUDE.md §3).

Each episode stores steps + 1 frames: frame i is the observation *before* action i, and the
last frame (after the final action) has zero actions. Episodes never end early.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from wm.data.expert import ScriptedPusher
from wm.envs.pusht import block_cog, circular_std_deg, get_state, make_env, render64, wrap_angles_rad

CONTACT_POS_PX = 0.5
CONTACT_ANG_DEG = 0.5
GRID = 8
TABLE = 512.0


class StepRefused(RuntimeError):
    """gym-pusht raised while stepping after it reported `terminated`."""


def seed_list(lo_hi: list[int]) -> list[int]:
    """Inclusive [lo, hi] -> [lo, ..., hi]."""
    lo, hi = lo_hi
    return list(range(int(lo), int(hi) + 1))


def collect_episode(env, seed: int, expert_cfg: dict[str, Any], steps: int) -> dict[str, Any]:
    """One episode of exactly `steps` actions and `steps + 1` frames."""
    obs, _ = env.reset(seed=seed)
    expert = ScriptedPusher(expert_cfg, np.random.default_rng(seed))
    frames = np.zeros((steps + 1, 64, 64, 3), np.uint8)
    states = np.zeros((steps + 1, 5), np.float32)
    actions = np.zeros((steps + 1, 2), np.float32)
    clean = np.zeros((steps + 1, 2), np.float32)
    terminated_seen = False
    for i in range(steps + 1):
        frames[i] = render64(obs["pixels"])
        states[i] = get_state(env)
        if i == steps:
            break
        actions[i], clean[i] = expert.act(states[i])
        try:
            obs, _, terminated, _, _ = env.step(actions[i])
        except Exception as e:
            if terminated_seen:
                raise StepRefused(f"seed {seed}: step {i} after termination: {e}") from e
            raise
        terminated_seen = terminated_seen or bool(terminated)
    return {"frames": frames, "states": states, "actions": actions, "clean_actions": clean,
            "terminated": terminated_seen, "sigma": expert.sigma, "eps": expert.eps}


def collect_split(seeds: list[int], expert_cfg: dict[str, Any], steps: int,
                  replacement_start: int = 2000, env=None) -> tuple[dict[str, np.ndarray], dict]:
    """Collect episodes for `seeds` and concatenate them into the §3 arrays (+ episode_seeds)."""
    own_env = env is None
    env = make_env() if own_env else env
    episodes, used_seeds, replaced = [], [], []
    next_replacement = replacement_start
    t0 = time.time()
    for seed in seeds:
        try:
            ep = collect_episode(env, seed, expert_cfg, steps)
        except StepRefused:
            replaced.append({"seed": seed, "replacement": next_replacement})
            seed, next_replacement = next_replacement, next_replacement + 1
            ep = collect_episode(env, seed, expert_cfg, steps)
        episodes.append(ep)
        used_seeds.append(seed)
    if own_env:
        env.close()
    lengths = np.array([len(e["frames"]) for e in episodes], np.int64)
    arrays = {
        "frames": np.concatenate([e["frames"] for e in episodes]),
        "states": np.concatenate([e["states"] for e in episodes]),
        "actions": np.concatenate([e["actions"] for e in episodes]),
        "clean_actions": np.concatenate([e["clean_actions"] for e in episodes]),
        "episode_starts": np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64),
        "episode_lengths": lengths,
        "episode_seeds": np.array(used_seeds, np.int64),
    }
    info = {
        "seconds": time.time() - t0,
        "replaced": replaced,
        "terminated_episodes": int(sum(e["terminated"] for e in episodes)),
        "sigmas": [e["sigma"] for e in episodes],
        "eps": [e["eps"] for e in episodes],
    }
    return arrays, info


def transitions(arrays: dict[str, np.ndarray]) -> int:
    """Number of (frame, action, next frame) transitions = frames - episodes."""
    return int(arrays["episode_lengths"].sum() - len(arrays["episode_lengths"]))


def step_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Bool [F]: True where frame i has a successor in the same episode."""
    mask = np.ones(len(arrays["states"]), bool)
    mask[arrays["episode_starts"] + arrays["episode_lengths"] - 1] = False
    return mask


def health(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Contact ratio, coverage and action statistics of one split."""
    states, mask = arrays["states"], step_mask(arrays)
    cog = block_cog(states)
    move = np.linalg.norm(cog[1:] - cog[:-1], axis=1)
    turn = np.abs(np.degrees(wrap_angles_rad(states[1:, 4] - states[:-1, 4])))
    contact = ((move > CONTACT_POS_PX) | (turn > CONTACT_ANG_DEG))[mask[:-1]]

    cells = np.clip((cog / (TABLE / GRID)).astype(int), 0, GRID - 1)
    visited = len({(int(x), int(y)) for x, y in cells})

    acts, clean = arrays["actions"][mask], arrays["clean_actions"][mask]
    at_edge = (acts <= 0.0) | (acts >= TABLE)
    angle_std_deg, angle_r = circular_std_deg(states[:, 4])  # wraps; unbounded (see docstring)
    return {
        "transitions": transitions(arrays),
        "episodes": len(arrays["episode_lengths"]),
        "contact_ratio": float(contact.mean()),
        "coverage_8x8": visited / GRID**2,
        "block_angle_std_deg": angle_std_deg,
        "block_angle_resultant_R": angle_r,
        "block_step_px_mean": float(move[mask[:-1]].mean()),
        "action_mean": [round(float(v), 2) for v in acts.mean(0)],
        "action_std": [round(float(v), 2) for v in acts.std(0)],
        "action_min": float(acts.min()),
        "action_max": float(acts.max()),
        "action_clip_frac": float(at_edge.any(1).mean()),
        "noisy_clean_gap_px": float(np.linalg.norm(acts - clean, axis=1).mean()),
    }

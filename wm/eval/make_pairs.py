"""Evaluation pairs (CLAUDE.md §9 Pairs): (start frame, goal frame = start + offset) of one episode.

Candidates: every episode of the split, starts t in {0, 5, ..., 75}; keep a pair if the block's
centre of gravity moves >= 30 px or the block turns >= 20 deg between start and goal; then a
seeded subset of n pairs. Each pair stores the start/goal states and global frame indices.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wm.envs.pusht import pose_error


def candidate_pairs(arrays: dict[str, np.ndarray], episodes: np.ndarray, offset: int = 20,
                    start_step: int = 5, start_max: int = 75, min_move_px: float = 30.0,
                    min_turn_deg: float = 20.0) -> list[dict[str, Any]]:
    """All pairs of the given episodes (bool mask over episodes) that pass the motion filter."""
    states = arrays["states"]
    out = []
    for e in np.flatnonzero(episodes):
        s0, n = int(arrays["episode_starts"][e]), int(arrays["episode_lengths"][e])
        for t in range(0, start_max + 1, start_step):
            if t + offset > n - 1:
                break
            i, g = s0 + t, s0 + t + offset
            move, turn = pose_error(states[g], states[i])
            if move >= min_move_px or abs(turn) >= min_turn_deg:
                out.append({"episode_seed": int(arrays["episode_seeds"][e]), "t": t, "offset": offset,
                            "start_idx": i, "goal_idx": g,
                            "start_state": [float(x) for x in states[i]],
                            "goal_state": [float(x) for x in states[g]],
                            "move_px": float(move), "turn_deg": float(turn)})
    return out


def make_pairs(arrays: dict[str, np.ndarray], episodes: np.ndarray, n: int, seed: int = 0,
               **filter_kw: Any) -> tuple[list[dict[str, Any]], int]:
    """(seeded subset of n candidate pairs in candidate order, number of candidates)."""
    cands = candidate_pairs(arrays, episodes, **filter_kw)
    if len(cands) < n:
        raise ValueError(f"only {len(cands)} candidate pairs, need {n}")
    pick = np.sort(np.random.default_rng(seed).choice(len(cands), n, replace=False))
    return [{"pair": j, **cands[i]} for j, i in enumerate(pick)], len(cands)

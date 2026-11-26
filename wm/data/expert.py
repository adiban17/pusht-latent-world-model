"""Scripted pusher that generates the training data (CLAUDE.md §2).

"Block position" b is the T's centre of gravity (see wm.envs.pusht.block_cog), so the contact
point sits behind the middle of the T rather than behind the top edge of its bar.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wm.envs.pusht import block_cog

TABLE = 512.0


def unit(v: np.ndarray) -> np.ndarray:
    """v / |v| (zero vector stays zero)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def rot90(v: np.ndarray) -> np.ndarray:
    """v rotated by +90 degrees."""
    return np.array([-v[1], v[0]])


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """v scaled down so that |v| <= max_norm."""
    n = float(np.linalg.norm(v))
    return v * (max_norm / n) if n > max_norm else v


def contact_point(b: np.ndarray, d: np.ndarray, u: float, offset: float) -> np.ndarray:
    """c = b - offset * d + u * n, with n = d rotated 90 degrees."""
    return b - offset * d + u * rot90(d)


def detour_point(b: np.ndarray, d: np.ndarray, side: float, dist: float) -> np.ndarray:
    """b + side * dist * n (side = +1 or -1)."""
    return b + side * dist * rot90(d)


def nearer_side(p: np.ndarray, b: np.ndarray, d: np.ndarray) -> float:
    """+1 if p is on the +n side of the block, else -1."""
    return 1.0 if float(np.dot(p - b, rot90(d))) >= 0 else -1.0


class ScriptedPusher:
    """Pushes the T towards random sub-goals; returns (noisy_action, clean_action) in px."""

    def __init__(self, cfg: dict[str, Any], rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        """Draw this episode's noise level and random-jump probability."""
        c = self.cfg
        self.sigma = float(self.rng.choice(c["noise_sigmas"]))
        self.eps = float(self.rng.choice(c["random_eps"]))
        self.t = 0
        self.subgoal = np.zeros(2)
        self.u = 0.0
        self.detour_side: float | None = None   # set while detouring around the block
        self.after_detour = False               # detour reached; head to c without re-checking

    def _new_segment(self) -> None:
        c = self.cfg
        self.subgoal = self.rng.uniform(c["subgoal_low"], c["subgoal_high"], size=2)
        self.u = float(self.rng.uniform(-c["lateral_max"], c["lateral_max"]))

    def clean_target(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Noise-free target (§2 steps 1-4) for pusher p and block centre b; updates detour state."""
        c = self.cfg
        d = unit(self.subgoal - b)
        contact = contact_point(b, d, self.u, c["contact_offset"])
        far_side = float(np.dot(p - b, d)) > 0
        if not far_side:
            self.after_detour = False
        if far_side and self.detour_side is None and not self.after_detour:
            self.detour_side = nearer_side(p, b, d)
        if self.detour_side is not None:
            way = detour_point(b, d, self.detour_side, c["detour_dist"])
            if np.linalg.norm(way - p) > c["reach_px"]:
                return p + clip_norm(way - p, c["approach_step"])
            self.detour_side, self.after_detour = None, True
        if np.linalg.norm(contact - p) > c["reach_px"]:
            return p + clip_norm(contact - p, c["approach_step"])
        return p + c["push_step"] * d

    def act(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One step from the env state [pusher x, y, block x, y, angle]."""
        if self.t % self.cfg["subgoal_every"] == 0:
            self._new_segment()
        self.t += 1
        p = np.asarray(state[:2], np.float64)
        target = self.clean_target(p, block_cog(state))
        clean = np.clip(target, 0.0, TABLE)
        noisy = target + self.rng.normal(0.0, self.sigma, size=2)
        if self.rng.random() < self.eps:
            jump = self.cfg["random_jump"]
            noisy = p + self.rng.uniform(-jump, jump, size=2)
        noisy = np.clip(noisy, 0.0, TABLE)
        return noisy.astype(np.float32), clean.astype(np.float32)

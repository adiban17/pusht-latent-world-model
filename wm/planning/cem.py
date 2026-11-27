"""Batched cross-entropy method over waypoint deltas (CLAUDE.md §8).

Decision variables δ ∈ R^{H×2} in px, each clipped to ±delta_clip; waypoints
w_k = clip(p + Σ_{j≤k} δ_j, 0, 512); the model action is w_k / 256 − 1.
Each round samples N candidates (one is always the current mean), keeps the K cheapest, and
refits μ, σ to them (σ floored). Returns the cheapest candidate seen over all rounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class CEMConfig:
    candidates: int = 128
    horizon: int = 10
    rounds: int = 4
    elites: int = 12
    delta_clip: float = 50.0
    sigma_init: float = 30.0
    sigma_floor: float = 2.0
    beta: float = 0.1
    warm_start: bool = True
    cost_space: str = "latent"   # Phase 6b diagnostic: "latent" (default, frozen) | "pixel"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CEMConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def deltas_to_waypoints(p: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Pusher p [2] px, deltas [N, H, 2] px -> waypoints [N, H, 2] px, clipped to [0, 512]."""
    return (p.to(deltas) + deltas.cumsum(dim=1)).clamp(0.0, 512.0)


def waypoints_to_actions(w: torch.Tensor) -> torch.Tensor:
    """Waypoints px -> model actions in [-1, 1] (normalize_pos)."""
    return w / 256.0 - 1.0


def horizon_cost(d: torch.Tensor, beta: float) -> torch.Tensor:
    """Per-step distances d [N, H] -> d_H + β Σ_{k=1}^{H−1} d_k [N]."""
    return d[:, -1] + beta * d[:, :-1].sum(-1)


def latent_distances(z_hat: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """||ẑ_k − z_g||²; z_hat [N, H, D], z_goal [D] -> [N, H]."""
    return ((z_hat.float() - z_goal.float()) ** 2).sum(-1)


def latent_cost(z_hat: torch.Tensor, z_goal: torch.Tensor, beta: float) -> torch.Tensor:
    """||ẑ_H − z_g||² + β Σ_{k=1}^{H−1} ||ẑ_k − z_g||²; z_hat [N, H, D], z_goal [D] -> [N]."""
    return horizon_cost(latent_distances(z_hat, z_goal), beta)


def pixel_distances(frames_hat: torch.Tensor, goal: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Phase 6b: mean over pixels and channels of w · (clip(frame) − goal)²;
    frames_hat [N, H, 3, 64, 64], goal [3, 64, 64], weight [1, 64, 64] -> [N, H]."""
    x = frames_hat.float().clamp(-0.5, 0.5)
    return (weight * (x - goal.float()) ** 2).mean(dim=(-3, -2, -1))


def pixel_cost(frames_hat: torch.Tensor, goal: torch.Tensor, weight: torch.Tensor, beta: float) -> torch.Tensor:
    """Same structure as latent_cost with the weighted pixel distance -> [N]."""
    return horizon_cost(pixel_distances(frames_hat, goal, weight), beta)


@torch.no_grad()
def cem(score: Callable[[torch.Tensor], torch.Tensor], mu0: torch.Tensor, cfg: CEMConfig,
        gen: torch.Generator) -> dict[str, Any]:
    """Minimise score(deltas [N, H, 2]) -> costs [N]; mu0 [H, 2] px (CPU).

    Returns best (deltas [H, 2]), best_cost, best_index (in its round's batch), mu, sigma and
    per-round minimum cost. Sampling uses the CPU generator `gen`, so results do not depend on
    the device that `score` runs on.
    """
    h = mu0.shape[0]
    mu = mu0.float().clamp(-cfg.delta_clip, cfg.delta_clip)
    sigma = torch.full((h, 2), float(cfg.sigma_init))
    best, best_cost, round_min = mu.clone(), float("inf"), []
    for _ in range(cfg.rounds):
        noise = torch.randn((cfg.candidates - 1, h, 2), generator=gen)
        cand = torch.cat([mu[None], mu + sigma * noise]).clamp(-cfg.delta_clip, cfg.delta_clip)
        costs = score(cand).float().cpu()
        i = int(torch.argmin(costs))
        round_min.append(float(costs[i]))
        if costs[i] < best_cost:
            best, best_cost = cand[i].clone(), float(costs[i])
        elite = cand[torch.topk(costs, cfg.elites, largest=False).indices]
        mu = elite.mean(0)
        sigma = elite.std(0, unbiased=False).clamp_min(cfg.sigma_floor)
    return {"best": best, "best_cost": best_cost, "mu": mu, "sigma": sigma, "round_min": round_min}


def shift_warm_start(prev: torch.Tensor | None, horizon: int) -> torch.Tensor:
    """Previous best shifted by one step (last = 0), or zeros; resized to `horizon`."""
    out = torch.zeros(horizon, 2)
    if prev is not None:
        n = min(horizon, prev.shape[0] - 1)
        out[:n] = prev[1:1 + n]
    return out

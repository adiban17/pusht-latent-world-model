"""State probe (CLAUDE.md §7): the "judge" that reads the state back from a (real or predicted) frame.

Encoder (§4 design, fresh weights) -> Linear 256 -> 6: pusher (x, y), block centre of gravity
(x, y) — both normalised x/256 - 1 — sin θ, cos θ. 1,740,518 params.
Block position is the T's centre of gravity, as in pose_error (Phase 2 decision).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from wm.envs.pusht import block_cog
from wm.models.encoder import Encoder

EXPECTED_PARAMS = {"encoder": 1_738_976, "head": 1_542, "total": 1_740_518}


class StateProbe(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.head = nn.Linear(latent_dim, 6)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [..., 3, 64, 64] -> raw outputs [..., 6] (normalised targets)."""
        return self.head(self.encoder(frames))

    @torch.no_grad()
    def predict_state(self, frames: torch.Tensor) -> dict[str, np.ndarray]:
        """frames [..., 3, 64, 64] -> pusher px, block (CoG) px, angle rad."""
        return decode_targets(self.forward(frames).float().cpu().numpy())


def decode_targets(y: np.ndarray) -> dict[str, np.ndarray]:
    """[..., 6] normalised -> {"pusher": px [..., 2], "block": CoG px [..., 2], "angle": rad [...]}."""
    y = np.asarray(y, np.float64)
    return {"pusher": (y[..., 0:2] + 1.0) * 256.0, "block": (y[..., 2:4] + 1.0) * 256.0,
            "angle": np.arctan2(y[..., 4], y[..., 5])}


def probe_errors(pred: dict[str, np.ndarray], states: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame errors vs true states [..., 5]: block CoG px, |block angle| deg (wrapped), pusher px."""
    s = np.asarray(states, np.float64)
    d_ang = np.degrees(pred["angle"] - s[..., 4])
    d_ang = (d_ang + 180.0) % 360.0 - 180.0
    return {"block_px": np.linalg.norm(pred["block"] - block_cog(s), axis=-1),
            "block_deg": np.abs(d_ang),
            "pusher_px": np.linalg.norm(pred["pusher"] - s[..., :2], axis=-1)}


def param_counts(model: StateProbe) -> dict[str, int]:
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"encoder": n(model.encoder), "head": n(model.head), "total": n(model)}


def build_probe(cfg: dict[str, Any]) -> StateProbe:
    return StateProbe(latent_dim=cfg["model"]["latent_dim"])

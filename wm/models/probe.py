"""State probe (CLAUDE.md §7): the "judge" that reads the state back from a (real or predicted) frame.

Encoder (§4 design, fresh weights) -> Linear 256 -> 6: pusher (x, y), block centre of gravity
(x, y) — both normalised x/256 - 1 — sin θ, cos θ. 1,740,518 params.
Block position is the T's centre of gravity, as in pose_error (Phase 2 decision).

`model.pool` (Phase 5b, judges only): "flatten" (default) is the §7 design above; "coord_gap" is
the documented fallback for probe_mix, used only if the flatten judge cannot read the angle:
CoordConv (x, y channels appended to the input), the same 4 convs, then global average pooling
of the 4x4x256 map in place of flatten -> Linear, then the same non-affine LayerNorm and head.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from wm.envs.pusht import block_cog
from wm.models.encoder import CHANNELS, Encoder, latent_norm

EXPECTED_PARAMS = {"encoder": 1_738_976, "head": 1_542, "total": 1_740_518}
EXPECTED_PARAMS_COORD_GAP = {"encoder": 691_168, "head": 1_542, "total": 692_710}


class CoordGapEncoder(nn.Module):
    """CoordConv input (+2 channels: x, y in [-1, 1]), 4 x [Conv(k4, s2, p1) + SiLU]
    5->32->64->128->256, global average pooling -> 256, non-affine LayerNorm."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        assert latent_dim == CHANNELS[-1], "coord_gap pools the last conv's channels"
        chans = (CHANNELS[0] + 2,) + CHANNELS[1:]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1), nn.SiLU()]
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), 0).expand(b, -1, -1, -1)
        z = self.convs(torch.cat([x, grid], 1)).mean(dim=(-2, -1))
        return latent_norm(z).reshape(*lead, -1)


class StateProbe(nn.Module):
    def __init__(self, latent_dim: int = 256, pool: str = "flatten"):
        super().__init__()
        if pool not in ("flatten", "coord_gap"):
            raise ValueError(f"unknown probe pool '{pool}'")
        self.pool = pool
        self.encoder = Encoder(latent_dim) if pool == "flatten" else CoordGapEncoder(latent_dim)
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


def expected_params(pool: str = "flatten") -> dict[str, int]:
    return EXPECTED_PARAMS if pool == "flatten" else EXPECTED_PARAMS_COORD_GAP


def build_probe(cfg: dict[str, Any]) -> StateProbe:
    return StateProbe(latent_dim=cfg["model"]["latent_dim"], pool=cfg["model"].get("pool", "flatten"))

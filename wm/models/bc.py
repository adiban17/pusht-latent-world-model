"""Goal-conditioned behaviour cloning (CLAUDE.md §6).

3 context frames + the goal frame go through one shared encoder (the §4 design, fresh weights)
-> 4 x 256; + pusher (x, y) normalised -> 1,026 -> MLP 512 -> 512 -> 2 (SiLU, dropout 0.1).
Output (dx, dy) / 50 px relative to the current pusher position. 2,528,482 params.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from wm.models.encoder import Encoder

DELTA_SCALE = 50.0
EXPECTED_PARAMS = {"encoder": 1_738_976, "mlp": 789_506, "total": 2_528_482}


class BCPolicyNet(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden: int = 512, context: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.context = context
        self.encoder = Encoder(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear((context + 1) * latent_dim + 2, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2))

    def forward(self, images: torch.Tensor, pusher: torch.Tensor) -> torch.Tensor:
        """images [B, C+1, 3, 64, 64] (context oldest first, then goal), pusher [B, 2] in [-1, 1]
        -> (dx, dy) / 50 [B, 2]."""
        z = self.encoder(images).flatten(1)
        return self.mlp(torch.cat([z, pusher], dim=1))


def delta_to_target(pusher_px, out, scale: float = DELTA_SCALE):
    """Executed absolute target = clip(p + 50 * output, 0, 512) (numpy or torch)."""
    return (pusher_px + scale * out).clip(0.0, 512.0)


def param_counts(model: BCPolicyNet) -> dict[str, int]:
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"encoder": n(model.encoder), "mlp": n(model.mlp), "total": n(model)}


def build_bc(cfg: dict[str, Any]) -> BCPolicyNet:
    m = cfg["model"]
    return BCPolicyNet(latent_dim=m["latent_dim"], hidden=m["hidden"], context=m["context"],
                       dropout=m["dropout"])

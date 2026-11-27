"""Latent judge (Phase 5b): MLP on world-model latents z = E_wm(frame) -> the 6 targets of §7.

256 -> 512 -> 512 -> 6 with SiLU and dropout 0.1 (397,318 params). It reads the same targets as
the image probe (pusher xy, block CoG xy normalised x/256 - 1, sin, cos), so `decode_targets`
and `probe_errors` apply. Trained on E_wm of real train frames + rendered judge frames; applied
to real E_wm(o) (floor), to imagined z_hat, and to E_wm of every method's predicted frames.
"""
from __future__ import annotations

from typing import Any

from torch import nn

EXPECTED_PARAMS = {"total": 397_318}


class LatentProbe(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 6))

    def forward(self, z):
        """z [..., 256] -> raw outputs [..., 6] (normalised targets)."""
        return self.net(z)


def param_counts(model: LatentProbe) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def build_latent_probe(cfg: dict[str, Any]) -> LatentProbe:
    m = cfg["model"]
    return LatentProbe(latent_dim=m["latent_dim"], hidden=m["hidden"], dropout=m["dropout"])

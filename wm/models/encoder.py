"""CNN encoder (CLAUDE.md §4): 64x64 RGB -> 256-d latent on the unit LayerNorm sphere."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CHANNELS = (3, 32, 64, 128, 256)


def latent_norm(z: torch.Tensor) -> torch.Tensor:
    """Non-affine LayerNorm over the last dim (same as nn.LayerNorm(d, elementwise_affine=False))."""
    return F.layer_norm(z, z.shape[-1:])


class Encoder(nn.Module):
    """4 x [Conv(k4, s2, p1) + SiLU] 3->32->64->128->256 (64->4 px), flatten, Linear 4096->256, LN."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        layers: list[nn.Module] = []
        for c_in, c_out in zip(CHANNELS[:-1], CHANNELS[1:]):
            layers += [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1), nn.SiLU()]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(CHANNELS[-1] * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., 3, 64, 64] (normalised to [-0.5, 0.5]) -> z [..., 256]."""
        lead = x.shape[:-3]
        h = self.convs(x.reshape(-1, *x.shape[-3:])).flatten(1)
        return latent_norm(self.fc(h)).reshape(*lead, -1)

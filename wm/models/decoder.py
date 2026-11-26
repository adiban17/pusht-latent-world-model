"""Decoder (CLAUDE.md §4): 256-d latent -> 64x64 RGB in normalised pixel space."""
from __future__ import annotations

import torch
from torch import nn

CHANNELS = (256, 128, 64, 32, 3)


class Decoder(nn.Module):
    """Linear 256->4096, reshape 256x4x4, 4 x ConvTranspose(k4, s2, p1) 256->128->64->32->3.

    SiLU after the linear layer and between the transposed convs; none at the end
    (the output is the normalised image, x/255 - 0.5).
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, CHANNELS[0] * 4 * 4)
        layers: list[nn.Module] = [nn.SiLU()]
        pairs = list(zip(CHANNELS[:-1], CHANNELS[1:]))
        for i, (c_in, c_out) in enumerate(pairs):
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            if i < len(pairs) - 1:
                layers.append(nn.SiLU())
        self.deconvs = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [..., 256] -> x [..., 3, 64, 64]."""
        lead = z.shape[:-1]
        h = self.fc(z.reshape(-1, z.shape[-1])).reshape(-1, CHANNELS[0], 4, 4)
        return self.deconvs(h).reshape(*lead, 3, 64, 64)

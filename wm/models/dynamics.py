"""Action-conditioned latent dynamics (CLAUDE.md §4): residual GRU in latent space."""
from __future__ import annotations

import torch
from torch import nn

from wm.models.encoder import latent_norm


class Dynamics(nn.Module):
    """phi = Linear(256+2 -> 256) + SiLU; GRUCell(256, 320); out = Linear(320 -> 256) gives delta.

    One step: h' = GRU(phi([z, a]), h);  z' = LN(z + out(h')).
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 320):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(nn.Linear(latent_dim + action_dim, latent_dim), nn.SiLU())
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def init_hidden(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)

    def observe(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update the hidden state with (z, a) without predicting: h' = GRU(phi([z, a]), h)."""
        x = self.phi(torch.cat([z, a], dim=-1))
        return self.gru(x, h.to(x.dtype))

    def step(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One imagined step: returns (z_next, h_next)."""
        h = self.observe(z, a, h)
        return latent_norm(z + self.out(h)), h

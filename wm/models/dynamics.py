"""Action-conditioned latent dynamics (CLAUDE.md §4): residual GRU in latent space."""
from __future__ import annotations

import torch
from torch import nn

from wm.models.encoder import latent_norm


class Dynamics(nn.Module):
    """phi = Linear(256+2 -> 256) + SiLU; GRUCell(256, 320); out = Linear(320 -> 256) gives delta.

    One step: h' = GRU(phi([z, a]), h);  z' = LN(z + out(h')).

    proprio=True (Phase 3b option B): the pusher position p (normalised, 2-d) is part of the
    model state. phi takes [z, p, a] (256+2+2 -> 256) and a second head p_out = Linear(320 -> 2)
    predicts its change: p' = p + p_out(h') (p_out starts at zero).
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 320,
                 proprio: bool = False, proprio_dim: int = 2):
        super().__init__()
        self.hidden_dim, self.proprio = hidden_dim, proprio
        in_dim = latent_dim + action_dim + (proprio_dim if proprio else 0)
        self.phi = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.SiLU())
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim)
        if proprio:
            # zero init: p' = p (copy-last) at the start, so a random head's large early pusher
            # error doesn't swamp the other terms under grad-clip (Phase 3b decision)
            self.p_out = nn.Linear(hidden_dim, proprio_dim)
            nn.init.zeros_(self.p_out.weight)
            nn.init.zeros_(self.p_out.bias)

    def init_hidden(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)

    def _inputs(self, z: torch.Tensor, a: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor:
        if self.proprio:
            if p is None:
                raise ValueError("this dynamics model has proprio=True: pass the pusher position p")
            return torch.cat([z, p.to(z.dtype), a.to(z.dtype)], dim=-1)
        return torch.cat([z, a], dim=-1)

    def observe(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor,
                p: torch.Tensor | None = None) -> torch.Tensor:
        """Update the hidden state with (z, [p,] a) without predicting: h' = GRU(phi(...), h)."""
        x = self.phi(self._inputs(z, a, p))
        return self.gru(x, h.to(x.dtype))

    def step(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One imagined step: returns (z_next, h_next)."""
        h = self.observe(z, a, h)
        return latent_norm(z + self.out(h)), h

    def step_proprio(self, z: torch.Tensor, p: torch.Tensor, a: torch.Tensor,
                     h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagined step of the proprio model: returns (z_next, p_next, h_next)."""
        h = self.observe(z, a, h, p)
        return latent_norm(z + self.out(h)), p + self.p_out(h).to(p.dtype), h

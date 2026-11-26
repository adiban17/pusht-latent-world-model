"""Rollout metrics (CLAUDE.md §9). Phase 3 needs only the latent NMSE; Phase 5 adds the rest."""
from __future__ import annotations

import torch


def latent_spread(z_ref: torch.Tensor) -> float:
    """mean ||z - z_bar||^2 over reference latents z_ref [M, D] (z_bar = their mean)."""
    z_ref = z_ref.float()
    return float(((z_ref - z_ref.mean(0)) ** 2).sum(-1).mean())


def latent_nmse(z_hat: torch.Tensor, z_true: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    """NMSE_k = mean_windows ||z_hat_k - z_k||^2 / mean ||z - z_bar||^2, for [N, H, D] -> [H]."""
    err = ((z_hat.float() - z_true.float()) ** 2).sum(-1).mean(0)
    return err / latent_spread(z_ref)

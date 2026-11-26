"""Rollout metrics (CLAUDE.md §9 Rollouts): open-loop 10-step predictions of every method, scored
physically by the state probe, in the wm encoder's latent space, and in pixels; bootstrap CIs.

Window = 13 frames o[t-2..t+10], 12 actions a[t-2..t+9]; context C = 3, k = 1..H (H = 10).
Every predicted frame is clipped to the valid normalised pixel range [-0.5, 0.5] before the
probe, the wm encoder or the pixel MSE see it (Phase 5 decision). The pixel U-Net's own
autoregressive loop is left as trained (unclipped).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.models.pixel_unet import PixelUNet
from wm.models.probe import StateProbe, decode_targets, probe_errors
from wm.models.world_model import WorldModel

PIXEL_LO, PIXEL_HI = -0.5, 0.5      # range of real normalised frames (x/255 - 0.5)
PROBE_KEYS = ("block_px", "block_deg", "pusher_px")


def latent_spread(z_ref: torch.Tensor) -> float:
    """mean ||z - z_bar||^2 over reference latents z_ref [M, D] (z_bar = their mean)."""
    z_ref = z_ref.float()
    return float(((z_ref - z_ref.mean(0)) ** 2).sum(-1).mean())


def latent_nmse(z_hat: torch.Tensor, z_true: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    """NMSE_k = mean_windows ||z_hat_k - z_k||^2 / mean ||z - z_bar||^2, for [N, H, D] -> [H]."""
    err = ((z_hat.float() - z_true.float()) ** 2).sum(-1).mean(0)
    return err / latent_spread(z_ref)


def clip_frames(x: torch.Tensor) -> torch.Tensor:
    """Clip predicted frames to the valid normalised pixel range."""
    return x.float().clamp(PIXEL_LO, PIXEL_HI)


# ---- predictions --------------------------------------------------------------------------
@torch.no_grad()
def predict_wm(model: WorldModel, frames: torch.Tensor, actions: torch.Tensor,
               pusher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Open-loop rollout of a world model on windows.

    frames [B, C+H, 3, 64, 64], actions [B, C+H-1, 2], pusher [B, C+H, 2] (normalised) ->
    (clipped decoded frames [B, H, 3, 64, 64], z_hat [B, H, D], p_hat [B, H, 2] or None).
    Teacher forcing (wm_1step) affects training only: here every model feeds back its predictions.
    """
    model.eval()
    c = model.context
    z = model.encode(frames[:, :c])
    if model.proprio:
        h = model.warmup(z[:, :c - 1], actions[:, :c - 1], pusher[:, :c - 1])
        z_hat, p_hat = model.rollout(z[:, c - 1], h, actions[:, c - 1:], p0=pusher[:, c - 1])
    else:
        h = model.warmup(z[:, :c - 1], actions[:, :c - 1])
        z_hat, p_hat = model.rollout(z[:, c - 1], h, actions[:, c - 1:]), None
    return clip_frames(model.decode(z_hat)), z_hat.float(), p_hat


@torch.no_grad()
def predict_pixel(model: PixelUNet, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Autoregressive U-Net rollout from frames t-2..t -> clipped frames [B, H, 3, 64, 64]."""
    model.eval()
    c = model.context
    return clip_frames(model.rollout(frames[:, :c], actions[:, c - 1:]))


def predict_copy_last(frames: torch.Tensor, context: int) -> torch.Tensor:
    """Frame t repeated for k = 1..H."""
    h = frames.shape[1] - context
    return frames[:, context - 1:context].expand(-1, h, -1, -1, -1)


@torch.no_grad()
def recon_wm(model: WorldModel, frames: torch.Tensor) -> torch.Tensor:
    """Autoencoder reference: clip(D(E(o))) for frames [B, H, 3, 64, 64]."""
    model.eval()
    return clip_frames(model.decode(model.encode(frames)))


# ---- per-window scores [B, H] -----------------------------------------------------------
@torch.no_grad()
def probe_scores(probe: StateProbe, frames: torch.Tensor, states: np.ndarray) -> dict[str, np.ndarray]:
    """Probe errors (block CoG px, block angle deg, pusher px) on frames [B, H, 3, 64, 64]
    vs the true states [B, H, 5]."""
    probe.eval()
    b, h = frames.shape[:2]
    pred = probe(frames.reshape(b * h, *frames.shape[2:])).float().cpu().numpy().reshape(b, h, 6)
    return probe_errors(decode_targets(pred), states)


def latent_sq_err(z_pred: torch.Tensor, z_true: torch.Tensor) -> np.ndarray:
    """||z_pred - z_true||^2 per window and k: [B, H, D] -> [B, H]."""
    return ((z_pred.detach().float() - z_true.detach().float()) ** 2).sum(-1).cpu().numpy().astype(np.float64)


def pixel_mse(pred: torch.Tensor, real: torch.Tensor) -> np.ndarray:
    """Mean squared pixel error per window and k: [B, H, 3, 64, 64] -> [B, H]."""
    return ((pred.detach().float() - real.detach().float()) ** 2).mean(dim=(2, 3, 4)).cpu().numpy().astype(np.float64)


def pusher_px(p_norm: torch.Tensor | np.ndarray, states: np.ndarray) -> np.ndarray:
    """Pusher error in px of normalised positions [B, H, 2] vs true states [B, H, 5]."""
    p = p_norm.detach().double().cpu().numpy() if isinstance(p_norm, torch.Tensor) else np.asarray(p_norm, np.float64)
    return np.linalg.norm((p + 1.0) * 256.0 - np.asarray(states, np.float64)[..., :2], axis=-1)


# ---- bootstrap ------------------------------------------------------------------------------
def bootstrap_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """Resample indices [R, n] over windows (shared by every method -> paired CIs)."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def _ci(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    q = 100.0 * (1.0 - ci) / 2.0
    return np.percentile(boot, q, axis=0), np.percentile(boot, 100.0 - q, axis=0)


def mean_ci(x: np.ndarray, idx: np.ndarray, ci: float = 0.95) -> dict[str, list[float]]:
    """Per-k mean and percentile-bootstrap CI of per-window values x [N, H]."""
    x = np.asarray(x, np.float64)
    lo, hi = _ci(x[idx].mean(axis=1), ci)
    return {"mean": x.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def diff_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray, ci: float = 0.95) -> dict[str, list[float]]:
    """Paired difference mean(a) - mean(b) per k, with a bootstrap CI (same resamples)."""
    return mean_ci(np.asarray(a, np.float64) - np.asarray(b, np.float64), idx, ci)


def reduction_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray, ci: float = 0.95) -> dict[str, list[float]]:
    """Relative reduction 1 - mean(a) / mean(b) per k, with a paired bootstrap CI."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    point = 1.0 - a.mean(0) / b.mean(0)
    lo, hi = _ci(1.0 - a[idx].mean(axis=1) / b[idx].mean(axis=1), ci)
    return {"mean": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def at_k(series: dict[str, list[float]], k: int) -> dict[str, float]:
    """{mean, lo, hi} at k (1-based)."""
    return {key: series[key][k - 1] for key in ("mean", "lo", "hi")}


def summarize(scores: dict[str, dict[str, np.ndarray]], idx: np.ndarray, ci: float,
              spread: float) -> dict[str, dict[str, dict[str, list[float]]]]:
    """scores[metric][row] = per-window values [N, H] -> metrics[metric][row] = mean/lo/hi per k.

    The metric `latent_sq` becomes `nmse` (divided by the wm latent spread of the test frames).
    """
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for metric, rows in scores.items():
        name = "nmse" if metric == "latent_sq" else metric
        scale = 1.0 / spread if metric == "latent_sq" else 1.0
        out[name] = {row: mean_ci(v * scale, idx, ci) for row, v in rows.items()}
    return out


def all_finite(obj: Any) -> bool:
    """True if every float in a nested dict/list is finite."""
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    return not isinstance(obj, float) or bool(np.isfinite(obj))

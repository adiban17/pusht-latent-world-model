"""World-model diagnostics (job diagnose_wm): where does the train/val gap come from?

Loss terms per split, single-frame reconstruction (all pixels and non-background pixels),
and a ridge-regression probe from z to state, applied to real and imagined latents.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.windows import WindowData, frames_to_model, state_targets
from wm.models.probe import decode_targets, probe_errors
from wm.models.world_model import WorldModel
from wm.train.train_wm import open_loop_z_hat

TERMS = ("loss", "rec", "lat", "img", "var")
FG_THRESHOLD = 0.1   # a pixel is "non-background" if any channel differs from the mean frame by > 0.1 (~25/255)


def subset(starts: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """A fixed, seeded subset of n indices (all if fewer), sorted."""
    perm = torch.randperm(len(starts), generator=torch.Generator().manual_seed(seed))[:n]
    return starts[perm].sort().values


@torch.no_grad()
def window_terms(model: WorldModel, data: WindowData, starts: torch.Tensor,
                 weights: dict[str, float], chunk: int = 64) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Mean loss terms over windows, plus z [N, C+H, D] and open-loop z_hat [N, H, D] (CPU)."""
    model.eval()
    sums = {k: 0.0 for k in TERMS}
    zs, z_hats = [], []
    for i in range(0, len(starts), chunk):
        frames, actions = data.gather(starts[i:i + chunk])
        out = model.window_loss(frames, actions, weights)
        for k in TERMS:
            sums[k] += float(out[k]) * len(frames)
        zs.append(out["z"].float().cpu())
        z_hats.append(open_loop_z_hat(model, out, actions).float().cpu())
    return {k: v / len(starts) for k, v in sums.items()}, torch.cat(zs), torch.cat(z_hats)


@torch.no_grad()
def encode_frames(model: WorldModel, frames_u8: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """uint8 frames [N, 64, 64, 3] (on the model's device) -> z [N, D] on CPU."""
    model.eval()
    return torch.cat([model.encode(frames_to_model(frames_u8[i:i + chunk])).float().cpu()
                      for i in range(0, len(frames_u8), chunk)])


@torch.no_grad()
def recon_errors(model: WorldModel, frames_u8: torch.Tensor, mean_frame: torch.Tensor,
                 chunk: int = 512) -> dict[str, float]:
    """Encode -> decode MSE on single frames: all pixels, non-background pixels only, and the
    same masked MSE for "predict the mean frame" (reference)."""
    model.eval()
    sq = sq_fg = sq_fg_mean = 0.0
    n_px = n_fg = 0
    for i in range(0, len(frames_u8), chunk):
        x = frames_to_model(frames_u8[i:i + chunk])
        err = (model.decode(model.encode(x)).float() - x) ** 2               # [B, 3, H, W]
        mask = ((x - mean_frame).abs().amax(dim=1, keepdim=True) > FG_THRESHOLD).float()
        sq += float(err.sum())
        n_px += err.numel()
        sq_fg += float((err * mask).sum())
        sq_fg_mean += float((((mean_frame - x) ** 2) * mask).sum())
        n_fg += float(mask.sum()) * 3
    return {"mse": sq / n_px, "mse_foreground": sq_fg / max(n_fg, 1),
            "mean_frame_mse_foreground": sq_fg_mean / max(n_fg, 1),
            "foreground_fraction": n_fg / n_px, "frames": len(frames_u8)}


def fit_ridge(z: torch.Tensor, y: np.ndarray, lam: float = 1.0) -> torch.Tensor:
    """Closed-form ridge regression with an (unpenalised) bias: returns W [D+1, T] (float64)."""
    zb = torch.cat([z.double(), torch.ones(len(z), 1, dtype=torch.float64)], dim=1)
    reg = lam * torch.eye(zb.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(zb.T @ zb + reg, zb.T @ torch.from_numpy(np.asarray(y, np.float64)))


def ridge_predict(w: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    zb = torch.cat([z.double(), torch.ones(len(z), 1, dtype=torch.float64)], dim=1)
    return (zb @ w).numpy()


def state_error_summary(pred: np.ndarray, states: np.ndarray) -> dict[str, float]:
    """Mean and median pusher px, block CoG px, block angle deg for normalised predictions [N, 6]."""
    errs = probe_errors(decode_targets(pred), states)
    out = {}
    for k, v in errs.items():
        out[f"{k}_mean"], out[f"{k}_median"] = float(v.mean()), float(np.median(v))
    return out


def ridge_probe(z_train: torch.Tensor, s_train: np.ndarray, z_val: torch.Tensor, s_val: np.ndarray,
                lam: float = 1.0) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit on train latents, report errors on train (in-sample), val, and a predict-mean baseline."""
    y = state_targets(s_train)
    w = fit_ridge(z_train, y, lam)
    base = np.broadcast_to(y.mean(0), (len(s_val), 6))
    return w, {"train": state_error_summary(ridge_predict(w, z_train), s_train),
               "val": state_error_summary(ridge_predict(w, z_val), s_val),
               "val_predict_mean": state_error_summary(base, s_val), "lambda": lam}

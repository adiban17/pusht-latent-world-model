"""Judges (Phase 5b): reading the state back from frames or latents, and summarising the errors.

  image judge  = a StateProbe (v1 `ckpt-probe` or `ckpt-probe-mix`) on [N, 3, 64, 64] frames;
  latent judge = a LatentProbe MLP on wm latents z = E_wm(frame) or imagined z_hat.
Both output the 6 normalised targets of §7 (pusher xy, block CoG xy, sin, cos), so both use
`decode_targets` and `probe_errors`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.windows import frames_to_model
from wm.models.probe import decode_targets, probe_errors

ERROR_KEYS = ("block_px", "block_deg", "pusher_px")


@torch.no_grad()
def predict_targets(model: torch.nn.Module, x: torch.Tensor, chunk: int = 512) -> np.ndarray:
    """Raw judge outputs [N, 6] for inputs x [N, ...] (frames or latents), in chunks."""
    model.eval()
    dev = next(model.parameters()).device
    out = [model(x[i:i + chunk].to(dev).float()).float().cpu().numpy() for i in range(0, len(x), chunk)]
    return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


def judge_errors(model: torch.nn.Module, x: torch.Tensor, states: np.ndarray,
                 chunk: int = 512) -> dict[str, np.ndarray]:
    """Per-sample errors (block CoG px, block angle deg, pusher px) of a judge on inputs x [N, ...]."""
    return probe_errors(decode_targets(predict_targets(model, x, chunk)), np.asarray(states))


def frames_u8_to_model(frames_u8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 [N, 64, 64, 3] (numpy or torch) -> float [N, 3, 64, 64] in [-0.5, 0.5] (CPU)."""
    t = torch.as_tensor(np.asarray(frames_u8)) if not isinstance(frames_u8, torch.Tensor) else frames_u8
    return frames_to_model(t.cpu())


def model_to_u8_float(x: torch.Tensor) -> np.ndarray:
    """Normalised float frames [N, 3, 64, 64] -> float [N, 64, 64, 3] on the 0..255 scale."""
    return ((x.float().clamp(-0.5, 0.5) + 0.5) * 255.0).movedim(-3, -1).cpu().numpy()


def summarize_errors(errs: dict[str, np.ndarray]) -> dict[str, float]:
    """{block_px_mean, block_px_median, ..., n} of per-sample errors."""
    out: dict[str, Any] = {"n": int(len(errs["block_px"]))}
    for k in ERROR_KEYS:
        v = np.asarray(errs[k], np.float64).reshape(-1)
        out[f"{k}_mean"] = float(v.mean()) if len(v) else float("nan")
        out[f"{k}_median"] = float(np.median(v)) if len(v) else float("nan")
    return out


def baseline_errors(mean_target: np.ndarray, states: np.ndarray) -> dict[str, float]:
    """Errors of predicting one constant target vector (the train mean) for every sample."""
    pred = np.broadcast_to(np.asarray(mean_target, np.float64), (len(states), 6))
    return summarize_errors(probe_errors(decode_targets(pred), np.asarray(states)))


def fmt_errs(s: dict[str, float]) -> str:
    """'block 10.4 px / 76.2 deg / pusher 24.3 px' from a summarize_errors dict."""
    return (f"block {s['block_px_mean']:.1f} px / {s['block_deg_mean']:.1f} deg / "
            f"pusher {s['pusher_px_mean']:.1f} px")

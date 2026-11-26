"""State probe training (CLAUDE.md §7): MSE on normalised (pusher xy, block CoG xy, sin, cos)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from wm.data.windows import FrameData
from wm.models.probe import StateProbe, decode_targets, probe_errors
from wm.train import common
from wm.train.common import autocast

LOSS_KEYS = ("loss",)
ERROR_KEYS = ("block_px", "block_deg", "pusher_px")


def probe_loss(model: StateProbe, frames: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"loss": F.mse_loss(model(frames).float(), targets)}


@torch.no_grad()
def evaluate(model: StateProbe, data: FrameData, idx: torch.Tensor, mean_target: torch.Tensor,
             device: torch.device, amp: bool, chunk: int = 512) -> dict[str, Any]:
    """MSE, per-frame errors (mean and median of block px, block deg, pusher px) for the probe
    and for the predict-the-train-mean baseline, plus z_std of the encoder latents (info only)."""
    model.eval()
    preds, zs, sq = [], [], 0.0
    for i in range(0, len(idx), chunk):
        frames, targets = data.gather(idx[i:i + chunk])
        with autocast(device, amp):
            z = model.encoder(frames)
            y = model.head(z).float()
        sq += float(((y - targets) ** 2).mean(1).sum())
        preds.append(y.cpu().numpy())
        zs.append(z.float())
    states = data.states[idx.numpy()]
    errs = probe_errors(decode_targets(np.concatenate(preds)), states)
    base = np.broadcast_to(mean_target.cpu().numpy(), (len(idx), 6))
    errs_base = probe_errors(decode_targets(base), states)
    out: dict[str, Any] = {"loss": sq / len(idx), "frames": len(idx),
                           "z_std": float(torch.cat(zs).std(0).mean())}
    for k in ERROR_KEYS:
        out[f"{k}_mean"], out[f"{k}_median"] = float(errs[k].mean()), float(np.median(errs[k]))
        out[f"baseline_{k}_mean"] = float(errs_base[k].mean())
    return out


def run_training(model: StateProbe, data: FrameData, tcfg: dict[str, Any], device: torch.device,
                 *, start: dict[str, Any] | None = None, stop_at: int | None = None,
                 on_checkpoint: Callable | None = None, on_log: Callable | None = None,
                 on_val: Callable | None = None) -> dict[str, Any]:
    val_idx = data.fixed_val(tcfg["val_windows"], tcfg["val_seed"])
    mean_target = data.mean_train_target()
    return common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen),
        loss_fn=lambda batch: probe_loss(model, *batch),
        validate=lambda: evaluate(model, data, val_idx, mean_target, device, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=LOSS_KEYS, start=start, stop_at=stop_at,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val)

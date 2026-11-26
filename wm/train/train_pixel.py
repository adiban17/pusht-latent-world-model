"""Pixel baseline training (CLAUDE.md §5): one-step, teacher-forced pixel MSE; open-loop validation."""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from wm.data.windows import WindowData
from wm.envs.pusht import denormalize_image
from wm.models.pixel_unet import PixelUNet
from wm.train import common
from wm.train.common import autocast

LOSS_KEYS = ("loss",)


def one_step_loss(model: PixelUNet, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """4-frame window (t-2..t+1) -> MSE of the predicted o_{t+1} (action a_t)."""
    c = model.context
    pred = model(frames[:, :c], actions[:, c - 1])
    return {"loss": F.mse_loss(pred.float(), frames[:, c])}


@torch.no_grad()
def validate(model: PixelUNet, data: WindowData, starts: torch.Tensor, horizon: int,
             device: torch.device, amp: bool, chunk: int = 64) -> dict[str, Any]:
    """Open-loop H-step rollouts on val windows: pixel MSE_k for the model and copy-last.

    `loss` = MSE at k = 1 (= the one-step training objective on val data).
    """
    model.eval()
    c = model.context
    err = torch.zeros(horizon, device=device)
    err_copy = torch.zeros(horizon, device=device)
    for i in range(0, len(starts), chunk):
        frames, actions = data.gather(starts[i:i + chunk], length=c + horizon)
        with autocast(device, amp):
            pred = model.rollout(frames[:, :c], actions[:, c - 1:])
        real = frames[:, c:]
        err += ((pred.float() - real) ** 2).mean(dim=(2, 3, 4)).sum(0)
        err_copy += ((frames[:, c - 1:c] - real) ** 2).mean(dim=(2, 3, 4)).sum(0)
    n = len(starts)
    mse, mse_copy = (err / n).tolist(), (err_copy / n).tolist()
    return {"loss": mse[0], "mse": mse, "mse_copy_last": mse_copy, "windows": n}


def run_training(model: PixelUNet, data: WindowData, tcfg: dict[str, Any], device: torch.device,
                 horizon: int, *, start: dict[str, Any] | None = None, stop_at: int | None = None,
                 on_checkpoint: Callable | None = None, on_log: Callable | None = None,
                 on_val: Callable | None = None) -> dict[str, Any]:
    val_starts = data.fixed_val_starts_of(model.context + horizon, tcfg["val_windows"], tcfg["val_seed"])
    return common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen),
        loss_fn=lambda batch: one_step_loss(model, *batch),
        validate=lambda: validate(model, data, val_starts, horizon, device, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=LOSS_KEYS, start=start, stop_at=stop_at,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val)


@torch.no_grad()
def predicted_rollout(model: PixelUNet, data: WindowData, start: torch.Tensor,
                      horizon: int) -> tuple:
    """Real frames t..t+H and [o_t, predicted o_{t+1..t+H}] for one val window, uint8."""
    model.eval()
    c = model.context
    frames, actions = data.gather(start.reshape(1), length=c + horizon)
    pred = model.rollout(frames[:, :c], actions[:, c - 1:])
    seq = torch.cat([frames[:, c - 1:c], pred.float()], dim=1)[0].cpu().numpy()
    model.train()
    return denormalize_image(frames[0, c - 1:].cpu().numpy()), denormalize_image(seq)

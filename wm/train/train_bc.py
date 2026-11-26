"""Goal-conditioned BC training (CLAUDE.md §6): MSE on the logged action as (a_t - p_t) / 50."""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from wm.data.windows import GoalData
from wm.models.bc import BCPolicyNet
from wm.train import common
from wm.train.common import autocast

LOSS_KEYS = ("loss",)


def bc_loss(model: BCPolicyNet, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = model(batch["images"], batch["pusher"])
    return {"loss": F.mse_loss(pred.float(), batch["target"])}


@torch.no_grad()
def validate(model: BCPolicyNet, data: GoalData, val: tuple[torch.Tensor, torch.Tensor],
             delta_scale: float, device: torch.device, amp: bool, chunk: int = 256) -> dict[str, Any]:
    """Val MSE (no augmentation), mean action error in px, the same for the constant
    mean-train-action baseline, and z_std of the encoder latents (info only)."""
    model.eval()
    mean_target = data.mean_train_target()
    sq = sq_base = px = px_base = 0.0
    zs = []
    t_all, k_all = val
    for i in range(0, len(t_all), chunk):
        b = data.gather(t_all[i:i + chunk], k_all[i:i + chunk])
        with autocast(device, amp):
            pred = model(b["images"], b["pusher"]).float()
            zs.append(model.encoder(b["images"]).float().reshape(-1, model.encoder.fc.out_features))
        tgt = b["target"]
        sq += float(((pred - tgt) ** 2).mean(1).sum())
        sq_base += float(((mean_target - tgt) ** 2).mean(1).sum())
        px += float((pred - tgt).norm(dim=1).sum()) * delta_scale
        px_base += float((mean_target - tgt).norm(dim=1).sum()) * delta_scale
    n = len(t_all)
    z = torch.cat(zs)
    return {"loss": sq / n, "baseline_mse": sq_base / n, "action_err_px": px / n,
            "baseline_err_px": px_base / n, "z_std": float(z.std(0).mean()), "samples": n}


def run_training(model: BCPolicyNet, data: GoalData, tcfg: dict[str, Any], device: torch.device,
                 shift_px: int, delta_scale: float, *, start: dict[str, Any] | None = None,
                 stop_at: int | None = None, on_checkpoint: Callable | None = None,
                 on_log: Callable | None = None, on_val: Callable | None = None) -> dict[str, Any]:
    val = data.fixed_val(tcfg["val_windows"], tcfg["val_seed"])
    return common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen, shift=shift_px),
        loss_fn=lambda batch: bc_loss(model, batch),
        validate=lambda: validate(model, data, val, delta_scale, device, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=LOSS_KEYS, start=start, stop_at=stop_at,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val)

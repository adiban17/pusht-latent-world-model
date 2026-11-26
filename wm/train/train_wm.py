"""World-model training and validation (CLAUDE.md §4) on top of the shared loop."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from wm.data.windows import WindowData
from wm.envs.pusht import denormalize_image
from wm.eval.rollouts import latent_nmse
from wm.models.world_model import WorldModel
from wm.train import common
from wm.train.common import autocast, resolve_config  # noqa: F401  (resolve_config re-exported)

LOSS_KEYS = ("loss", "rec", "lat", "img", "var", "z_std")


def loss_keys(model: WorldModel) -> tuple[str, ...]:
    """Logged loss terms: LOSS_KEYS, plus `proprio` for a proprio model (Phase 3b B)."""
    return LOSS_KEYS + ("proprio",) if model.proprio else LOSS_KEYS


def gather(model: WorldModel, data: WindowData, starts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(frames, actions) windows, + pusher positions for a proprio model."""
    return data.gather(starts, pusher=model.proprio)


def batch_loss(model: WorldModel, batch: tuple[torch.Tensor, ...],
               weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """window_loss on a (frames, actions[, pusher]) batch."""
    pusher = batch[2] if len(batch) > 2 else None
    return model.window_loss(batch[0], batch[1], weights, pusher=pusher)


def open_loop(model: WorldModel, out: dict[str, torch.Tensor], actions: torch.Tensor,
              pusher: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(z_hat, p_hat) with predictions fed back (no teacher forcing), from a window_loss output.

    p_hat is None unless the model is proprio. For the default model this is window_loss's own
    z_hat; a teacher-forced model (wm_1step) is re-rolled open-loop, so val NMSE_k and
    filmstrips always measure k-step imagination.
    """
    if not model.teacher_forcing:
        return out["z_hat"], out.get("p_hat")
    c, z = model.context, out["z"]
    if model.proprio:
        h = model.warmup(z[:, :c - 1], actions[:, :c - 1], pusher[:, :c - 1])
        return model.rollout(z[:, c - 1], h, actions[:, c - 1:], p0=pusher[:, c - 1])
    h = model.warmup(z[:, :c - 1], actions[:, :c - 1])
    return model.rollout(z[:, c - 1], h, actions[:, c - 1:]), None


def open_loop_z_hat(model: WorldModel, out: dict[str, torch.Tensor], actions: torch.Tensor,
                    pusher: torch.Tensor | None = None) -> torch.Tensor:
    """Open-loop z_hat only (see open_loop)."""
    return open_loop(model, out, actions, pusher)[0]


@torch.no_grad()
def validate(model: WorldModel, data: WindowData, starts: torch.Tensor, device: torch.device,
             weights: dict[str, float], amp: bool, chunk: int = 64) -> dict[str, Any]:
    """Val losses (training objective), z_std and open-loop latent NMSE_k (k = 1..H) for the
    model and for copy-last. A proprio model also gets pusher_px[k] = mean |p_hat - p| in px
    (and pusher_px_copy_last)."""
    model.eval()
    c = model.context
    sums = {k: 0.0 for k in loss_keys(model) if k != "z_std"}
    zs, z_hats, p_errs, p_copy = [], [], [], []
    for i in range(0, len(starts), chunk):
        batch = gather(model, data, starts[i:i + chunk])
        frames, actions = batch[0], batch[1]
        pusher = batch[2] if model.proprio else None
        with autocast(device, amp):
            out = batch_loss(model, batch, weights)
        for k in sums:
            sums[k] += float(out[k]) * len(frames)
        zs.append(out["z"].float())
        with autocast(device, amp):
            z_hat, p_hat = open_loop(model, out, actions, pusher)
        z_hats.append(z_hat.float())
        if p_hat is not None:
            p_errs.append(((p_hat.float() - pusher[:, c:]).norm(dim=-1) * 256.0))
            p_copy.append(((pusher[:, c - 1:c] - pusher[:, c:]).norm(dim=-1) * 256.0))
    z, z_hat = torch.cat(zs), torch.cat(z_hats)
    with autocast(device, amp):
        z_ref = torch.cat([model.encode(f).float() for f in data.val_frames().split(256)])
    z_true = z[:, c:]
    copy_last = z[:, c - 1:c].expand_as(z_true)
    n = len(starts)
    model.train()
    res = {
        **{k: v / n for k, v in sums.items()},
        "z_std": float(z.reshape(-1, z.shape[-1]).std(0).mean()),
        "nmse": latent_nmse(z_hat, z_true, z_ref).tolist(),
        "nmse_copy_last": latent_nmse(copy_last, z_true, z_ref).tolist(),
        "windows": n,
    }
    if p_errs:
        res["pusher_px"] = torch.cat(p_errs).mean(0).tolist()
        res["pusher_px_copy_last"] = torch.cat(p_copy).mean(0).tolist()
    return res


def select_nmse_k10(v: dict[str, Any]) -> float:
    """keep_best criterion: val NMSE at the last horizon step (k = 10)."""
    return v["nmse"][-1]


def run_training(model: WorldModel, data: WindowData, tcfg: dict[str, Any], device: torch.device,
                 weights: dict[str, float], *, start: dict[str, Any] | None = None,
                 stop_at: int | None = None, keep_best: bool = False,
                 on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
                 on_log: Callable[[int, dict[str, float]], None] | None = None,
                 on_val: Callable[[int, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train the world model with the shared loop (wm.train.common.train_loop); see there for
    the checkpoint / resume contract. keep_best tracks the weights with the best val NMSE@10."""
    val_starts = data.fixed_val_starts(tcfg["val_windows"], tcfg["val_seed"])
    return common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen, pusher=model.proprio),
        loss_fn=lambda batch: batch_loss(model, batch, weights),
        validate=lambda: validate(model, data, val_starts, device, weights, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=loss_keys(model), start=start, stop_at=stop_at,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val,
        select=select_nmse_k10 if keep_best else None)


def prepare_model(model: WorldModel, data: WindowData) -> WorldModel:
    """Set the dataset-dependent buffers (input mean frame, foreground mean frame) if present."""
    if model.input_mean is not None or model.fg_mean is not None:
        mean = data.train_mean_frame()
        if model.input_mean is not None:
            model.set_input_mean(mean)
        if model.fg_mean is not None:
            model.set_fg_mean(mean)
    return model


def loss_window_means(log: list[dict[str, float]], n: int,
                      keys: tuple[str, ...] = LOSS_KEYS) -> tuple[dict, dict]:
    """Mean of each loss term over the first n and the last n logged steps."""
    return common.loss_window_means(log, n, keys)


def all_finite(log: list[dict[str, float]], keys: tuple[str, ...] = LOSS_KEYS) -> bool:
    return common.all_finite(log, keys)


@torch.no_grad()
def decoded_rollout(model: WorldModel, data: WindowData, start: torch.Tensor,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Real frames t..t+H and decoded [D(z_t), D(z_hat_{t+1..t+H})] for one window, uint8."""
    model.eval()
    batch = gather(model, data, start.reshape(1))
    frames, actions = batch[0], batch[1]
    pusher = batch[2] if model.proprio else None
    out = batch_loss(model, batch, {"rec": 1.0, "lat": 1.0, "img": 1.0})
    c = model.context
    z_seq = torch.cat([out["z"][:, c - 1:c], open_loop_z_hat(model, out, actions, pusher)], dim=1)
    decoded = model.decode(z_seq)[0].float().cpu().numpy()
    real = frames[0, c - 1:].float().cpu().numpy()
    model.train()
    return denormalize_image(real), denormalize_image(decoded)

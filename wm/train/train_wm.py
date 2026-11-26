"""World-model training and validation (CLAUDE.md §4) on top of the shared loop."""
from __future__ import annotations

from typing import Any, Callable

import torch

from wm.data.windows import WindowData
from wm.envs.pusht import denormalize_image
from wm.eval.rollouts import latent_nmse
from wm.models.world_model import WorldModel
from wm.train import common
from wm.train.common import autocast, resolve_config  # noqa: F401  (resolve_config re-exported)

LOSS_KEYS = ("loss", "rec", "lat", "img", "var", "z_std")


def open_loop_z_hat(model: WorldModel, out: dict[str, torch.Tensor],
                    actions: torch.Tensor) -> torch.Tensor:
    """z_hat with predictions fed back (no teacher forcing), from a window_loss output.

    For the default model this is window_loss's own z_hat; a teacher-forced model (wm_1step)
    is re-rolled open-loop, so val NMSE_k and filmstrips always measure k-step imagination.
    """
    if not model.teacher_forcing:
        return out["z_hat"]
    c, z = model.context, out["z"]
    h = model.warmup(z[:, :c - 1], actions[:, :c - 1])
    return model.rollout(z[:, c - 1], h, actions[:, c - 1:])


@torch.no_grad()
def validate(model: WorldModel, data: WindowData, starts: torch.Tensor, device: torch.device,
             weights: dict[str, float], amp: bool, chunk: int = 64) -> dict[str, Any]:
    """Val losses (training objective), z_std and open-loop latent NMSE_k (k = 1..H) for the
    model and for copy-last."""
    model.eval()
    c = model.context
    sums = {k: 0.0 for k in LOSS_KEYS if k != "z_std"}
    zs, z_hats = [], []
    for i in range(0, len(starts), chunk):
        frames, actions = data.gather(starts[i:i + chunk])
        with autocast(device, amp):
            out = model.window_loss(frames, actions, weights)
        for k in sums:
            sums[k] += float(out[k]) * len(frames)
        zs.append(out["z"].float())
        with autocast(device, amp):
            z_hats.append(open_loop_z_hat(model, out, actions).float())
    z, z_hat = torch.cat(zs), torch.cat(z_hats)
    with autocast(device, amp):
        z_ref = torch.cat([model.encode(f).float() for f in data.val_frames().split(256)])
    z_true = z[:, c:]
    copy_last = z[:, c - 1:c].expand_as(z_true)
    n = len(starts)
    model.train()
    return {
        **{k: v / n for k, v in sums.items()},
        "z_std": float(z.reshape(-1, z.shape[-1]).std(0).mean()),
        "nmse": latent_nmse(z_hat, z_true, z_ref).tolist(),
        "nmse_copy_last": latent_nmse(copy_last, z_true, z_ref).tolist(),
        "windows": n,
    }


def run_training(model: WorldModel, data: WindowData, tcfg: dict[str, Any], device: torch.device,
                 weights: dict[str, float], *, start: dict[str, Any] | None = None,
                 stop_at: int | None = None,
                 on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
                 on_log: Callable[[int, dict[str, float]], None] | None = None,
                 on_val: Callable[[int, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train the world model with the shared loop (wm.train.common.train_loop); see there for
    the checkpoint / resume contract."""
    val_starts = data.fixed_val_starts(tcfg["val_windows"], tcfg["val_seed"])
    return common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen),
        loss_fn=lambda batch: model.window_loss(*batch, weights),
        validate=lambda: validate(model, data, val_starts, device, weights, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=LOSS_KEYS, start=start, stop_at=stop_at,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val)


def loss_window_means(log: list[dict[str, float]], n: int) -> tuple[dict, dict]:
    """Mean of each loss term over the first n and the last n logged steps."""
    return common.loss_window_means(log, n, LOSS_KEYS)


def all_finite(log: list[dict[str, float]]) -> bool:
    return common.all_finite(log, LOSS_KEYS)


@torch.no_grad()
def decoded_rollout(model: WorldModel, data: WindowData, start: torch.Tensor,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Real frames t..t+H and decoded [D(z_t), D(z_hat_{t+1..t+H})] for one window, uint8."""
    model.eval()
    frames, actions = data.gather(start.reshape(1))
    out = model.window_loss(frames, actions, {"rec": 1.0, "lat": 1.0, "img": 1.0})
    c = model.context
    z_seq = torch.cat([out["z"][:, c - 1:c], open_loop_z_hat(model, out, actions)], dim=1)
    decoded = model.decode(z_seq)[0].float().cpu().numpy()
    real = frames[0, c - 1:].float().cpu().numpy()
    model.train()
    return denormalize_image(real), denormalize_image(decoded)

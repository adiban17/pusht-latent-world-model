"""World-model training loop and validation (CLAUDE.md §4)."""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np
import torch

from wm.data.windows import WindowData
from wm.envs.pusht import denormalize_image
from wm.eval.rollouts import latent_nmse
from wm.models.world_model import WorldModel
from wm.train.common import (StepTimer, autocast, lr_at, make_optimizer, make_scaler, set_lr)

LOSS_KEYS = ("loss", "rec", "lat", "img", "var", "z_std")


def resolve_config(cfg: dict[str, Any], mode: str, platform: str) -> dict[str, Any]:
    """The `train` settings for this run: smoke mode applies smoke.local / smoke.remote on top.

    Returns a flat dict: train keys + seed + val_windows + val_seed.
    """
    tcfg = {**cfg["train"], "seed": cfg["seed"], "val_windows": cfg["val"]["windows"],
            "val_seed": cfg["val"]["seed"]}
    if mode == "smoke":
        tcfg.update(cfg["smoke"]["local" if platform == "local" else "remote"])
    return tcfg


@torch.no_grad()
def validate(model: WorldModel, data: WindowData, starts: torch.Tensor, device: torch.device,
             weights: dict[str, float], amp: bool, chunk: int = 64) -> dict[str, Any]:
    """Val losses, z_std and latent NMSE_k (k = 1..H) for the model and for copy-last."""
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
        z_hats.append(out["z_hat"].float())
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
    """Train to tcfg["steps"] (or `stop_at`), resuming from checkpoint `start` if given.

    Every ckpt_every steps (before the end) on_checkpoint gets the full training state, which
    `start` accepts back: model, optimiser, scaler, sampler RNG, step and logs. Resuming gives
    bit-identical results to an uninterrupted run (CPU).
    """
    opt = make_optimizer(model, tcfg["lr"], tcfg["weight_decay"])
    scaler = make_scaler(device, tcfg["amp"])
    gen = torch.Generator().manual_seed(int(tcfg["seed"]))
    step, log, val_log = 0, [], []
    if start is not None:
        model.load_state_dict(start["model"])
        opt.load_state_dict(start["opt"])
        scaler.load_state_dict(start["scaler"])
        gen.set_state(start["rng"])
        step, log, val_log = int(start["step"]), list(start["log"]), list(start["val_log"])
    val_starts = data.fixed_val_starts(tcfg["val_windows"], tcfg["val_seed"])
    total = int(tcfg["steps"])
    end = total if stop_at is None else min(int(stop_at), total)

    def state() -> dict[str, Any]:
        return {"step": step, "model": copy.deepcopy(model.state_dict()),
                "opt": copy.deepcopy(opt.state_dict()), "scaler": scaler.state_dict(),
                "rng": gen.get_state(), "log": list(log), "val_log": list(val_log)}

    timer = StepTimer(skip=min(10, max(0, (end - step) // 5)))
    model.train()
    while step < end:
        set_lr(opt, lr_at(step, tcfg["lr"], tcfg["warmup_steps"], total))
        frames, actions = data.sample(tcfg["batch_size"], gen)
        timer.start()
        with autocast(device, tcfg["amp"]):
            out = model.window_loss(frames, actions, weights)
        opt.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        entry = {k: float(out[k].detach()) for k in LOSS_KEYS}  # .item() syncs, so timing is honest
        timer.stop()
        entry["lr"] = opt.param_groups[0]["lr"]
        step += 1
        log.append(entry)
        if on_log is not None and step % tcfg["log_every"] == 0:
            on_log(step, entry)
        if step % tcfg["val_every"] == 0 or step == total:
            v = validate(model, data, val_starts, device, weights, tcfg["amp"])
            val_log.append({"step": step, **v})
            if on_val is not None:
                on_val(step, v)
        if on_checkpoint is not None and step % tcfg["ckpt_every"] == 0 and step < total:
            on_checkpoint(state())
    return {"step": step, "complete": step >= total, "log": log, "val_log": val_log,
            "it_per_s": timer.rate(), "steps_this_run": timer.seen, "state": state()}


def loss_window_means(log: list[dict[str, float]], n: int) -> tuple[dict, dict]:
    """Mean of each loss term over the first n and the last n logged steps."""
    def mean(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in LOSS_KEYS}
    return mean(log[:n]), mean(log[-n:])


def all_finite(log: list[dict[str, float]]) -> bool:
    return all(math.isfinite(r[k]) for r in log for k in LOSS_KEYS)


@torch.no_grad()
def decoded_rollout(model: WorldModel, data: WindowData, start: torch.Tensor,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Real frames t..t+H and decoded [D(z_t), D(z_hat_{t+1..t+H})] for one window, uint8."""
    model.eval()
    frames, actions = data.gather(start.reshape(1))
    out = model.window_loss(frames, actions, {"rec": 1.0, "lat": 1.0, "img": 1.0})
    c = model.context
    z_seq = torch.cat([out["z"][:, c - 1:c], out["z_hat"]], dim=1)
    decoded = model.decode(z_seq)[0].float().cpu().numpy()
    real = frames[0, c - 1:].float().cpu().numpy()
    model.train()
    return denormalize_image(real), denormalize_image(decoded)

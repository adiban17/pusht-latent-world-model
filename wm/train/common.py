"""Shared training utilities: optimiser, LR schedule, AMP, checkpoints, timing."""
from __future__ import annotations

import copy
import math
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """LR for 0-indexed `step`: linear warm-up to base_lr over `warmup` steps, then cosine to 0."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def amp_enabled(device: torch.device, amp: bool) -> bool:
    """fp16 mixed precision only on CUDA."""
    return bool(amp) and device.type == "cuda"


def autocast(device: torch.device, amp: bool) -> AbstractContextManager:
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=amp_enabled(device, amp))


def make_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=amp_enabled(device, amp))


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    """torch.save to a temp file, then rename (a kill mid-write never leaves a broken file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, device="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def peak_gpu_mb(device: torch.device) -> float | None:
    """Peak allocated GPU memory in MiB (None on CPU)."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


class StepTimer:
    """Training-step throughput, ignoring the first `skip` steps (warm-up / cudnn autotune)."""

    def __init__(self, skip: int = 10):
        self.skip, self.seen, self.timed, self.seconds = skip, 0, 0, 0.0
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        dt = time.perf_counter() - self._t0
        self.seen += 1
        if self.seen > self.skip:
            self.timed += 1
            self.seconds += dt

    def rate(self) -> float | None:
        """Steps per second after warm-up (None if no step was timed)."""
        return self.timed / self.seconds if self.timed and self.seconds > 0 else None


# ---- shared training loop -------------------------------------------------------------------

def resolve_config(cfg: dict[str, Any], mode: str, platform: str) -> dict[str, Any]:
    """The `train` settings for this run: smoke mode applies smoke.local / smoke.remote on top.

    Returns a flat dict: train keys + seed + val_windows + val_seed. Every training config
    (wm, wm_1step, pixel, bc, probe) has the same train / val / smoke layout.
    """
    tcfg = {**cfg["train"], "seed": cfg["seed"], "val_windows": cfg["val"]["windows"],
            "val_seed": cfg["val"]["seed"]}
    if mode == "smoke":
        tcfg.update(cfg["smoke"]["local" if platform == "local" else "remote"])
    return tcfg


def _global_rng_state() -> dict[str, Any]:
    """Torch's global RNG (dropout uses it); saved so a resumed run stays identical."""
    state = {"torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def train_loop(model: nn.Module, sample: Callable[[torch.Generator], Any],
               loss_fn: Callable[[Any], dict[str, torch.Tensor]],
               validate: Callable[[], dict[str, Any]], tcfg: dict[str, Any],
               device: torch.device, log_keys: tuple[str, ...], *,
               start: dict[str, Any] | None = None, stop_at: int | None = None,
               on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
               on_log: Callable[[int, dict[str, float]], None] | None = None,
               on_val: Callable[[int, dict[str, Any]], None] | None = None,
               select: Callable[[dict[str, Any]], float] | None = None) -> dict[str, Any]:
    """Train to tcfg["steps"] (or `stop_at`), resuming from checkpoint `start` if given.

    sample(gen) draws a batch with the (CPU) sampler generator; loss_fn(batch) returns a dict
    with "loss" plus the other `log_keys`; validate() returns a dict of val metrics.
    Every ckpt_every steps (before the end) on_checkpoint gets the full training state, which
    `start` accepts back: model, optimiser, scaler, sampler RNG, global torch RNG, step and
    logs. Resuming gives bit-identical results to an uninterrupted run (CPU).

    select(val_metrics) -> float (lower is better; keep_best, Phase 3b C): after every validation
    the weights with the lowest value so far are kept as {"step", "value", "model"}; they are
    in the checkpoint state under "best" and returned as result["best"]. Without `select` there
    is no "best" key anywhere.
    """
    opt = make_optimizer(model, tcfg["lr"], tcfg["weight_decay"])
    scaler = make_scaler(device, tcfg["amp"])
    gen = torch.Generator().manual_seed(int(tcfg["seed"]))
    step, log, val_log = 0, [], []
    best: dict[str, Any] | None = None
    if start is not None:
        model.load_state_dict(start["model"])
        opt.load_state_dict(start["opt"])
        scaler.load_state_dict(start["scaler"])
        gen.set_state(start["rng"])
        if "torch_rng" in start:   # checkpoints written before Phase 4 don't have it
            torch.set_rng_state(start["torch_rng"])
        if "cuda_rng" in start and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(start["cuda_rng"])
        step, log, val_log = int(start["step"]), list(start["log"]), list(start["val_log"])
        best = start.get("best")
    total = int(tcfg["steps"])
    end = total if stop_at is None else min(int(stop_at), total)

    def state() -> dict[str, Any]:
        s = {"step": step, "model": copy.deepcopy(model.state_dict()),
             "opt": copy.deepcopy(opt.state_dict()), "scaler": scaler.state_dict(),
             "rng": gen.get_state(), **_global_rng_state(),
             "log": list(log), "val_log": list(val_log)}
        if select is not None:
            s["best"] = best
        return s

    timer = StepTimer(skip=min(10, max(0, (end - step) // 5)))
    model.train()
    while step < end:
        set_lr(opt, lr_at(step, tcfg["lr"], tcfg["warmup_steps"], total))
        batch = sample(gen)
        timer.start()
        with autocast(device, tcfg["amp"]):
            out = loss_fn(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        entry = {k: float(out[k].detach()) for k in log_keys}  # .item() syncs, so timing is honest
        timer.stop()
        entry["lr"] = opt.param_groups[0]["lr"]
        step += 1
        log.append(entry)
        if on_log is not None and step % tcfg["log_every"] == 0:
            on_log(step, entry)
        if step % tcfg["val_every"] == 0 or step == total:
            v = validate()
            model.train()
            val_log.append({"step": step, **v})
            if select is not None:
                value = float(select(v))
                if best is None or value < best["value"]:
                    best = {"step": step, "value": value, "model": copy.deepcopy(model.state_dict())}
            if on_val is not None:
                on_val(step, v)
        if on_checkpoint is not None and step % tcfg["ckpt_every"] == 0 and step < total:
            on_checkpoint(state())
    result = {"step": step, "complete": step >= total, "log": log, "val_log": val_log,
              "it_per_s": timer.rate(), "steps_this_run": timer.seen, "state": state()}
    if select is not None:
        result["best"] = best
    return result


def loss_window_means(log: list[dict[str, float]], n: int, keys: tuple[str, ...]) -> tuple[dict, dict]:
    """Mean of each logged term over the first n and the last n steps."""
    def mean(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return mean(log[:n]), mean(log[-n:])


def all_finite(log: list[dict[str, float]], keys: tuple[str, ...]) -> bool:
    return all(math.isfinite(r[k]) for r in log for k in keys)

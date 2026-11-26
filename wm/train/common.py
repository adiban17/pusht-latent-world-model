"""Shared training utilities: optimiser, LR schedule, AMP, checkpoints, timing."""
from __future__ import annotations

import math
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

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

"""Training of the latent judge (Phase 5b): MSE on the §7 targets from precomputed wm latents."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from wm.data.windows import state_targets
from wm.eval.judges import judge_errors, summarize_errors
from wm.models.latent_probe import LatentProbe
from wm.train import common

LOSS_KEYS = ("loss",)


class LatentData:
    """Latents z [N, 256] and their §7 targets, on the device."""

    def __init__(self, z: torch.Tensor, states: np.ndarray, device):
        self.device = torch.device(device)
        self.z = z.float().to(self.device)
        self.states = np.asarray(states)
        self.targets = torch.from_numpy(state_targets(self.states)).to(self.device)

    def sample(self, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.randint(len(self.z), (batch,), generator=gen).to(self.device)
        return self.z[i], self.targets[i]

    def fixed(self, n: int, seed: int = 0) -> torch.Tensor:
        if n >= len(self.z):
            return torch.arange(len(self.z))
        return torch.randperm(len(self.z), generator=torch.Generator().manual_seed(seed))[:n].sort().values


def latent_loss(model: LatentProbe, z: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"loss": F.mse_loss(model(z).float(), targets)}


@torch.no_grad()
def validate(model: LatentProbe, data: LatentData, idx: torch.Tensor) -> dict[str, Any]:
    model.eval()
    z, y = data.z[idx.to(data.device)], data.targets[idx.to(data.device)]
    out: dict[str, Any] = {"loss": float(((model(z).float() - y) ** 2).mean())}
    out.update(summarize_errors(judge_errors(model, z, data.states[idx.numpy()])))
    return out


def run_training(model: LatentProbe, train: LatentData, val: LatentData, tcfg: dict[str, Any],
                 device: torch.device, *, start: dict[str, Any] | None = None,
                 on_checkpoint: Callable | None = None, on_log: Callable | None = None,
                 on_val: Callable | None = None) -> dict[str, Any]:
    """train_loop with keep-best on the rendered-val loss; the model ends with the best weights."""
    val_idx = val.fixed(int(tcfg["val_windows"]), int(tcfg["val_seed"]))
    result = common.train_loop(
        model, sample=lambda gen: train.sample(tcfg["batch_size"], gen),
        loss_fn=lambda batch: latent_loss(model, *batch),
        validate=lambda: validate(model, val, val_idx), tcfg=tcfg, device=device,
        log_keys=LOSS_KEYS, start=start, on_checkpoint=on_checkpoint, on_log=on_log,
        on_val=on_val, select=lambda v: v["loss"])
    if result.get("best") is not None:
        model.load_state_dict(result["best"]["model"])
    return result

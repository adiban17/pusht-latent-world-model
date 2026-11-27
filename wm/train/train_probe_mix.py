"""Training and evaluation of the fair image judge probe_mix (Phase 5b).

Same loss as §7 (MSE on the 6 normalised targets), balanced batches over the four frame types,
±4 px shift with matching label shift, keep-best by the rendered-val loss (mean over types).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from wm.data.judge_mix import FRAME_TYPES, MixData
from wm.eval.judges import ERROR_KEYS, judge_errors, summarize_errors
from wm.models.probe import StateProbe, decode_targets, probe_errors
from wm.train import common
from wm.train.common import autocast
from wm.train.train_probe import LOSS_KEYS, probe_loss


def fixed_val_set(data: MixData, n: int, seed: int) -> dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Fixed validation images per frame type (generated once)."""
    return {k: data.images(k, data.fixed(k, n, seed)) for k in FRAME_TYPES}


@torch.no_grad()
def validate(model: StateProbe, val: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]],
             device: torch.device, amp: bool, chunk: int = 512) -> dict[str, Any]:
    """Per type: MSE and mean errors; `loss` = mean MSE over types (the keep-best criterion)."""
    model.eval()
    out: dict[str, Any] = {}
    for kind, (x, y, states) in val.items():
        preds = []
        for i in range(0, len(x), chunk):
            with autocast(device, amp):
                preds.append(model(x[i:i + chunk]).float())
        p = torch.cat(preds)
        out[f"{kind}_loss"] = float(((p - y) ** 2).mean())
        errs = probe_errors(decode_targets(p.cpu().numpy()), states)
        for k in ERROR_KEYS:
            out[f"{kind}_{k}_mean"] = float(errs[k].mean())
    out["loss"] = float(np.mean([out[f"{k}_loss"] for k in val]))
    return out


def run_training(model: StateProbe, data: MixData, val: dict, tcfg: dict[str, Any], device: torch.device,
                 pad: int, *, start: dict[str, Any] | None = None, on_checkpoint: Callable | None = None,
                 on_log: Callable | None = None, on_val: Callable | None = None) -> dict[str, Any]:
    """train_loop with keep-best on the rendered-val loss; the model ends with the best weights."""
    result = common.train_loop(
        model, sample=lambda gen: data.sample(tcfg["batch_size"], gen, pad),
        loss_fn=lambda batch: probe_loss(model, *batch),
        validate=lambda: validate(model, val, device, tcfg["amp"]),
        tcfg=tcfg, device=device, log_keys=LOSS_KEYS, start=start,
        on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val, select=lambda v: v["loss"])
    if result.get("best") is not None:
        model.load_state_dict(result["best"]["model"])
    return result


@torch.no_grad()
def typed_errors(model: StateProbe, data: MixData, kinds: tuple[str, ...] = FRAME_TYPES,
                 n: int | None = None, seed: int = 0, chunk: int = 256) -> dict[str, dict[str, float]]:
    """Mean/median errors of an image judge on every frame type of `data` (all valid frames
    unless n is given), built chunk by chunk."""
    out = {}
    for kind in kinds:
        idx = data.fixed(kind, n, seed)
        errs: dict[str, list[np.ndarray]] = {k: [] for k in ERROR_KEYS}
        for i in range(0, len(idx), chunk):
            x, _, states = data.images(kind, idx[i:i + chunk])
            e = judge_errors(model, x, states)
            for k in ERROR_KEYS:
                errs[k].append(e[k])
        out[kind] = summarize_errors({k: np.concatenate(v) for k, v in errs.items()})
    return out


def acceptance(test: dict[str, dict[str, float]], acc: dict[str, float]) -> dict[str, Any]:
    """Pre-registered acceptance (PROGRESS.md, Phase 5b): real angle < acc.real_angle_deg, real
    block <= acc.real_block_px, block and pusher px of every derived type <= ratio x real's."""
    real = test["real"]
    checks = {"real_angle": real["block_deg_mean"] < acc["real_angle_deg"],
              "real_block": real["block_px_mean"] <= acc["real_block_px"]}
    for kind in test:
        if kind == "real":
            continue
        for k in ("block_px", "pusher_px"):
            checks[f"{kind}_{k}"] = test[kind][f"{k}_mean"] <= acc["derived_ratio"] * real[f"{k}_mean"]
    return {"accepted": all(checks.values()), "checks": checks}

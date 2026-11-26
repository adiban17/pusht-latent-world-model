"""Shared plumbing for the training jobs (train_wm, train_wm_1step, train_pixel, train_bc,
train_probe): data loading, resume, checkpoint pushes, the final artifact and common REPORT lines.

Store layout of every `ckpt-*` artifact: intermediate pushes hold `checkpoint.pt` (full training
state) with complete=false, step; the final push holds `model.pt` (weights + config + params +
git), `train_summary.json`, `config.yaml` (+ job-specific images) with complete=true.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wm.jobs import JobContext
from wm.report import Report, format_duration
from wm.repro import seed_all
from wm.train.common import (all_finite, load_checkpoint, loss_window_means, peak_gpu_mb,
                             resolve_config, save_checkpoint)


def load_npz(ctx: JobContext, filename: str = "pusht_train.npz") -> tuple[dict[str, np.ndarray], str]:
    """An npz from the store's `pusht-data` (or `pusht-data-smoke` in smoke mode)."""
    name = ctx.store.resolve("pusht-data")
    path = ctx.store.pull("pusht-data")
    if path is None:
        raise RuntimeError("store artifact 'pusht-data' not found: run collect_data first")
    with np.load(path / filename) as d:
        return {k: d[k] for k in d.files}, name


def find_resume(ctx: JobContext, artifact: str) -> dict[str, Any] | None:
    """The incomplete checkpoint this mode wrote earlier with the same config_hash, if any."""
    meta = ctx.store.own_metadata(artifact)
    if not meta or meta.get("complete") or meta.get("config_hash") != ctx.config_hash:
        return None
    path = ctx.store.pull_own(artifact)
    if path is None or not (path / "checkpoint.pt").exists():
        return None
    return load_checkpoint(path / "checkpoint.pt", ctx.device)


class TrainingJob:
    """One training job's run: config, resume state, artifact dir and store pushes."""

    def __init__(self, ctx: JobContext, artifact: str):
        self.t0 = time.time()
        self.ctx, self.artifact, self.cfg = ctx, artifact, ctx.config
        self.tcfg = resolve_config(self.cfg, ctx.mode, ctx.platform)
        self.full_steps = int(self.cfg["train"]["steps"])
        seed_all(self.cfg["seed"])
        self.counts: dict[str, int] = {}
        self.data_name = ""
        self.extra_info: dict[str, Any] = {}   # e.g. best_step (keep_best), saved with model.pt

        self.art_dir = ctx.out_dir / "artifact"
        if self.art_dir.exists():
            shutil.rmtree(self.art_dir)
        self.art_dir.mkdir(parents=True)
        (self.art_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))

        self.start = find_resume(ctx, artifact)
        self.elapsed_before = float(self.start.get("wall_clock_s", 0.0)) if self.start else 0.0
        if self.start is not None:
            print(f"resuming from step {self.start['step']}", flush=True)

    @property
    def resumed_from(self) -> int:
        return int(self.start["step"]) if self.start else 0

    def set_params(self, report: Report, counts: dict[str, int]) -> None:
        """Record parameter counts (REPORT header + startup print)."""
        self.counts = counts
        report.params = f"{counts['total']:,}"
        print("params: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)

    def info(self, step: int) -> dict[str, Any]:
        """Reproducibility info saved with every checkpoint (CLAUDE.md hard rule 5)."""
        return {"step": step, "config": self.cfg, "train_config": self.tcfg,
                "config_hash": self.ctx.config_hash, "git": self.ctx.git, "params": self.counts,
                "data": self.data_name, "wall_clock_s": self.elapsed_before + time.time() - self.t0,
                **self.extra_info}

    # ---- callbacks for train_loop --------------------------------------------------------
    def on_checkpoint(self, state: dict[str, Any]) -> None:
        save_checkpoint(self.art_dir / "checkpoint.pt", {**state, **self.info(state["step"])})
        self.ctx.store.push(self.artifact, self.art_dir, self.ctx.meta(complete=False, step=state["step"]))
        print(f"step {state['step']}: checkpoint pushed", flush=True)

    def on_log(self, step: int, entry: dict[str, float]) -> None:
        print(f"step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in entry.items()), flush=True)
        self.wandb_log({f"train/{k}": v for k, v in entry.items()}, step)

    def wandb_log(self, row: dict[str, float], step: int) -> None:
        if self.ctx.wandb_run is not None:
            self.ctx.wandb_run.log(row, step=step)

    # ---- end of training -----------------------------------------------------------------
    def save_model(self, model: torch.nn.Module, step: int) -> Path:
        """model.pt = weights + reproducibility info; removes the intermediate checkpoint."""
        path = save_checkpoint(self.art_dir / "model.pt", {"model": model.state_dict(), **self.info(step)})
        (self.art_dir / "checkpoint.pt").unlink(missing_ok=True)
        return path

    def write_summary(self, result: dict[str, Any], keys: tuple[str, ...], **extra: Any) -> dict[str, Any]:
        """train_summary.json: info + timing + first/last loss windows + logs + `extra`."""
        log = result["log"]
        n = min(100, max(1, len(log) // 5))
        first, last = loss_window_means(log, n, keys)
        summary = {**self.info(result["step"]), "complete": result["complete"],
                   "resumed_from_step": self.resumed_from, "it_per_s": result["it_per_s"],
                   "peak_gpu_mb": peak_gpu_mb(self.ctx.device), "loss_first": first,
                   "loss_last": last, "loss_window": n, **extra,
                   "val_log": result["val_log"], "train_log": log}
        (self.art_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def loss_checks(self, report: Report, summary: dict[str, Any], log: list[dict[str, float]],
                    keys: tuple[str, ...]) -> None:
        """CHECKS losses_finite and loss_fell."""
        ok = all_finite(log, keys)
        report.check_true("losses_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
        first, last, n = summary["loss_first"], summary["loss_last"], summary["loss_window"]
        report.check_true("loss_fell", last["loss"] < first["loss"],
                          f"{first['loss']:.4f} -> {last['loss']:.4f}",
                          f"mean of last {n} < first {n} steps")

    def timing_metrics(self, report: Report, summary: dict[str, Any]) -> None:
        """it/s, projected time for the full run, peak GPU memory."""
        it_s, peak = summary["it_per_s"], summary["peak_gpu_mb"]
        report.metric("it_per_s", f"{it_s:.2f}" if it_s else "n/a")
        report.metric(f"projected_{self.full_steps // 1000}k_steps",
                      format_duration(self.full_steps / it_s) if it_s else "n/a")
        report.metric("peak_gpu_mem", f"{peak:.0f} MiB" if peak is not None else "n/a (cpu)")

    def push_final(self, report: Report, result: dict[str, Any], files: list[Path]) -> None:
        """Final store push (complete only if training finished and no CHECK failed)."""
        saved = self.ctx.store.push(self.artifact, self.art_dir,
                                    self.ctx.meta(complete=result["complete"] and not report.failed,
                                                  step=result["step"]))
        for p in files:
            report.save(p)
        report.save(saved)


def check_param_counts(report: Report, counts: dict[str, int], expected: dict[str, int],
                       where: str) -> None:
    """CHECK param_counts == expected."""
    report.check_true("param_counts", counts == expected,
                      " / ".join(f"{k} {v:,}" for k, v in counts.items()), f"== {where}")


def same_outputs(fn_a, fn_b, x) -> bool:
    """True if two models give identical outputs on x (checkpoint round-trip check)."""
    with torch.no_grad():
        a, b = fn_a(x), fn_b(x)
    return torch.equal(a, b)

"""Job train_wm (Phase 3): train the latent world model (CLAUDE.md §4) -> store `ckpt-wm`.

Checkpoints go to the store every ckpt_every steps with complete=false, so a killed run resumes
from the latest one (same config_hash). The final artifact holds model.pt (weights + config +
param counts + git), train_summary.json, config.yaml and rollout_filmstrip.png.
"""
from __future__ import annotations

import json
import shutil
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm.data.windows import WindowData
from wm.jobs import JobContext
from wm.models.world_model import EXPECTED_PARAMS, build_world_model, param_counts
from wm.report import Report, format_duration
from wm.repro import seed_all
from wm.train.common import load_checkpoint, peak_gpu_mb, save_checkpoint
from wm.train.train_wm import (all_finite, decoded_rollout, loss_window_means, resolve_config,
                               run_training)
from wm.viz.media import rollout_filmstrip

ARTIFACT = "ckpt-wm"
FULL_STEPS = 30_000
KS = (1, 5, 10)


def load_train_arrays(ctx: JobContext) -> tuple[dict[str, np.ndarray], str]:
    """pusht_train.npz from the store (`pusht-data`, or `pusht-data-smoke` in smoke mode)."""
    name = ctx.store.resolve("pusht-data")
    path = ctx.store.pull("pusht-data")
    if path is None:
        raise RuntimeError("store artifact 'pusht-data' not found: run collect_data first")
    with np.load(path / "pusht_train.npz") as d:
        return {k: d[k] for k in d.files}, name


def find_resume(ctx: JobContext) -> dict[str, Any] | None:
    """The incomplete checkpoint this mode wrote earlier with the same config_hash, if any."""
    meta = ctx.store.own_metadata(ARTIFACT)
    if not meta or meta.get("complete") or meta.get("config_hash") != ctx.config_hash:
        return None
    path = ctx.store.pull_own(ARTIFACT)
    if path is None or not (path / "checkpoint.pt").exists():
        return None
    return load_checkpoint(path / "checkpoint.pt", ctx.device)


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg, device = ctx.config, ctx.device
    tcfg = resolve_config(cfg, ctx.mode, ctx.platform)
    weights = dict(cfg["loss"])
    seed_all(cfg["seed"])

    arrays, data_name = load_train_arrays(ctx)
    data = WindowData(arrays, device, cfg["val"]["seeds"],
                      window=cfg["window"]["context"] + cfg["window"]["horizon"])
    model = build_world_model(cfg).to(device)
    if model.input_mean is not None:
        model.set_input_mean(data.train_mean_frame())
    counts = param_counts(model)
    report.params = f"{counts['total']:,}"
    print("params: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)
    report.check_true("param_counts", counts == EXPECTED_PARAMS,
                      " / ".join(f"{k} {v:,}" for k, v in counts.items()), "== CLAUDE.md §4")

    art_dir = ctx.out_dir / "artifact"
    if art_dir.exists():
        shutil.rmtree(art_dir)
    art_dir.mkdir(parents=True)
    (art_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    elapsed_before = 0.0

    start = find_resume(ctx)
    if start is not None:
        elapsed_before = float(start.get("wall_clock_s", 0.0))
        print(f"resuming from step {start['step']}", flush=True)

    def info(step: int) -> dict[str, Any]:
        """Reproducibility info saved with every checkpoint (CLAUDE.md hard rule 5)."""
        return {"step": step, "config": cfg, "train_config": tcfg, "config_hash": ctx.config_hash,
                "git": ctx.git, "params": counts, "data": data_name,
                "wall_clock_s": elapsed_before + time.time() - t0}

    def on_checkpoint(state: dict[str, Any]) -> None:
        save_checkpoint(art_dir / "checkpoint.pt", {**state, **info(state["step"])})
        ctx.store.push(ARTIFACT, art_dir, ctx.meta(complete=False, step=state["step"]))
        print(f"step {state['step']}: checkpoint pushed", flush=True)

    def on_log(step: int, entry: dict[str, float]) -> None:
        print(f"step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in entry.items()), flush=True)
        if ctx.wandb_run is not None:
            ctx.wandb_run.log({f"train/{k}": v for k, v in entry.items()}, step=step)

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: val loss {v['loss']:.4g} nmse@10 {v['nmse'][-1]:.3f} "
              f"(copy-last {v['nmse_copy_last'][-1]:.3f}) z_std {v['z_std']:.3f}", flush=True)
        if ctx.wandb_run is not None:
            row = {f"val/{k}": v[k] for k in ("loss", "rec", "lat", "img", "z_std")}
            row.update({f"val/nmse_k{k}": v["nmse"][k - 1] for k in KS})
            row.update({f"val/copy_last_nmse_k{k}": v["nmse_copy_last"][k - 1] for k in KS})
            ctx.wandb_run.log(row, step=step)

    result = run_training(model, data, tcfg, device, weights, start=start,
                          on_checkpoint=on_checkpoint, on_log=on_log, on_val=on_val)
    step, log, val = result["step"], result["log"], result["val_log"][-1]

    # ---- final artifact: weights only + round-trip check ----
    model_path = art_dir / "model.pt"
    save_checkpoint(model_path, {"model": model.state_dict(), **info(step)})
    reloaded = build_world_model(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    probe = data.gather(data.val_starts[:2])[0]
    model.eval(); reloaded.eval()
    with torch.no_grad():
        same = torch.equal(model.encode(probe), reloaded.encode(probe))
    (art_dir / "checkpoint.pt").unlink(missing_ok=True)

    real, decoded = decoded_rollout(model, data, data.fixed_val_starts(1, tcfg["val_seed"])[0], device)
    strip_path = art_dir / "rollout_filmstrip.png"
    imageio.imwrite(strip_path, rollout_filmstrip(real, decoded))

    n = min(100, max(1, len(log) // 5))
    first, last = loss_window_means(log, n)
    it_s = result["it_per_s"]
    peak = peak_gpu_mb(device)
    summary = {
        **info(step), "complete": result["complete"], "resumed_from_step": start["step"] if start else 0,
        "data_summary": data.summary(), "it_per_s": it_s, "peak_gpu_mb": peak,
        "loss_first": first, "loss_last": last, "loss_window": n,
        "val": val, "val_log": result["val_log"], "train_log": log,
    }
    (art_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- checks ----
    report.check_true("losses_finite", all_finite(log), "all finite" if all_finite(log) else "NaN/inf",
                      "no NaN/inf")
    report.check_true("loss_fell", last["loss"] < first["loss"],
                      f"{first['loss']:.4f} -> {last['loss']:.4f}", f"mean of last {n} < first {n} steps")
    report.check_true("z_std", val["z_std"] > 0.1, f"{val['z_std']:.3f}", "> 0.1, collapse alarm")
    report.check_true("checkpoint_roundtrip", same, "identical latents" if same else "mismatch",
                      "model.pt reload == trained model")
    beat = val["nmse"][-1] < val["nmse_copy_last"][-1]
    report.check("nmse_k10_vs_copy_last", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"{val['nmse'][-1]:.3f} vs {val['nmse_copy_last'][-1]:.3f}",
                 "model < copy-last; WARN only in smoke")

    # ---- metrics ----
    ds = data.summary()
    report.metric("data", data_name)
    report.metric("steps", f"{step} (batch {tcfg['batch_size']})")
    report.metric("resumed_from_step", start["step"] if start else 0)
    report.metric("train_windows", ds["train_windows"])
    report.metric("val_source", ds["val_source"])
    report.metric("val_windows", val["windows"])
    for k in ("loss", "rec", "lat", "img"):
        report.metric(f"{k}_first_last", f"{first[k]:.4f} -> {last[k]:.4f}")
    report.metric("it_per_s", f"{it_s:.2f}" if it_s else "n/a")
    report.metric("projected_30k_steps", format_duration(FULL_STEPS / it_s) if it_s else "n/a")
    report.metric("peak_gpu_mem", f"{peak:.0f} MiB" if peak is not None else "n/a (cpu)")
    for k in KS:
        report.metric(f"val_nmse_k{k}", f"{val['nmse'][k - 1]:.3f} (copy-last {val['nmse_copy_last'][k - 1]:.3f})")
    report.metric("val_z_std", f"{val['z_std']:.3f}")
    report.metric("val_loss", f"{val['loss']:.4f}")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")

    saved = ctx.store.push(ARTIFACT, art_dir, ctx.meta(complete=result["complete"] and not report.failed,
                                                       step=step))
    for p in (model_path, art_dir / "train_summary.json", art_dir / "config.yaml", strip_path):
        report.save(p)
    report.save(saved)

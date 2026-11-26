"""Job diagnose_wm (Phase 3, eval only): where does the world model's train/val gap come from?

Pulls `ckpt-wm` (smoke mode falls back to `ckpt-wm-smoke`) and compares train episodes with val
episodes (seeds 190-199):
  1. each loss term (rec, lat, img, var) on 256 train and 256 val windows (+ open-loop NMSE_k);
  2. single-frame encode -> decode MSE, all pixels and non-background pixels only;
  3. a ridge probe z -> state fit on train frames, evaluated on val frames, and applied to the
     imagined latents at k = 1, 5, 10 (next to real z_{t+k} and copy-last z_t);
  4. a filmstrip of encode -> decode reconstructions of 8 val frames.
Writes diagnose_wm.json + recon_filmstrip_val.png to the job's output dir. Nothing is trained,
nothing goes to the store (jobs.yaml `makes: []`, so it reruns every time).
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from wm.data.windows import WindowData, episode_frames, frames_to_model, split_episodes
from wm.envs.pusht import denormalize_image
from wm.eval.diagnose import (TERMS, encode_frames, imagined_probe, recon_errors, ridge_probe,
                              subset, window_terms)
from wm.eval.rollouts import latent_nmse
from wm.jobs import JobContext
from wm.jobs._trainer import load_npz
from wm.models.world_model import load_world_model, param_counts
from wm.report import Report
from wm.viz.media import rollout_filmstrip

WINDOWS = 256          # per split, for loss terms and imagined-latent probes
MAX_FRAMES = 4096      # train frames for reconstruction + ridge fit (val uses all of its frames)
RIDGE_LAMBDA = 1.0
FILM_FRAMES = 8
KS = (1, 5, 10)


def fmt_state(e: dict[str, float]) -> str:
    return (f"pusher {e['pusher_px_mean']:.1f} px / block {e['block_px_mean']:.1f} px / "
            f"angle {e['block_deg_mean']:.1f} deg (medians {e['pusher_px_median']:.1f} / "
            f"{e['block_px_median']:.1f} / {e['block_deg_median']:.1f})")


def all_finite(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    return not isinstance(obj, float) or math.isfinite(obj)


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    device = ctx.device
    ckpt_name = ctx.store.resolve("ckpt-wm")
    path = ctx.store.pull("ckpt-wm")
    if path is None or not (path / "model.pt").exists():
        raise RuntimeError("no finished 'ckpt-wm' (model.pt) in the store: run train_wm first")
    ckpt_meta = ctx.store.metadata("ckpt-wm") or {}
    model, blob = load_world_model(path / "model.pt", device)
    cfg = blob["config"]
    counts = param_counts(model)
    report.params = f"{counts['total']:,}"
    weights = dict(cfg["loss"])

    arrays, data_name = load_npz(ctx)
    c, h = cfg["window"]["context"], cfg["window"]["horizon"]
    data = WindowData(arrays, device, cfg["val"]["seeds"], window=c + h)
    states = arrays["states"]
    train_eps, val_eps, _ = split_episodes(arrays["episode_seeds"], cfg["val"]["seeds"])
    starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
    train_frames = subset(torch.from_numpy(episode_frames(starts[train_eps], lengths[train_eps])), MAX_FRAMES)
    val_frames = torch.from_numpy(episode_frames(starts[val_eps], lengths[val_eps]))

    # ---- 1. loss terms + open-loop NMSE on train vs val windows ----
    z_ref = encode_frames(model, data.frames[val_frames.to(device)])
    splits = {"train": subset(data.train_starts, WINDOWS), "val": subset(data.val_starts, WINDOWS)}
    terms, nmse, windows = {}, {}, {}
    for split, ws in splits.items():
        terms[split], z, z_hat = window_terms(model, data, ws, weights)
        nmse[split] = {"model": latent_nmse(z_hat, z[:, c:], z_ref).tolist(),
                       "copy_last": latent_nmse(z[:, c - 1:c].expand_as(z[:, c:]), z[:, c:], z_ref).tolist()}
        windows[split] = (ws, z, z_hat)

    # ---- 2. single-frame reconstruction ----
    mean_frame = data.train_mean_frame()
    recon = {"train": recon_errors(model, data.frames[train_frames.to(device)], mean_frame),
             "val": recon_errors(model, data.frames[val_frames.to(device)], mean_frame)}

    # ---- 3. ridge probe z -> state; then on imagined latents ----
    z_tr = encode_frames(model, data.frames[train_frames.to(device)])
    w, ridge = ridge_probe(z_tr, states[train_frames.numpy()], z_ref, states[val_frames.numpy()], RIDGE_LAMBDA)
    imagined = {split: imagined_probe(w, ws.numpy(), z, z_hat, states, c, KS)
                for split, (ws, z, z_hat) in windows.items()}

    # ---- 4. reconstruction filmstrip of 8 val frames ----
    pick = val_frames[np.linspace(0, len(val_frames) - 1, FILM_FRAMES).round().astype(int)].to(device)
    with torch.no_grad():
        x = data.frames[pick]
        recon_imgs = model.decode(model.encode(frames_to_model(x))).float().cpu().numpy()
    strip = rollout_filmstrip(x.cpu().numpy(), denormalize_image(recon_imgs))
    strip_path = ctx.out_dir / "recon_filmstrip_val.png"
    imageio.imwrite(strip_path, strip)
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({"diagnose/recon_filmstrip_val": wandb.Image(strip)})

    results = {
        "job": "diagnose_wm", "mode": ctx.mode, "git": ctx.git, "config_hash": ctx.config_hash,
        "ckpt": {"name": ckpt_name, "meta": ckpt_meta, "git": blob.get("git"), "step": blob.get("step"),
                 "config_hash": blob.get("config_hash")},
        "wm_config": cfg, "params": counts, "data": data_name, "val_source": data.val_source,
        "settings": {"windows": WINDOWS, "max_train_frames": MAX_FRAMES, "ridge_lambda": RIDGE_LAMBDA,
                     "fg_threshold": 0.1, "ks": list(KS)},
        "loss_terms": terms, "nmse": nmse, "recon": recon, "ridge": ridge, "ridge_imagined": imagined,
        "wall_clock_s": time.time() - t0,
    }
    json_path = ctx.out_dir / "diagnose_wm.json"
    json_path.write_text(json.dumps(results, indent=2))

    # ---- REPORT ----
    complete = bool(ckpt_meta.get("complete"))
    report.check("ckpt_complete", "PASS" if complete else "WARN",
                 f"{ckpt_name} step {blob.get('step')}", "complete: true")
    ok = all_finite({k: results[k] for k in ("loss_terms", "nmse", "recon", "ridge", "ridge_imagined")})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")

    report.metric("ckpt", f"{ckpt_name} (step {blob.get('step')}, git {blob.get('git')})")
    report.metric("data", data_name)
    report.metric("val_source", data.val_source)
    report.metric("windows", f"train {len(splits['train'])} / val {len(splits['val'])}")
    for split in ("train", "val"):
        tt = terms[split]
        report.metric(f"loss_terms_{split}", " ".join(f"{k} {tt[k]:.4f}" for k in TERMS))
    for split in ("train", "val"):
        report.metric(f"nmse_{split}", " / ".join(
            f"k{k} {nmse[split]['model'][k - 1]:.3f} (copy {nmse[split]['copy_last'][k - 1]:.3f})" for k in KS))
    for split in ("train", "val"):
        r = recon[split]
        report.metric(f"recon_{split}", f"mse {r['mse']:.5f} / non-background {r['mse_foreground']:.5f} "
                      f"(mean-frame {r['mean_frame_mse_foreground']:.5f}; {r['foreground_fraction']:.1%} "
                      f"of pixels; {r['frames']} frames)")
    report.metric("ridge_train_fit", fmt_state(ridge["train"]))
    report.metric("ridge_val", fmt_state(ridge["val"]))
    report.metric("ridge_val_predict_mean", fmt_state(ridge["val_predict_mean"]))
    for split in ("train", "val"):
        for k in KS:
            e = imagined[split][f"k{k}"]
            report.metric(f"ridge_imagined_{split}_k{k}",
                          f"block {e['imagined']['block_px_mean']:.1f} px / {e['imagined']['block_deg_mean']:.1f} deg, "
                          f"pusher {e['imagined']['pusher_px_mean']:.1f} px "
                          f"(real z: {e['real_z']['block_px_mean']:.1f} px / {e['real_z']['block_deg_mean']:.1f} deg / "
                          f"{e['real_z']['pusher_px_mean']:.1f} px; copy-last: {e['copy_last']['block_px_mean']:.1f} px / "
                          f"{e['copy_last']['block_deg_mean']:.1f} deg / {e['copy_last']['pusher_px_mean']:.1f} px)")
    report.save(json_path)
    report.save(strip_path)

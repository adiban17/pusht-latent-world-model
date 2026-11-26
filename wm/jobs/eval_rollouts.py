"""Job eval_rollouts (Phase 5, CLAUDE.md §9 Rollouts): 10-step open-loop prediction on test windows.

Methods: wm (A+B), wm_1step, pixel, copy_last; every checkpoint is the complete one whose
config_hash matches its producer's current config (`_ckpt.pull_matching`).
Pre-registered (PROGRESS.md, Phase 5):
  primary   = the CNN state probe on predicted frames: block CoG px, block angle deg, pusher px,
              plus pusher px of p_hat for proprio models;
  secondary = NMSE in the wm encoder's space (headline row `wm_via_frames` = E_wm(D_wm(z_hat)),
              like-for-like with the other methods, whose frames are re-encoded by E_wm; direct
              z_hat NMSE is row `wm`, secondary) and pixel MSE;
  references = `real` (probe on o_{t+k}: the judge's noise floor), `wm_recon` (D_wm(E_wm(o_{t+k}))),
              and the probe on all real test frames;
  95% percentile bootstrap CIs over windows, same resamples for every row (paired CIs).
Every predicted frame is clipped to [-0.5, 0.5] first.
Writes rollouts.json, plots, filmstrips; pushes `results-rollouts`; full mode also writes
results/rollouts.json in the checkout.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm.data.windows import FrameData, WindowData, window_starts
from wm.envs.pusht import denormalize_image
from wm.eval.diagnose import encode_frames, subset
from wm.eval.rollouts import (PROBE_KEYS, all_finite, at_k, bootstrap_indices, diff_ci, latent_spread,
                              latent_sq_err, pixel_mse, predict_copy_last, predict_pixel, predict_wm,
                              probe_scores, pusher_px, recon_wm, reduction_ci, summarize)
from wm.jobs import JobContext
from wm.jobs._ckpt import pull_matching
from wm.jobs._trainer import load_npz
from wm.models.pixel_unet import build_pixel_unet
from wm.models.pixel_unet import param_counts as pixel_params
from wm.models.probe import build_probe
from wm.models.probe import param_counts as probe_params
from wm.models.world_model import load_world_model
from wm.models.world_model import param_counts as wm_params
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, seed_all
from wm.report import Report
from wm.train.train_probe import evaluate as probe_evaluate
from wm.viz.media import multirow_filmstrip
from wm.viz.plots import metric_vs_k

ARTIFACT = "results-rollouts"
RESULTS_NAME = "rollouts.json"
CONTEXT, HORIZON = 3, 10
WINDOW = CONTEXT + HORIZON
METHODS = ("wm", "wm_1step", "pixel", "copy_last")
CKPTS = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "pixel": "ckpt-pixel", "probe": "ckpt-probe"}
UNITS = {"block_px": "px", "block_deg": "deg", "pusher_px": "px", "p_hat_pusher_px": "px",
         "nmse": "", "pixel_mse": ""}
# Rows shown per metric, in table order (the first rows are the methods, then references).
ROWS = {
    "block_px": ("wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon"),
    "block_deg": ("wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon"),
    "pusher_px": ("wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon"),
    "p_hat_pusher_px": ("wm", "wm_1step", "copy_last"),
    "nmse": ("wm_via_frames", "wm_1step", "pixel", "copy_last", "wm", "wm_recon"),
    "pixel_mse": ("wm", "wm_1step", "pixel", "copy_last", "wm_recon"),
}


def load_model_pt(path: Path, build, device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Weights + saved info from a training job's model.pt (eval mode)."""
    blob = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model = build(blob["config"]).to(device)
    model.load_state_dict(blob["model"])
    return model.eval(), blob


def load_checkpoints(ctx: JobContext) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    """The four hash-matched checkpoints and a description of each (name, step, git, hashes)."""
    models, info = {}, {}
    for key, artifact in CKPTS.items():
        path, name, meta, expected = pull_matching(ctx, artifact)
        if not (path / "model.pt").exists():
            raise RuntimeError(f"'{name}' has no model.pt")
        if key in ("wm", "wm_1step"):
            model, blob = load_world_model(path / "model.pt", ctx.device)
            params = wm_params(model)
        else:
            model, blob = load_model_pt(path, build_pixel_unet if key == "pixel" else build_probe, ctx.device)
            params = (pixel_params if key == "pixel" else probe_params)(model)
        models[key] = model
        info[key] = {"artifact": name, "step": blob.get("step"), "git": blob.get("git"),
                     "config_hash": blob.get("config_hash"), "expected_config_hash": expected,
                     "meta_config_hash": meta.get("config_hash"), "params": params,
                     "proprio": getattr(model, "proprio", None),
                     "teacher_forcing": getattr(model, "teacher_forcing", None)}
    return models, info


@torch.no_grad()
def score_windows(models: dict[str, torch.nn.Module], data: WindowData, starts: torch.Tensor,
                  states: np.ndarray, chunk: int) -> dict[str, dict[str, np.ndarray]]:
    """Per-window scores [N, H] for every metric (block_px, ..., latent_sq, pixel_mse) and row."""
    wm, wm1, pixel, probe = models["wm"], models["wm_1step"], models["pixel"], models["probe"]
    acc: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in
                                                    ("block_px", "block_deg", "pusher_px",
                                                     "p_hat_pusher_px", "latent_sq", "pixel_mse")}

    def add(metric: str, row: str, v: np.ndarray) -> None:
        acc[metric].setdefault(row, []).append(v)

    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        frames, actions, pusher = data.gather(s, pusher=True)
        st = states[s.numpy()[:, None] + np.arange(CONTEXT, WINDOW)]          # true states at t+k
        real = frames[:, CONTEXT:]
        z_real = wm.encode(real)
        preds, p_hats = {}, {}
        preds["wm"], z_hat, p_hats["wm"] = predict_wm(wm, frames, actions, pusher)
        preds["wm_1step"], _, p_hats["wm_1step"] = predict_wm(wm1, frames, actions, pusher)
        preds["pixel"] = predict_pixel(pixel, frames, actions)
        preds["copy_last"] = predict_copy_last(frames, CONTEXT)
        preds["wm_recon"] = recon_wm(wm, real)
        for row, pred in {**preds, "real": real}.items():
            for key, v in probe_scores(probe, pred, st).items():
                add(key, row, v)
        for row, pred in preds.items():
            add("pixel_mse", row, pixel_mse(pred, real))
            # wm's own decoded frames re-encoded = the like-for-like headline row
            add("latent_sq", "wm_via_frames" if row == "wm" else row, latent_sq_err(wm.encode(pred), z_real))
        add("latent_sq", "wm", latent_sq_err(z_hat, z_real))                 # direct z_hat (secondary)
        for row, p in p_hats.items():
            if p is not None:
                add("p_hat_pusher_px", row, pusher_px(p, st))
        add("p_hat_pusher_px", "copy_last", pusher_px(pusher[:, CONTEXT - 1:CONTEXT].expand(-1, HORIZON, -1), st))
    return {m: {r: np.concatenate(v) for r, v in rows.items()} for m, rows in acc.items()}


@torch.no_grad()
def filmstrip_rows(models: dict[str, torch.nn.Module], data: WindowData, start: torch.Tensor) -> dict[str, np.ndarray]:
    """Frames t..t+10 for real | wm | wm_1step | pixel (predicted rows start with the real o_t)."""
    frames, actions, pusher = data.gather(start.reshape(1), pusher=True)
    o_t = frames[:, CONTEXT - 1:CONTEXT]
    rows = {"real": frames[:, CONTEXT - 1:],
            "wm": torch.cat([o_t, predict_wm(models["wm"], frames, actions, pusher)[0]], 1),
            "wm_1step": torch.cat([o_t, predict_wm(models["wm_1step"], frames, actions, pusher)[0]], 1),
            "pixel": torch.cat([o_t, predict_pixel(models["pixel"], frames, actions)], 1)}
    return {k: denormalize_image(v[0].float().cpu().numpy()) for k, v in rows.items()}


def fmt_ci(s: dict[str, float], digits: int) -> str:
    return f"{s['mean']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"


DIGITS = {"block_px": 1, "block_deg": 1, "pusher_px": 1, "p_hat_pusher_px": 1, "nmse": 3, "pixel_mse": 5}


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    n_windows = int(cfg["smoke"]["windows"] if ctx.mode == "smoke" else cfg["windows"])
    seed, ci, chunk = int(cfg["seed"]), float(cfg["ci"]), int(cfg["chunk"])
    seed_all(seed)
    device = ctx.device

    models, ckpts = load_checkpoints(ctx)
    report.params = " / ".join(f"{k} {v['params']['total']:,}" for k, v in ckpts.items())
    test_arrays, data_name = load_npz(ctx, "pusht_test.npz")
    train_arrays, _ = load_npz(ctx)
    data = WindowData(test_arrays, device, val_seeds=[190, 199], window=WINDOW)
    states = test_arrays["states"]
    all_starts = torch.from_numpy(window_starts(test_arrays["episode_starts"],
                                                test_arrays["episode_lengths"], WINDOW))
    starts = subset(all_starts, n_windows, seed)

    # ---- references: probe on every real test frame; wm latent spread of the test frames ----
    test_frames = FrameData(test_arrays, device, None)
    train_frames = FrameData(train_arrays, device, [190, 199])
    probe_all = probe_evaluate(models["probe"], test_frames, test_frames.val_idx,
                               train_frames.mean_train_target(), device, amp=False)
    z_test = encode_frames(models["wm"], test_frames.frames)
    spread = latent_spread(z_test)

    # ---- per-window scores and bootstrap ----
    scores = score_windows(models, data, starts, states, chunk)
    idx = bootstrap_indices(len(starts), int(cfg["bootstrap"]), seed)
    metrics = summarize(scores, idx, ci, spread)
    metrics = {m: {r: metrics[m][r] for r in ROWS[m] if r in metrics[m]} for m in ROWS}
    nm = {r: v / spread for r, v in scores["latent_sq"].items()}
    ks = list(cfg["report_ks"])
    paired = {
        "nmse_reduction_wm_via_frames_vs_pixel": reduction_ci(nm["wm_via_frames"], nm["pixel"], idx, ci),
        "nmse_reduction_wm_via_frames_vs_wm_1step": reduction_ci(nm["wm_via_frames"], nm["wm_1step"], idx, ci),
        "block_px_reduction_wm_vs_pixel": reduction_ci(scores["block_px"]["wm"], scores["block_px"]["pixel"], idx, ci),
        "block_px_diff_wm_minus_pixel": diff_ci(scores["block_px"]["wm"], scores["block_px"]["pixel"], idx, ci),
        "block_px_diff_wm_minus_copy_last": diff_ci(scores["block_px"]["wm"], scores["block_px"]["copy_last"], idx, ci),
    }

    # ---- outputs ----
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    (art / "plots").mkdir(parents=True)
    (art / "filmstrips").mkdir()
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    floor = ("probe, all real test frames", None)
    plots = []
    for metric, title, ylabel, log_y in (
            ("block_px", "Block position error (probe on predicted frames)", "block CoG error (px)", False),
            ("block_deg", "Block angle error (probe on predicted frames)", "block angle error (deg)", False),
            ("pusher_px", "Pusher error (probe; dashed = p_hat)", "pusher error (px)", False),
            ("nmse", "Latent NMSE in the wm encoder's space", "NMSE", False),
            ("pixel_mse", "Pixel MSE", "pixel MSE (normalised)", True)):
        dashed = ({f"{r} p_hat": v for r, v in metrics["p_hat_pusher_px"].items() if r != "copy_last"}
                  if metric == "pusher_px" else None)
        fl = (floor[0], probe_all[f"{metric}_mean"]) if metric in PROBE_KEYS else None
        plots.append(metric_vs_k(metrics[metric], title, ylabel, art / "plots" / f"{metric}_vs_k.png",
                                 dashed=dashed, floor=fl, log_y=log_y))
    picks = np.linspace(0, len(starts) - 1, min(int(cfg["filmstrips"]), len(starts))).round().astype(int)
    strips = []
    for j, p in enumerate(picks):
        strip = multirow_filmstrip(filmstrip_rows(models, data, starts[p]))
        strips.append(art / "filmstrips" / f"window_{j}_start{int(starts[p])}.png")
        imageio.imwrite(strips[-1], strip)

    entry = load_registry()["jobs"][ctx.name]
    inputs = input_hashes(entry)
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
        "checkpoints": ckpts,
        "windows": {"n": len(starts), "available": len(all_starts), "seed": seed,
                    "starts": starts.tolist(), "length": WINDOW, "context": CONTEXT, "horizon": HORIZON},
        "bootstrap": {"resamples": int(cfg["bootstrap"]), "ci": ci, "seed": seed, "paired": True},
        "pixel_clip": [-0.5, 0.5], "latent_spread_test": spread,
        "headline": {"nmse_row": "wm_via_frames", "nmse_note": "wm frames re-encoded by E_wm, like "
                     "wm_1step and pixel; row 'wm' (direct z_hat) is secondary"},
        "probe_test_all": probe_all,
        "ks": list(range(1, HORIZON + 1)), "metrics": metrics, "paired": paired,
        "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({**{f"rollouts/{p.stem}": wandb.Image(str(p)) for p in plots},
                           **{f"rollouts/filmstrip_{j}": wandb.Image(str(p)) for j, p in enumerate(strips)}})

    # ---- CHECKS ----
    for key, c in ckpts.items():
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    ok = all_finite({"metrics": metrics, "paired": paired, "probe": probe_all})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    report.check_true("windows", len(starts) == n_windows, f"{len(starts)} of {len(all_starts)} test windows",
                      f"== {n_windows}")
    beat = probe_all["block_px_mean"] < probe_all["baseline_block_px_mean"]
    report.check("probe_floor", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"block {probe_all['block_px_mean']:.1f} px vs predict-mean {probe_all['baseline_block_px_mean']:.1f}",
                 "probe on all real test frames < predict-train-mean; WARN only in smoke")
    ang = probe_all["block_deg_mean"]
    floor_deg = float(cfg["probe_angle_floor_deg"])
    report.check("probe_angle_floor", "PASS" if ang < floor_deg else "WARN",
                 f"{ang:.1f} deg" + ("" if ang < floor_deg else
                                     " - angle columns unreliable — judge cannot read angle from real frames"),
                 f"< {floor_deg:g} deg mean on all real test frames; WARN only")
    nmse_ok = [metrics["nmse"]["wm_via_frames"]["mean"][k] < metrics["nmse"]["copy_last"]["mean"][k]
               for k in range(HORIZON)]
    direct_ok = sum(metrics["nmse"]["wm"]["mean"][k] < metrics["nmse"]["copy_last"]["mean"][k] for k in range(HORIZON))
    report.check("nmse_wm_vs_copy_last_all_k", "PASS" if all(nmse_ok) else "WARN",
                 f"wm_via_frames < copy_last at {sum(nmse_ok)} of {HORIZON} k (direct z_hat: {direct_ok} of {HORIZON})",
                 "all k; WARN only (a result, not a bug)")
    b10 = {r: metrics["block_px"][r]["mean"][HORIZON - 1] for r in ("wm", "copy_last")}
    report.check("block_px_wm_vs_copy_last_k10", "PASS" if b10["wm"] < b10["copy_last"] else "WARN",
                 f"{b10['wm']:.1f} vs {b10['copy_last']:.1f} px", "wm < copy_last; WARN only")

    # ---- METRICS ----
    for key, c in ckpts.items():
        extra = f", proprio {c['proprio']}" if c["proprio"] is not None else ""
        report.metric(f"ckpt_{key}", f"{c['artifact']} step {c['step']}, git {c['git']}, "
                                     f"config_hash {c['config_hash']}{extra}")
    report.metric("data", data_name)
    report.metric("windows", f"{len(starts)} test windows of {WINDOW} frames (seed {seed}); "
                             f"bootstrap {cfg['bootstrap']} paired resamples, {ci:.0%} CI")
    report.metric("probe_test_all", " / ".join(
        f"{k} mean {probe_all[f'{k}_mean']:.1f} median {probe_all[f'{k}_median']:.1f}" for k in PROBE_KEYS)
        + f" ({probe_all['frames']} frames; predict-mean block {probe_all['baseline_block_px_mean']:.1f} px)")
    report.metric("note", "rows real / wm_recon = probe on o_{t+k} / on D_wm(E_wm(o_{t+k})) (noise floors); "
                          "headline NMSE = wm_via_frames; nmse 'wm' = direct z_hat (secondary)")
    for metric in ROWS:
        for k in ks:
            rows = metrics[metric]
            report.metric(f"{metric}_k{k}", " | ".join(
                f"{r} {fmt_ci(at_k(s, k), DIGITS[metric])}" for r, s in rows.items()))
    for name, d in (("nmse_reduction_vs_pixel", "nmse_reduction_wm_via_frames_vs_pixel"),
                    ("nmse_reduction_vs_wm_1step", "nmse_reduction_wm_via_frames_vs_wm_1step"),
                    ("block_px_reduction_vs_pixel", "block_px_reduction_wm_vs_pixel")):
        report.metric(name, " | ".join(f"k{k} {fmt_ci(at_k(paired[d], k), 3)}" for k in (5, 10)))
    report.metric("block_px_wm_minus_pixel", " | ".join(
        f"k{k} {fmt_ci(at_k(paired['block_px_diff_wm_minus_pixel'], k), 1)} px" for k in (5, 10)))
    report.metric("block_px_wm_minus_copy_last_k10",
                  f"{fmt_ci(at_k(paired['block_px_diff_wm_minus_copy_last'], HORIZON), 1)} px")

    # ---- SAVED ----
    report.save(json_path)
    report.save(art / "plots")
    report.save(art / "filmstrips")
    if ctx.mode == "full":   # smoke never writes into the repo's results/
        repo_json = REPO_ROOT / "results" / RESULTS_NAME
        repo_json.parent.mkdir(exist_ok=True)
        shutil.copy2(json_path, repo_json)
        report.save(repo_json)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

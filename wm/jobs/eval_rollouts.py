"""Job eval_rollouts (Phase 5, judges Phase 5b; CLAUDE.md §9 Rollouts): 10-step open-loop
prediction on test windows.

Methods: wm (A+B), wm_1step, pixel, copy_last; every checkpoint is the complete one whose
config_hash matches its producer's current config (`_ckpt.load_matching`).
Judges (Phase 5b, pre-registered in PROGRESS.md), reported side by side:
  probe_mix = image judge (`ckpt-probe-mix`) on predicted frames; rows wm, wm_1step, pixel,
              copy_last + references real (o_{t+k}), wm_recon, wm1step_recon;
  latent    = MLP on wm latents (`ckpt-latent-probe`); rows wm (direct z_hat), wm_via_frames
              (E_wm(D_wm(z_hat))), wm_1step / pixel (E_wm of their frames), copy_last (E_wm(o_t)),
              real (E_wm(o_{t+k}): the floor);
  probe_v1  = the Phase 5 probe (`ckpt-probe`), kept as "deprecated (judge failure)".
Headline rule: "wm beats X on block px at k" only if both judges agree in sign, both paired 95%
CIs exclude 0, and the image judge passed acceptance. The REPORT also answers "is block angle
in the wm latent?" (latent judge on real test latents; < 20 deg present, > 45 absent) beside a
non-learned template oracle, and gives every judge's angle error on rendered grid poses.
Secondary (unchanged): NMSE in the wm encoder's space (headline row `wm_via_frames`), pixel MSE,
p_hat pusher error. 95% percentile bootstrap CIs, same resamples for every row (paired).
Every predicted frame is clipped to [-0.5, 0.5] first.
Writes rollouts.json (schema 2), plots, filmstrips; pushes `results-rollouts`; full mode also
writes results/rollouts.json in the checkout.
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

from wm.data.windows import (FrameData, WindowData, episode_frames, split_episodes, state_targets,
                             window_starts)
from wm.envs.pusht import denormalize_image, make_env
from wm.eval.angle_audit import angle_err_deg, scoring_grid, template_angle
from wm.eval.diagnose import encode_frames, subset
from wm.eval.judges import baseline_errors, fmt_errs, frames_u8_to_model, judge_errors, summarize_errors
from wm.eval.rollouts import (PROBE_KEYS, all_finite, at_k, bootstrap_indices, diff_ci, headline_claim,
                              judge_scores, latent_spread, latent_sq_err, pixel_mse, predict_copy_last,
                              predict_pixel, predict_wm, pusher_px, recon_wm, reduction_ci, summarize)
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._judges import angle_verdict
from wm.jobs._trainer import load_npz
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, seed_all
from wm.report import Report
from wm.train.train_probe import evaluate as probe_evaluate
from wm.viz.media import multirow_filmstrip
from wm.viz.plots import metric_vs_k

ARTIFACT = "results-rollouts"
RESULTS_NAME = "rollouts.json"
SCHEMA = 2
CONTEXT, HORIZON = 3, 10
WINDOW = CONTEXT + HORIZON
METHODS = ("wm", "wm_1step", "pixel", "copy_last")
JUDGE_CKPTS = {"probe_v1": "ckpt-probe", "probe_mix": "ckpt-probe-mix", "latent": "ckpt-latent-probe"}
CKPTS = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "pixel": "ckpt-pixel", **JUDGE_CKPTS}
JUDGES = tuple(JUDGE_CKPTS)
IMAGE_JUDGES = ("probe_v1", "probe_mix")
DEPRECATED = "deprecated (judge failure)"
# Rows per judge, in table order (methods first, then references).
JUDGE_ROWS = {
    "probe_v1": ("wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon"),
    "probe_mix": ("wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon", "wm1step_recon"),
    "latent": ("wm", "wm_via_frames", "wm_1step", "pixel", "copy_last", "real"),
}
OTHER_ROWS = {
    "p_hat_pusher_px": ("wm", "wm_1step", "copy_last"),
    "nmse": ("wm_via_frames", "wm_1step", "pixel", "copy_last", "wm", "wm_recon"),
    "pixel_mse": ("wm", "wm_1step", "pixel", "copy_last", "wm_recon"),
}
ROWS = {**{f"{j}/{k}": JUDGE_ROWS[j] for j in JUDGES for k in PROBE_KEYS}, **OTHER_ROWS}
DIGITS = {"block_px": 1, "block_deg": 1, "pusher_px": 1, "p_hat_pusher_px": 1, "nmse": 3, "pixel_mse": 5}
UNITS = {"block_px": "px", "block_deg": "deg", "pusher_px": "px"}


def digits(metric: str) -> int:
    return DIGITS[metric.split("/")[-1]]


def load_checkpoints(ctx: JobContext) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    """Every hash-matched checkpoint and a description of each (name, step, git, hashes, summary)."""
    models, info = {}, {}
    for key, artifact in CKPTS.items():
        models[key], info[key] = load_matching(ctx, artifact)
    return models, info


@torch.no_grad()
def score_windows(models: dict[str, torch.nn.Module], data: WindowData, starts: torch.Tensor,
                  states: np.ndarray, chunk: int) -> dict[str, dict[str, np.ndarray]]:
    """Per-window scores [N, H] for every metric (`<judge>/block_px`, ..., latent_sq, pixel_mse)
    and row."""
    wm, wm1, pixel = models["wm"], models["wm_1step"], models["pixel"]
    acc: dict[str, dict[str, list[np.ndarray]]] = {}

    def add(metric: str, row: str, v: np.ndarray) -> None:
        acc.setdefault(metric, {}).setdefault(row, []).append(v)

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
        preds["wm1step_recon"] = recon_wm(wm1, real)
        frames_by_row = {**preds, "real": real}
        z_frames = {row: wm.encode(preds[row]) for row in ("wm", "wm_1step", "pixel", "copy_last", "wm_recon")}
        # image judges on frames
        for judge in IMAGE_JUDGES:
            for row in JUDGE_ROWS[judge]:
                for key, v in judge_scores(models[judge], frames_by_row[row], st).items():
                    add(f"{judge}/{key}", row, v)
        # latent judge on latents
        latents = {"wm": z_hat, "wm_via_frames": z_frames["wm"], "wm_1step": z_frames["wm_1step"],
                   "pixel": z_frames["pixel"], "copy_last": z_frames["copy_last"], "real": z_real}
        for row in JUDGE_ROWS["latent"]:
            for key, v in judge_scores(models["latent"], latents[row], st).items():
                add(f"latent/{key}", row, v)
        # pixel MSE and latent NMSE (Phase 5 rows, unchanged)
        for row in OTHER_ROWS["pixel_mse"]:
            add("pixel_mse", row, pixel_mse(preds[row], real))
        for row, z in z_frames.items():
            add("latent_sq", "wm_via_frames" if row == "wm" else row, latent_sq_err(z, z_real))
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


def fmt_ci(s: dict[str, float], n: int) -> str:
    return f"{s['mean']:.{n}f} [{s['lo']:.{n}f}, {s['hi']:.{n}f}]"


def judge_status(key: str, info: dict[str, Any]) -> dict[str, Any]:
    """Status of a judge for the JSON and REPORT."""
    if key == "probe_v1":
        return {"status": DEPRECATED, "accepted": False}
    if key == "probe_mix":
        s = info.get("summary", {})
        ok = bool(s.get("accepted", False))
        return {"status": "accepted" if ok else "NOT accepted (rows unreliable)", "accepted": ok,
                "pool": s.get("chosen"), "acceptance": s.get("acceptance"), "test_floors": s.get("test"),
                "best_step": s.get("best_step")}
    return {"status": "reported (no gate)", "accepted": True, "best_step": info.get("summary", {}).get("best_step")}


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    smoke = ctx.mode == "smoke"
    n_windows = int(cfg["smoke"]["windows"] if smoke else cfg["windows"])
    n_oracle = int(cfg["smoke"]["oracle_frames"] if smoke else cfg["oracle_frames"])
    seed, ci, chunk = int(cfg["seed"]), float(cfg["ci"]), int(cfg["chunk"])
    jcfg = cfg["judges"]
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
    status = {j: judge_status(j, ckpts[j]) for j in JUDGES}
    image_ok = status["probe_mix"]["accepted"]

    # ---- references: every judge on all real test frames; grid poses; template oracle ----
    test_frames = FrameData(test_arrays, device, None)
    train_frames = FrameData(train_arrays, device, [190, 199])
    tr_eps, _, _ = split_episodes(train_arrays["episode_seeds"], [190, 199])
    mean_target = state_targets(train_arrays["states"][episode_frames(train_arrays["episode_starts"][tr_eps],
                                                                      train_arrays["episode_lengths"][tr_eps])]).mean(0)
    z_test = encode_frames(models["wm"], test_frames.frames)
    spread = latent_spread(z_test)
    test_all = {"probe_v1": probe_evaluate(models["probe_v1"], test_frames, test_frames.val_idx,
                                           train_frames.mean_train_target(), device, amp=False)}
    for j, x in (("probe_mix", frames_u8_to_model(test_arrays["frames"])), ("latent", z_test)):
        test_all[j] = {**summarize_errors(judge_errors(models[j], x, states)),
                       "baseline": baseline_errors(mean_target, states)}
    env = make_env()
    env.reset(seed=seed)
    grid_f, grid_s = scoring_grid(env, cfg["grid"]["step_deg"], cfg["grid"]["positions"])
    grid_x = frames_u8_to_model(grid_f)
    grid = {j: summarize_errors(judge_errors(models[j], grid_x if j != "latent" else
                                             encode_frames(models["wm"], torch.from_numpy(grid_f).to(device)),
                                             grid_s)) for j in JUDGES}
    o_idx = np.sort(np.random.default_rng(seed).choice(len(states), min(n_oracle, len(states)), replace=False))
    o_err = np.array([float(angle_err_deg(template_angle(env, test_arrays["frames"][i].astype(np.float32),
                                                         states[i]), states[i, 4])) for i in o_idx])
    oracle = {"n": int(len(o_err)), "angle_deg_mean": float(o_err.mean()), "angle_deg_median": float(np.median(o_err))}
    av = jcfg["angle_verdict"]
    lat_ang = test_all["latent"]["block_deg_mean"]
    angle_in_latent = {"angle_deg_mean": lat_ang,
                       "verdict": angle_verdict(lat_ang, float(av["present_deg"]), float(av["absent_deg"])),
                       "template_oracle_deg": oracle["angle_deg_mean"],
                       "predict_mean_deg": test_all["latent"]["baseline"]["block_deg_mean"], "thresholds": av}

    # ---- per-window scores and bootstrap ----
    scores = score_windows(models, data, starts, states, chunk)
    idx = bootstrap_indices(len(starts), int(cfg["bootstrap"]), seed)
    metrics = summarize(scores, idx, ci, spread)
    metrics = {m: {r: metrics[m][r] for r in ROWS[m] if r in metrics[m]} for m in ROWS}
    nm = {r: v / spread for r, v in scores["latent_sq"].items()}
    paired = {
        "nmse_reduction_wm_via_frames_vs_pixel": reduction_ci(nm["wm_via_frames"], nm["pixel"], idx, ci),
        "nmse_reduction_wm_via_frames_vs_wm_1step": reduction_ci(nm["wm_via_frames"], nm["wm_1step"], idx, ci),
    }
    for j in JUDGES:
        bp = scores[f"{j}/block_px"]
        for other in ("pixel", "wm_1step", "copy_last"):
            paired[f"{j}/block_px_diff_wm_minus_{other}"] = diff_ci(bp["wm"], bp[other], idx, ci)
        paired[f"{j}/block_px_reduction_wm_vs_pixel"] = reduction_ci(bp["wm"], bp["pixel"], idx, ci)
    headline = {f"wm_vs_{other}_k{k}": headline_claim(
        at_k(paired[f"probe_mix/block_px_diff_wm_minus_{other}"], k),
        at_k(paired[f"latent/block_px_diff_wm_minus_{other}"], k), image_ok)
        for other in ("pixel", "wm_1step", "copy_last") for k in jcfg["headline_ks"]}

    # ---- outputs ----
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    (art / "plots").mkdir(parents=True)
    (art / "filmstrips").mkdir()
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    plots = []
    for j in JUDGES:
        tag = " [deprecated: judge failure]" if j == "probe_v1" else ""
        on = "predicted frames" if j != "latent" else "wm latents"
        for key, title, ylabel in (("block_px", "Block position error", "block CoG error (px)"),
                                   ("block_deg", "Block angle error", "block angle error (deg)"),
                                   ("pusher_px", "Pusher error (dashed = p_hat)", "pusher error (px)")):
            dashed = ({f"{r} p_hat": v for r, v in metrics["p_hat_pusher_px"].items() if r != "copy_last"}
                      if key == "pusher_px" else None)
            plots.append(metric_vs_k(metrics[f"{j}/{key}"], f"{title} ({j} on {on}){tag}", ylabel,
                                     art / "plots" / f"{j}_{key}_vs_k.png", dashed=dashed,
                                     floor=(f"{j}, all real test frames", test_all[j][f"{key}_mean"])))
    plots.append(metric_vs_k(metrics["nmse"], "Latent NMSE in the wm encoder's space", "NMSE",
                             art / "plots" / "nmse_vs_k.png"))
    plots.append(metric_vs_k(metrics["pixel_mse"], "Pixel MSE", "pixel MSE (normalised)",
                             art / "plots" / "pixel_mse_vs_k.png", log_y=True))
    picks = np.linspace(0, len(starts) - 1, min(int(cfg["filmstrips"]), len(starts))).round().astype(int)
    strips = []
    for n_, p in enumerate(picks):
        strips.append(art / "filmstrips" / f"window_{n_}_start{int(starts[p])}.png")
        imageio.imwrite(strips[-1], multirow_filmstrip(filmstrip_rows(models, data, starts[p])))

    entry = load_registry()["jobs"][ctx.name]
    inputs = input_hashes(entry)
    results = {
        "schema": SCHEMA, "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
        "checkpoints": ckpts,
        "windows": {"n": len(starts), "available": len(all_starts), "seed": seed,
                    "starts": starts.tolist(), "length": WINDOW, "context": CONTEXT, "horizon": HORIZON},
        "bootstrap": {"resamples": int(cfg["bootstrap"]), "ci": ci, "seed": seed, "paired": True},
        "pixel_clip": [-0.5, 0.5], "latent_spread_test": spread,
        "judges": {j: {**status[j], "test_all": test_all[j], "grid": grid[j], "rows": list(JUDGE_ROWS[j])}
                   for j in JUDGES},
        "angle_in_wm_latent": angle_in_latent, "template_oracle_test": oracle,
        "headline": {"nmse_row": "wm_via_frames", "nmse_note": "wm frames re-encoded by E_wm, like "
                     "wm_1step and pixel; row 'wm' (direct z_hat) is secondary",
                     "judge_rule": "claim only if probe_mix (accepted) and latent agree in sign and both "
                                   "paired 95% CIs exclude 0; latent 'wm' = direct z_hat",
                     "block_px": headline},
        "ks": list(range(1, HORIZON + 1)), "metrics": metrics, "paired": paired,
        "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({**{f"rollouts/{p.stem}": wandb.Image(str(p)) for p in plots},
                           **{f"rollouts/filmstrip_{n_}": wandb.Image(str(p)) for n_, p in enumerate(strips)}})

    # ---- CHECKS ----
    for key, c in ckpts.items():
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    ok = all_finite({"metrics": metrics, "paired": paired, "judges": results["judges"]})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    report.check_true("windows", len(starts) == n_windows, f"{len(starts)} of {len(all_starts)} test windows",
                      f"== {n_windows}")
    floor_deg = float(cfg["probe_angle_floor_deg"])
    for j in ("probe_mix", "latent"):
        ta = test_all[j]
        beat = ta["block_px_mean"] < ta["baseline"]["block_px_mean"]
        report.check(f"judge_floor_{j}", "PASS" if beat else ("WARN" if smoke else "FAIL"),
                     f"block {ta['block_px_mean']:.1f} px vs predict-mean {ta['baseline']['block_px_mean']:.1f}",
                     "judge on all real test frames < predict-train-mean; WARN only in smoke")
        a = ta["block_deg_mean"]
        report.check(f"probe_angle_floor_{j}", "PASS" if a < floor_deg else "WARN",
                     f"{a:.1f} deg" + ("" if a < floor_deg else " - angle columns unreliable for this judge"),
                     f"< {floor_deg:g} deg mean on all real test frames; WARN only")
    report.check("judge_accepted_probe_mix", "PASS" if image_ok else "WARN", status["probe_mix"]["status"],
                 "pre-registered acceptance in train_probe_mix; WARN only (no headline claim without it)")
    nmse_ok = [metrics["nmse"]["wm_via_frames"]["mean"][k] < metrics["nmse"]["copy_last"]["mean"][k]
               for k in range(HORIZON)]
    direct_ok = sum(metrics["nmse"]["wm"]["mean"][k] < metrics["nmse"]["copy_last"]["mean"][k] for k in range(HORIZON))
    report.check("nmse_wm_vs_copy_last_all_k", "PASS" if all(nmse_ok) else "WARN",
                 f"wm_via_frames < copy_last at {sum(nmse_ok)} of {HORIZON} k (direct z_hat: {direct_ok} of {HORIZON})",
                 "all k; WARN only (a result, not a bug)")
    for j in ("probe_mix", "latent"):
        b10 = {r: metrics[f"{j}/block_px"][r]["mean"][HORIZON - 1] for r in ("wm", "copy_last")}
        report.check(f"block_px_wm_vs_copy_last_k10_{j}", "PASS" if b10["wm"] < b10["copy_last"] else "WARN",
                     f"{b10['wm']:.1f} vs {b10['copy_last']:.1f} px", "wm < copy_last; WARN only")

    # ---- METRICS ----
    for key, c in ckpts.items():
        extra = f", proprio {c['proprio']}" if c["proprio"] is not None else ""
        report.metric(f"ckpt_{key}", f"{c['artifact']} step {c['step']}, git {c['git']}, "
                                     f"config_hash {c['config_hash']}{extra}")
    report.metric("data", data_name)
    report.metric("windows", f"{len(starts)} test windows of {WINDOW} frames (seed {seed}); "
                             f"bootstrap {cfg['bootstrap']} paired resamples, {ci:.0%} CI")
    report.metric("angle_in_wm_latent", f"{lat_ang:.1f} deg -> {angle_in_latent['verdict']} | template oracle "
                                        f"{oracle['angle_deg_mean']:.1f} deg ({oracle['n']} real test frames, true "
                                        f"position given) | predict-mean {angle_in_latent['predict_mean_deg']:.1f} deg")
    for j in JUDGES:
        st = status[j]
        extra = f", pool {st.get('pool')}" if j == "probe_mix" else ""
        report.metric(f"judge_{j}", f"{st['status']}{extra}; all real test frames: {fmt_errs(test_all[j])}")
    report.metric("grid_angle_deg", " | ".join(
        f"{j} {grid[j]['block_deg_mean']:.1f}" + (" (before)" if j == "probe_v1" else "") for j in JUDGES)
        + f" ({grid['latent']['n']} rendered poses)")
    report.metric("note", "probe_mix rows real / wm_recon / wm1step_recon and latent row real = floors; "
                          "latent 'wm' = direct z_hat, 'wm_via_frames' = E_wm(D_wm(z_hat)); "
                          "headline NMSE = wm_via_frames")
    for j in ("probe_mix", "latent"):
        for key in PROBE_KEYS:
            for k in cfg["report_ks"]:
                m = f"{j}/{key}"
                report.metric(f"{j}_{key}_k{k}", " | ".join(
                    f"{r} {fmt_ci(at_k(s, k), digits(m))}" for r, s in metrics[m].items()))
    for key in PROBE_KEYS:
        m = f"probe_v1/{key}"
        report.metric(f"probe_v1_{key}_k10", f"[{DEPRECATED}] " + " | ".join(
            f"{r} {at_k(s, HORIZON)['mean']:.{digits(m)}f}" for r, s in metrics[m].items()))
    for metric in OTHER_ROWS:
        for k in cfg["report_ks"]:
            report.metric(f"{metric}_k{k}", " | ".join(
                f"{r} {fmt_ci(at_k(s, k), digits(metric))}" for r, s in metrics[metric].items()))
    for name, d in (("nmse_reduction_vs_pixel", "nmse_reduction_wm_via_frames_vs_pixel"),
                    ("nmse_reduction_vs_wm_1step", "nmse_reduction_wm_via_frames_vs_wm_1step")):
        report.metric(name, " | ".join(f"k{k} {fmt_ci(at_k(paired[d], k), 3)}" for k in (5, 10)))
    for j in ("probe_mix", "latent"):
        for other in ("pixel", "wm_1step", "copy_last"):
            report.metric(f"{j}_block_px_wm_minus_{other}", " | ".join(
                f"k{k} {fmt_ci(at_k(paired[f'{j}/block_px_diff_wm_minus_{other}'], k), 1)} px" for k in (5, 10)))
    for name, h in headline.items():
        report.metric(f"headline_{name}", f"{h['claim']} (image {h['image']}, latent {h['latent']})")

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

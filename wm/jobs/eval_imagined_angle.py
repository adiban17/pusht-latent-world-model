"""Job eval_imagined_angle (Phase 7, pre-registered in PROGRESS.md): does an imagined frame still
show the block's orientation? The non-learned template oracle reads the angle of every frame type
on eval_rollouts' 500 test windows (same windows, same bootstrap resamples) at k = 1, 5, 10.

Rows: real o_{t+k} and wm_recon = clip(D_wm(E_wm(o_{t+k}))) (floors), wm (decoded z_hat),
wm_1step (decoded), pixel (predicted frames), copy_last (o_t); every frame clipped to [-0.5, 0.5].
Readers (candidates rendered every 5 deg with the block's CoG and the pusher at their true
positions at t+k, rendered once per window and k and shared by all rows):
  search   (primary)   best (shift, angle) over every integer shift |dy|, |dx| <= 4 image px,
                       scored on the central 56x56 crop: separates angle from position error;
  truepos  (secondary) shift 0 on the full frame (= wm.eval.angle_audit.template_angle).
Reference row copy_last_exact = |theta_t - theta_{t+k}| (copy_last read perfectly).
Also: the oracle-found shift (world px) and the share of windows that hit the search limit.
Verdict (pre-registered): wm.eval.final.rollout_angle_verdict at k = 10 on the search row.
Writes imagined_angle.json + a plot, pushes `results-imagined-angle`; full mode also writes
results/final/imagined_angle.json in the checkout.
"""
from __future__ import annotations

import json
import shutil
import time
from typing import Any

import numpy as np
import torch
import yaml

from wm.data.windows import WindowData, window_starts
from wm.envs.pusht import make_env
from wm.eval.angle_audit import PX_PER_CELL, angle_err_deg, template_candidates, template_match
from wm.eval.diagnose import subset
from wm.eval.final import rollout_angle_verdict
from wm.eval.judges import model_to_u8_float
from wm.eval.rollouts import (all_finite, bootstrap_indices, diff_ci, mean_ci, predict_copy_last, predict_pixel,
                              predict_wm, recon_wm)
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching, pull_matching
from wm.jobs._trainer import load_npz
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.report import Report
from wm.viz.plots import metric_vs_k

ARTIFACT = "results-imagined-angle"
RESULTS_NAME = "imagined_angle.json"
CONTEXT, HORIZON = 3, 10
WINDOW = CONTEXT + HORIZON
ROWS = ("real", "wm_recon", "wm", "wm_1step", "pixel", "copy_last")
READERS = ("search", "truepos")
CKPTS = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "pixel": "ckpt-pixel"}
COMPARE = ("copy_last", "wm_1step", "pixel", "wm_recon")     # paired: wm - other


@torch.no_grad()
def predicted_frames(models: dict[str, torch.nn.Module], data: WindowData, starts: torch.Tensor,
                     ks: list[int]) -> dict[str, np.ndarray]:
    """{row: float [B, len(ks), 64, 64, 3] on the 0..255 scale} for a chunk of windows."""
    frames, actions, pusher = data.gather(starts, pusher=True)
    real = frames[:, CONTEXT:]
    out = {"real": real, "wm_recon": recon_wm(models["wm"], real),
           "wm": predict_wm(models["wm"], frames, actions, pusher)[0],
           "wm_1step": predict_wm(models["wm_1step"], frames, actions, pusher)[0],
           "pixel": predict_pixel(models["pixel"], frames, actions),
           "copy_last": predict_copy_last(frames, CONTEXT)}
    sel = [k - 1 for k in ks]
    res = {}
    for row, x in out.items():
        x = x[:, sel]
        b, h = x.shape[:2]
        res[row] = model_to_u8_float(x.reshape(b * h, *x.shape[2:])).reshape(b, h, 64, 64, 3)
    return res


def read_angles(env, frames: dict[str, np.ndarray], states: np.ndarray, starts: np.ndarray, ks: list[int],
                step_deg: float, search_px: int, device) -> dict[str, dict[str, np.ndarray]]:
    """Per-window, per-k oracle readings: angle error (deg) per reader and row, the search shift
    (world px) and edge flags per row. states = all test states; starts = window starts [B]."""
    b, h = len(starts), len(ks)
    out: dict[str, dict[str, np.ndarray]] = {
        "angle_search": {r: np.zeros((b, h)) for r in ROWS}, "angle_truepos": {r: np.zeros((b, h)) for r in ROWS},
        "shift_px": {r: np.zeros((b, h)) for r in ROWS}, "edge": {r: np.zeros((b, h), bool) for r in ROWS}}
    for i, s0 in enumerate(starts):
        for j, k in enumerate(ks):
            st = states[s0 + CONTEXT - 1 + k].astype(np.float64)
            angles, cands = template_candidates(env, st, step_deg)
            stack = np.stack([frames[r][i, j] for r in ROWS])
            ms = template_match(cands, stack, search_px, device)
            m0 = template_match(cands, stack, 0, device)
            for n_, r in enumerate(ROWS):
                out["angle_search"][r][i, j] = angle_err_deg(angles[ms["index"][n_]], st[4])
                out["angle_truepos"][r][i, j] = angle_err_deg(angles[m0["index"][n_]], st[4])
                out["shift_px"][r][i, j] = PX_PER_CELL * float(np.hypot(ms["dy"][n_], ms["dx"][n_]))
                out["edge"][r][i, j] = bool(ms["edge"][n_])
    return out


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    rcfg = load_yaml(REPO_ROOT / cfg["rollouts_config"])
    smoke, device = ctx.mode == "smoke", ctx.device
    n_windows = int(rcfg["smoke"]["windows"] if smoke else rcfg["windows"])
    seed, ci, chunk, resamples = int(rcfg["seed"]), float(rcfg["ci"]), int(rcfg["chunk"]), int(rcfg["bootstrap"])
    ks, vk = [int(k) for k in cfg["ks"]], int(cfg["verdict_k"])
    vj = ks.index(vk)
    seed_all(seed)

    models, ckpts = {}, {}
    for key, art in CKPTS.items():
        models[key], ckpts[key] = load_matching(ctx, art)
    report.params = " / ".join(f"{k} {v['params']['total']:,}" for k, v in ckpts.items())
    test, data_name = load_npz(ctx, "pusht_test.npz")
    data = WindowData(test, device, val_seeds=[190, 199], window=WINDOW)
    states = test["states"]
    all_starts = torch.from_numpy(window_starts(test["episode_starts"], test["episode_lengths"], WINDOW))
    starts = subset(all_starts, n_windows, seed)

    # same windows as eval_rollouts? (WARN if its results are unavailable or of another size)
    try:
        rpath, rname, _, _ = pull_matching(ctx, "results-rollouts")
        ref = json.loads((rpath / "rollouts.json").read_text())["windows"]["starts"]
        same = {"available": True, "artifact": rname, "same_n": len(ref) == len(starts),
                "identical": ref == starts.tolist()}
    except (RuntimeError, OSError, KeyError) as e:
        same = {"available": False, "reason": str(e)[:300]}

    env = make_env()
    env.reset(seed=seed)
    step, search = float(cfg["step_deg"]), int(cfg["search_px"])
    acc: dict[str, dict[str, list[np.ndarray]]] = {}
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        frames = predicted_frames(models, data, s, ks)
        part = read_angles(env, frames, states, s.numpy(), ks, step, search, device)
        for metric, rows in part.items():
            for r, v in rows.items():
                acc.setdefault(metric, {}).setdefault(r, []).append(v)
        print(f"eval_imagined_angle: {min(i + chunk, len(starts))}/{len(starts)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)
    per = {m: {r: np.concatenate(v) for r, v in rows.items()} for m, rows in acc.items()}
    sn = starts.numpy()
    theta_t = states[sn + CONTEXT - 1, 4]
    exact = np.stack([angle_err_deg(theta_t, states[sn + CONTEXT - 1 + k, 4]) for k in ks], 1).astype(np.float64)

    # ---- bootstrap (eval_rollouts' resamples: same n, count and seed) ----
    idx = bootstrap_indices(len(starts), resamples, seed)
    metrics: dict[str, Any] = {}
    for reader in READERS:
        metrics[f"angle_{reader}"] = {r: mean_ci(per[f"angle_{reader}"][r], idx, ci) for r in ROWS}
    metrics["angle_search"]["copy_last_exact"] = mean_ci(exact, idx, ci)
    metrics["angle_truepos"]["copy_last_exact"] = metrics["angle_search"]["copy_last_exact"]
    metrics["shift_px"] = {r: {"mean": per["shift_px"][r].mean(0).tolist(),
                               "median": np.median(per["shift_px"][r], 0).tolist()} for r in ROWS}
    metrics["edge_share"] = {r: per["edge"][r].mean(0).tolist() for r in ROWS}
    paired: dict[str, Any] = {}
    for reader in READERS:
        a = per[f"angle_{reader}"]
        for other in COMPARE:
            paired[f"{reader}/wm_minus_{other}"] = diff_ci(a["wm"], a[other], idx, ci)
        paired[f"{reader}/wm_minus_copy_last_exact"] = diff_ci(a["wm"], exact, idx, ci)

    def at(series: dict[str, list[float]], j: int) -> dict[str, float]:
        return {key: series[key][j] for key in ("mean", "lo", "hi")}

    verdict = rollout_angle_verdict(at(paired["search/wm_minus_wm_recon"], vj),
                                    at(paired["search/wm_minus_copy_last"], vj),
                                    at(paired["search/wm_minus_copy_last_exact"], vj))
    verdict["k"] = vk

    # ---- outputs ----
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump({**cfg, "rollouts": rcfg}, sort_keys=False))
    plot = metric_vs_k({r: metrics["angle_search"][r] for r in ("wm", "wm_1step", "pixel", "copy_last",
                                                                "real", "wm_recon")},
                       "Template-oracle angle error on imagined frames (search reader)", "angle error (deg)",
                       art / "imagined_angle_vs_k.png", dashed={"copy_last exact": metrics["angle_search"]["copy_last_exact"]},
                       ks=ks)
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "rollouts_config": rcfg, "inputs": inputs,
        "data": data_name, "checkpoints": ckpts,
        "windows": {"n": len(starts), "available": len(all_starts), "seed": seed, "starts": starts.tolist(),
                    "same_as_eval_rollouts": same},
        "bootstrap": {"resamples": resamples, "ci": ci, "seed": seed, "paired": True},
        "ks": ks, "rows": list(ROWS), "readers": {
            "search": f"primary: best (shift, angle), every integer shift |dy|,|dx| <= {search} image px "
                      f"({search * PX_PER_CELL:g} world px), central {64 - 2 * search}x{64 - 2 * search} crop",
            "truepos": "secondary: shift 0, full frame (= template_angle)",
            "copy_last_exact": "reference: |theta_t - theta_{t+k}| (copy_last read perfectly)"},
        "step_deg": step, "pixel_clip": [-0.5, 0.5],
        "metrics": metrics, "paired": paired, "verdict": verdict,
        "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({"imagined_angle/plot": wandb.Image(str(plot))})

    # ---- CHECKS ----
    for key, c in ckpts.items():
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    report.check_true("windows", len(starts) == n_windows, f"{len(starts)} of {len(all_starts)} test windows",
                      f"== {n_windows}")
    if same["available"] and same["same_n"]:
        report.check_true("same_windows_as_eval_rollouts", same["identical"], f"{same['artifact']}: "
                          f"{'identical' if same['identical'] else 'DIFFERENT'} starts", "identical window starts")
    else:
        report.check("same_windows_as_eval_rollouts", "WARN",
                     same.get("reason", f"{same.get('artifact')} has a different window count"),
                     "identical window starts; WARN if not comparable")
    floor = float(np.mean(metrics["angle_search"]["real"]["mean"]))
    report.check("real_frame_floor", "PASS" if floor < float(cfg["real_floor_deg"]) else ("WARN" if smoke else "FAIL"),
                 f"search oracle on real o_t+k: {floor:.2f} deg mean", f"< {cfg['real_floor_deg']:g} deg (the instrument "
                 "reads sharp frames); WARN only in smoke")
    for r in ROWS:
        share = metrics["edge_share"][r][vj]
        report.check(f"search_edge_{r}", "PASS" if share <= float(cfg["edge_warn_share"]) else "WARN",
                     f"{100 * share:.1f}% of windows at k = {vk}", f"<= {100 * float(cfg['edge_warn_share']):g}% hit "
                     f"the +-{search} px limit; WARN only")
    ok = all_finite({"metrics": metrics, "paired": paired})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")

    # ---- METRICS ----
    for key, c in ckpts.items():
        report.metric(f"ckpt_{key}", f"{c['artifact']} step {c['step']}, git {c['git']}, config_hash {c['config_hash']}")
    report.metric("data", data_name)
    report.metric("windows", f"{len(starts)} test windows (seed {seed}, as eval_rollouts); bootstrap {resamples}, "
                             f"{ci:.0%} CI; candidates every {step:g} deg; search +-{search} image px")
    for reader in READERS:
        for j, k in enumerate(ks):
            rows = list(ROWS) + (["copy_last_exact"] if reader == "search" else [])
            report.metric(f"angle_{reader}_k{k}", " | ".join(
                f"{r} {at(metrics[f'angle_{reader}'][r], j)['mean']:.1f} [{at(metrics[f'angle_{reader}'][r], j)['lo']:.1f}, "
                f"{at(metrics[f'angle_{reader}'][r], j)['hi']:.1f}]" for r in rows))
    for j, k in enumerate(ks):
        report.metric(f"shift_px_k{k}", " | ".join(
            f"{r} {metrics['shift_px'][r]['mean'][j]:.1f} (edge {100 * metrics['edge_share'][r][j]:.0f}%)" for r in ROWS))
    for name in [f"search/wm_minus_{o}" for o in COMPARE] + ["search/wm_minus_copy_last_exact"]:
        report.metric(name.replace("/", "_"), " | ".join(
            f"k{k} {at(paired[name], j)['mean']:+.1f} [{at(paired[name], j)['lo']:+.1f}, {at(paired[name], j)['hi']:+.1f}]"
            for j, k in enumerate(ks)))
    report.metric("verdict", f"imagined rollouts lose the angle: {verdict['verdict']} (k = {vk}; A rollout adds error "
                             f"{verdict['A_rollout_adds_error']}, B worse than copy_last {verdict['B_worse_than_copy_last']} "
                             f"[search {verdict['B_search']}, exact {verdict['B_exact']}])")

    # ---- SAVED ----
    report.save(json_path)
    report.save(plot)
    if ctx.mode == "full":
        repo_json = REPO_ROOT / "results" / "final" / RESULTS_NAME
        repo_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, repo_json)
        report.save(repo_json)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

"""Job angle_audit (Phase 5b, eval only): is the v1 probe's 76 deg angle error a label bug?

  1. labelled grid image: the block at 0/30/90/180/270 deg with the state angle, the angle the
     mask moments read, and an arrow along the body +y axis computed from the state;
  2. convention checks: re-render stored states vs stored frames; mask-moment angle vs state
     angle on a 288-pose grid; mask centroid nearer block_cog than the body origin; target
     encode/decode round trip and wrap cases. Any failure = a bug (FAIL);
  3. template oracle (non-learned, true position known) on real test frames and on
     every frame type the judges see (real, wm_recon, wm1step_recon, pixel_1step): does a 64x64
     frame contain the angle, and does each model's output keep it?
  4. v1 probe (`ckpt-probe`) errors on train / val / test frames and on the 288 grid frames
     ("before"; train << test = memorisation).
Writes angle_audit.json + angle_grid.png, pushes `results-angle-audit`; full mode also writes
results/angle_audit.json and results/angle_audit/angle_grid.png in the checkout.
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

from wm.data.judge_mix import FRAME_TYPES, MixData
from wm.data.windows import episode_frames, split_episodes, state_targets
from wm.envs.pusht import make_env
from wm.eval.angle_audit import (angle_err_deg, grid_image, orientation_check, rerender_consistency,
                                 scoring_grid, target_roundtrip, template_angle)
from wm.eval.diagnose import subset
from wm.eval.judges import (baseline_errors, fmt_errs, frames_u8_to_model, judge_errors,
                            model_to_u8_float, summarize_errors)
from wm.eval.rollouts import all_finite
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._trainer import load_npz
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, seed_all
from wm.report import Report

ARTIFACT = "results-angle-audit"
RESULTS_NAME = "angle_audit.json"


def oracle_errors(env, frames: np.ndarray, states: np.ndarray, step: float) -> dict[str, float]:
    """Template-oracle angle error (deg) on frames (0..255 scale) with known true states."""
    err = np.array([float(angle_err_deg(template_angle(env, f, s, step), s[4])) for f, s in zip(frames, states)])
    return {"n": int(len(err)), "angle_deg_mean": float(err.mean()), "angle_deg_median": float(np.median(err)),
            "share_under_10deg": float((err < 10).mean())}


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    smoke = ctx.mode == "smoke"
    n = {k: int(cfg["smoke"][k] if smoke else cfg[k]) for k in ("rerender_frames", "oracle_frames",
                                                                  "probe_train_frames")}
    seed = int(cfg["seed"])
    seed_all(seed)
    th = cfg["thresholds"]
    rng = np.random.default_rng(seed)
    env = make_env()
    env.reset(seed=seed)

    probe, probe_info = load_matching(ctx, "ckpt-probe")
    wm, wm_info = load_matching(ctx, "ckpt-wm")
    report.params = f"probe {probe_info['params']['total']:,} / wm {wm_info['params']['total']:,}"
    train, data_name = load_npz(ctx)
    test, _ = load_npz(ctx, "pusht_test.npz")

    # ---- 2. convention checks ----
    both_f = np.concatenate([train["frames"], test["frames"]])
    both_s = np.concatenate([train["states"], test["states"]])
    pick = np.sort(rng.choice(len(both_f), min(n["rerender_frames"], len(both_f)), replace=False))
    rerender = rerender_consistency(env, both_f[pick], both_s[pick])
    angles_all = list(np.arange(0.0, 360.0, float(cfg["score_step_deg"])))
    orient = orientation_check(env, angles_all, [tuple(p) for p in cfg["score_positions"]])
    roundtrip = target_roundtrip(both_s)

    # ---- 3. template oracle on every frame type the judges will see ----
    wm1, wm1_info = load_matching(ctx, "ckpt-wm-1step")
    pixel, pixel_info = load_matching(ctx, "ckpt-pixel")
    mix = MixData(test, ctx.device, {"wm": wm, "wm_1step": wm1, "pixel": pixel})
    step = float(cfg["oracle_step_deg"])
    oracle = {}
    for kind in FRAME_TYPES:
        x, _, o_states = mix.images(kind, mix.fixed(kind, n["oracle_frames"], seed))
        oracle[kind] = oracle_errors(env, model_to_u8_float(x), o_states, step)

    # ---- 4. the v1 probe on train / val / test / grid frames ----
    tr_eps, va_eps, val_source = split_episodes(train["episode_seeds"], [190, 199])
    starts, lengths = train["episode_starts"], train["episode_lengths"]
    tr_idx = subset(torch.from_numpy(episode_frames(starts[tr_eps], lengths[tr_eps])),
                    n["probe_train_frames"], seed).numpy()
    va_idx = episode_frames(starts[va_eps], lengths[va_eps])
    grid_f, grid_s = scoring_grid(env, cfg["score_step_deg"], cfg["score_positions"])
    mean_target = state_targets(train["states"][episode_frames(starts[tr_eps], lengths[tr_eps])]).mean(0)
    v1: dict[str, Any] = {}
    for split, frames, states in (("train", train["frames"][tr_idx], train["states"][tr_idx]),
                                  ("val", train["frames"][va_idx], train["states"][va_idx]),
                                  ("test", test["frames"], test["states"]),
                                  ("grid", grid_f, grid_s)):
        v1[split] = summarize_errors(judge_errors(probe, frames_u8_to_model(frames), states))
        v1[split]["baseline"] = baseline_errors(mean_target, states)

    # ---- verdict ----
    bug = {"rerender": rerender["mean_abs_diff"] > th["rerender_mean_abs_diff"],
           "orientation": orient["angle_err_mean_deg"] > th["orientation_err_deg"],
           "cog_convention": orient["cog_closer_share"] < 1.0,
           "roundtrip": max(roundtrip["max_block_deg"], abs(roundtrip["wrap_cases_deg"][0] - 2.0),
                            abs(roundtrip["wrap_cases_deg"][1] - 2.0), roundtrip["wrap_cases_deg"][2]) > th["roundtrip_deg"]}
    memorised = v1["train"]["block_deg_mean"] < th["memorisation_ratio"] * v1["test"]["block_deg_mean"]
    if any(bug.values()):
        verdict = "BUG: " + ", ".join(k for k, v in bug.items() if v)
    elif memorised:
        verdict = "labels consistent; failure = probe generalisation (memorisation: train angle << test)"
    else:
        verdict = "labels consistent; the probe fails to read angle even on train frames"

    # ---- outputs ----
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    grid_png = art / "angle_grid.png"
    imageio.imwrite(grid_png, grid_image(env, [float(a) for a in cfg["grid_angles_deg"]]))
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    results = {"job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
               "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
               "checkpoints": {"probe": probe_info, "wm": wm_info, "wm_1step": wm1_info, "pixel": pixel_info},
               "convention": {"angle_units": "radians, wrapped to [-pi, pi)",
                              "image_axes": "column = x/8, row = y/8 (y down); angle increases clockwise on screen",
                              "zero_angle": "bar on top, stem pointing down (+y)",
                              "block_xy": "stored (x, y) = body origin; block_cog = origin + R(theta) (0, 45)"},
               "rerender": rerender, "orientation": orient, "target_roundtrip": roundtrip,
               "template_oracle": oracle, "probe_v1": v1, "bug": bug, "memorised": bool(memorised),
               "verdict": verdict, "val_source": val_source, "wall_clock_s": time.time() - t0}
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({"angle_audit/grid": wandb.Image(str(grid_png))})

    # ---- CHECKS ----
    for key, c in (("probe", probe_info), ("wm", wm_info), ("wm_1step", wm1_info), ("pixel", pixel_info)):
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    report.check_true("rerender_consistency", not bug["rerender"],
                      f"mean abs diff {rerender['mean_abs_diff']:.3f} over {rerender['frames']} stored frames "
                      f"({rerender['exact_share']:.0%} exact)", f"<= {th['rerender_mean_abs_diff']} uint8 levels")
    report.check_true("angle_orientation", not bug["orientation"],
                      f"mask-moment angle vs state angle: mean {orient['angle_err_mean_deg']:.1f}, "
                      f"max {orient['angle_err_max_deg']:.1f} deg over {orient['poses']} poses",
                      f"mean <= {th['orientation_err_deg']:g} deg")
    report.check_true("cog_convention", not bug["cog_convention"],
                      f"mask centroid to block_cog {orient['centroid_to_cog_mean_px']:.1f} px vs to body origin "
                      f"{orient['centroid_to_origin_mean_px']:.1f} px", "centroid nearer block_cog on every pose")
    report.check_true("target_roundtrip", not bug["roundtrip"],
                      f"max {roundtrip['max_block_deg']:.2e} deg / {roundtrip['max_block_px']:.2e} px; wrap "
                      f"359v1, -179v179, 540v180 -> " + ", ".join(f"{x:.1f}" for x in roundtrip["wrap_cases_deg"]),
                      "0 / 0; wraps 2, 2, 0 deg")
    ok = all_finite(results)
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")

    # ---- METRICS ----
    report.metric("verdict", verdict)
    report.metric("ckpt_probe", f"{probe_info['artifact']} step {probe_info['step']}, git {probe_info['git']}")
    report.metric("ckpt_wm", f"{wm_info['artifact']} step {wm_info['step']}, git {wm_info['git']}, "
                             f"proprio {wm_info['proprio']}")
    report.metric("convention", "radians wrapped to [-pi, pi); 0 = bar on top, stem down; angle grows "
                                "clockwise on screen (y down); stored xy = body origin, CoG = origin + R(0, 45)")
    for name, o in oracle.items():
        report.metric(f"template_oracle_{name}", f"angle {o['angle_deg_mean']:.1f} deg mean / "
                                                 f"{o['angle_deg_median']:.1f} median, {o['share_under_10deg']:.0%} "
                                                 f"under 10 deg ({o['n']} test frames, true position given)")
    report.metric("template_oracle_note", "non-learned reader given the true position: real = does a 64x64 "
                                          "frame hold the angle; derived types = is it still in that frame type")
    for split in ("train", "val", "test", "grid"):
        s = v1[split]
        report.metric(f"probe_v1_{split}", f"{fmt_errs(s)} ({s['n']} frames; predict-mean "
                                           f"{s['baseline']['block_deg_mean']:.1f} deg)")
    report.metric("memorisation", f"train angle {v1['train']['block_deg_mean']:.1f} vs test "
                                  f"{v1['test']['block_deg_mean']:.1f} deg -> {'yes' if memorised else 'no'} "
                                  f"(train < {th['memorisation_ratio']} x test)")
    report.metric("data", data_name)

    # ---- SAVED ----
    report.save(json_path)
    report.save(grid_png)
    if ctx.mode == "full":   # smoke never writes into the repo's results/
        repo = REPO_ROOT / "results"
        (repo / "angle_audit").mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, repo / RESULTS_NAME)
        shutil.copy2(grid_png, repo / "angle_audit" / "angle_grid.png")
        report.save(repo / RESULTS_NAME)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

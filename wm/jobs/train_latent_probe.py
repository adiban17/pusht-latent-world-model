"""Job train_latent_probe (Phase 5b): the latent judge -> store `ckpt-latent-probe`.

An MLP reads the §7 state from z = E_wm(frame). It trains on E_wm of real train-split frames +
rendered judge frames (latents computed once; the wm is frozen), with keep-best on the
rendered-val loss, and is reported on real E_wm(o) train / val / test frames and on the
rendered grid. Its mean angle error on real test latents answers "is block angle in the wm
latent?" with the pre-registered verdict (< 20 deg present, 20-45 weak, > 45 absent).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.judge_mix import concat_arrays, train_split, val_split
from wm.data.windows import state_targets
from wm.envs.pusht import make_env
from wm.eval.angle_audit import scoring_grid
from wm.eval.diagnose import encode_frames, subset
from wm.eval.judges import baseline_errors, fmt_errs, judge_errors, summarize_errors
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._judges import angle_verdict, load_judge_npz
from wm.jobs._trainer import TrainingJob, check_param_counts, check_roundtrip, load_npz, roundtrip
from wm.models.latent_probe import EXPECTED_PARAMS, build_latent_probe, param_counts
from wm.report import Report
from wm.train.train_latent_probe import LOSS_KEYS, LatentData, run_training

ARTIFACT = "ckpt-latent-probe"
REPORT_TRAIN_FRAMES = 4096


def encode(wm, frames: np.ndarray, device) -> torch.Tensor:
    """E_wm of uint8 frames [N, 64, 64, 3] -> z [N, 256] (CPU)."""
    return encode_frames(wm, torch.from_numpy(frames).to(device))


def run(ctx: JobContext, report: Report) -> None:
    job = TrainingJob(ctx, ARTIFACT)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device

    wm, wm_info = load_matching(ctx, "ckpt-wm")
    for p in wm.parameters():
        p.requires_grad_(False)
    real, data_name = load_npz(ctx)
    test, _ = load_npz(ctx, "pusht_test.npz")
    r_train, judge_name = load_judge_npz(ctx, "train")
    r_val, _ = load_judge_npz(ctx, "val")
    job.data_name = f"{data_name} + {judge_name}"
    real_train, real_val = train_split(real, cfg["val_seeds"]), val_split(real, cfg["val_seeds"])
    train_arrays = concat_arrays(real_train, r_train)
    train = LatentData(encode(wm, train_arrays["frames"], device), train_arrays["states"], device)
    val = LatentData(encode(wm, r_val["frames"], device), r_val["states"], device)
    env = make_env()
    env.reset(seed=0)
    grid_f, grid_s = scoring_grid(env, cfg["grid"]["step_deg"], cfg["grid"]["positions"])

    model = build_latent_probe(cfg).to(device)
    counts = param_counts(model)
    job.set_params(report, counts)
    check_param_counts(report, counts, EXPECTED_PARAMS, "documented latent MLP")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: rendered-val loss {v['loss']:.4f} | {fmt_errs(v)}", flush=True)
        job.wandb_log({f"val/{k}": x for k, x in v.items() if isinstance(x, float)}, step)

    result = run_training(model, train, val, tcfg, device, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    best_step = (result.get("best") or {}).get("step")
    job.extra_info = {"best_step": best_step}

    # ---- real latents: train subset / val episodes / test; rendered val; grid ----
    mean_target = state_targets(real_train["states"]).mean(0)
    tr_pick = subset(torch.arange(len(real_train["frames"])), REPORT_TRAIN_FRAMES, 0).numpy()
    sets = {"real_train": (real_train["frames"][tr_pick], real_train["states"][tr_pick]),
            "real_val": (real_val["frames"], real_val["states"]),
            "real_test": (test["frames"], test["states"]),
            "rendered_val": (r_val["frames"], r_val["states"]),
            "grid": (grid_f, grid_s)}
    evals = {}
    for name, (frames, states) in sets.items():
        evals[name] = summarize_errors(judge_errors(model, encode(wm, frames, device), states))
        evals[name]["baseline"] = baseline_errors(mean_target, states)
    vcfg = cfg["verdict"]
    ang = evals["real_test"]["block_deg_mean"]
    verdict = angle_verdict(ang, float(vcfg["present_deg"]), float(vcfg["absent_deg"]))

    step, log = result["step"], result["log"]
    model_path = job.save_model(model, step)
    reloaded = build_latent_probe(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    rt = roundtrip(model, reloaded, model, reloaded, val.z[:2])
    summary = job.write_summary(result, LOSS_KEYS, best_step=best_step, evals=evals,
                                angle_in_wm_latent={"angle_deg_mean": ang, "verdict": verdict,
                                                    "predict_mean_deg": evals["real_test"]["baseline"]["block_deg_mean"],
                                                    "thresholds": vcfg},
                                wm_checkpoint=wm_info, train_latents=int(len(train.z)),
                                rendered_val_latents=int(len(val.z)), accepted=True)

    # ---- checks ----
    report.check_true("ckpt_config_hash_wm", wm_info["config_hash"] == wm_info["expected_config_hash"],
                      f"{wm_info['artifact']} step {wm_info['step']}: {wm_info['config_hash']}",
                      f"== {wm_info['expected_config_hash']}, current config; complete: true")
    job.loss_checks(report, summary, log, LOSS_KEYS)
    check_roundtrip(report, rt, "outputs")
    beat = evals["real_test"]["block_px_mean"] < evals["real_test"]["baseline"]["block_px_mean"]
    report.check("real_test_block_vs_mean", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"{evals['real_test']['block_px_mean']:.1f} vs {evals['real_test']['baseline']['block_px_mean']:.1f} px",
                 "block error on real E_wm(o) test < predict-train-mean; WARN only in smoke")

    # ---- metrics ----
    report.metric("angle_in_wm_latent", f"{ang:.1f} deg -> {verdict} (latent MLP on real E_wm(o) test frames; "
                                        f"predict-mean {evals['real_test']['baseline']['block_deg_mean']:.1f} deg; "
                                        f"thresholds < {vcfg['present_deg']:g} present, > {vcfg['absent_deg']:g} absent)")
    report.metric("ckpt_wm", f"{wm_info['artifact']} step {wm_info['step']}, git {wm_info['git']}, "
                             f"config_hash {wm_info['config_hash']}, proprio {wm_info['proprio']}")
    report.metric("data", f"{job.data_name}: {len(train.z):,} train latents; rendered val {len(val.z):,}")
    report.metric("steps", f"{step} (batch {tcfg['batch_size']}, best step {best_step})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("loss_first_last", f"{summary['loss_first']['loss']:.4f} -> {summary['loss_last']['loss']:.4f}")
    job.timing_metrics(report, summary)
    for name, e in evals.items():
        report.metric(name, f"{fmt_errs(e)} ({e['n']} frames; predict-mean {fmt_errs(e['baseline'])})")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")
    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json", job.art_dir / "config.yaml"])

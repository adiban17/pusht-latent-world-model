"""Job train_probe_mix (Phase 5b): the fair image judge -> store `ckpt-probe-mix`.

Trains a state probe on four frame types (real, D_wm(E_wm(.)), D_wm1step(E_wm1step(.)), pixel
1-step predictions) of real train-split frames + rendered judge frames, all labelled with the
true state (wm/data/judge_mix.py), with ±4 px shifts and keep-best on the rendered-val loss.
Pre-registered (PROGRESS.md, Phase 5b): the §7 encoder first; only if its real-frame test angle
is > 30 deg, the same run also trains the coord_gap variant (no intermediate checkpoints) and
uses it if it passes acceptance. Acceptance on test frames: real angle < 15 deg, real block
<= 13.4 px, block and pusher px of each derived type <= 1.5 x real's (WARN if not; the
summary's `accepted` tells eval_rollouts).
"""
from __future__ import annotations

import copy
from typing import Any

import torch

from wm.data.judge_mix import FRAME_TYPES, MixData, concat_arrays, train_split, val_split
from wm.envs.pusht import make_env
from wm.eval.angle_audit import scoring_grid
from wm.eval.judges import fmt_errs, frames_u8_to_model, judge_errors, summarize_errors
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._judges import fmt_typed, load_judge_npz
from wm.jobs._trainer import TrainingJob, check_param_counts, check_roundtrip, load_npz, roundtrip
from wm.models.probe import build_probe, expected_params, param_counts
from wm.report import Report
from wm.repro import seed_all
from wm.train.train_probe import LOSS_KEYS
from wm.train.train_probe_mix import acceptance, fixed_val_set, run_training, typed_errors

ARTIFACT = "ckpt-probe-mix"
FROZEN = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "pixel": "ckpt-pixel"}


def evaluate_variant(model, test: MixData, real_val: MixData, grid: tuple, acc: dict[str, float]) -> dict[str, Any]:
    """Test errors per frame type, real val-episode errors, grid errors, acceptance."""
    t = typed_errors(model, test)
    v = typed_errors(model, real_val, kinds=("real",))
    g = summarize_errors(judge_errors(model, frames_u8_to_model(grid[0]), grid[1]))
    return {"test": t, "real_val": v["real"], "grid": g, "acceptance": acceptance(t, acc)}


def run(ctx: JobContext, report: Report) -> None:
    job = TrainingJob(ctx, ARTIFACT)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device
    pad, acc = int(cfg["shift_px"]), cfg["accept"]

    frozen, frozen_info = {}, {}
    for key, art in FROZEN.items():
        frozen[key], frozen_info[key] = load_matching(ctx, art)
        for p in frozen[key].parameters():
            p.requires_grad_(False)
    real, data_name = load_npz(ctx)
    test_arrays, _ = load_npz(ctx, "pusht_test.npz")
    rendered_train, judge_name = load_judge_npz(ctx, "train")
    rendered_val, _ = load_judge_npz(ctx, "val")
    job.data_name = f"{data_name} + {judge_name}"
    real_train = train_split(real, cfg["val_seeds"])
    train = MixData(concat_arrays(real_train, rendered_train), device, frozen)
    val_data = MixData(rendered_val, device, frozen)
    test = MixData(test_arrays, device, frozen)
    real_val = MixData(val_split(real, cfg["val_seeds"]), device, frozen)
    val = fixed_val_set(val_data, int(tcfg["val_windows"]), int(tcfg["val_seed"]))
    env = make_env()
    env.reset(seed=0)
    grid = scoring_grid(env, cfg["grid"]["step_deg"], cfg["grid"]["positions"])

    model = build_probe(cfg).to(device)
    counts = param_counts(model)
    job.set_params(report, counts)
    check_param_counts(report, counts, expected_params("flatten"), "CLAUDE.md §7 (flatten)")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: rendered-val loss {v['loss']:.4f} | angle deg " +
              " ".join(f"{k} {v[f'{k}_block_deg_mean']:.1f}" for k in FRAME_TYPES), flush=True)
        job.wandb_log({f"val/{k}": x for k, x in v.items()}, step)

    result = run_training(model, train, val, tcfg, device, pad, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    variants = {"flatten": {**evaluate_variant(model, test, real_val, grid, acc),
                            "best_step": (result.get("best") or {}).get("step"), "params": counts}}
    chosen, chosen_model = "flatten", model

    # ---- pre-registered fallback: coord_gap, only if the §7 judge cannot read the angle ----
    fb = cfg["fallback"]
    triggered = variants["flatten"]["test"]["real"]["block_deg_mean"] > float(fb["trigger_real_angle_deg"])
    if triggered:
        print("fallback: flatten real-frame angle > trigger; training coord_gap", flush=True)
        seed_all(cfg["seed"])
        fb_cfg = copy.deepcopy(cfg)
        fb_cfg["model"]["pool"] = fb["pool"]
        fb_model = build_probe(fb_cfg).to(device)
        fb_counts = param_counts(fb_model)
        report.check_true("param_counts_coord_gap", fb_counts == expected_params(fb["pool"]),
                          " / ".join(f"{k} {v:,}" for k, v in fb_counts.items()), "== documented coord_gap")
        fb_result = run_training(fb_model, train, val, tcfg, device, pad, on_val=on_val)
        variants[fb["pool"]] = {**evaluate_variant(fb_model, test, real_val, grid, acc),
                                "best_step": (fb_result.get("best") or {}).get("step"), "params": fb_counts}
        if variants[fb["pool"]]["acceptance"]["accepted"] and not variants["flatten"]["acceptance"]["accepted"]:
            chosen, chosen_model, job.cfg = fb["pool"], fb_model, fb_cfg   # model.pt saves job.cfg
            job.set_params(report, fb_counts)
    best = variants[chosen]
    job.extra_info = {"pool": chosen, "best_step": best["best_step"]}

    step, log = result["step"], result["log"]
    model_path = job.save_model(chosen_model, step)
    reloaded = build_probe(job.cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    rt = roundtrip(chosen_model, reloaded, chosen_model, reloaded, test.images("real", test.fixed("real", 2))[0])
    summary = job.write_summary(result, LOSS_KEYS, chosen=chosen, accepted=best["acceptance"]["accepted"],
                                acceptance=best["acceptance"], test=best["test"], real_val=best["real_val"],
                                grid=best["grid"], variants=variants, fallback_triggered=triggered,
                                accept_thresholds=acc, frozen_checkpoints=frozen_info,
                                train_frames=int(len(train.frame_idx)),
                                rendered_val_frames=int(len(val_data.frame_idx)))

    # ---- checks ----
    for key, c in frozen_info.items():
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    job.loss_checks(report, summary, log, LOSS_KEYS)
    check_roundtrip(report, rt, "outputs")
    real_t = best["test"]["real"]
    status = lambda ok: "PASS" if ok else "WARN"      # noqa: E731  (acceptance is WARN-only)
    a = best["acceptance"]["checks"]
    report.check("accept_real_angle", status(a["real_angle"]), f"{real_t['block_deg_mean']:.1f} deg",
                 f"< {acc['real_angle_deg']:g} deg on real test frames; WARN only")
    report.check("accept_real_block", status(a["real_block"]), f"{real_t['block_px_mean']:.1f} px",
                 f"<= {acc['real_block_px']:g} px (1.25 x v1 probe); WARN only")
    for kind in FRAME_TYPES[1:]:
        t = best["test"][kind]
        ok = a[f"{kind}_block_px"] and a[f"{kind}_pusher_px"]
        report.check(f"accept_{kind}", status(ok),
                     f"block {t['block_px_mean']:.1f} / pusher {t['pusher_px_mean']:.1f} px vs real "
                     f"{real_t['block_px_mean']:.1f} / {real_t['pusher_px_mean']:.1f}",
                     f"<= {acc['derived_ratio']:g} x real; WARN only")

    # ---- metrics ----
    first, last = summary["loss_first"], summary["loss_last"]
    report.metric("judge", f"{chosen} ({'accepted' if best['acceptance']['accepted'] else 'NOT accepted'}); "
                           f"fallback {'triggered' if triggered else 'not triggered'}")
    report.metric("data", f"{job.data_name}: {len(train.frame_idx):,} train frames (real train split + rendered); "
                          f"rendered val {len(val_data.frame_idx):,}")
    report.metric("steps", f"{step} (batch {tcfg['batch_size']}, best step {best['best_step']})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("loss_first_last", f"{first['loss']:.4f} -> {last['loss']:.4f}")
    job.timing_metrics(report, summary)
    for name, v in variants.items():
        for key, unit in (("block_px", "px"), ("block_deg", "deg"), ("pusher_px", "px")):
            report.metric(f"test_{key}_{name}", fmt_typed(v["test"], key) + f" {unit}")
        report.metric(f"real_val_{name}", fmt_errs(v["real_val"]))
        report.metric(f"grid_{name}", fmt_errs(v["grid"]) + f" ({v['grid']['n']} rendered poses)")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")
    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json", job.art_dir / "config.yaml"])

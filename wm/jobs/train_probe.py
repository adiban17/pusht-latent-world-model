"""Job train_probe (Phase 4): state probe (CLAUDE.md §7) -> store `ckpt-probe`.

Its error on real test frames is the judge's noise floor for the rollout evaluation (Phase 5).
"""
from __future__ import annotations

from typing import Any

import torch

from wm.data.windows import FrameData
from wm.jobs import JobContext
from wm.jobs._trainer import TrainingJob, check_param_counts, load_npz, same_outputs
from wm.models.probe import EXPECTED_PARAMS, build_probe, param_counts
from wm.report import Report
from wm.train.train_probe import ERROR_KEYS, LOSS_KEYS, evaluate, run_training

ARTIFACT = "ckpt-probe"
UNITS = {"block_px": "px", "block_deg": "deg", "pusher_px": "px"}


def run(ctx: JobContext, report: Report) -> None:
    job = TrainingJob(ctx, ARTIFACT)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device

    arrays, job.data_name = load_npz(ctx)
    data = FrameData(arrays, device, cfg["val"]["seeds"])
    test_arrays, _ = load_npz(ctx, "pusht_test.npz")
    test = FrameData(test_arrays, device, None)
    model = build_probe(cfg).to(device)
    counts = param_counts(model)
    job.set_params(report, counts)
    check_param_counts(report, counts, EXPECTED_PARAMS, "CLAUDE.md §7")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: val mse {v['loss']:.4f} block {v['block_px_mean']:.1f} px / "
              f"{v['block_deg_mean']:.1f} deg, pusher {v['pusher_px_mean']:.1f} px", flush=True)
        job.wandb_log({f"val/{k}": v[k] for k in ("loss", "block_px_mean", "block_deg_mean",
                                                   "pusher_px_mean", "z_std")}, step)

    result = run_training(model, data, tcfg, device, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    step, log, val = result["step"], result["log"], result["val_log"][-1]
    test_eval = evaluate(model, test, test.val_idx, data.mean_train_target(), device, tcfg["amp"])

    model_path = job.save_model(model, step)
    reloaded = build_probe(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    model.eval(); reloaded.eval()
    same = same_outputs(model, reloaded, test.gather(test.val_idx[:2])[0])
    summary = job.write_summary(result, LOSS_KEYS, val=val, test=test_eval,
                                block_position="centre of gravity (block_cog)")

    # ---- checks ----
    job.loss_checks(report, summary, log, LOSS_KEYS)
    report.check_true("checkpoint_roundtrip", same, "identical outputs" if same else "mismatch",
                      "model.pt reload == trained model")
    beat = test_eval["block_px_mean"] < test_eval["baseline_block_px_mean"]
    report.check("test_block_px_vs_mean", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"{test_eval['block_px_mean']:.1f} vs {test_eval['baseline_block_px_mean']:.1f}",
                 "mean block error < predict-train-mean baseline; WARN only in smoke")

    # ---- metrics ----
    first, last = summary["loss_first"], summary["loss_last"]
    report.metric("data", job.data_name)
    report.metric("steps", f"{step} (batch {tcfg['batch_size']})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("train_frames", len(data.train_idx))
    report.metric("val_source", data.val_source)
    report.metric("test_frames", test_eval["frames"])
    report.metric("loss_first_last", f"{first['loss']:.4f} -> {last['loss']:.4f}")
    job.timing_metrics(report, summary)
    for k in ERROR_KEYS:
        report.metric(f"test_{k}", f"mean {test_eval[f'{k}_mean']:.2f} / median "
                                   f"{test_eval[f'{k}_median']:.2f} {UNITS[k]} "
                                   f"(mean baseline {test_eval[f'baseline_{k}_mean']:.2f})")
    report.metric("val_block_px_mean", f"{val['block_px_mean']:.2f}")
    report.metric("test_z_std", f"{test_eval['z_std']:.3f} (info only; no variance hinge)")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")

    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json",
                                    job.art_dir / "config.yaml"])

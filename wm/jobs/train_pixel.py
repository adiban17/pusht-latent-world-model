"""Job train_pixel (Phase 4): pixel-space U-Net baseline (CLAUDE.md §5) -> store `ckpt-pixel`."""
from __future__ import annotations

from typing import Any

import imageio.v2 as imageio
import torch

from wm.data.windows import WindowData
from wm.jobs import JobContext
from wm.jobs._trainer import TrainingJob, check_roundtrip, load_npz, roundtrip
from wm.models.pixel_unet import (EXPECTED_PARAMS, PARAM_TOLERANCE, WM_PARAMS, build_pixel_unet,
                                  param_counts, within_tolerance)
from wm.report import Report
from wm.train.train_pixel import LOSS_KEYS, predicted_rollout, run_training
from wm.viz.media import rollout_filmstrip

ARTIFACT = "ckpt-pixel"
KS = (1, 5, 10)


def run(ctx: JobContext, report: Report) -> None:
    job = TrainingJob(ctx, ARTIFACT)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device
    horizon = cfg["window"]["horizon"]

    arrays, job.data_name = load_npz(ctx)
    data = WindowData(arrays, device, cfg["val"]["seeds"], window=cfg["window"]["context"] + 1)
    model = build_pixel_unet(cfg).to(device)
    counts = param_counts(model)
    job.set_params(report, counts)
    total = counts["total"]
    report.check_true("param_counts", total == EXPECTED_PARAMS and within_tolerance(total),
                      f"{total:,} ({total / WM_PARAMS - 1:+.1%} vs world model {WM_PARAMS:,})",
                      f"== {EXPECTED_PARAMS:,}, within ±{PARAM_TOLERANCE:.0%}")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: val mse@1 {v['mse'][0]:.5f} mse@10 {v['mse'][-1]:.5f} "
              f"(copy-last {v['mse_copy_last'][0]:.5f} / {v['mse_copy_last'][-1]:.5f})", flush=True)
        row = {f"val/mse_k{k}": v["mse"][k - 1] for k in KS}
        row.update({f"val/copy_last_mse_k{k}": v["mse_copy_last"][k - 1] for k in KS})
        job.wandb_log(row, step)

    result = run_training(model, data, tcfg, device, horizon, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    step, log, val = result["step"], result["log"], result["val_log"][-1]

    model_path = job.save_model(model, step)
    reloaded = build_pixel_unet(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    frames, actions = data.gather(data.val_starts[:2])
    c = model.context
    rt = roundtrip(model, reloaded, lambda f: model(f, actions[:, c - 1]),
                   lambda f: reloaded(f, actions[:, c - 1]), frames[:, :c])

    vstart = data.fixed_val_starts_of(c + horizon, 1, tcfg["val_seed"])[0]
    real, pred = predicted_rollout(model, data, vstart, horizon)
    strip_path = job.art_dir / "rollout_filmstrip.png"
    imageio.imwrite(strip_path, rollout_filmstrip(real, pred))
    summary = job.write_summary(result, LOSS_KEYS, data_summary=data.summary(), val=val)

    # ---- checks ----
    job.loss_checks(report, summary, log, LOSS_KEYS)
    check_roundtrip(report, rt, "predictions")
    beat = val["mse"][0] < val["mse_copy_last"][0]
    report.check("one_step_vs_copy_last", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"{val['mse'][0]:.5f} vs {val['mse_copy_last'][0]:.5f}",
                 "val pixel MSE at k=1 < copy-last; WARN only in smoke")

    # ---- metrics ----
    first, last = summary["loss_first"], summary["loss_last"]
    report.metric("data", job.data_name)
    report.metric("steps", f"{step} (batch {tcfg['batch_size']})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("train_windows", data.summary()["train_windows"])
    report.metric("val_source", data.summary()["val_source"])
    report.metric("val_windows", val["windows"])
    report.metric("loss_first_last", f"{first['loss']:.5f} -> {last['loss']:.5f}")
    job.timing_metrics(report, summary)
    for k in KS:
        report.metric(f"val_pixel_mse_k{k}",
                      f"{val['mse'][k - 1]:.5f} (copy-last {val['mse_copy_last'][k - 1]:.5f})")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")

    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json",
                                    job.art_dir / "config.yaml", strip_path])

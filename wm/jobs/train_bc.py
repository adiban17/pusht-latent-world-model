"""Job train_bc (Phase 4): goal-conditioned behaviour cloning (CLAUDE.md §6) -> store `ckpt-bc`."""
from __future__ import annotations

from typing import Any

import torch

from wm.data.windows import GoalData
from wm.jobs import JobContext
from wm.jobs._trainer import TrainingJob, check_param_counts, load_npz, same_outputs
from wm.models.bc import EXPECTED_PARAMS, build_bc, param_counts
from wm.report import Report
from wm.train.train_bc import LOSS_KEYS, run_training

ARTIFACT = "ckpt-bc"


def run(ctx: JobContext, report: Report) -> None:
    job = TrainingJob(ctx, ARTIFACT)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device
    scale = float(cfg["delta_scale"])

    arrays, job.data_name = load_npz(ctx)
    data = GoalData(arrays, device, cfg["val"]["seeds"], context=cfg["model"]["context"],
                    k_min=cfg["goal"]["k_min"], k_max=cfg["goal"]["k_max"], delta_scale=scale,
                    use_clean_actions=bool(cfg["use_clean_actions"]))
    model = build_bc(cfg).to(device)
    counts = param_counts(model)
    job.set_params(report, counts)
    check_param_counts(report, counts, EXPECTED_PARAMS, "CLAUDE.md §6 (≈2.53M)")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: val mse {v['loss']:.4f} (baseline {v['baseline_mse']:.4f}) "
              f"err {v['action_err_px']:.1f} px (baseline {v['baseline_err_px']:.1f} px)", flush=True)
        job.wandb_log({f"val/{k}": v[k] for k in ("loss", "baseline_mse", "action_err_px",
                                                   "baseline_err_px", "z_std")}, step)

    result = run_training(model, data, tcfg, device, int(cfg["shift_px"]), scale, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    step, log, val = result["step"], result["log"], result["val_log"][-1]

    model_path = job.save_model(model, step)
    reloaded = build_bc(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    model.eval(); reloaded.eval()
    b = data.gather(*[x[:2] for x in data.fixed_val(2, 0)])
    same = same_outputs(lambda im: model(im, b["pusher"]), lambda im: reloaded(im, b["pusher"]), b["images"])
    summary = job.write_summary(result, LOSS_KEYS, data_summary=data.summary(), val=val)

    # ---- checks ----
    job.loss_checks(report, summary, log, LOSS_KEYS)
    report.check_true("checkpoint_roundtrip", same, "identical actions" if same else "mismatch",
                      "model.pt reload == trained model")
    beat = val["loss"] < val["baseline_mse"]
    report.check("val_mse_vs_mean_action", "PASS" if beat else ("WARN" if ctx.mode == "smoke" else "FAIL"),
                 f"{val['loss']:.4f} vs {val['baseline_mse']:.4f}",
                 "val MSE < constant mean-action baseline; WARN only in smoke")

    # ---- metrics ----
    first, last = summary["loss_first"], summary["loss_last"]
    ds = data.summary()
    report.metric("data", job.data_name)
    report.metric("steps", f"{step} (batch {tcfg['batch_size']})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("train_samples", ds["train_samples"])
    report.metric("val_source", ds["val_source"])
    report.metric("val_samples", val["samples"])
    report.metric("actions", "clean" if cfg["use_clean_actions"] else "logged (noisy)")
    report.metric("loss_first_last", f"{first['loss']:.4f} -> {last['loss']:.4f}")
    job.timing_metrics(report, summary)
    report.metric("val_mse", f"{val['loss']:.4f} (mean-action baseline {val['baseline_mse']:.4f})")
    report.metric("val_action_err_px", f"{val['action_err_px']:.1f} (mean-action baseline "
                                       f"{val['baseline_err_px']:.1f})")
    report.metric("val_z_std", f"{val['z_std']:.3f} (info only; no variance hinge)")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")

    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json",
                                    job.art_dir / "config.yaml"])

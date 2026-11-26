"""Job train_wm (Phase 3): train the latent world model (CLAUDE.md §4) -> store `ckpt-wm`.

Also runs train_wm_1step (Phase 4) with artifact `ckpt-wm-1step` and the teacher-forced config.
Checkpoints go to the store every ckpt_every steps with complete=false, so a killed run resumes
from the latest one (same config_hash). The final artifact holds model.pt (weights + config +
param counts + git), train_summary.json, config.yaml and rollout_filmstrip.png.
"""
from __future__ import annotations

from typing import Any

import imageio.v2 as imageio
import torch

from wm.data.windows import WindowData
from wm.jobs import JobContext
from wm.jobs._trainer import TrainingJob, check_param_counts, load_npz, same_outputs
from wm.models.world_model import EXPECTED_PARAMS, build_world_model, param_counts
from wm.report import Report
from wm.train.train_wm import LOSS_KEYS, decoded_rollout, run_training
from wm.viz.media import rollout_filmstrip

ARTIFACT = "ckpt-wm"
KS = (1, 5, 10)


def run(ctx: JobContext, report: Report, artifact: str = ARTIFACT,
        copy_last_must_win: bool = True) -> None:
    """Train, validate, save. copy_last_must_win=False (wm_1step) makes the k=10 check WARN-only."""
    job = TrainingJob(ctx, artifact)
    cfg, tcfg, device = job.cfg, job.tcfg, ctx.device
    weights = dict(cfg["loss"])

    arrays, job.data_name = load_npz(ctx)
    data = WindowData(arrays, device, cfg["val"]["seeds"],
                      window=cfg["window"]["context"] + cfg["window"]["horizon"])
    model = build_world_model(cfg).to(device)
    if model.input_mean is not None:
        model.set_input_mean(data.train_mean_frame())
    counts = param_counts(model)
    job.set_params(report, counts)
    check_param_counts(report, counts, EXPECTED_PARAMS, "CLAUDE.md §4")

    def on_val(step: int, v: dict[str, Any]) -> None:
        print(f"step {step}: val loss {v['loss']:.4g} nmse@10 {v['nmse'][-1]:.3f} "
              f"(copy-last {v['nmse_copy_last'][-1]:.3f}) z_std {v['z_std']:.3f}", flush=True)
        row = {f"val/{k}": v[k] for k in ("loss", "rec", "lat", "img", "z_std")}
        row.update({f"val/nmse_k{k}": v["nmse"][k - 1] for k in KS})
        row.update({f"val/copy_last_nmse_k{k}": v["nmse_copy_last"][k - 1] for k in KS})
        job.wandb_log(row, step)

    result = run_training(model, data, tcfg, device, weights, start=job.start,
                          on_checkpoint=job.on_checkpoint, on_log=job.on_log, on_val=on_val)
    step, log, val = result["step"], result["log"], result["val_log"][-1]

    # ---- final artifact: weights only + round-trip check ----
    model_path = job.save_model(model, step)
    reloaded = build_world_model(cfg).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model"])
    model.eval(); reloaded.eval()
    same = same_outputs(model.encode, reloaded.encode, data.gather(data.val_starts[:2])[0])

    real, decoded = decoded_rollout(model, data, data.fixed_val_starts(1, tcfg["val_seed"])[0], device)
    strip_path = job.art_dir / "rollout_filmstrip.png"
    imageio.imwrite(strip_path, rollout_filmstrip(real, decoded))
    summary = job.write_summary(result, LOSS_KEYS, data_summary=data.summary(), val=val)
    first, last = summary["loss_first"], summary["loss_last"]

    # ---- checks ----
    job.loss_checks(report, summary, log, LOSS_KEYS)
    report.check_true("z_std", val["z_std"] > 0.1, f"{val['z_std']:.3f}", "> 0.1, collapse alarm")
    report.check_true("checkpoint_roundtrip", same, "identical latents" if same else "mismatch",
                      "model.pt reload == trained model")
    beat = val["nmse"][-1] < val["nmse_copy_last"][-1]
    strict = copy_last_must_win and ctx.mode == "full"
    report.check("nmse_k10_vs_copy_last", "PASS" if beat else ("FAIL" if strict else "WARN"),
                 f"{val['nmse'][-1]:.3f} vs {val['nmse_copy_last'][-1]:.3f}",
                 "model < copy-last; WARN only in smoke" if copy_last_must_win
                 else "model < copy-last; WARN only for the 1-step baseline")

    # ---- metrics ----
    ds = data.summary()
    report.metric("data", job.data_name)
    report.metric("steps", f"{step} (batch {tcfg['batch_size']})")
    report.metric("resumed_from_step", job.resumed_from)
    report.metric("train_windows", ds["train_windows"])
    report.metric("val_source", ds["val_source"])
    report.metric("val_windows", val["windows"])
    for k in ("loss", "rec", "lat", "img"):
        report.metric(f"{k}_first_last", f"{first[k]:.4f} -> {last[k]:.4f}")
    job.timing_metrics(report, summary)
    for k in KS:
        report.metric(f"val_nmse_k{k}", f"{val['nmse'][k - 1]:.3f} (copy-last {val['nmse_copy_last'][k - 1]:.3f})")
    report.metric("val_z_std", f"{val['z_std']:.3f}")
    report.metric("val_loss", f"{val['loss']:.4f}")
    report.metric("lr_final", f"{log[-1]['lr']:.2e}")

    job.push_final(report, result, [model_path, job.art_dir / "train_summary.json",
                                    job.art_dir / "config.yaml", strip_path])

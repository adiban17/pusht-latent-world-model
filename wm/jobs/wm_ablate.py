"""Job wm_ablate (Phase 3b): short runs of the world-model fixes, compared on val.

Variants (configs/wm_ablate.yaml), each 5,000 steps on the full data with the same seed and the
same batches: v1 (all off), A (fg_weight), AB (fg_weight + proprio, trained with keep_best).
AB gives two rows: its last weights ("A+B") and its best-val-NMSE@10 weights ("A+B+C"), because
keep_best (C) only changes which checkpoint is kept. Per row, on val episodes:
  - latent NMSE@5/@10 and copy-last (each variant normalised by its own latent spread);
  - ridge probe z -> state fit on train frames: on real val z, and on imagined z at k = 10 next
    to copy-last z_t (block px, angle deg, pusher px);
  - B rows: pusher px of the predicted p_hat at k = 10 (and copy-last p_t);
  - non-background reconstruction MSE (same mask as diagnose_wm); it/s.
The winner is picked by the pre-registered rule in `select_winner` (PROGRESS.md, Phase 3b).
Writes ablation_wm_fixes.json (+ results/ablation_wm_fixes.json in full mode) and one val
reconstruction filmstrip per row; pushes `results-wm-ablate` after every variant, so a rerun
skips the variants that already finished.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from wm.data.windows import WindowData, episode_frames, frames_to_model, split_episodes
from wm.envs.pusht import denormalize_image
from wm.eval.diagnose import (encode_frames, imagined_probe, pusher_errors, recon_errors,
                              ridge_probe, subset, window_terms)
from wm.eval.rollouts import latent_nmse
from wm.jobs import JobContext
from wm.jobs._trainer import load_npz
from wm.models.world_model import build_world_model, expected_params, param_counts
from wm.report import Report, format_duration
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.common import loss_window_means, resolve_config
from wm.train.train_wm import loss_keys, prepare_model, run_training
from wm.viz.media import rollout_filmstrip

ARTIFACT = "results-wm-ablate"
RESULTS_NAME = "ablation_wm_fixes.json"
FILM_FRAMES = 8
ROW_LABELS = {"v1": ["v1"], "A": ["A"], "AB": ["A+B", "A+B+C"]}   # training run -> table rows
TIE_PX = 1.0


# ---- configuration ---------------------------------------------------------------------------

def variant_config(abl: dict[str, Any], base: dict[str, Any], name: str) -> dict[str, Any]:
    """Full world-model config of one variant: base + flags (+ option values) + train overrides;
    the ablation's smoke block replaces the base one."""
    cfg = copy.deepcopy(base)
    flags = dict(abl["variants"][name] or {})
    cfg.update(flags)
    opts = abl["options"]
    if flags.get("fg_weight"):
        cfg["fg_alpha"], cfg["fg_threshold"] = float(opts["fg_alpha"]), float(opts["fg_threshold"])
    if flags.get("proprio"):
        cfg["loss"]["proprio"] = float(opts["loss_proprio"])
    cfg["train"].update(abl["train"])
    cfg["smoke"] = copy.deepcopy(abl["smoke"])
    return cfg


def flags_of(cfg: dict[str, Any]) -> dict[str, bool]:
    return {k: bool(cfg.get(k, False)) for k in ("fg_weight", "proprio", "keep_best")}


# ---- evaluation ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data: WindowData, arrays: dict[str, np.ndarray], weights: dict[str, float],
             val_starts: torch.Tensor, val_seeds: list[int], ecfg: dict[str, Any],
             device) -> dict[str, Any]:
    """Every per-row metric of the table (see module docstring), on val."""
    model.eval()
    c, ks = model.context, tuple(int(k) for k in ecfg["ks"])
    states = arrays["states"]
    train_eps, _, _ = split_episodes(arrays["episode_seeds"], val_seeds)
    starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
    train_frames = subset(torch.from_numpy(episode_frames(starts[train_eps], lengths[train_eps])),
                          int(ecfg["max_train_frames"]))
    val_frames = data.val_frame_idx

    terms, z, z_hat, p_hat = window_terms(model, data, val_starts, weights, with_p=True)
    z_ref = encode_frames(model, data.frames[val_frames.to(device)])
    nmse = latent_nmse(z_hat, z[:, c:], z_ref).tolist()
    nmse_copy = latent_nmse(z[:, c - 1:c].expand_as(z[:, c:]), z[:, c:], z_ref).tolist()

    z_tr = encode_frames(model, data.frames[train_frames.to(device)])
    w, ridge = ridge_probe(z_tr, states[train_frames.numpy()], z_ref, states[val_frames.numpy()],
                           float(ecfg["ridge_lambda"]))
    imagined = imagined_probe(w, val_starts.numpy(), z, z_hat, states, c, ks)
    pusher = pusher_errors(p_hat, val_starts.numpy(), states, c, ks) if p_hat is not None else None
    recon = recon_errors(model, data.frames[val_frames.to(device)], data.train_mean_frame())
    return {"loss_terms": terms, "z_std": float(z.reshape(-1, z.shape[-1]).std(0).mean()),
            "nmse": {f"k{k}": nmse[k - 1] for k in ks},
            "nmse_copy_last": {f"k{k}": nmse_copy[k - 1] for k in ks},
            "ridge": ridge, "ridge_imagined": imagined, "p_hat": pusher, "recon": recon,
            "windows": len(val_starts)}


@torch.no_grad()
def recon_strip(model, data: WindowData, device) -> np.ndarray:
    """Encode -> decode filmstrip of 8 evenly spaced val frames (real row over decoded row)."""
    idx = data.val_frame_idx
    pick = idx[np.linspace(0, len(idx) - 1, FILM_FRAMES).round().astype(int)].to(device)
    x = data.frames[pick]
    rec = model.decode(model.encode(frames_to_model(x))).float().cpu().numpy()
    return rollout_filmstrip(x.cpu().numpy(), denormalize_image(rec))


def flat_row(label: str, run: str, ev: dict[str, Any], train: dict[str, Any],
             checkpoint: str) -> dict[str, Any]:
    """One table row (flat, means) from an evaluate() result; `details` keeps everything."""
    k10 = ev["ridge_imagined"]["k10"]
    ph = ev["p_hat"]["k10"] if ev["p_hat"] else None

    def three(e: dict[str, float]) -> dict[str, float]:
        return {"block_px": e["block_px_mean"], "angle_deg": e["block_deg_mean"], "pusher_px": e["pusher_px_mean"]}

    return {
        "row": label, "run": run, "checkpoint": checkpoint, **train,
        "z_std": ev["z_std"],
        "nmse_k5": ev["nmse"]["k5"], "nmse_k10": ev["nmse"]["k10"],
        "copy_last_nmse_k5": ev["nmse_copy_last"]["k5"], "copy_last_nmse_k10": ev["nmse_copy_last"]["k10"],
        "ridge_real_val": three(ev["ridge"]["val"]),
        "imagined_k10": three(k10["imagined"]),
        "copy_last_k10": three(k10["copy_last"]),
        "real_z_k10": three(k10["real_z"]),
        "p_hat_pusher_px_k10": ph["p_hat_mean"] if ph else None,
        "copy_last_pusher_px_k10": ph["copy_last_mean"] if ph else None,
        "recon_fg_mse_val": ev["recon"]["mse_foreground"],
        "recon_fg_mean_frame_mse_val": ev["recon"]["mean_frame_mse_foreground"],
        "recon_mse_val": ev["recon"]["mse"],
        "details": ev,
    }


# ---- pre-registered selection rule (PROGRESS.md, Phase 3b) -----------------------------------

def pusher_score(row: dict[str, Any]) -> float:
    """Tie-break: p_hat pusher px at k = 10 for proprio rows, else the ridge pusher px on real z."""
    p = row.get("p_hat_pusher_px_k10")
    return p if p is not None else row["ridge_real_val"]["pusher_px"]


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest imagined block px at k = 10 among rows that (a) beat copy-last on block px at k = 10
    and (b) have imagined angle error at k = 10 no worse than copy-last's. If none satisfies both,
    say so and use the rows satisfying (a). Ties (within 1 px of the lowest): lower pusher_score."""
    a = [r for r in rows if r["imagined_k10"]["block_px"] < r["copy_last_k10"]["block_px"]]
    ab = [r for r in a if r["imagined_k10"]["angle_deg"] <= r["copy_last_k10"]["angle_deg"]]
    if ab:
        pool, note = ab, "satisfies (a) block px < copy-last and (b) angle <= copy-last at k=10"
    elif a:
        pool, note = a, "NO row satisfies both (a) and (b); picked among rows satisfying (a) only"
    else:
        return {"winner": None, "note": "NO row beats copy-last on imagined block px at k=10",
                "a": [], "ab": []}
    best_px = min(r["imagined_k10"]["block_px"] for r in pool)
    ties = [r for r in pool if r["imagined_k10"]["block_px"] <= best_px + TIE_PX]
    win = min(ties, key=pusher_score)
    if len(ties) > 1:
        note += f"; tie within {TIE_PX:g} px among {[r['row'] for r in ties]} broken by pusher error"
    return {"winner": win["row"], "note": note, "a": [r["row"] for r in a], "ab": [r["row"] for r in ab]}


# ---- table -----------------------------------------------------------------------------------

TABLE_HEADER = ("nmse@5 (copy) | nmse@10 (copy) | ridge real z: block px / angle / pusher px | "
                "imagined@10 block px (copy) | angle deg (copy) | pusher px (copy) | "
                "p_hat@10 px (copy) | recon fg mse | it/s")


def fmt(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def table_line(r: dict[str, Any]) -> str:
    im, cl, rr = r["imagined_k10"], r["copy_last_k10"], r["ridge_real_val"]
    return (f"{r['nmse_k5']:.3f} ({r['copy_last_nmse_k5']:.3f}) | {r['nmse_k10']:.3f} ({r['copy_last_nmse_k10']:.3f}) | "
            f"{rr['block_px']:.1f} / {rr['angle_deg']:.1f} / {rr['pusher_px']:.1f} | "
            f"{im['block_px']:.1f} ({cl['block_px']:.1f}) | {im['angle_deg']:.1f} ({cl['angle_deg']:.1f}) | "
            f"{im['pusher_px']:.1f} ({cl['pusher_px']:.1f}) | "
            f"{fmt(r['p_hat_pusher_px_k10'])} ({fmt(r['copy_last_pusher_px_k10'])}) | "
            f"{r['recon_fg_mse_val']:.5f} | {fmt(r['it_per_s'], 2)}")


def all_finite(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    return not isinstance(obj, float) or math.isfinite(obj)


# ---- partial results (resume) ----------------------------------------------------------------

def load_partial(ctx: JobContext, art_dir: Path) -> dict[str, Any]:
    """Finished runs from an earlier, incomplete push with the same config_hash ({} if none);
    their files are copied into art_dir."""
    meta = ctx.store.own_metadata(ARTIFACT)
    if not meta or meta.get("complete") or meta.get("config_hash") != ctx.config_hash:
        return {}
    path = ctx.store.pull_own(ARTIFACT)
    if path is None or not (Path(path) / RESULTS_NAME).exists():
        return {}
    shutil.copytree(path, art_dir, dirs_exist_ok=True)
    return json.loads((art_dir / RESULTS_NAME).read_text()).get("runs", {})


# ---- job -------------------------------------------------------------------------------------

def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    abl, device = ctx.config, ctx.device
    base = load_yaml(REPO_ROOT / abl["base_config"])
    art_dir = ctx.out_dir / "artifact"
    if art_dir.exists():
        shutil.rmtree(art_dir)
    art_dir.mkdir(parents=True)

    arrays, data_name = load_npz(ctx)
    c, h = base["window"]["context"], base["window"]["horizon"]
    data = WindowData(arrays, device, base["val"]["seeds"], window=c + h)

    runs = load_partial(ctx, art_dir)
    if runs:
        print(f"resuming: runs already finished: {', '.join(runs)}", flush=True)

    def save_json(complete: bool) -> dict[str, Any]:
        rows = [r for name in abl["variants"] if name in runs for r in runs[name]["rows"]]
        results = {
            "job": "wm_ablate", "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
            "config_hash": ctx.config_hash, "data": data_name, "val_source": data.val_source,
            "ablation_config": abl, "base_config": base, "complete": complete,
            "note": "NMSE is normalised by each variant's own latent spread, so it is not compared "
                    "across variants; state errors (px, deg) are. Errors are means over val windows.",
            "table": [{k: v for k, v in r.items() if k != "details"} for r in rows],
            "selection": select_winner(rows) if complete and rows else None,
            "runs": runs, "wall_clock_s": time.time() - t0,
        }
        (art_dir / RESULTS_NAME).write_text(json.dumps(results, indent=2, default=float))
        return results

    for name in abl["variants"]:
        if name in runs:
            continue
        cfg = variant_config(abl, base, name)
        tcfg = resolve_config(cfg, ctx.mode, ctx.platform)
        weights = dict(cfg["loss"])
        seed_all(cfg["seed"])
        model = prepare_model(build_world_model(cfg).to(device), data)
        counts = param_counts(model)
        print(f"[{name}] flags {flags_of(cfg)} params " + ", ".join(f"{k} {v:,}" for k, v in counts.items()),
              flush=True)
        val_starts = data.fixed_val_starts(tcfg["val_windows"], tcfg["val_seed"])

        def on_log(step: int, e: dict[str, float], name=name) -> None:
            print(f"[{name}] step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in e.items()), flush=True)
            if ctx.wandb_run is not None:
                ctx.wandb_run.log({f"{name}/train/{k}": v for k, v in e.items()})

        def on_val(step: int, v: dict[str, Any], name=name) -> None:
            print(f"[{name}] step {step}: val nmse@10 {v['nmse'][-1]:.3f} (copy-last "
                  f"{v['nmse_copy_last'][-1]:.3f}) z_std {v['z_std']:.3f}", flush=True)

        t_run = time.time()
        keep_best = bool(cfg.get("keep_best", False))
        res = run_training(model, data, tcfg, device, weights, keep_best=keep_best,
                           on_log=on_log, on_val=on_val)
        it_s = res["it_per_s"]
        train = {"steps": res["step"], "batch_size": tcfg["batch_size"], "it_per_s": it_s,
                 "projected_30k_steps": format_duration(30000 / it_s) if it_s else None,
                 "train_wall_clock_s": time.time() - t_run, "flags": flags_of(cfg), "params": counts,
                 "params_expected": expected_params(model), "best_step": None, "best_nmse10_during_training": None}
        checkpoints = [("last", None)]
        if keep_best:
            train["best_step"] = res["best"]["step"]
            train["best_nmse10_during_training"] = res["best"]["value"]
            checkpoints.append((f"best@{res['best']['step']}", res["best"]["model"]))
        rows = []
        for label, (ckpt, state) in zip(ROW_LABELS[name], checkpoints):
            if state is not None:
                model.load_state_dict(state)
            ev = evaluate(model, data, arrays, weights, val_starts, base["val"]["seeds"], abl["eval"], device)
            rows.append(flat_row(label, name, ev, train, ckpt))
            strip = recon_strip(model, data, device)
            imageio.imwrite(art_dir / f"recon_filmstrip_{label.replace('+', '')}.png", strip)
            if ctx.wandb_run is not None:
                import wandb
                ctx.wandb_run.log({f"wm_ablate/recon_{label}": wandb.Image(strip)})
        n = min(100, max(1, len(res["log"]) // 5))
        first, last = loss_window_means(res["log"], n, loss_keys(model))
        runs[name] = {"rows": rows, "loss_first": first, "loss_last": last, "loss_window": n,
                      "val_log": res["val_log"]}
        save_json(complete=False)
        ctx.store.push(ARTIFACT, art_dir, ctx.meta(complete=False, done=list(runs)))
        print(f"[{name}] done in {format_duration(time.time() - t_run)}", flush=True)
        del model, res

    results = save_json(complete=True)
    rows = [r for name in abl["variants"] for r in runs[name]["rows"]]
    json_path = art_dir / RESULTS_NAME
    saved = [json_path]
    if ctx.mode == "full":   # smoke never writes into the repo's results/
        repo_json = REPO_ROOT / "results" / RESULTS_NAME
        repo_json.parent.mkdir(exist_ok=True)
        shutil.copy2(json_path, repo_json)
        saved.append(repo_json)

    # ---- REPORT ----
    ok = all_finite(results["table"])
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    for name in abl["variants"]:
        r = runs[name]["rows"][0]
        report.check_true(f"param_counts_{name}", r["params"] == r["params_expected"],
                          f"total {r['params']['total']:,} / planner {r['params']['planner']:,}",
                          "== CLAUDE.md §4" + (" + proprio" if r["flags"]["proprio"] else ""))
    for r in rows:
        report.check("z_std_" + r["row"], "PASS" if r["z_std"] > 0.1 else "WARN", f"{r['z_std']:.3f}",
                     "> 0.1, collapse alarm; WARN only in the ablation")
    report.params = " / ".join(f"{n} {runs[n]['rows'][0]['params']['total']:,}" for n in abl["variants"])

    report.metric("data", data_name)
    report.metric("val_source", data.val_source)
    report.metric("steps", f"{rows[0]['steps']} per run (batch {rows[0]['batch_size']}), same seed and batches")
    report.metric("val_windows", rows[0]["details"]["windows"])
    report.metric("table", TABLE_HEADER)
    for r in rows:
        report.metric(f"{r['row']:<6}", table_line(r))
    for name in abl["variants"]:
        r = runs[name]["rows"][0]
        report.metric(f"run_{name}", f"{fmt(r['it_per_s'], 2)} it/s, projected 30k steps "
                      f"{r['projected_30k_steps'] or 'n/a'}, trained in {format_duration(r['train_wall_clock_s'])}")
    ab = runs.get("AB", {}).get("rows", [])
    if len(ab) == 2:
        report.metric("keep_best", f"best step {ab[1]['best_step']} of {ab[0]['steps']} "
                      f"(val nmse@10 best {ab[1]['nmse_k10']:.3f} vs last {ab[0]['nmse_k10']:.3f})")
    sel = results["selection"]
    report.metric("winner", f"{sel['winner'] or 'none'} ({sel['note']})")
    for p in saved:
        report.save(p)
    report.save(ctx.store.push(ARTIFACT, art_dir, ctx.meta(complete=not report.failed, done=list(runs))))

"""Phase 3 investigation: does a collapse fix keep z_std > 0.1? (local CPU, not part of the pipeline)

Same 22 episodes (train seeds 0-19 + val seeds 190, 191; collected once and cached), same seed,
400 steps, batch 16. Architecture and param counts are unchanged in every variant.
  base: CLAUDE.md §4 as written
  a:    subtract the train mean frame from encoder inputs
  b:    + VICReg-style variance hinge on z, lambda_var = 1.0
  c:    detach encoder outputs used as warm-up / rollout inputs (dynamics-only latent loss)
  d:    b + c

Usage: python scripts/compare_collapse_fixes.py --variant a      (one run -> <out>/<variant>.json)
       python scripts/compare_collapse_fixes.py --table          (markdown table of finished runs)
       python scripts/compare_collapse_fixes.py --export         (-> results/ablation_collapse_local.json)
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wm.data.collect import collect_split  # noqa: E402
from wm.data.windows import WindowData  # noqa: E402
from wm.models.world_model import build_world_model, param_counts  # noqa: E402
from wm.repro import REPO_ROOT, git_info, load_yaml, seed_all  # noqa: E402
from wm.train.train_wm import resolve_config, run_training  # noqa: E402

OUT = REPO_ROOT / "outputs" / "collapse_compare"
SEEDS = list(range(20)) + [190, 191]
VARIANTS = {
    "base": {},
    "a": {"input_mean_frame": True},
    "b": {"loss_var": 1.0},
    "c": {"detach_dynamics_inputs": True},
    "d": {"loss_var": 1.0, "detach_dynamics_inputs": True},
}
TRAIN = {"steps": 400, "batch_size": 16, "warmup_steps": 40, "val_every": 100,
         "ckpt_every": 10**9, "log_every": 10**9, "val_windows": 64}


def load_data() -> dict[str, np.ndarray]:
    path = OUT / "data_22eps.npz"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        arrays, _ = collect_split(SEEDS, load_yaml(REPO_ROOT / "configs/data.yaml")["expert"], 100)
        np.savez_compressed(path, **arrays)
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def run_variant(name: str, threads: int) -> dict:
    torch.set_num_threads(threads)
    cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/wm.yaml"))
    v = VARIANTS[name]
    for k in ("input_mean_frame", "detach_dynamics_inputs"):
        if k in v:
            cfg[k] = v[k]
    weights = {**cfg["loss"], "var": v.get("loss_var", 0.0)}  # explicit: base/a/c run without it
    data = WindowData(load_data(), "cpu", cfg["val"]["seeds"])
    seed_all(0)
    model = build_world_model(cfg)
    if model.input_mean is not None:
        model.set_input_mean(data.train_mean_frame())
    tcfg = {**resolve_config(cfg, "smoke", "local"), **TRAIN}
    t0 = time.time()
    curve = []

    def on_val(step, val):
        curve.append({"step": step, "z_std": val["z_std"], "rec": val["rec"], "img": val["img"],
                      "nmse1": val["nmse"][0], "nmse10": val["nmse"][-1]})
        print(f"[{name}] step {step} ({time.time() - t0:.0f}s) z_std {val['z_std']:.4f} "
              f"nmse10 {val['nmse'][-1]:.2f} copy {val['nmse_copy_last'][-1]:.2f}", flush=True)

    res = run_training(model, data, tcfg, torch.device("cpu"), weights, on_val=on_val)
    val = res["val_log"][-1]
    out = {"variant": name, "settings": v, "train": TRAIN, "seeds": SEEDS, "git": git_info(),
           "params": param_counts(model), "wall_clock_s": time.time() - t0,
           "z_std": val["z_std"], "rec": val["rec"], "img": val["img"], "lat": val["lat"],
           "nmse": val["nmse"], "nmse_copy_last": val["nmse_copy_last"], "curve": curve}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
    return out


def table() -> str:
    rows = ["| variant | z_std | rec | img | NMSE@1 (copy-last) | NMSE@10 (copy-last) |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name in VARIANTS:
        p = OUT / f"{name}.json"
        if not p.exists():
            rows.append(f"| {name} | pending | | | | |")
            continue
        r = json.loads(p.read_text())
        rows.append(f"| {name} | {r['z_std']:.4f} | {r['rec']:.4f} | {r['img']:.4f} | "
                    f"{r['nmse'][0]:.3f} ({r['nmse_copy_last'][0]:.3f}) | "
                    f"{r['nmse'][-1]:.3f} ({r['nmse_copy_last'][-1]:.3f}) |")
    return "\n".join(rows)


def export(path: Path = REPO_ROOT / "results" / "ablation_collapse_local.json") -> Path:
    """Collect the finished runs into one results file (CLAUDE.md hard rule 1: numbers come from here)."""
    runs = {}
    for name in VARIANTS:
        r = json.loads((OUT / f"{name}.json").read_text())
        runs[name] = {k: r[k] for k in ("settings", "z_std", "rec", "img", "lat", "nmse",
                                        "nmse_copy_last", "wall_clock_s", "git", "params", "curve")}
    doc = {
        "what": "Phase 3 latent-collapse comparison, local CPU",
        "date": "2026-11-26",
        "data": f"{len(SEEDS)} expert episodes collected locally: train seeds 0-19, val seeds 190-191",
        "train": TRAIN, "seed": 0,
        "metrics_at": f"step {TRAIN['steps']}, {TRAIN['val_windows']} fixed val windows",
        "nmse_note": "NMSE_k is normalised by each variant's own latent spread over the val frames",
        "variants": {"base": "CLAUDE.md §4 as written", "a": "subtract train mean frame from inputs",
                     "b": "VICReg variance hinge on z, lambda_var = 1.0",
                     "c": "detach encoder outputs used as warm-up/rollout inputs", "d": "b + c"},
        "chosen": "b",
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.variant:
        run_variant(args.variant, args.threads)
    if args.table:
        print(table())
    if args.export:
        print(export())

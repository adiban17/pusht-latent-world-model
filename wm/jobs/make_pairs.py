"""Job make_pairs (Phase 6; CLAUDE.md §9 Pairs) -> store `eval-pairs`.

eval_pairs_val.json: 50 pairs from the val episodes (seeds 190-199) of pusht_train.npz (smoke data
has none: its last train episode stands in, as in training). eval_pairs_test.json: 200 pairs from
pusht_test.npz; only final_eval (Phase 7) reads them. Full mode also writes both into results/.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import yaml

from wm.data.windows import split_episodes
from wm.jobs import JobContext
from wm.jobs._trainer import load_npz
from wm.eval.make_pairs import make_pairs
from wm.repro import REPO_ROOT
from wm.report import Report

ARTIFACT = "eval-pairs"
FILES = {"val": "eval_pairs_val.json", "test": "eval_pairs_test.json"}


def run(ctx: JobContext, report: Report) -> None:
    cfg = ctx.config
    smoke = ctx.mode == "smoke"
    n = {s: int((cfg["smoke"] if smoke else cfg)[f"n_{s}"]) for s in FILES}
    kw = {k: cfg[k] for k in ("offset", "start_step", "start_max", "min_move_px", "min_turn_deg")}
    train, data_name = load_npz(ctx)
    test, _ = load_npz(ctx, "pusht_test.npz")
    _, val_eps, val_source = split_episodes(train["episode_seeds"], cfg["val_seeds"])
    sources = {"val": (train, val_eps, val_source),
               "test": (test, np.ones(len(test["episode_seeds"]), bool), "test episodes (pusht_test.npz)")}

    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = {}
    for split, (arrays, eps, source) in sources.items():
        pairs, n_cand = make_pairs(arrays, eps, n[split], seed=int(cfg["seed"]), **kw)
        out[split] = pairs
        doc = {"split": split, "source": source, "data": data_name, "seed": int(cfg["seed"]),
               "filter": kw, "candidates": n_cand, "n": len(pairs), "git": ctx.git,
               "config_hash": ctx.config_hash, "pairs": pairs}
        (art / FILES[split]).write_text(json.dumps(doc, indent=2))
        seeds = sorted({p["episode_seed"] for p in pairs})
        report.check_true(f"pairs_{split}", len(pairs) == n[split],
                          f"{len(pairs)} of {n_cand} candidates from {len(seeds)} episodes ({source})",
                          f"== {n[split]}")
        rot = sum(abs(p["turn_deg"]) >= cfg["min_turn_deg"] for p in pairs)
        report.metric(f"{split}_pairs", f"{len(pairs)}: rotation (|turn| >= {cfg['min_turn_deg']:g} deg) {rot}, "
                                        f"translation {len(pairs) - rot}; median move "
                                        f"{np.median([p['move_px'] for p in pairs]):.1f} px, median |turn| "
                                        f"{np.median([abs(p['turn_deg']) for p in pairs]):.1f} deg")
    if "val" in out and not smoke:
        ok = all(190 <= p["episode_seed"] <= 199 for p in out["val"])
        report.check_true("val_pairs_from_val_episodes", ok, "all seeds in 190-199" if ok else "outside",
                          "val pairs only from seeds 190-199")
    report.metric("data", data_name)
    for f in FILES.values():
        report.save(art / f)
    if ctx.mode == "full":
        (REPO_ROOT / "results").mkdir(exist_ok=True)
        for f in FILES.values():
            shutil.copy2(art / f, REPO_ROOT / "results" / f)
            report.save(REPO_ROOT / "results" / f)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed)))

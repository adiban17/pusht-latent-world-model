"""Job plan_sweep (Phase 6; CLAUDE.md §9): cem_wm on the val pairs, one CEM setting changed at a
time from configs/plan.yaml `cem`: N in {32, 64, 128, 256}, rounds in {1, 2, 4}, H in {5, 10, 15},
beta in {0, 0.1}. The pre-registered freeze rule (`wm.eval.planning.select_setting`) picks the
setting that Phase 6 writes into plan.yaml. Writes plan_sweep.json, pushes `results-plan-sweep`;
full mode also writes results/plan_sweep.json in the checkout.
"""
from __future__ import annotations

import json
import shutil
import time
from typing import Any

import yaml

from wm.envs.pusht import make_env
from wm.eval.planning import select_setting, summarize
from wm.eval.rollouts import all_finite
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._planning import (all_ran, check_ckpts, ckpt_metrics, episode_records, fmt_rate, load_pairs,
                               run_method)
from wm.jobs._trainer import load_npz
from wm.planning.cem import CEMConfig
from wm.planning.mpc import CEMPolicy
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, seed_all
from wm.report import Report

ARTIFACT = "results-plan-sweep"
RESULTS_NAME = "plan_sweep.json"
SWEEP_KEYS = (("N", "candidates"), ("rounds", "rounds"), ("H", "horizon"), ("beta", "beta"))


def sweep_settings(base: dict[str, Any], grid: dict[str, list]) -> dict[str, dict[str, Any]]:
    """{'default': base, 'N=32': base with candidates 32, ...}: one change at a time."""
    out = {"default": dict(base)}
    for short, key in SWEEP_KEYS:
        for v in grid[key]:
            if v != base[key]:
                out[f"{short}={v:g}"] = {**base, key: v}
    return out


def compute_cost(c: dict[str, Any]) -> int:
    """Model rollout work per planning call: N x rounds x H."""
    return int(c["candidates"]) * int(c["rounds"]) * int(c["horizon"])


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    smoke, device = ctx.mode == "smoke", ctx.device
    seed = int(cfg["seed"])
    seed_all(seed)
    budget = int(cfg["smoke"]["budget"] if smoke else cfg["budget"])
    model, info = load_matching(ctx, "ckpt-wm")
    ckpts = {"wm": info}
    report.params = f"wm {info['params']['total']:,}"
    pairs, pairs_info = load_pairs(ctx, "val")
    if smoke:
        pairs = pairs[:int(cfg["smoke"]["pairs"])]
    arrays, data_name = load_npz(ctx)
    env = make_env()
    env.reset(seed=seed)

    settings = sweep_settings(cfg["cem"], cfg["sweep"])
    episodes, summaries = {}, {}
    for name, c in settings.items():
        pol = CEMPolicy(model, CEMConfig.from_dict(c), device, f"cem_wm[{name}]")
        episodes[name] = run_method(env, pol, pairs, arrays["frames"], budget, seed)
        summaries[name] = summarize(episodes[name], pairs)
    costs = {n: compute_cost(c) for n, c in settings.items()}
    sel = select_setting(summaries, costs, "default", int(cfg["selection"]["sweep_margin"]),
                         int(cfg["selection"]["sweep_floor"]))
    frozen = {**cfg["cem"], **{k: v for k, v in settings[sel["selected"]].items()}}

    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
        "checkpoints": ckpts, "pairs": {**pairs_info, "n": len(pairs)}, "budget": budget, "seed": seed,
        "method": "cem_wm", "settings": settings, "compute_cost": costs, "summaries": summaries,
        "selection": {**sel, "rule": f"a variant replaces the default only with >= "
                                     f"{cfg['selection']['sweep_margin']} more successes (of {len(pairs)}); "
                                     f"several -> the most, ties -> cheaper N x rounds x H; if every setting has "
                                     f"<= {cfg['selection']['sweep_floor']} successes, the rule runs on "
                                     f"position-only success (start >= 20 px)", "frozen_cem": frozen},
        "episodes": {n: episode_records(e, pairs) for n, e in episodes.items()},
        "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))

    check_ckpts(report, ckpts, pairs_info)
    n_exp = int(cfg["smoke"]["pairs"]) if smoke else pairs_info["n_expected"]
    report.check_true("pairs", len(pairs) == n_exp, f"{len(pairs)} val pairs ({pairs_info['source']})", f"== {n_exp}")
    report.check_true("all_settings_ran", all_ran(episodes, len(pairs)), f"{len(settings)} settings",
                      f"{len(pairs)} episodes each")
    ok = all_finite(summaries)
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")

    ckpt_metrics(report, ckpts)
    report.metric("data", data_name)
    report.metric("pairs", f"{len(pairs)} val pairs ({pairs_info['source']}); budget {budget}; seed {seed}; cem_wm only")
    for name, s in summaries.items():
        report.metric(f"setting_{name}", f"success {fmt_rate(s['success'])} | position-only "
                                         f"{s['pos_success']['k']}/{s['pos_success']['n']} (start >= 20 px) | "
                                         f"median pos {s['median_final_pos_px']:.1f} px | "
                                         f"{s['seconds_per_episode']:.2f} s/ep | cost {costs[name]}")
    report.metric("selected", f"{sel['selected']} by {sel['metric']} ({sel['reason']})")
    report.metric("frozen_cem", ", ".join(f"{k} {v}" for k, v in frozen.items()))
    report.save(json_path)
    if ctx.mode == "full":
        repo_json = REPO_ROOT / "results" / RESULTS_NAME
        repo_json.parent.mkdir(exist_ok=True)
        shutil.copy2(json_path, repo_json)
        report.save(repo_json)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

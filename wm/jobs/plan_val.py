"""Job plan_val (Phase 6; CLAUDE.md §8, §9 Planning): every method on the val pairs, 1 seed.

Methods (configs/plan.yaml `methods`): cem_wm, cem_wm_1step, bc, bc_clean, random, noop, and
cem_pixel behind a flag. Reports per method: success (primary), position-only success (secondary,
headline on pairs starting >= 20 px away), both split into rotation / translation pairs, Wilson
95% CIs, median final errors, steps, s/episode. Also: which BC variant goes to the test
(`select_bc`), the imagined-success threshold tau (val-calibrated, for Phase 7), and a full-size
timing probe for every CEM method (projected s/episode for the full config).
Only val pairs are read. Writes plan_val.json (+ filmstrips), pushes `results-plan-val`; full
mode also writes results/plan_val.json in the checkout.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm.data.windows import episode_frames, split_episodes
from wm.envs.pusht import denormalize_image, get_state, make_env, render64, reset_to_state
from wm.eval.diagnose import encode_frames
from wm.eval.planning import calibrate_tau, select_bc, summarize
from wm.eval.rollouts import all_finite
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching
from wm.jobs._planning import (all_ran, check_ckpts, ckpt_metrics, episode_records, fmt_rate, load_pairs,
                               method_metrics, run_method, timed_calls)
from wm.jobs._trainer import load_npz
from wm.planning.cem import CEMConfig
from wm.planning.mpc import BCPolicy, CEMPolicy, History, NoopPolicy, PixelCEMPolicy, RandomPolicy
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, seed_all
from wm.report import Report
from wm.viz.media import multirow_filmstrip

ARTIFACT = "results-plan-val"
RESULTS_NAME = "plan_val.json"
CEM_METHODS = ("cem_wm", "cem_wm_1step", "cem_pixel")
# checkpoint key -> artifact, and which methods need it (ckpt-wm is always loaded: tau, cem_pixel's encoder)
CKPTS = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "bc": "ckpt-bc", "bc_clean": "ckpt-bc-clean",
         "pixel": "ckpt-pixel"}
NEEDS = {"cem_wm_1step": "wm_1step", "bc": "bc", "bc_clean": "bc_clean", "cem_pixel": "pixel"}
RENDER_TOL = 0.5     # mean abs uint8 difference, rendered vs stored frame


def render_alignment(env, pairs: list[dict[str, Any]], frames: np.ndarray) -> dict[str, list[float]]:
    """Mean abs uint8 difference between the frame rendered after reset_to_state(start / goal
    state) and the stored start / goal frame, per pair."""
    out: dict[str, list[float]] = {"start": [], "goal": []}
    for p in pairs:
        for key, s, i in (("start", p["start_state"], p["start_idx"]), ("goal", p["goal_state"], p["goal_idx"])):
            obs = reset_to_state(env, np.array(s, np.float32))
            out[key].append(float(np.abs(render64(obs["pixels"]).astype(np.float32) - frames[i]).mean()))
    return out


def plan_filmstrip(model, ep: dict[str, Any], goal: np.ndarray) -> np.ndarray:
    """Rows: real frames 1..H | decoded imagined plan of the first step | goal."""
    z = ep["plans"][0]
    h = z.shape[0]
    with torch.no_grad():
        imagined = denormalize_image(model.decode(z.to(next(model.parameters()).device)).float()
                                     .clamp(-0.5, 0.5).cpu().numpy())
    real = ep["frames"][1:h + 1]
    if len(real) < h:
        real = np.concatenate([real, np.repeat(real[-1:], h - len(real), 0)])
    return multirow_filmstrip({"real": real, "imagined": imagined, "goal": np.repeat(goal[None], h, 0)})


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    smoke, device = ctx.mode == "smoke", ctx.device
    seed = int(cfg["seed"])
    seed_all(seed)
    budget = int(cfg["smoke"]["budget"] if smoke else cfg["budget"])
    methods = [m for m, on in cfg["methods"].items() if on]
    cem_cfg = CEMConfig.from_dict(cfg["cem"])
    pixel_cfg = cem_cfg
    if smoke and ctx.platform == "local":
        pixel_cfg = dataclasses.replace(cem_cfg, **cfg["smoke"]["local_pixel"])

    keys = ["wm"] + [NEEDS[m] for m in methods if m in NEEDS]
    models, ckpts = {}, {}
    for k in keys:
        models[k], ckpts[k] = load_matching(ctx, CKPTS[k])
    report.params = " / ".join(f"{k} {v['params']['total']:,}" for k, v in ckpts.items())
    pairs, pairs_info = load_pairs(ctx, "val")
    if smoke:
        pairs = pairs[:int(cfg["smoke"]["pairs"])]
    arrays, data_name = load_npz(ctx)
    frames = arrays["frames"]
    env = make_env()
    env.reset(seed=seed)
    align = render_alignment(env, pairs, frames)

    def bc_scale(k: str) -> float:
        return float(ckpts[k]["summary"].get("config", {}).get("delta_scale", 50.0))

    def make_policy(m: str, full_size: bool = False):
        if m == "cem_wm":
            return CEMPolicy(models["wm"], cem_cfg, device, "cem_wm")
        if m == "cem_wm_1step":
            return CEMPolicy(models["wm_1step"], cem_cfg, device, "cem_wm_1step")
        if m == "cem_pixel":
            return PixelCEMPolicy(models["pixel"], models["wm"], cem_cfg if full_size else pixel_cfg, device,
                                  amp=bool(cfg["pixel_amp"]))
        if m in ("bc", "bc_clean"):
            return BCPolicy(models[m], device, m, bc_scale(m))
        if m == "random":
            return RandomPolicy(float(cfg["random_sigma"]), cem_cfg.delta_clip)
        if m == "noop":
            return NoopPolicy()
        raise KeyError(m)

    n_film = int(cfg["filmstrips"])
    episodes = {m: run_method(env, make_policy(m), pairs, frames, budget, seed,
                              keep=n_film if m == "cem_wm" else 0) for m in methods}
    summaries = {m: summarize(eps, pairs) for m, eps in episodes.items()}
    bc_sel = select_bc(summaries["bc"], summaries["bc_clean"]) if {"bc", "bc_clean"} <= set(summaries) else None

    # ---- timing probe: full-size CEM calls on the first pair's start -------------------------
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    obs = reset_to_state(env, np.array(pairs[0]["start_state"], np.float32))
    hist = History(render64(obs["pixels"]), get_state(env)[:2])
    calls = int(cfg["smoke"]["timing_calls"] if smoke else cfg["timing"]["calls"])
    full_budget = int(cfg["budget"])
    timing = {}
    for m in methods:
        if m not in CEM_METHODS:
            continue
        call_s = timed_calls(make_policy(m, full_size=True), hist, frames[pairs[0]["goal_idx"]], calls,
                             warmup=device.type == "cuda", sync=sync)
        env_s = summaries[m]["env_s_per_step"]
        per_ep = full_budget * (call_s + env_s)
        timing[m] = {"call_s": call_s, "env_s_per_step": env_s, "calls": calls,
                     "config": dataclasses.asdict(cem_cfg),
                     "projected_s_per_episode_full": per_ep,
                     "projected_s_50_pairs_upper": 50 * per_ep,
                     "note": f"upper bound: every episode uses all {full_budget} steps"}

    # ---- tau: imagined-success threshold on val frames --------------------------------------
    _, val_eps, val_source = split_episodes(arrays["episode_seeds"], [190, 199])
    starts, lengths = arrays["episode_starts"][val_eps], arrays["episode_lengths"][val_eps]
    vidx = episode_frames(starts, lengths)
    z_val = encode_frames(models["wm"], torch.from_numpy(frames[vidx]).to(device)).numpy()
    ep_of = np.repeat(np.arange(len(starts)), lengths)
    tau = calibrate_tau(z_val, arrays["states"][vidx], ep_of, float(cfg["tau_percentile"]),
                        int(cfg["smoke"]["tau_max_pairs"] if smoke else cfg["tau_max_pairs"]), seed)
    tau["source"] = val_source
    imagined = {}
    if tau["tau"] is not None:
        for m in methods:
            if m in CEM_METHODS:
                hit = [bool(e["terminal_costs"]) and min(e["terminal_costs"]) < tau["tau"] for e in episodes[m]]
                imagined[m] = {"imagined_success": int(sum(hit)),
                               "imagined_success_real_fail": int(sum(h and not e["success"]
                                                                     for h, e in zip(hit, episodes[m]))),
                               "n": len(hit)}

    # ---- outputs -----------------------------------------------------------------------------
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    (art / "filmstrips").mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    strips = []
    if "cem_wm" in episodes:
        for i, ep in enumerate(episodes["cem_wm"][:n_film]):
            if ep.get("plans"):
                strips.append(art / "filmstrips" / f"cem_wm_val_pair{pairs[i]['pair']}.png")
                imageio.imwrite(strips[-1], plan_filmstrip(models["wm"], ep, frames[pairs[i]["goal_idx"]]))
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    n_rot = sum(abs(p["turn_deg"]) >= 20.0 for p in pairs)
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
        "checkpoints": ckpts, "pairs": {**pairs_info, "n": len(pairs), "rotation_pairs": n_rot,
                                        "translation_pairs": len(pairs) - n_rot},
        "budget": budget, "seed": seed, "cem": dataclasses.asdict(cem_cfg),
        "cem_pixel_episode_config": dataclasses.asdict(pixel_cfg) if "cem_pixel" in methods else None,
        "success_definition": "block CoG < 20 px AND |angle| < 20 deg, checked after every step (primary)",
        "pos_success_definition": "block CoG < 20 px after any step (secondary); headline on pairs whose start "
                                  "CoG error >= 20 px; rotation pairs: |goal angle - start angle| >= 20 deg",
        "summaries": summaries, "bc_selected": ({"variant": bc_sel[0], "reason": bc_sel[1]} if bc_sel else None),
        "timing": timing, "tau": tau, "imagined_success": imagined,
        "render_alignment": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in align.items()},
        "episodes": {m: episode_records(eps, pairs) for m, eps in episodes.items()},
        "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None and strips:
        import wandb
        ctx.wandb_run.log({f"plan_val/{p.stem}": wandb.Image(str(p)) for p in strips})

    # ---- CHECKS ------------------------------------------------------------------------------
    check_ckpts(report, ckpts, pairs_info)
    n_exp = int(cfg["smoke"]["pairs"]) if smoke else pairs_info["n_expected"]
    report.check_true("pairs", len(pairs) == n_exp, f"{len(pairs)} val pairs ({pairs_info['source']})", f"== {n_exp}")
    report.check_true("all_methods_ran", all_ran(episodes, len(pairs)),
                      ", ".join(f"{m} {len(e)}" for m, e in episodes.items()), f"{len(pairs)} episodes each")
    for k, v in align.items():
        report.check_true(f"render_alignment_{k}", float(np.mean(v)) <= RENDER_TOL,
                          f"mean abs diff {np.mean(v):.3f} (max {np.max(v):.3f}) over {len(v)} pairs",
                          f"mean <= {RENDER_TOL} uint8 levels: rendered {k} state == stored {k} frame")
    if "noop" in summaries:
        k = summaries["noop"]["success"]["k"]
        report.check_true("noop_success_zero", k == 0, f"{k}/{len(pairs)}",
                          "== 0 (impossible by the pair filter; > 0 means a bug)")
    ok = all_finite({"summaries": summaries, "timing": timing})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    if {"cem_wm", "random"} <= set(summaries):
        a, b = summaries["cem_wm"]["success"]["k"], summaries["random"]["success"]["k"]
        report.check("cem_wm_vs_random", "PASS" if a > b else "WARN", f"{a} vs {b} successes",
                     "cem_wm > random; WARN only (a result, not a bug)")

    # ---- METRICS -----------------------------------------------------------------------------
    ckpt_metrics(report, ckpts)
    report.metric("data", data_name)
    report.metric("pairs", f"{len(pairs)} val pairs from {pairs_info['artifact']} ({pairs_info['source']}): "
                           f"rotation {n_rot}, translation {len(pairs) - n_rot}; budget {budget}; seed {seed}")
    report.metric("cem", f"N {cem_cfg.candidates}, H {cem_cfg.horizon}, rounds {cem_cfg.rounds}, K {cem_cfg.elites}, "
                         f"beta {cem_cfg.beta:g}, sigma {cem_cfg.sigma_init:g} (floor {cem_cfg.sigma_floor:g}), "
                         f"clip {cem_cfg.delta_clip:g} px" + (f"; cem_pixel episodes at N {pixel_cfg.candidates}, "
                                                              f"rounds {pixel_cfg.rounds} (local smoke)"
                                                              if pixel_cfg is not cem_cfg else ""))
    for m, s in summaries.items():
        method_metrics(report, m, s)
    if bc_sel:
        report.metric("bc_selected", f"{bc_sel[0]} ({bc_sel[1]}); success bc {fmt_rate(summaries['bc']['success'])} | "
                                     f"bc_clean {fmt_rate(summaries['bc_clean']['success'])}")
    for m, t in timing.items():
        report.metric(f"timing_{m}", f"{t['call_s']:.3f} s per full-size planning call (N {cem_cfg.candidates}, "
                                     f"{cem_cfg.rounds} rounds, H {cem_cfg.horizon}; mean of {t['calls']}) + env "
                                     f"{t['env_s_per_step'] * 1000:.1f} ms/step -> projected "
                                     f"{t['projected_s_per_episode_full']:.1f} s/episode, "
                                     f"{t['projected_s_50_pairs_upper'] / 60:.1f} min for 50 pairs (upper bound)")
    if tau["tau"] is not None:
        report.metric("tau", f"{tau['tau']:.4f} = p{tau['percentile']:g} of ||E(o_i) - E(o_j)||^2 over "
                             f"{tau['n_success_pairs']} same-episode val frame pairs that are successes "
                             f"(median {tau['median_success']:.4f}; non-success median "
                             f"{tau['median_non_success']:.4f}) [{val_source}]")
    else:
        report.metric("tau", "none: no same-episode success pairs")
    for m, v in imagined.items():
        report.metric(f"imagined_success_{m}", f"{v['imagined_success']}/{v['n']} episodes reached terminal cost "
                                               f"< tau; {v['imagined_success_real_fail']} of them failed for real")

    # ---- SAVED -------------------------------------------------------------------------------
    report.save(json_path)
    if strips:
        report.save(art / "filmstrips")
    if ctx.mode == "full":
        repo_json = REPO_ROOT / "results" / RESULTS_NAME
        repo_json.parent.mkdir(exist_ok=True)
        shutil.copy2(json_path, repo_json)
        report.save(repo_json)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

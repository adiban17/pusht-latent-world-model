"""Job final_eval (Phase 7; CLAUDE.md §9 final_eval, pre-registered in PROGRESS.md): every method on
the 200 TEST pairs with the frozen configs; the only job that reads test pairs.

Methods: cem_wm, cem_wm_1step, cem_pixel (seed 0 only), the BC arm `bc_selected` (from
results-plan-val), random, noop; seeds 0, 1, 2. Reports per method: success and position-only
success (Wilson per seed; pooled with pair-clustered bootstrap CIs), turning / non-turning
groups, medians, angle once position is reached, failure tags, imagined vs real (tau per
encoder), s/episode, s per planning call. Primary tests (pre-registered): cem_wm vs bc_selected,
cem_wm_1step and cem_pixel on success and position-only success: exact McNemar on seed 0, Holm
over the 6, paired bootstrap of the per-pair mean over each method's seeds. Secondary:
difficulty bins, a seeded gallery, an exploratory goal-distance sweep (offsets 10 / 30).
Crash safety: cheap arms first, cem_pixel last; partial_final_eval.json is rewritten after every
(method, seed). Writes final_eval.json, summary.md (from the JSONs only), plots, gallery and
results_bundle.zip; pushes results-final and results-bundle; full mode copies to results/final/.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import time
import zipfile
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm.data.windows import episode_frames, split_episodes
from wm.envs.pusht import denormalize_image, get_state, make_env, render64, reset_to_state
from wm.eval.diagnose import encode_frames
from wm.eval.final import (TAGS, believed_success, difficulty_table, failure_tag, imagined_counts, pick_gallery,
                           pooled_summary, primary_tests)
from wm.eval.final_summary import write_summary
from wm.eval.make_pairs import candidate_pairs
from wm.eval.planning import calibrate_tau, summarize
from wm.eval.rollouts import all_finite
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching, pull_matching
from wm.jobs._planning import (all_ran, check_ckpts, ckpt_metrics, episode_records, load_pairs, run_method,
                               timed_calls)
from wm.jobs._trainer import load_npz
from wm.jobs.plan_val import RENDER_TOL, render_alignment
from wm.planning.cem import CEMConfig
from wm.planning.mpc import BCPolicy, CEMPolicy, History, NoopPolicy, PixelCEMPolicy, RandomPolicy
from wm.registry import input_hashes, load_job_config, load_registry
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.report import Report
from wm.viz.media import multirow_filmstrip
from wm.viz.plots import rate_bars, rate_by_bin

ARTIFACT, BUNDLE = "results-final", "results-bundle"
RESULTS_NAME, PARTIAL_NAME = "final_eval.json", "partial_final_eval.json"
CEM_METHODS = ("cem_wm", "cem_wm_1step", "cem_pixel")
CKPTS = {"wm": "ckpt-wm", "wm_1step": "ckpt-wm-1step", "pixel": "ckpt-pixel", "bc": "ckpt-bc",
         "bc_clean": "ckpt-bc-clean"}
TAU_OF = {"cem_wm": "wm", "cem_pixel": "wm", "cem_wm_1step": "wm_1step"}   # tau per encoder


def read_bc_selected(ctx: JobContext) -> dict[str, str]:
    """bc_selected from the hash-matched results-plan-val (raises if missing)."""
    path, name, _, _ = pull_matching(ctx, "results-plan-val")
    doc = json.loads((path / "plan_val.json").read_text())
    sel = doc.get("bc_selected")
    if not sel or sel.get("variant") not in ("bc", "bc_clean"):
        raise RuntimeError(f"{name} has no bc_selected; run plan_val with both BC variants first")
    return {"variant": sel["variant"], "reason": sel["reason"], "source": f"{name} (git {doc.get('git')})"}


def sweep_pairs(arrays: dict[str, np.ndarray], offset: int, n: int, seed: int,
                pcfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """make_pairs' filter and seeded subset at another goal offset; all candidates if fewer than n."""
    cands = candidate_pairs(arrays, np.ones(len(arrays["episode_starts"]), bool), offset=offset,
                            start_step=int(pcfg["start_step"]), start_max=int(pcfg["start_max"]),
                            min_move_px=float(pcfg["min_move_px"]), min_turn_deg=float(pcfg["min_turn_deg"]))
    k = min(n, len(cands))
    pick = np.sort(np.random.default_rng(seed).choice(len(cands), k, replace=False))
    return [{"pair": j, **cands[i]} for j, i in enumerate(pick)], len(cands)


def gallery_png(rows: dict[str, list[np.ndarray]]) -> np.ndarray:
    """Labelled rows (one per episode) of start | goal | final [| imagined terminal]."""
    return multirow_filmstrip({k: np.stack(v) for k, v in rows.items()}, scale=3)


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    plan_cfg = load_yaml(REPO_ROOT / cfg["plan_config"])
    smoke, device = ctx.mode == "smoke", ctx.device
    sm = cfg["smoke"]
    seeds = [int(s) for s in (sm["seeds"] if smoke else cfg["seeds"])]
    pixel_seeds = [int(s) for s in cfg["pixel_seeds"]]
    budget = int(sm["budget"] if smoke else plan_cfg["budget"])
    resamples, bseed = int(sm["bootstrap"] if smoke else cfg["bootstrap"]), int(cfg["bootstrap_seed"])
    cem_cfg = CEMConfig.from_dict(plan_cfg["cem"])
    pixel_cfg = cem_cfg
    if smoke and ctx.platform == "local":
        pixel_cfg = dataclasses.replace(cem_cfg, **plan_cfg["smoke"]["local_pixel"])
    seed_all(seeds[0])

    bc_sel = read_bc_selected(ctx)
    bc_name = bc_sel["variant"]
    resolve = lambda m: bc_name if m == "bc_selected" else m                              # noqa: E731
    order = [resolve(m) for m in cfg["order"]]
    methods = [m for m in order if m != "sweep"]                                          # table order
    models, ckpts = {}, {}
    for key in ("wm", "wm_1step", "pixel", bc_name):
        models[key], ckpts[key] = load_matching(ctx, CKPTS[key])
    report.params = " / ".join(f"{k} {v['params']['total']:,}" for k, v in ckpts.items())
    pairs, pairs_info = load_pairs(ctx, "test")
    if smoke:
        pairs = pairs[:int(sm["pairs"])]
    test, data_name = load_npz(ctx, "pusht_test.npz")
    train, _ = load_npz(ctx)
    frames = test["frames"]
    env = make_env()
    env.reset(seed=seeds[0])
    align = render_alignment(env, pairs, frames)

    # ---- tau per encoder, on val frames (as plan_val), before any episode ----------------------
    _, val_eps, val_source = split_episodes(train["episode_seeds"], [190, 199])
    starts, lengths = train["episode_starts"][val_eps], train["episode_lengths"][val_eps]
    vidx = episode_frames(starts, lengths)
    ep_of = np.repeat(np.arange(len(starts)), lengths)
    max_pairs = int(sm["tau_max_pairs"] if smoke else plan_cfg["tau_max_pairs"])
    tau = {}
    for enc in ("wm", "wm_1step"):
        z = encode_frames(models[enc], torch.from_numpy(train["frames"][vidx]).to(device)).numpy()
        tau[enc] = {**calibrate_tau(z, train["states"][vidx], ep_of, float(plan_cfg["tau_percentile"]), max_pairs,
                                    int(plan_cfg["seed"])), "source": val_source}

    def make_policy(m: str, full_size: bool = False):
        if m == "cem_wm":
            return CEMPolicy(models["wm"], cem_cfg, device, "cem_wm")
        if m == "cem_wm_1step":
            return CEMPolicy(models["wm_1step"], cem_cfg, device, "cem_wm_1step")
        if m == "cem_pixel":
            return PixelCEMPolicy(models["pixel"], models["wm"], cem_cfg if full_size else pixel_cfg, device,
                                  amp=bool(plan_cfg["pixel_amp"]))
        if m in ("bc", "bc_clean"):
            scale = float(ckpts[m]["summary"].get("config", {}).get("delta_scale", 50.0))
            return BCPolicy(models[m], device, m, scale)
        if m == "random":
            return RandomPolicy(float(plan_cfg["random_sigma"]), cem_cfg.delta_clip)
        if m == "noop":
            return NoopPolicy()
        raise KeyError(m)

    def records_of(m: str, eps: list[dict[str, Any]], prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recs = episode_records(eps, prs)
        t = tau[TAU_OF[m]]["tau"] if m in TAU_OF else None
        for r, e, p in zip(recs, eps, prs):
            r.update(final_state=e["final_state"], move_px=p["move_px"], turn_deg=p["turn_deg"],
                     tag=failure_tag(e["success"], e["final_pos_px"], e["final_ang_deg"]),
                     believed=believed_success(e["terminal_costs"], t) if m in TAU_OF else False)
        return recs

    # ---- episodes, in crash-safe order; partial JSON after every (method, seed) ----------------
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    (art / "plots").mkdir(parents=True)
    (art / "gallery").mkdir()
    partial_path = ctx.out_dir / PARTIAL_NAME
    gallery_methods = [resolve(m) for m in cfg["gallery"]["methods"]]
    eps_all: dict[str, list[list[dict[str, Any]]]] = {}
    recs_all: dict[str, list[list[dict[str, Any]]]] = {}
    sweep: dict[str, Any] = {}
    pcfg = load_job_config(load_registry()["jobs"]["make_pairs"])
    units = sum(len(pixel_seeds if m == "cem_pixel" else seeds) for m in methods) + (1 if "sweep" in order else 0)
    done = 0

    def write_partial(note: str) -> None:
        nonlocal done
        done += 1
        partial_path.write_text(json.dumps({
            "job": ctx.name, "mode": ctx.mode, "git": ctx.git, "config_hash": ctx.config_hash,
            "bc_selected": bc_sel, "tau": tau, "budget": budget, "pairs": [p["pair"] for p in pairs],
            "completed": [[m, i] for m, v in recs_all.items() for i in range(len(v))],
            "records": recs_all, "sweep": sweep, "wall_clock_s": time.time() - t0}, indent=1))
        print(f"[final_eval] {done}/{units} done: {note} ({time.time() - t0:.0f}s elapsed)", flush=True)

    for m in order:
        if m == "sweep":
            scfg = cfg["sweep"]
            n_sw = int(sm["sweep_pairs"] if smoke else scfg["n_pairs"])
            s_seed = int(scfg["seed"])
            sweep = {"methods": [resolve(x) for x in scfg["methods"]], "seed": s_seed, "budget": budget,
                     "label": "exploratory", "offsets": []}
            for off in scfg["offsets"]:
                sp, n_cand = sweep_pairs(test, int(off), n_sw, s_seed, pcfg)
                entry = {"offset": int(off), "n": len(sp), "candidates": n_cand,
                         "median_move_px": float(np.median([p["move_px"] for p in sp])) if sp else None,
                         "median_abs_turn_deg": float(np.median([abs(p["turn_deg"]) for p in sp])) if sp else None,
                         "summaries": {}, "records": {}}
                for sm_name in sweep["methods"]:
                    e = run_method(env, make_policy(sm_name), sp, frames, budget, s_seed)
                    entry["summaries"][sm_name] = summarize(e, sp)
                    entry["records"][sm_name] = records_of(sm_name, e, sp)
                sweep["offsets"].append(entry)
            write_partial("goal-distance sweep")
            continue
        for s in (pixel_seeds if m == "cem_pixel" else seeds):
            keep_last = m == "cem_wm" and s == seeds[0] and m in gallery_methods
            eps = run_method(env, make_policy(m), pairs, frames, budget, s, keep_last_plan=keep_last)
            eps_all.setdefault(m, []).append(eps)
            recs_all.setdefault(m, []).append(records_of(m, eps, pairs))
            k = sum(e["success"] for e in eps)
            write_partial(f"{m} seed {s}: success {k}/{len(eps)}")

    # ---- summaries, tests, bins ----------------------------------------------------------------
    per_seed = {m: [summarize(e, pairs) for e in v] for m, v in eps_all.items()}
    pooled = {m: pooled_summary(v, pairs, resamples, bseed) for m, v in recs_all.items()}
    imagined = {m: imagined_counts(recs_all[m], tau[TAU_OF[m]]["tau"]) for m in methods if m in TAU_OF}
    comparisons = [[resolve(a), resolve(b)] for a, b in cfg["comparisons"]]
    tests = primary_tests(recs_all, pairs, comparisons, resamples, bseed, float(cfg["alpha"]))
    difficulty = {key: difficulty_table(pairs, recs_all, key, cfg["bins"][key], resamples, bseed)
                  for key in ("move_px", "turn_deg")}

    # ---- timing probe: full-size planning calls -------------------------------------------------
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    obs = reset_to_state(env, np.array(pairs[0]["start_state"], np.float32))
    hist = History(render64(obs["pixels"]), get_state(env)[:2])
    calls = int(sm["timing_calls"] if smoke else cfg["timing_calls"])
    timing = {m: {"call_s": timed_calls(make_policy(m, full_size=True), hist, frames[pairs[0]["goal_idx"]], calls,
                                        warmup=device.type == "cuda", sync=sync), "calls": calls}
              for m in methods if m in CEM_METHODS}

    # ---- gallery (seed 0, seeded picks per tag) ------------------------------------------------
    gseed, per_tag = int(cfg["gallery"]["seed"]), int(cfg["gallery"]["per_tag"])
    gallery, strips = {}, []
    for m in gallery_methods:
        eps0, recs0 = eps_all[m][0], recs_all[m][0]
        tags = [r["tag"] for r in recs0]
        gallery[m] = {}
        for tag in TAGS:
            pick = pick_gallery(tags, tag, per_tag, gseed)
            gallery[m][tag] = [pairs[i]["pair"] for i in pick]
            if not pick:
                continue
            rows = {}
            for i in pick:
                p, e = pairs[i], eps0[i]
                tiles = [frames[p["start_idx"]], frames[p["goal_idx"]], e["final_frame"]]
                if e.get("last_plan_z") is not None:
                    with torch.no_grad():
                        dec = models["wm"].decode(e["last_plan_z"][-1:].to(device)).float().clamp(-0.5, 0.5)
                    tiles.append(denormalize_image(dec.cpu().numpy())[0])
                rows[f"pair {p['pair']}"] = tiles
            strips.append(art / "gallery" / f"{m}_{tag}.png")
            imageio.imwrite(strips[-1], gallery_png(rows))

    # ---- outputs -------------------------------------------------------------------------------
    (art / "config.yaml").write_text(yaml.safe_dump({**cfg, "plan": plan_cfg}, sort_keys=False))
    plots = [rate_bars({m: {"success": pooled[m]["success"], "pos_success": pooled[m]["pos_success"]} for m in methods},
                       "Test success (pooled over seeds, 95% pair-clustered CI)", art / "plots" / "success.png")]
    for key in ("move_px", "turn_deg"):
        for metric in ("success", "pos_success"):
            plots.append(rate_by_bin(difficulty[key], metric, f"{metric} vs {key}",
                                     art / "plots" / f"{metric}_by_{key}.png"))
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    n_rot = sum(abs(p["turn_deg"]) >= 20.0 for p in pairs)
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "plan_config": plan_cfg, "inputs": inputs,
        "data": data_name, "checkpoints": ckpts,
        "pairs": {**pairs_info, "n": len(pairs), "rotation_pairs": n_rot, "translation_pairs": len(pairs) - n_rot},
        "budget": budget, "seeds": seeds, "pixel_seeds": pixel_seeds, "cem": dataclasses.asdict(cem_cfg),
        "cem_pixel_episode_config": dataclasses.asdict(pixel_cfg), "bc_selected": bc_sel,
        "methods": methods, "tags": list(TAGS), "alpha": float(cfg["alpha"]),
        "bootstrap": {"resamples": resamples, "seed": bseed, "unit": "pair"},
        "success_definition": "block CoG < 20 px AND |angle| < 20 deg, checked after every step (primary)",
        "pos_success_definition": "block CoG < 20 px after any step; on pairs whose start CoG error >= 20 px",
        "tag_definition": "from the final state: success | position_only (CoG < 20 px, angle >= 20 deg) | "
                          "angle_only | both_failed; believed = CEM terminal cost < tau (own encoder) at some step",
        "tau": tau, "per_seed": per_seed, "pooled": pooled, "imagined": imagined,
        "primary_tests": tests, "difficulty": difficulty, "timing": timing, "gallery": gallery, "sweep": sweep,
        "render_alignment": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in align.items()},
        "episodes": recs_all, "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=1))
    angle = None
    try:
        apath, _, _, _ = pull_matching(ctx, "results-imagined-angle")
        angle = json.loads((apath / "imagined_angle.json").read_text())
        shutil.copy2(apath / "imagined_angle.json", art / "imagined_angle.json")
        if (apath / "imagined_angle_vs_k.png").exists():
            shutil.copy2(apath / "imagined_angle_vs_k.png", art / "plots" / "imagined_angle_vs_k.png")
    except (RuntimeError, OSError) as e:
        print(f"[final_eval] imagined angle results unavailable: {str(e)[:200]}", flush=True)
    (art / "summary.md").write_text(write_summary(results, angle))
    if ctx.wandb_run is not None:
        import wandb
        ctx.wandb_run.log({f"final/{p.stem}": wandb.Image(str(p)) for p in plots + strips})

    # bundle: the final files at the top level + every earlier results file under all_results/
    bundle_dir = ctx.out_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir()
    bundle = bundle_dir / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(art.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(art).as_posix())
        res = REPO_ROOT / "results"
        for p in sorted(res.rglob("*")) if res.exists() else []:
            if p.is_file() and "final" not in p.relative_to(res).parts and p.suffix in (".json", ".png"):
                z.write(p, f"all_results/{p.relative_to(res).as_posix()}")
    shutil.copy2(bundle, ctx.out_dir / "results_bundle.zip")
    with zipfile.ZipFile(bundle) as z:
        bundle_files = z.namelist()

    # ---- CHECKS --------------------------------------------------------------------------------
    check_ckpts(report, ckpts, pairs_info)
    pf = cfg["plan_config"]
    report.check_true("plan_config_frozen", bool(plan_cfg.get("frozen")), f"frozen: {plan_cfg.get('frozen')}",
                      f"{pf} frozen: true")
    report.check_true("bc_selected", bc_name in models, f"{bc_name} ({bc_sel['reason']}; {bc_sel['source']})",
                      "read from results-plan-val, checkpoint hash-matched")
    n_exp = int(sm["pairs"]) if smoke else pairs_info["n_expected"]
    report.check_true("pairs", len(pairs) == n_exp, f"{len(pairs)} test pairs ({pairs_info['source']})", f"== {n_exp}")
    ran = all(all_ran({m: e}, len(pairs)) for m, vs in eps_all.items() for e in vs)
    report.check_true("all_methods_ran", ran and set(eps_all) == set(methods),
                      ", ".join(f"{m} {len(v)}x{len(v[0])}" for m, v in eps_all.items()),
                      f"every method x seed ran {len(pairs)} pairs")
    report.check_true("seeds", all(len(eps_all[m]) == len(pixel_seeds if m == "cem_pixel" else seeds) for m in methods),
                      " | ".join(f"{m} {len(eps_all[m])}" for m in methods),
                      f"{len(seeds)} seeds per method, cem_pixel {pixel_seeds}")
    for k, v in align.items():
        report.check_true(f"render_alignment_{k}", float(np.mean(v)) <= RENDER_TOL,
                          f"mean abs diff {np.mean(v):.3f} (max {np.max(v):.3f}) over {len(v)} pairs",
                          f"mean <= {RENDER_TOL} uint8 levels")
    if "noop" in pooled:
        k = pooled["noop"]["success"]["k"]
        report.check_true("noop_success_zero", k == 0, f"{k}/{pooled['noop']['success']['cells']}",
                          "== 0 (impossible by the pair filter)")
    ok = all_finite({"pooled": pooled, "tests": tests, "timing": timing, "difficulty": difficulty})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    report.check_true("bundle", RESULTS_NAME in bundle_files and "summary.md" in bundle_files,
                      f"{len(bundle_files)} files", "results_bundle.zip has final_eval.json and summary.md")
    if {"cem_wm", "random"} <= set(pooled):
        a, b = pooled["cem_wm"]["success"]["mean"], pooled["random"]["success"]["mean"]
        report.check("cem_wm_vs_random", "PASS" if a > b else "WARN", f"{a:.3f} vs {b:.3f} pooled success",
                     "cem_wm > random; WARN only")

    # ---- METRICS -------------------------------------------------------------------------------
    ckpt_metrics(report, ckpts)
    report.metric("data", data_name)
    report.metric("setup", f"{len(pairs)} test pairs ({pairs_info['artifact']}): turning {n_rot}, non-turning "
                           f"{len(pairs) - n_rot}; budget {budget}; seeds {seeds} (cem_pixel {pixel_seeds}); "
                           f"bc_selected {bc_name}; CEM N {cem_cfg.candidates}, H {cem_cfg.horizon}, rounds "
                           f"{cem_cfg.rounds} (frozen {pf})")
    fmt = lambda r: "n/a" if r["mean"] is None else f"{100 * r['mean']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"  # noqa: E731
    for m in methods:
        p = pooled[m]
        s0 = per_seed[m][0]["success"]
        report.metric(f"success_{m}", f"pooled {fmt(p['success'])} ({p['seeds']} seeds) | seed 0 {s0['k']}/{s0['n']} | "
                                      f"position-only pooled {fmt(p['pos_success'])}")
        g = p["groups"]
        report.metric(f"groups_{m}", f"turning ({g['rotation']['pairs']}) {fmt(g['rotation']['success'])} | non-turning "
                                     f"({g['translation']['pairs']}{', low power' if g['translation']['low_power'] else ''}) "
                                     f"{fmt(g['translation']['success'])}")
        reached = p["pos_reached_median_final_ang_deg"]
        reached_s = "n/a" if reached is None else f"{reached:.1f}"
        report.metric(f"final_err_{m}", f"median pos {p['median_final_pos_px']:.1f} px / angle "
                                        f"{p['median_final_ang_deg']:.1f} deg; angle once position reached {reached_s} "
                                        f"deg (n {p['pos_reached_n']}); {p['seconds_per_episode']:.2f} s/episode")
        report.metric(f"tags_{m}", " | ".join(f"{t} {p['tags'][t]}" for t in TAGS)
                      + (f" | believed-but-failed {imagined[m]['believed_success_failed']} of {imagined[m]['believed']} believed"
                         if m in imagined else ""))
    for t in tests:
        d = t["diff_bootstrap"]
        report.metric(f"test_{t['a']}_vs_{t['b']}_{t['metric']}",
                      f"discordant {t['a_only']}/{t['b_only']}, McNemar p {t['p']:.3g}, Holm p {t['p_holm']:.3g} -> "
                      f"{t['claim']}; diff {100 * d['mean']:+.1f} pp [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]")
    for enc, v in tau.items():
        report.metric(f"tau_{enc}", "none" if v["tau"] is None else f"{v['tau']:.3f} (p{v['percentile']:g} over "
                                                                   f"{v['n_success_pairs']} val success pairs)")
    for m, tt in timing.items():
        report.metric(f"timing_{m}", f"{tt['call_s']:.3f} s per full-size planning call (mean of {tt['calls']})")
    for o in sweep.get("offsets", []):
        report.metric(f"sweep_offset{o['offset']}", f"{o['n']} pairs (of {o['candidates']}; median move "
                                                    f"{o['median_move_px'] or 0:.1f} px, |turn| {o['median_abs_turn_deg'] or 0:.1f} deg): "
                      + " | ".join(f"{m} {s['success']['k']}/{s['success']['n']}" for m, s in o["summaries"].items()))
    report.metric("imagined_angle", "pending (no results-imagined-angle)" if angle is None
                  else f"verdict {angle['verdict']['verdict']} (from {angle['mode']} imagined_angle.json)")

    # ---- SAVED ---------------------------------------------------------------------------------
    report.save(json_path)
    report.save(art / "summary.md")
    report.save(art / "plots")
    report.save(art / "gallery")
    report.save(partial_path)
    report.save(ctx.out_dir / "results_bundle.zip")
    if ctx.mode == "full":
        dest = REPO_ROOT / "results" / "final"
        dest.mkdir(parents=True, exist_ok=True)
        for p in art.rglob("*"):
            if p.is_file() and p.name != "config.yaml":
                (dest / p.relative_to(art)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / p.relative_to(art))
        report.save(dest)
    meta = ctx.meta(complete=not report.failed, inputs=inputs)
    report.save(ctx.store.push(ARTIFACT, art, meta))
    report.save(ctx.store.push(BUNDLE, bundle_dir, meta))

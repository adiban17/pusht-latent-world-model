"""Job plan_cost_ablation (Phase 6b; pre-registered in PROGRESS.md): cem_wm with cost_space
"latent" (the frozen default) vs "pixel" on the val pairs, same seed, budget and CEM settings
(from the frozen configs/plan.yaml).

pixel cost: per-step distance = mean of w * (clip(D_wm(z_hat_k)) - goal)^2, w = 1 + 20 * (fg AND
NOT pusher) from the goal image only (wm.planning.goal_mask); same horizon weighting and beta.
Reports what plan_val reports per arm, plus the median final angle over episodes that reached
position (direct test of the angle-insensitivity hypothesis), imagined-vs-real with each arm's
own tau, paired discordant counts, the pusher-mask share, a debug PNG, the reproduction check
of the latent arm against plan_val's cem_wm, and the pre-registered decision line.
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

from wm.data.windows import episode_frames, frames_to_model, split_episodes
from wm.envs.pusht import denormalize_image, get_state, make_env, render64, reset_to_state
from wm.eval.diagnose import encode_frames
from wm.eval.planning import (calibrate_tau, calibrate_tau_pixel, compare_runs, select_cost_arm, summarize)
from wm.eval.rollouts import all_finite
from wm.jobs import JobContext
from wm.jobs._ckpt import load_matching, pull_matching
from wm.jobs._planning import (all_ran, check_ckpts, ckpt_metrics, episode_records, load_pairs, method_metrics,
                               run_method, timed_calls)
from wm.jobs._trainer import load_npz
from wm.planning.cem import CEMConfig
from wm.planning.goal_mask import goal_weight
from wm.planning.mpc import CEMPolicy, History
from wm.registry import input_hashes, load_registry, producer
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.report import Report
from wm.viz.media import upscale

ARTIFACT = "results-plan-cost-ablation"
RESULTS_NAME = "plan_cost_ablation.json"


@torch.no_grad()
def pixel_tau(model, frames_u8: np.ndarray, states: np.ndarray, episode_of: np.ndarray, device,
              percentile: float, max_pairs: int, seed: int, chunk: int = 512) -> tuple[dict[str, Any], float]:
    """τ_pixel over val frames (decoded o_i vs real o_j, weight from o_j); also the mean share of
    fg pixels the pusher mask removes on these frames."""
    x = frames_to_model(torch.from_numpy(frames_u8)).to(device)
    dec = torch.cat([model.decode(model.encode(x[i:i + chunk])).float().clamp(-0.5, 0.5)
                     for i in range(0, len(x), chunk)])
    ws, shares = [], []
    for f in frames_u8:
        w, info = goal_weight(model, f, device)
        ws.append(w)
        shares.append(info["removed_share"])
    w = torch.stack(ws)

    def distance(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        out = []
        for a in range(0, len(i), 4096):
            ii, jj = torch.as_tensor(i[a:a + 4096]), torch.as_tensor(j[a:a + 4096])
            out.append((w[jj] * (dec[ii] - x[jj]) ** 2).mean(dim=(1, 2, 3)).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    return calibrate_tau_pixel(distance, states, episode_of, percentile, max_pairs, seed), float(np.mean(shares))


def debug_png(model, ep: dict[str, Any], goal_u8: np.ndarray, device) -> np.ndarray:
    """goal frame | weight mask (white x21, grey = pusher pixels removed, black x1) | decoded
    predicted terminal frame of the first planning step; 4x upscaled."""
    _, info = goal_weight(model, goal_u8, device)
    mask = np.zeros((64, 64, 3), np.uint8)
    mask[info["fg"] & ~info["pusher"]] = 255
    mask[info["fg"] & info["pusher"]] = 128
    with torch.no_grad():
        dec = model.decode(ep["plans"][0][-1:].to(device)).float().clamp(-0.5, 0.5).cpu().numpy()
    pred = denormalize_image(dec)[0]
    gap = np.full((64, 2, 3), 255, np.uint8)
    return upscale(np.concatenate([goal_u8, gap, mask, gap, pred], axis=1)[None], 4)[0]


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    smoke, device = ctx.mode == "smoke", ctx.device
    plan_cfg = load_yaml(REPO_ROOT / cfg["plan_config"])
    seed = int(plan_cfg["seed"])
    seed_all(seed)
    budget = int(cfg["smoke"]["budget"] if smoke else plan_cfg["budget"])
    model, info = load_matching(ctx, "ckpt-wm")
    ckpts = {"wm": info}
    report.params = f"wm {info['params']['total']:,}"
    pairs, pairs_info = load_pairs(ctx, "val")
    if smoke:
        pairs = pairs[:int(cfg["smoke"]["pairs"])]
    arrays, data_name = load_npz(ctx)
    frames = arrays["frames"]
    env = make_env()
    env.reset(seed=seed)

    arms = {name: CEMConfig.from_dict({**plan_cfg["cem"], "cost_space": space}) for name, space in cfg["arms"].items()}
    episodes = {}
    for name, c in arms.items():
        episodes[name] = run_method(env, CEMPolicy(model, c, device, name), pairs, frames, budget, seed,
                                    keep=1 if c.cost_space == "pixel" else 0)
    summaries = {m: summarize(e, pairs) for m, e in episodes.items()}
    latent_arm = next(n for n, c in arms.items() if c.cost_space == "latent")
    pixel_arm = next(n for n, c in arms.items() if c.cost_space == "pixel")
    joins, decision = select_cost_arm(summaries[latent_arm], summaries[pixel_arm], int(cfg["margin"]))
    records = {m: episode_records(e, pairs) for m, e in episodes.items()}
    setup = {"budget": budget, "seed": seed}
    discordant = compare_runs(records[pixel_arm], records[latent_arm], setup, setup)

    # ---- reproduction: latent arm vs plan_val's cem_wm (WARN only) ---------------------------
    try:
        ref_path, ref_name, _, _ = pull_matching(ctx, cfg["reference"])
        ref = json.loads((ref_path / "plan_val.json").read_text())
        repro = compare_runs(records[latent_arm], ref["episodes"]["cem_wm"][:len(pairs)], setup,
                             {"budget": ref["budget"], "seed": ref["seed"]})
        if repro["comparable"] and len(ref["episodes"]["cem_wm"]) != len(pairs):
            repro = {"comparable": False, "reason": f"plan_val ran {len(ref['episodes']['cem_wm'])} pairs"}
        repro["reference"] = f"{ref_name} (git {ref.get('git')})"
    except RuntimeError as e:
        repro = {"comparable": False, "reason": f"no matching {cfg['reference']}: {e}"}

    # ---- timing (full-size CEM calls on the first pair's start) ------------------------------
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    obs = reset_to_state(env, np.array(pairs[0]["start_state"], np.float32))
    hist = History(render64(obs["pixels"]), get_state(env)[:2])
    calls = int(cfg["smoke"]["timing_calls"] if smoke else cfg["timing_calls"])
    timing = {n: {"call_s": timed_calls(CEMPolicy(model, c, device, n), hist, frames[pairs[0]["goal_idx"]], calls,
                                        warmup=device.type == "cuda", sync=sync), "calls": calls}
              for n, c in arms.items()}

    # ---- tau (latent) and tau_pixel on val frames --------------------------------------------
    _, val_eps, val_source = split_episodes(arrays["episode_seeds"], [190, 199])
    starts, lengths = arrays["episode_starts"][val_eps], arrays["episode_lengths"][val_eps]
    vidx = episode_frames(starts, lengths)
    ep_of = np.repeat(np.arange(len(starts)), lengths)
    max_pairs = int(cfg["smoke"]["tau_max_pairs"] if smoke else cfg["tau_max_pairs"])
    z_val = encode_frames(model, torch.from_numpy(frames[vidx]).to(device)).numpy()
    taus = {"latent": calibrate_tau(z_val, arrays["states"][vidx], ep_of, float(cfg["tau_percentile"]), max_pairs, seed)}
    taus["pixel"], val_removed = pixel_tau(model, frames[vidx], arrays["states"][vidx], ep_of, device,
                                           float(cfg["tau_percentile"]), max_pairs, seed)
    imagined = {}
    for n, c in arms.items():
        tau = taus[c.cost_space]["tau"]
        if tau is None:
            continue
        hit = [bool(e["terminal_costs"]) and min(e["terminal_costs"]) < tau for e in episodes[n]]
        imagined[n] = {"imagined_success": int(sum(hit)), "n": len(hit), "tau": tau,
                       "imagined_success_real_fail": int(sum(h and not e["success"] for h, e in zip(hit, episodes[n])))}
    goal_removed = [goal_weight(model, frames[p["goal_idx"]], device)[1]["removed_share"] for p in pairs]

    # ---- outputs -----------------------------------------------------------------------------
    art = ctx.out_dir / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    png = art / "goal_mask_debug.png"
    ep0 = episodes[pixel_arm][0]
    if ep0.get("plans"):
        imageio.imwrite(png, debug_png(model, ep0, frames[pairs[0]["goal_idx"]], device))
    inputs = input_hashes(load_registry()["jobs"][ctx.name])
    results = {
        "job": ctx.name, "mode": ctx.mode, "platform": ctx.platform, "git": ctx.git,
        "config_hash": ctx.config_hash, "config": cfg, "inputs": inputs, "data": data_name,
        "plan_config": {"path": cfg["plan_config"], "frozen": plan_cfg.get("frozen"),
                        "plan_val_config_hash": producer("results-plan-val")[1]},
        "checkpoints": ckpts, "pairs": {**pairs_info, "n": len(pairs)}, "budget": budget, "seed": seed,
        "arms": {n: dataclasses.asdict(c) for n, c in arms.items()},
        "summaries": summaries, "timing": timing, "tau": taus, "imagined_success": imagined,
        "pusher_mask_removed_share": {"goal_frames_mean": float(np.mean(goal_removed)), "val_frames_mean": val_removed},
        "discordant": {"pixel_only": discordant.get("only_ours"), "latent_only": discordant.get("only_ref"),
                       "agree": discordant.get("agree")},
        "reproduction": repro,
        "decision": {"pixel_joins_phase7": joins, "reason": decision,
                     "rule": f"pixel joins Phase 7 as 'post-hoc diagnostic, chosen on val' only with >= "
                             f"{cfg['margin']} more successes than latent; the frozen latent default stays the headline"},
        "episodes": records, "wall_clock_s": time.time() - t0,
    }
    json_path = art / RESULTS_NAME
    json_path.write_text(json.dumps(results, indent=2))
    if ctx.wandb_run is not None and png.exists():
        import wandb
        ctx.wandb_run.log({"plan_cost_ablation/goal_mask_debug": wandb.Image(str(png))})

    # ---- CHECKS ------------------------------------------------------------------------------
    check_ckpts(report, ckpts, pairs_info)
    report.check_true("plan_config_frozen", plan_cfg.get("frozen") is True, f"frozen: {plan_cfg.get('frozen')}",
                      f"{cfg['plan_config']} frozen: true")
    report.check_true("fg_mean_present", getattr(model, "fg_mean", None) is not None,
                      "ckpt-wm has fg_mean" if getattr(model, "fg_mean", None) is not None else "missing",
                      "pixel cost needs the fg_weight mean frame")
    n_exp = int(cfg["smoke"]["pairs"]) if smoke else pairs_info["n_expected"]
    report.check_true("pairs", len(pairs) == n_exp, f"{len(pairs)} val pairs ({pairs_info['source']})", f"== {n_exp}")
    report.check_true("all_arms_ran", all_ran(episodes, len(pairs)),
                      ", ".join(f"{m} {len(e)}" for m, e in episodes.items()), f"{len(pairs)} episodes each")
    ok = all_finite({"summaries": summaries, "timing": timing})
    report.check_true("metrics_finite", ok, "all finite" if ok else "NaN/inf", "no NaN/inf")
    if repro["comparable"]:
        same = repro["agree"] == repro["n"] and repro["ours_success"] == repro["ref_success"]
        report.check("latent_arm_reproduces_plan_val", "PASS" if same else "WARN",
                     f"{repro['ours_success']}/{repro['n']} vs plan_val {repro['ref_success']}/{repro['n']}; "
                     f"{repro['agree']}/{repro['n']} pairs agree", "identical per-pair outcomes; WARN only")
    else:
        report.check("latent_arm_reproduces_plan_val", "WARN", f"not comparable: {repro['reason']}",
                     "identical per-pair outcomes; WARN only")

    # ---- METRICS -----------------------------------------------------------------------------
    ckpt_metrics(report, ckpts)
    report.metric("data", data_name)
    c0 = arms[latent_arm]
    report.metric("setup", f"{len(pairs)} val pairs ({pairs_info['source']}); budget {budget}; seed {seed}; CEM N "
                           f"{c0.candidates}, H {c0.horizon}, rounds {c0.rounds}, K {c0.elites}, beta {c0.beta:g} "
                           f"(frozen {cfg['plan_config']}); fp32")
    for n, s in summaries.items():
        method_metrics(report, n, s)
        med, nr = s["pos_reached_median_final_ang_deg"], s["pos_reached_n"]
        medf, nf = s["pos_reached_failed_median_final_ang_deg"], s["pos_reached_failed_n"]
        report.metric(f"angle_when_position_reached_{n}",
                      f"median final angle {'n/a' if med is None else f'{med:.1f}'} deg over {nr} position-reached "
                      f"episodes; {'n/a' if medf is None else f'{medf:.1f}'} deg over the {nf} of them that failed "
                      f"(direct test of the angle-insensitivity hypothesis)")
        report.metric(f"timing_{n}", f"{timing[n]['call_s']:.3f} s per full-size planning call (mean of {calls})")
        if n in imagined:
            v = imagined[n]
            report.metric(f"imagined_success_{n}", f"{v['imagined_success']}/{v['n']} episodes reached terminal "
                                                   f"distance < tau ({v['tau']:.4g}); {v['imagined_success_real_fail']} "
                                                   f"of them failed for real")
    for space, t in taus.items():
        report.metric(f"tau_{space}", "none" if t["tau"] is None else
                      f"{t['tau']:.5g} = p{t['percentile']:g} over {t['n_success_pairs']} same-episode val success "
                      f"pairs (median {t['median_success']:.4g}; non-success median {t['median_non_success']:.4g}) "
                      f"[{val_source}]")
    report.metric("pusher_mask_removed", f"mean {100 * np.mean(goal_removed):.1f}% of fg-mask pixels removed over "
                                         f"{len(pairs)} goal frames ({100 * val_removed:.1f}% over the val frames)")
    report.metric("discordant", f"pixel-only successes {discordant.get('only_ours')}, latent-only "
                                f"{discordant.get('only_ref')}, same outcome {discordant.get('agree')}/{len(pairs)}")
    report.metric("reproduction", (f"latent arm {repro['ours_success']}/{repro['n']} vs {repro['reference']} cem_wm "
                                   f"{repro['ref_success']}/{repro['n']}; agree {repro['agree']}/{repro['n']} "
                                   f"(only ours {repro['only_ours']}, only plan_val {repro['only_ref']})")
                  if repro["comparable"] else f"not comparable: {repro['reason']}")
    report.metric("cost_ablation_decision", decision)

    report.save(json_path)
    if png.exists():
        report.save(png)
    if ctx.mode == "full":
        repo_json = REPO_ROOT / "results" / RESULTS_NAME
        repo_json.parent.mkdir(exist_ok=True)
        shutil.copy2(json_path, repo_json)
        report.save(repo_json)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed, inputs=inputs)))

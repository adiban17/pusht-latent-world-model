"""Shared plumbing for the planning jobs (plan_val, plan_sweep; final_eval in Phase 7)."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
import torch

from wm.eval.planning import is_rotation_pair, rate
from wm.jobs import JobContext
from wm.jobs._ckpt import pull_matching
from wm.planning.mpc import Policy, run_episode
from wm.registry import load_job_config, load_registry
from wm.report import Report

PAIR_FILES = {"val": "eval_pairs_val.json", "test": "eval_pairs_test.json"}
SEED_STRIDE = 100003   # episode generator seed = planner seed * SEED_STRIDE + pair index


def load_pairs(ctx: JobContext, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pairs of `split` from the hash-matched `eval-pairs`, and info (artifact, hashes, expected n)."""
    path, name, meta, expected = pull_matching(ctx, "eval-pairs")
    doc = json.loads((path / PAIR_FILES[split]).read_text())
    pcfg = load_job_config(load_registry()["jobs"]["make_pairs"])
    n_expected = int((pcfg["smoke"] if ctx.mode == "smoke" else pcfg)[f"n_{split}"])
    info = {"artifact": name, "config_hash": meta.get("config_hash"), "expected_config_hash": expected,
            "source": doc["source"], "candidates": doc["candidates"], "n_expected": n_expected}
    return doc["pairs"], info


def episode_gen(seed: int, i: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * SEED_STRIDE + i)


def run_method(env, policy: Policy, pairs: list[dict[str, Any]], frames: np.ndarray, budget: int,
               seed: int, keep: int = 0) -> list[dict[str, Any]]:
    """Run one policy on every pair (in order); the first `keep` episodes keep frames and plans."""
    t0, out = time.time(), []
    for i, p in enumerate(pairs):
        out.append(run_episode(env, np.array(p["start_state"], np.float32), frames[p["goal_idx"]],
                               np.array(p["goal_state"], np.float32), policy, budget, episode_gen(seed, i),
                               keep_frames=i < keep, keep_plans=i < keep))
    k = sum(e["success"] for e in out)
    kp = sum(e["pos_success"] for e in out)
    print(f"{policy.name}: success {k}/{len(out)}, position-only {kp}/{len(out)} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return out


def episode_records(episodes: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-pair JSON records (no frames / plan latents)."""
    keys = ("success", "pos_success", "steps", "start_pos_px", "start_ang_deg", "final_pos_px",
            "final_ang_deg", "min_pos_px", "seconds")
    return [{"pair": p["pair"], "episode_seed": p["episode_seed"], "t": p["t"],
             "rotation_pair": is_rotation_pair(p), **{k: e[k] for k in keys},
             "costs": [round(c, 4) for c in e["costs"]],
             "terminal_costs": [round(c, 4) for c in e["terminal_costs"]]}
            for e, p in zip(episodes, pairs)]


def fmt_rate(r: dict[str, Any]) -> str:
    return f"{r['k']}/{r['n']} = {100 * r['rate']:.1f}% [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]" \
        if r["n"] else "0/0"


def check_ckpts(report: Report, ckpts: dict[str, dict[str, Any]], pairs_info: dict[str, Any]) -> None:
    for key, c in ckpts.items():
        report.check_true(f"ckpt_config_hash_{key}", c["config_hash"] == c["expected_config_hash"],
                          f"{c['artifact']} step {c['step']}: {c['config_hash']}",
                          f"== {c['expected_config_hash']}, current config; complete: true")
    report.check_true("eval_pairs_hash", pairs_info["config_hash"] == pairs_info["expected_config_hash"],
                      f"{pairs_info['artifact']}: {pairs_info['config_hash']}",
                      f"== {pairs_info['expected_config_hash']} (make_pairs current config)")


def ckpt_metrics(report: Report, ckpts: dict[str, dict[str, Any]]) -> None:
    for key, c in ckpts.items():
        extra = f", proprio {c['proprio']}" if c.get("proprio") is not None else ""
        report.metric(f"ckpt_{key}", f"{c['artifact']} step {c['step']}, git {c['git']}, "
                                     f"config_hash {c['config_hash']}{extra}")


def method_metrics(report: Report, name: str, s: dict[str, Any]) -> None:
    """REPORT lines for one method's (or setting's) summary."""
    report.metric(f"success_{name}", fmt_rate(s["success"]))
    report.metric(f"pos_success_{name}", f"{fmt_rate(s['pos_success'])} (start >= 20 px) | all pairs "
                                         f"{s['pos_success_all']['k']}/{s['pos_success_all']['n']}")
    g = s["groups"]
    report.metric(f"groups_{name}", " | ".join(
        f"{k} ({v['pairs']} pairs): success {v['success']['k']}/{v['success']['n']}, position-only "
        f"{v['pos_success']['k']}/{v['pos_success']['n']}" for k, v in g.items()))
    steps = s["mean_steps_to_success"]
    report.metric(f"final_err_{name}", f"median pos {s['median_final_pos_px']:.1f} px / angle "
                                       f"{s['median_final_ang_deg']:.1f} deg; mean steps-to-success "
                                       f"{'n/a' if steps is None else f'{steps:.1f}'}; "
                                       f"{s['seconds_per_episode']:.2f} s/episode")


def all_ran(episodes: dict[str, list], n: int) -> bool:
    return all(len(v) == n for v in episodes.values())


def timed_calls(policy: Policy, hist, goal_frame: np.ndarray, calls: int, warmup: bool = True,
                sync: Callable[[], None] = lambda: None) -> float:
    """Mean seconds per policy.act on a fixed history (after one warm-up call if `warmup`)."""
    policy.reset(goal_frame, episode_gen(0, 0))
    if warmup:
        policy.act(hist)
    sync()
    t0 = time.perf_counter()
    for _ in range(calls):
        policy.act(hist)
    sync()
    return (time.perf_counter() - t0) / calls

"""Job collect_data (Phase 2): expert episodes -> pusht_train.npz / pusht_test.npz + health report.

Full: 200 train episodes (seeds 0-199, val = 190-199) and 30 test episodes (seeds 1000-1029),
100 actions and 101 frames each. Smoke: 2 train + 1 test episode.
"""
from __future__ import annotations

import json
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

from wm.data.collect import collect_split, health, seed_list, transitions
from wm.jobs import JobContext
from wm.report import Report, format_duration
from wm.repro import seed_all
from wm.viz.media import filmstrip, save_gif, save_histograms

CONTACT_PASS, CONTACT_WARN = 0.35, 0.25
TOTAL_FULL_EPISODES = 230


def contact_status(ratio: float) -> str:
    """PASS >= 35%, WARN 25-35%, FAIL < 25%."""
    if ratio >= CONTACT_PASS:
        return "PASS"
    return "WARN" if ratio >= CONTACT_WARN else "FAIL"


def split_seeds(cfg: dict[str, Any], mode: str) -> tuple[list[int], list[int]]:
    """(train seeds, test seeds) for this mode."""
    src = cfg["smoke"] if mode == "smoke" else cfg
    return seed_list(src["train_seeds"]), seed_list(src["test_seeds"])


def save_media(ctx: JobContext, report: Report, arrays: dict, tag: str, n: int, seed: int) -> None:
    """GIF + filmstrip for n randomly chosen episodes of `arrays`."""
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(arrays["episode_lengths"]), size=n, replace=False))
    for e in picks:
        start, length = arrays["episode_starts"][e], arrays["episode_lengths"][e]
        frames = arrays["frames"][start:start + length]
        name = f"{tag}_seed{int(arrays['episode_seeds'][e]):04d}"
        report.save(save_gif(frames, ctx.out_dir / f"episode_{name}.gif"))
        path = ctx.out_dir / f"filmstrip_{name}.png"
        imageio.imwrite(path, filmstrip(frames))
        report.save(path)


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = ctx.config
    seed_all(0)
    steps, expert = cfg["steps_per_episode"], cfg["expert"]
    train_seeds, test_seeds = split_seeds(cfg, ctx.mode)
    val_seeds = set(seed_list(cfg["val_seeds"]))

    train, train_info = collect_split(train_seeds, expert, steps, cfg["replacement_seed_start"])
    replacement_after = max([cfg["replacement_seed_start"],
                             *[r["replacement"] + 1 for r in train_info["replaced"]]])
    test, test_info = collect_split(test_seeds, expert, steps, replacement_after)
    n_eps = len(train_seeds) + len(test_seeds)
    sec_per_ep = (train_info["seconds"] + test_info["seconds"]) / n_eps

    for name, arrays in (("pusht_train.npz", train), ("pusht_test.npz", test)):
        np.savez_compressed(ctx.out_dir / name, **arrays)
        report.save(ctx.out_dir / name)

    # ---- checks ----
    want_train, want_test = len(train_seeds) * steps, len(test_seeds) * steps
    report.check_true("train_transitions", transitions(train) == want_train, transitions(train),
                      f"== {len(train_seeds)} episodes x {steps}")
    report.check_true("test_transitions", transitions(test) == want_test, transitions(test),
                      f"== {len(test_seeds)} episodes x {steps}")
    lengths = np.concatenate([train["episode_lengths"], test["episode_lengths"]])
    report.check_true("frames_per_episode", bool((lengths == steps + 1).all()),
                      f"{int(lengths.min())}..{int(lengths.max())}", f"all == {steps + 1}, no early end")
    lo = float(min(train["actions"].min(), test["actions"].min()))
    hi = float(max(train["actions"].max(), test["actions"].max()))
    report.check_true("actions_in_bounds", lo >= 0.0 and hi <= 512.0, f"[{lo:.1f}, {hi:.1f}]",
                      "[0, 512]")
    h_train, h_test = health(train), health(test)
    ratio = h_train["contact_ratio"]
    report.check("contact_ratio", contact_status(ratio), f"{ratio:.1%}",
                 ">= 35% PASS, 25-35% WARN, < 25% FAIL")

    # ---- health file + media ----
    n_val = int(sum(int(s) in val_seeds for s in train["episode_seeds"]))
    replaced = train_info["replaced"] + test_info["replaced"]
    env_success = train_info["terminated_episodes"] + test_info["terminated_episodes"]
    health_doc = {
        "train": h_train, "test": h_test, "val_episodes": n_val,
        "env_success_episodes": env_success, "replaced_episodes": replaced,
        "sec_per_episode": sec_per_ep, "mode": ctx.mode, "config": cfg,
        "config_hash": ctx.config_hash, "git": ctx.git, "wall_clock_s": time.time() - t0,
    }
    (ctx.out_dir / "data_health.json").write_text(json.dumps(health_doc, indent=2))
    (ctx.out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    report.save(ctx.out_dir / "data_health.json")
    report.save(save_histograms(train, ctx.out_dir / "histograms.png"))
    n_media = cfg["media_episodes"]
    if len(train_seeds) >= n_media:
        save_media(ctx, report, train, "train", n_media, cfg["media_seed"])
    else:  # smoke: 2 train + 1 test episode
        save_media(ctx, report, train, "train", len(train_seeds), cfg["media_seed"])
        save_media(ctx, report, test, "test", n_media - len(train_seeds), cfg["media_seed"])

    # ---- metrics ----
    report.metric("train_episodes", len(train_seeds))
    report.metric("val_episodes", n_val)
    report.metric("test_episodes", len(test_seeds))
    report.metric("contact_ratio_train", f"{h_train['contact_ratio']:.3f}")
    report.metric("contact_ratio_test", f"{h_test['contact_ratio']:.3f}")
    report.metric("coverage_8x8", f"{h_train['coverage_8x8']:.3f}")
    report.metric("block_angle_std_deg", f"{h_train['block_angle_std_deg']:.1f}")
    report.metric("block_step_px_mean", f"{h_train['block_step_px_mean']:.2f}")
    report.metric("action_mean", h_train["action_mean"])
    report.metric("action_std", h_train["action_std"])
    report.metric("action_clip_frac", f"{h_train['action_clip_frac']:.3f}")
    report.metric("noisy_clean_gap_px", f"{h_train['noisy_clean_gap_px']:.1f}")
    report.metric("env_success_episodes", env_success)
    report.metric("replaced_episodes", len(replaced))
    report.metric("sec_per_episode", f"{sec_per_ep:.2f}")
    report.metric("projected_full_time", format_duration(sec_per_ep * TOTAL_FULL_EPISODES))

    saved = ctx.store.push("pusht-data", ctx.out_dir, ctx.meta(complete=not report.failed))
    report.save(saved)

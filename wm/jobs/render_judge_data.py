"""Job render_judge_data (Phase 5b): rendered micro-episodes for the judges -> store `judge-data`.

judge_train.npz / judge_val.npz in the pusht_*.npz layout (see wm/data/judge_render.py), plus a
filmstrip of a few micro-episodes and judge_data.json (counts, rejection stats, coverage).
"""
from __future__ import annotations

import json
import time

import imageio.v2 as imageio
import numpy as np

from wm.data.judge_render import render_split
from wm.envs.pusht import block_cog, circular_std_deg, make_env
from wm.jobs import JobContext
from wm.report import Report
from wm.repro import seed_all
from wm.viz.media import filmstrip

ARTIFACT = "judge-data"


def run(ctx: JobContext, report: Report) -> None:
    t0 = time.time()
    cfg = dict(ctx.config)
    n_micro = int(cfg["smoke"]["micro_episodes"] if ctx.mode == "smoke" else cfg["micro_episodes"])
    seed_all(int(cfg["seed_start"]))
    seeds = list(range(int(cfg["seed_start"]), int(cfg["seed_start"]) + n_micro))
    n_val = max(1, int(round(float(cfg["val_frac"]) * n_micro)))
    env = make_env()
    env.reset(seed=int(cfg["seed_start"]))

    art = ctx.out_dir / "artifact"
    art.mkdir(parents=True, exist_ok=True)
    stats = {}
    splits = {}
    for split, ss in (("val", seeds[:n_val]), ("train", seeds[n_val:])):
        arrays, stats[split] = render_split(env, ss, cfg)
        np.savez_compressed(art / f"judge_{split}.npz", **arrays)
        splits[split] = arrays

    tr = splits["train"]
    cog = block_cog(tr["states"].astype(np.float64))
    cells = np.clip(((cog - 0.0) / 512.0 * 8).astype(int), 0, 7)
    coverage = len({(int(a), int(b)) for a, b in cells}) / 64.0
    ang_std, ang_r = circular_std_deg(tr["states"][:, 4])
    frames_per = int(cfg["steps"])
    strip = np.concatenate([filmstrip(tr["frames"][i * frames_per:(i + 1) * frames_per], n=frames_per, scale=3)
                            for i in range(min(4, stats["train"]["episodes"]))], 0)
    strip_path = art / "judge_micro_episodes.png"
    imageio.imwrite(strip_path, strip)
    summary = {"config": cfg, "stats": stats, "block_cog_coverage_8x8": coverage,
               "block_angle_circular_std_deg": ang_std, "block_angle_resultant_R": ang_r,
               "seconds": time.time() - t0}
    (art / "judge_data.json").write_text(json.dumps(summary, indent=2))

    exp_frames = {s: stats[s]["episodes"] * frames_per for s in stats}
    total = sum(stats[s]["frames"] for s in stats)
    report.check_true("frame_counts", all(stats[s]["frames"] == exp_frames[s] for s in stats),
                      f"train {stats['train']['frames']:,} / val {stats['val']['frames']:,} frames "
                      f"({stats['train']['episodes']:,} / {stats['val']['episodes']:,} micro-episodes)",
                      f"== {frames_per} x micro-episodes; {n_micro * frames_per:,} total")
    report.check_true("angles_uniform", ang_r < 0.1 or ctx.mode == "smoke",
                      f"resultant R {ang_r:.3f} (circular std {ang_std:.0f} deg)", "R < 0.1 (full only)")
    report.metric("frames_total", f"{total:,}")
    report.metric("rejection", f"mean {stats['train']['mean_tries']:.2f} draws per valid start "
                               f"(max {max(stats[s]['max_tries'] for s in stats)})")
    report.metric("block_cog_coverage_8x8", f"{coverage:.3f}")
    report.metric("seeds", f"{seeds[0]}-{seeds[-1]} (val = first {n_val})")
    report.metric("seconds", f"{time.time() - t0:.1f}")
    for p in (art / "judge_train.npz", art / "judge_val.npz", strip_path, art / "judge_data.json"):
        report.save(p)
    report.save(ctx.store.push(ARTIFACT, art, ctx.meta(complete=not report.failed)))

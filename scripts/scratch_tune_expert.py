"""TOOL, not part of the pipeline: quick expert-tuning loop for Phase 2.

Collects N episodes in memory (no npz, no store) with the expert settings from
configs/data.yaml, optionally overridden, and prints the data-health numbers.

    .venv/bin/python scripts/scratch_tune_expert.py --episodes 20 --set push_step=25
    .venv/bin/python scripts/scratch_tune_expert.py --strips /some/dir   # also save filmstrips
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import yaml

from wm.data.collect import collect_split, health
from wm.repro import REPO_ROOT, load_yaml
from wm.viz.media import filmstrip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="expert overrides, key=value (YAML value)")
    ap.add_argument("--strips", type=Path, default=None, help="save a filmstrip per episode here")
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    expert = dict(cfg["expert"])
    for kv in args.set:
        key, value = kv.split("=", 1)
        expert[key] = yaml.safe_load(value)
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    arrays, info = collect_split(seeds, expert, cfg["steps_per_episode"])
    print(json.dumps({**health(arrays), "env_success_episodes": info["terminated_episodes"],
                      "seconds": round(info["seconds"], 1), "overrides": args.set}, indent=1))
    if args.strips:
        args.strips.mkdir(parents=True, exist_ok=True)
        for s, start, n in zip(arrays["episode_seeds"], arrays["episode_starts"],
                               arrays["episode_lengths"]):
            imageio.imwrite(args.strips / f"seed{s:04d}.png",
                            filmstrip(arrays["frames"][start:start + n]))


if __name__ == "__main__":
    main()

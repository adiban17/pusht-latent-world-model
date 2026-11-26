"""The only entry point: python run.py --job <name> --mode smoke|full [--force]"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame; must precede any pygame import

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import Any

from wm.jobs import JobContext
from wm.platform import detect_platform, device_name, get_device, out_dir
from wm.report import Report
from wm.repro import REPO_ROOT, config_hash, git_info, load_yaml
from wm.store import WANDB_PROJECT, make_store

LOCAL_FULL_ALLOWED = {"make_report"}


def load_registry(path: Path = REPO_ROOT / "jobs.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_job_config(entry: dict[str, Any]) -> dict[str, Any]:
    """The job's YAML config, or {} if it has none."""
    cfg = entry.get("config")
    if not cfg:
        return {}
    path = REPO_ROOT / cfg
    return load_yaml(path) if path.exists() else {}


def job_config_hash(name: str, entry: dict[str, Any], cfg: dict[str, Any]) -> str:
    """config_hash = hash of the job name, its jobs.yaml entry and its config file."""
    return config_hash({"job": name, "entry": entry, "config": cfg})


def is_finished(store, entry: dict[str, Any], chash: str) -> bool:
    """True if every artifact the job makes exists, is complete and has the same config_hash."""
    makes = entry.get("makes") or []
    if not makes:
        return False
    for name in makes:
        meta = store.own_metadata(name)
        if not meta or not meta.get("complete") or meta.get("config_hash") != chash:
            return False
    return True


def expand_group(name: str, groups: dict[str, Any]) -> list[str]:
    """The jobs of a group in order; a member that is itself a group is expanded in place."""
    out: list[str] = []
    for member in groups[name]["jobs"]:
        out += expand_group(member, groups) if member in groups else [member]
    return out


def run_job(name: str, mode: str, force: bool, registry: dict[str, Any]) -> int:
    """Run one job; returns its exit code."""
    entry = registry["jobs"][name]
    platform = detect_platform()
    if mode == "full" and platform == "local" and name not in LOCAL_FULL_ALLOWED:
        print(f"REFUSED: '{name}' --mode full does not run on the local Mac (CPU, smoke only). "
              "Run it on Kaggle/Colab with the runner notebook.")
        return 2

    try:
        module = importlib.import_module(entry["module"])
    except ModuleNotFoundError as e:
        if e.name == entry["module"]:
            print(f"job '{name}' is not built yet (Phase {entry['phase']})")
            return 2
        raise

    cfg = load_job_config(entry)
    chash = job_config_hash(name, entry, cfg)
    store = make_store(platform, mode)
    if not force and is_finished(store, entry, chash):
        print(f"{name}: already finished, skipping (use --force to rerun)")
        return 0

    wandb_run = None
    if platform != "local":
        import wandb

        wandb_run = wandb.init(project=WANDB_PROJECT, job_type=name, name=f"{name}-{mode}",
                               config={"mode": mode, "config_hash": chash, **cfg})
        store.backend.run = wandb_run

    device = get_device()
    git = git_info()
    ctx = JobContext(name=name, mode=mode, platform=platform, device=device,
                     out_dir=out_dir(platform) / name, store=store, config=cfg,
                     config_hash=chash, git=git, wandb_run=wandb_run)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    report = Report(job=name, mode=mode, platform=platform, device=device_name(device), git=git,
                    wandb=wandb_run.url if wandb_run is not None else "off")
    try:
        module.run(ctx, report)
    except Exception as e:  # still print a REPORT so the failure gets pasted back
        traceback.print_exc()
        report.check("job_ran", "FAIL", f"{type(e).__name__}: {e}", "no exception")
    code = report.finish(ctx.out_dir)
    if wandb_run is not None:
        wandb_run.finish(exit_code=code)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--force", action="store_true", help="rerun even if already finished")
    args = parser.parse_args(argv)

    registry = load_registry()
    groups = registry.get("groups", {})
    if args.job in groups:
        for member in expand_group(args.job, groups):
            code = run_job(member, args.mode, args.force, registry)
            if code != 0:
                return code
        return 0
    if args.job not in registry["jobs"]:
        known = sorted([*registry["jobs"], *groups])
        print(f"unknown job '{args.job}'. Known jobs: {', '.join(known)}")
        return 2
    return run_job(args.job, args.mode, args.force, registry)


if __name__ == "__main__":
    sys.exit(main())

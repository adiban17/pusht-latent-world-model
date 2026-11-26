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
from wm.registry import (expand_group, input_hashes, job_config_hash, load_job_config,  # noqa: F401
                         load_registry)
from wm.repro import git_info
from wm.store import WANDB_PROJECT, make_store

LOCAL_FULL_ALLOWED = {"make_report"}


def is_finished(store, entry: dict[str, Any], chash: str,
                registry: dict[str, Any] | None = None) -> bool:
    """True if every artifact the job makes exists, is complete and has the same config_hash,
    and (for a job that needs `ckpt-*` checkpoints) was made from checkpoints whose producers'
    config_hashes are still the current ones (metadata `inputs`)."""
    makes = entry.get("makes") or []
    if not makes:
        return False
    inputs = input_hashes(entry, registry)
    for name in makes:
        meta = store.own_metadata(name)
        if not meta or not meta.get("complete") or meta.get("config_hash") != chash:
            return False
        if inputs and meta.get("inputs") != inputs:
            return False
    return True


def run_job(name: str, mode: str, force: bool, registry: dict[str, Any]) -> int:
    """Run one job; returns its exit code."""
    return run_job_status(name, mode, force, registry)[0]


def run_job_status(name: str, mode: str, force: bool, registry: dict[str, Any]) -> tuple[int, str]:
    """Run one job; returns (exit code, status), status in PASS / SKIP / FAIL / REFUSED."""
    entry = registry["jobs"][name]
    platform = detect_platform()
    if mode == "full" and platform == "local" and name not in LOCAL_FULL_ALLOWED:
        print(f"REFUSED: '{name}' --mode full does not run on the local Mac (CPU, smoke only). "
              "Run it on Kaggle/Colab with the runner notebook.")
        return 2, "REFUSED"

    try:
        module = importlib.import_module(entry["module"])
    except ModuleNotFoundError as e:
        if e.name == entry["module"]:
            print(f"job '{name}' is not built yet (Phase {entry['phase']})")
            return 2, "FAIL"
        raise

    cfg = load_job_config(entry)
    chash = job_config_hash(name, entry, cfg)
    store = make_store(platform, mode)
    if not force and is_finished(store, entry, chash, registry):
        print(f"{name}: already finished, skipping (use --force to rerun)")
        return 0, "SKIP"

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
    return code, "PASS" if code == 0 else "FAIL"


def run_group(group: str, mode: str, force: bool, registry: dict[str, Any]) -> int:
    """Run every job of a group in order, continuing after a failure; prints a summary line and
    returns 0 only if no job failed (skipped = already finished counts as fine)."""
    results: list[tuple[str, str]] = []
    for member in expand_group(group, registry.get("groups", {})):
        try:
            status = run_job_status(member, mode, force, registry)[1]
        except Exception:  # e.g. W&B or store errors outside the job itself: record, go on
            traceback.print_exc()
            status = "FAIL"
        results.append((member, status))
        print(f"[group {group}] {member}: {status}", flush=True)
    by = {s: [m for m, st in results if st == s] for s in ("PASS", "SKIP", "FAIL", "REFUSED")}
    bad = by["FAIL"] + by["REFUSED"]
    parts = [f"{s} {', '.join(ms)}" for s, ms in by.items() if ms]
    print(f"GROUP {group}: {'FAILED' if bad else 'OK'} ({len(results) - len(bad)} of {len(results)} "
          f"passed or skipped) | " + " | ".join(parts), flush=True)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--force", action="store_true", help="rerun even if already finished")
    args = parser.parse_args(argv)

    registry = load_registry()
    groups = registry.get("groups", {})
    if args.job in groups:
        return run_group(args.job, args.mode, args.force, registry)
    if args.job not in registry["jobs"]:
        known = sorted([*registry["jobs"], *groups])
        print(f"unknown job '{args.job}'. Known jobs: {', '.join(known)}")
        return 2
    return run_job(args.job, args.mode, args.force, registry)


if __name__ == "__main__":
    sys.exit(main())

"""Loading a trained checkpoint (`ckpt-wm`, `ckpt-wm-1step`, ...) for an eval job.

Rule (fix after the Kaggle overnight smoke, 2026-11-27): an eval job uses only the checkpoint
whose config_hash equals what the producing job's *current* config would give. Otherwise, in
smoke mode, the store's "read `<name>` before `<name>-smoke`" rule silently picks an older
recipe (e.g. `ckpt-wm:v5`, the v1 model, after wm.yaml switched to A+B).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wm.jobs import JobContext
from wm.registry import producer


def pull_matching(ctx: JobContext, artifact: str) -> tuple[Path, str, dict[str, Any], str]:
    """(local path, artifact name, metadata, expected config_hash) of the complete checkpoint
    made with the current config. Raises RuntimeError naming what was found otherwise."""
    job, expected = producer(artifact)
    name = ctx.store.resolve_matching(artifact, expected)
    if name is None:
        found = "; ".join(
            f"{cand}: " + (f"config_hash {m.get('config_hash')}, complete {m.get('complete')}, "
                           f"step {m.get('step')}" if m else "missing")
            for cand, m in ctx.store.candidates(artifact))
        raise RuntimeError(f"no complete '{artifact}' with config_hash {expected} (the current "
                           f"configs of job {job}). Found: {found}. Fix: run {job} first.")
    path = ctx.store.backend.pull(name)
    if path is None:
        raise RuntimeError(f"'{name}' has matching metadata but could not be pulled")
    return path, name, ctx.store.backend.metadata(name) or {}, expected

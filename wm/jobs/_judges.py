"""Shared pieces of the Phase 5b judge jobs: rendered judge data and the angle verdict."""
from __future__ import annotations

from typing import Any

import numpy as np

from wm.jobs import JobContext


def load_judge_npz(ctx: JobContext, split: str) -> tuple[dict[str, np.ndarray], str]:
    """judge_<split>.npz from the store's `judge-data` (smoke: falls back to `judge-data-smoke`)."""
    name = ctx.store.resolve("judge-data")
    path = ctx.store.pull("judge-data")
    if path is None:
        raise RuntimeError("store artifact 'judge-data' not found: run render_judge_data first")
    with np.load(path / f"judge_{split}.npz") as d:
        return {k: d[k] for k in d.files}, name


def angle_verdict(angle_deg: float, present_deg: float, absent_deg: float) -> str:
    """Pre-registered (Phase 5b): < present = present, > absent = absent, else weak."""
    if angle_deg < present_deg:
        return "present (rotation goals plausible)"
    if angle_deg > absent_deg:
        return "absent (Phase 6 must restrict or flag rotation goals)"
    return "weak"


def fmt_typed(test: dict[str, dict[str, Any]], key: str, digits: int = 1) -> str:
    """'real 10.4 | wm_recon 12.0 | ...' for one error key over frame types."""
    return " | ".join(f"{k} {v[f'{key}_mean']:.{digits}f}" for k, v in test.items())

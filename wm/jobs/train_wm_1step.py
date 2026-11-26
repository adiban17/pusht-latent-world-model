"""Job train_wm_1step (Phase 4): the §4 world model trained teacher-forced -> `ckpt-wm-1step`.

Same model, loss (incl. the variance hinge) and schedule as train_wm; only the rollout during
training is fed the real z_{t+k} instead of its own prediction. Validation is open-loop, and
"beats copy-last at k = 10" is WARN-only (losing it is a legitimate baseline result).
"""
from __future__ import annotations

from wm.jobs import JobContext
from wm.jobs import train_wm
from wm.report import Report

ARTIFACT = "ckpt-wm-1step"


def run(ctx: JobContext, report: Report) -> None:
    tf = bool(ctx.config.get("teacher_forcing", False))
    report.check_true("teacher_forcing", tf, str(tf).lower(), "true for wm_1step")
    train_wm.run(ctx, report, artifact=ARTIFACT, copy_last_must_win=False)

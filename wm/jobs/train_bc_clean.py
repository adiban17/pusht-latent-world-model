"""Job train_bc_clean (Phase 6): the train_bc model and recipe on the expert's clean actions
(`configs/bc_clean.yaml`, use_clean_actions: true) -> store `ckpt-bc-clean`.

BC's fair second try: plan_val runs both BC variants on the val pairs and the better one
(pre-registered rule, `wm.eval.planning.select_bc`) goes to the final test.
"""
from __future__ import annotations

from wm.jobs import JobContext
from wm.jobs.train_bc import run_bc
from wm.report import Report

ARTIFACT = "ckpt-bc-clean"


def run(ctx: JobContext, report: Report) -> None:
    if not ctx.config.get("use_clean_actions"):
        raise RuntimeError("train_bc_clean needs use_clean_actions: true (configs/bc_clean.yaml)")
    run_bc(ctx, report, ARTIFACT)

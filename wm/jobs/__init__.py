"""One module per job in jobs.yaml. Each exposes `run(ctx: JobContext, report: Report) -> None`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wm.store import Store


@dataclass
class JobContext:
    """Everything a job needs, built once by run.py."""

    name: str
    mode: str          # "smoke" | "full"
    platform: str      # "local" | "kaggle" | "colab"
    device: Any        # torch.device
    out_dir: Path      # <platform out dir>/<job>
    store: Store
    config: dict[str, Any]
    config_hash: str
    git: str
    wandb_run: Any = None

    def meta(self, complete: bool, **extra: Any) -> dict[str, Any]:
        """Standard store metadata: complete, config_hash, git (+ anything extra, e.g. step)."""
        return {"complete": complete, "config_hash": self.config_hash, "git": self.git, **extra}

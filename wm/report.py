"""The REPORT block every job prints at the end (and saves to <out_dir>/<job>/REPORT.md)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEADER = "==================== RUN REPORT ===================="
FOOTER = "=" * len(HEADER)
STATUSES = ("PASS", "WARN", "FAIL")


def format_duration(seconds: float) -> str:
    """'4.2s', '3m05s' or '1h02m03s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


@dataclass
class Report:
    """Collects CHECKS / METRICS / SAVED lines for one job run."""

    job: str
    mode: str
    platform: str
    device: str
    git: str
    params: str = "n/a"
    wandb: str = "off"
    checks: list[tuple[str, str, str, str]] = field(default_factory=list)
    metrics: list[tuple[str, str]] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)
    start: float = field(default_factory=time.time)

    def check(self, name: str, status: str, value: Any, threshold: str) -> None:
        """Add a CHECK line; status is PASS, WARN or FAIL."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        self.checks.append((status, name, str(value), threshold))

    def check_true(self, name: str, ok: bool, value: Any, threshold: str) -> bool:
        """PASS if ok else FAIL. Returns ok."""
        self.check(name, "PASS" if ok else "FAIL", value, threshold)
        return ok

    def metric(self, name: str, value: Any) -> None:
        self.metrics.append((name, str(value)))

    def save(self, item: str | Path) -> None:
        self.saved.append(str(item))

    @property
    def failed(self) -> bool:
        return any(c[0] == "FAIL" for c in self.checks)

    def render(self, elapsed: float | None = None) -> str:
        """The exact REPORT text."""
        elapsed = time.time() - self.start if elapsed is None else elapsed
        lines = [
            HEADER,
            f"job: {self.job}   mode: {self.mode}   platform: {self.platform}   device: {self.device}",
            f"git: {self.git}   time: {format_duration(elapsed)}   params: {self.params}",
            "CHECKS",
            *[f" [{st}] {name}: {val} ({thr})" for st, name, val, thr in self.checks],
            "METRICS",
            *[f" {name} = {val}" for name, val in self.metrics],
            "SAVED",
            *[f" {item}" for item in self.saved],
            f"W&B: {self.wandb}",
            FOOTER,
        ]
        return "\n".join(lines)

    def finish(self, job_out_dir: str | Path) -> int:
        """Print the REPORT, write REPORT.md, return the exit code (1 if any FAIL)."""
        text = self.render()
        print(text)
        path = Path(job_out_dir) / "REPORT.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        return 1 if self.failed else 0

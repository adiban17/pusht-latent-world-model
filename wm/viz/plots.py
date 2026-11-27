"""Metric-vs-k plots for the rollout evaluation (Phase 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed colour per method (never by rank); references are grey. Categorical slots 1-4 of the
# dataviz reference palette (blue, orange, aqua, yellow).
COLORS = {"wm": "#2a78d6", "wm_via_frames": "#2a78d6", "wm_1step": "#eb6834", "pixel": "#1baf7a",
          "copy_last": "#eda100"}
REF_STYLE = {"real": ("#52514c", "--"), "wm_recon": ("#8a897f", ":"), "wm1step_recon": ("#b0afa6", "-.")}
LINE_STYLE = {"wm_via_frames": "--"}   # same colour as wm (latent judge: direct z_hat vs re-encoded frames)
INK, MUTED, GRID = "#1f1e1c", "#6b6a63", "#e6e5e0"


def metric_vs_k(series: dict[str, dict[str, list[float]]], title: str, ylabel: str,
                path: str | Path, dashed: dict[str, dict[str, list[float]]] | None = None,
                floor: tuple[str, float] | None = None, log_y: bool = False) -> Path:
    """One line per row (mean) with its 95% CI band, k on the x axis.

    `series` rows are methods (coloured) or references `real` / `wm_recon` (grey, dashed/dotted);
    `dashed` adds rows drawn dashed in their method's colour (e.g. p_hat pusher error);
    `floor` = (label, value) draws a horizontal grey line (e.g. the probe on all test frames).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    def draw(label: str, s: dict[str, list[float]], color: str, style: str) -> None:
        k = np.arange(1, len(s["mean"]) + 1)
        ax.fill_between(k, s["lo"], s["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(k, s["mean"], style, color=color, linewidth=2, marker="o", markersize=4, label=label)

    for row, s in series.items():
        color, style = REF_STYLE.get(row, (COLORS.get(row, MUTED), LINE_STYLE.get(row, "-")))
        draw(row, s, color, style)
    for row, s in (dashed or {}).items():
        draw(row, s, COLORS.get(row.split(" ")[0], MUTED), "--")
    if floor is not None:
        ax.axhline(floor[1], color=MUTED, linewidth=1, linestyle=(0, (2, 3)), label=floor[0])
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("prediction step k", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)

"""Metric-vs-k plots for the rollout evaluation (Phase 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed colour per method (never by rank); references are grey. Categorical slots 1-4 of the
# dataviz reference palette (blue, orange, aqua, yellow).
COLORS = {"wm": "#2a78d6", "wm_via_frames": "#2a78d6", "wm_1step": "#eb6834", "pixel": "#1baf7a",
          "copy_last": "#eda100",
          # planning methods (Phase 7): CEM arms share their dynamics model's colour
          "cem_wm": "#2a78d6", "cem_wm_1step": "#eb6834", "cem_pixel": "#1baf7a", "bc_clean": "#8a5cc2",
          "bc": "#8a5cc2", "random": "#8a897f", "noop": "#b0afa6"}
REF_STYLE = {"real": ("#52514c", "--"), "wm_recon": ("#8a897f", ":"), "wm1step_recon": ("#b0afa6", "-.")}
LINE_STYLE = {"wm_via_frames": "--"}   # same colour as wm (latent judge: direct z_hat vs re-encoded frames)
INK, MUTED, GRID = "#1f1e1c", "#6b6a63", "#e6e5e0"


def metric_vs_k(series: dict[str, dict[str, list[float]]], title: str, ylabel: str,
                path: str | Path, dashed: dict[str, dict[str, list[float]]] | None = None,
                floor: tuple[str, float] | None = None, log_y: bool = False,
                ks: list[int] | None = None) -> Path:
    """One line per row (mean) with its 95% CI band, k on the x axis (1..len, or `ks`).

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
        k = np.asarray(ks) if ks is not None else np.arange(1, len(s["mean"]) + 1)
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
    ax.set_xticks(list(ks) if ks is not None else range(1, 11))
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


def _style(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def rate_bars(rows: dict[str, dict[str, dict[str, float]]], title: str, path: str | Path) -> Path:
    """Grouped bars: per method, success and position-only success (pooled mean, 95% CI whiskers).
    rows[method] = {"success": {mean, lo, hi}, "pos_success": {mean, lo, hi}}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for off, key, label, alpha in ((-0.2, "success", "success", 1.0), (0.2, "pos_success", "position-only", 0.45)):
        m = np.array([rows[n][key]["mean"] or 0.0 for n in names])
        lo = np.array([rows[n][key]["lo"] or 0.0 for n in names])
        hi = np.array([rows[n][key]["hi"] or 0.0 for n in names])
        ax.bar(x + off, m, 0.38, color=[COLORS.get(n, MUTED) for n in names], alpha=alpha, label=label)
        ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate (pooled over seeds)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def rate_by_bin(table: dict, metric: str, title: str, path: str | Path) -> Path:
    """One line per method: pooled `metric` rate per difficulty bin (95% CI band)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{l}\n(n {n})" for l, n in zip(table["labels"], table["pairs"])]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for m, bins in table["methods"].items():
        mean = np.array([np.nan if b[metric]["mean"] is None else b[metric]["mean"] for b in bins])
        lo = np.array([np.nan if b[metric]["lo"] is None else b[metric]["lo"] for b in bins])
        hi = np.array([np.nan if b[metric]["hi"] is None else b[metric]["hi"] for b in bins])
        c = COLORS.get(m, MUTED)
        ax.fill_between(x, lo, hi, color=c, alpha=0.10, linewidth=0)
        ax.plot(x, mean, "-", color=c, marker="o", markersize=4, linewidth=2, label=m)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel(table["key"], color=INK)
    ax.set_ylabel(f"{metric} (pooled)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)

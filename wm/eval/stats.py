"""Paired tests and bootstrap CIs for the final evaluation (Phase 7, pre-registered in PROGRESS.md).

  mcnemar_exact          exact McNemar test on paired binary outcomes (binomial test on discordant pairs)
  holm                   Holm step-down adjustment of a family of p-values
  cluster_bootstrap_rate success rate pooled over seeds, CI from resampling pairs (pair-clustered)
  paired_bootstrap_diff  difference of per-pair means of two methods, CI from resampling pairs
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test of paired outcomes a, b (same pairs, same order).
    b_only / c_only = pairs where only a / only b succeeded; p = binomtest(b_only, b_only + c_only, 0.5);
    no discordant pairs -> p = 1."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    assert a.shape == b.shape
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "discordant": n10 + n01, "n": int(len(a)), "p": p}


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, made monotone, capped at 1), in the input order."""
    p = np.asarray(pvals, np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _ci(boot: np.ndarray, ci: float) -> tuple[float, float]:
    q = 100.0 * (1.0 - ci) / 2.0
    return float(np.percentile(boot, q)), float(np.percentile(boot, 100.0 - q))


def pair_resamples(n_pairs: int, resamples: int, seed: int) -> np.ndarray:
    """Bootstrap indices over pairs [R, n_pairs] (shared by every method -> paired CIs)."""
    return np.random.default_rng(seed).integers(0, n_pairs, size=(resamples, n_pairs))


def cluster_bootstrap_rate(succ: np.ndarray, resamples: int = 10000, seed: int = 0,
                           ci: float = 0.95) -> dict[str, Any]:
    """succ [P, S] (pairs x the seeds this method has) -> pooled mean success with a
    pair-clustered bootstrap CI: resample pairs, keep all seeds of a pair together."""
    succ = np.asarray(succ, np.float64)
    if succ.ndim == 1:
        succ = succ[:, None]
    per_pair = succ.mean(1)
    if len(per_pair) == 0:
        return {"mean": None, "lo": None, "hi": None, "n_pairs": 0, "n_seeds": int(succ.shape[1])}
    lo, hi = _ci(per_pair[pair_resamples(len(per_pair), resamples, seed)].mean(1), ci)
    return {"mean": float(per_pair.mean()), "lo": lo, "hi": hi, "n_pairs": int(len(per_pair)),
            "n_seeds": int(succ.shape[1]), "k": int(succ.sum()), "cells": int(succ.size)}


def paired_bootstrap_diff(x: np.ndarray, y: np.ndarray, resamples: int = 10000, seed: int = 0,
                          ci: float = 0.95) -> dict[str, float]:
    """Per-pair values x, y [P] (e.g. mean success over a method's seeds) -> mean(x - y) with a
    percentile bootstrap CI over pairs."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    lo, hi = _ci(d[pair_resamples(len(d), resamples, seed)].mean(1), ci)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi, "n_pairs": int(len(d))}

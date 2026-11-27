"""Final evaluation rules (Phase 7, pre-registered in PROGRESS.md).

  failure_tag        success | position_only | angle_only | both_failed, from the final state
  believed_success   CEM terminal cost fell below the imagined-success threshold tau at some step
  bin_index          difficulty bin of a pair (left-inclusive edges, last bin closed)
  pooled_summary     one method over its seeds: pooled rates with pair-clustered bootstrap CIs
  difficulty_table   success / position-only success per bin of move_px or |turn_deg|
  primary_tests      the 6 pre-registered comparisons: exact McNemar (seed 0), Holm, paired bootstrap
  pick_gallery       a fixed, seeded choice of episodes per tag
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from wm.envs.pusht import SUCCESS_ANG_DEG, SUCCESS_POS_PX
from wm.eval.planning import is_rotation_pair, rate
from wm.eval.stats import cluster_bootstrap_rate, holm, mcnemar_exact, paired_bootstrap_diff

TAGS = ("success", "position_only", "angle_only", "both_failed")
LOW_POWER_N = 30    # a group with fewer pairs is flagged "low power"


def failure_tag(success: bool, final_pos_px: float, final_ang_deg: float) -> str:
    """Partition of episodes by their final state (success is checked after every step with the
    same thresholds, so a failed episode never ends with both errors under them)."""
    if success:
        return "success"
    pos_ok, ang_ok = final_pos_px < SUCCESS_POS_PX, abs(final_ang_deg) < SUCCESS_ANG_DEG
    if pos_ok and ang_ok:
        raise ValueError("final state satisfies is_success but the episode is not a success")
    return "position_only" if pos_ok else ("angle_only" if ang_ok else "both_failed")


def believed_success(terminal_costs: Sequence[float], tau: float | None) -> bool:
    """Imagined success: the best plan's terminal cost was below tau at some step (CEM only)."""
    return tau is not None and len(terminal_costs) > 0 and min(terminal_costs) < tau


def _edges(edges: Sequence[float | None]) -> list[float]:
    """Config edges -> floats; a trailing null (YAML) means open-ended (inf)."""
    return [float("inf") if e is None else float(e) for e in edges]


def bin_index(value: float, edges: Sequence[float | None]) -> int:
    """Index of the bin [edges[i], edges[i+1]) holding value; the last bin is closed and
    open-ended values above the last edge fall in it. Values below edges[0] are an error."""
    e = _edges(edges)
    if value < e[0]:
        raise ValueError(f"{value} below the first bin edge {e[0]}")
    return int(min(np.searchsorted(np.asarray(e[1:-1], np.float64), value, side="right"), len(e) - 2))


def bin_labels(edges: Sequence[float | None]) -> list[str]:
    e = _edges(edges)
    out = [f"[{e[i]:g}, {e[i + 1]:g})" for i in range(len(e) - 2)]
    return out + [f">= {e[-2]:g}" if np.isinf(e[-1]) else f"[{e[-2]:g}, {e[-1]:g}]"]


def far_mask(pairs: list[dict[str, Any]]) -> np.ndarray:
    """Pairs whose start CoG error is >= 20 px (the position-only headline subset)."""
    return np.array([p["move_px"] >= SUCCESS_POS_PX for p in pairs])


def outcome_matrix(records_by_seed: list[list[dict[str, Any]]], key: str) -> np.ndarray:
    """[P, S] bool of records[seed][pair][key]."""
    return np.array([[r[key] for r in recs] for recs in records_by_seed], bool).T


def pooled_summary(records_by_seed: list[list[dict[str, Any]]], pairs: list[dict[str, Any]],
                   resamples: int, seed: int) -> dict[str, Any]:
    """One method over its seeds: pooled success / position-only success (pair-clustered bootstrap
    CIs), the same per turning group (low-power flag), pooled medians, angle once position is
    reached, failure-tag counts and seconds per episode."""
    succ, pos = outcome_matrix(records_by_seed, "success"), outcome_matrix(records_by_seed, "pos_success")
    far = far_mask(pairs)
    rot = np.array([is_rotation_pair(p) for p in pairs])
    allr = [r for recs in records_by_seed for r in recs]
    reached = [abs(r["final_ang_deg"]) for r in allr if r["pos_success"]]
    reached_fail = [abs(r["final_ang_deg"]) for r in allr if r["pos_success"] and not r["success"]]
    groups = {}
    for g, want in (("rotation", True), ("translation", False)):
        m = rot == want
        groups[g] = {"pairs": int(m.sum()), "low_power": bool(m.sum() < LOW_POWER_N),
                     "success": cluster_bootstrap_rate(succ[m], resamples, seed),
                     "pos_success": cluster_bootstrap_rate(pos[m & far], resamples, seed)}
    tags = [r["tag"] for r in allr]
    return {
        "seeds": len(records_by_seed),
        "success": cluster_bootstrap_rate(succ, resamples, seed),
        "pos_success": cluster_bootstrap_rate(pos[far], resamples, seed),
        "pos_success_all": cluster_bootstrap_rate(pos, resamples, seed),
        "groups": groups,
        "median_final_pos_px": float(np.median([r["final_pos_px"] for r in allr])),
        "median_final_ang_deg": float(np.median([abs(r["final_ang_deg"]) for r in allr])),
        "pos_reached_n": len(reached),
        "pos_reached_median_final_ang_deg": float(np.median(reached)) if reached else None,
        "pos_reached_failed_n": len(reached_fail),
        "pos_reached_failed_median_final_ang_deg": float(np.median(reached_fail)) if reached_fail else None,
        "tags": {t: int(tags.count(t)) for t in TAGS},
        "tags_seed0": {t: int(sum(r["tag"] == t for r in records_by_seed[0])) for t in TAGS},
        "seconds_per_episode": float(np.mean([r["seconds"] for r in allr])),
    }


def imagined_counts(records_by_seed: list[list[dict[str, Any]]], tau: float | None) -> dict[str, Any]:
    """Imagined vs real (CEM): episodes believed successful, and how many of them failed."""
    allr = [r for recs in records_by_seed for r in recs]
    return {"tau": tau, "n": len(allr), "believed": int(sum(r["believed"] for r in allr)),
            "believed_success_failed": int(sum(r["believed"] and not r["success"] for r in allr)),
            "per_seed": [{"believed": int(sum(r["believed"] for r in recs)),
                          "believed_success_failed": int(sum(r["believed"] and not r["success"] for r in recs))}
                         for recs in records_by_seed]}


def difficulty_table(pairs: list[dict[str, Any]], records: dict[str, list[list[dict[str, Any]]]], key: str,
                     edges: Sequence[float], resamples: int, seed: int) -> dict[str, Any]:
    """Per bin of `key` (move_px or |turn_deg|): pair count, and for every method pooled success
    and position-only success with pair-clustered CIs, plus seed-0 counts with Wilson CIs."""
    vals = np.array([abs(p[key]) for p in pairs])
    b = np.array([bin_index(v, edges) for v in vals])
    far = far_mask(pairs)
    out = {"key": key, "edges": [None if e is None or np.isinf(e) else e for e in edges],
           "labels": bin_labels(edges), "pairs": [int((b == i).sum()) for i in range(len(edges) - 1)],
           "methods": {}}
    for m, recs in records.items():
        succ, pos = outcome_matrix(recs, "success"), outcome_matrix(recs, "pos_success")
        out["methods"][m] = [{"success": cluster_bootstrap_rate(succ[b == i], resamples, seed),
                              "pos_success": cluster_bootstrap_rate(pos[(b == i) & far], resamples, seed),
                              "success_seed0": rate(succ[b == i, 0].tolist()),
                              "pos_success_seed0": rate(pos[(b == i) & far, 0].tolist())}
                             for i in range(len(edges) - 1)]
    return out


def primary_tests(records: dict[str, list[list[dict[str, Any]]]], pairs: list[dict[str, Any]],
                  comparisons: Sequence[Sequence[str]], resamples: int, seed: int,
                  alpha: float = 0.05) -> list[dict[str, Any]]:
    """Pre-registered: for each comparison (a, b) and metric (success; position-only success on
    pairs starting >= 20 px away): exact McNemar on seed 0 (the first seed of every method),
    Holm over the whole family, and a paired bootstrap CI of mean(a) - mean(b), where each
    pair's value is its mean over the seeds that method has."""
    far = far_mask(pairs)
    rows = []
    for metric, key, mask in (("success", "success", np.ones(len(pairs), bool)),
                              ("pos_success", "pos_success", far)):
        for a, b in comparisons:
            x, y = outcome_matrix(records[a], key)[mask], outcome_matrix(records[b], key)[mask]
            t = mcnemar_exact(x[:, 0], y[:, 0])
            rows.append({"metric": metric, "a": a, "b": b, **t,
                         "rate_a_seed0": float(x[:, 0].mean()), "rate_b_seed0": float(y[:, 0].mean()),
                         "seeds_a": int(x.shape[1]), "seeds_b": int(y.shape[1]),
                         "diff_bootstrap": paired_bootstrap_diff(x.mean(1), y.mean(1), resamples, seed)})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
        r["significant"] = bool(p < alpha)
        if not r["significant"]:
            r["claim"] = f"no claim (Holm p = {p:.3g})"
        else:
            r["claim"] = f"{r['a']} better" if r["a_only"] > r["b_only"] else f"{r['b']} better"
    return rows


def rollout_angle_verdict(wm_minus_recon: dict[str, float], wm_minus_copy_search: dict[str, float],
                          wm_minus_copy_exact: dict[str, float]) -> dict[str, Any]:
    """Pre-registered verdict on "imagined rollouts lose the angle" (paired 95% CIs at verdict_k,
    search oracle). A: the rollout adds angle error beyond the autoencoder (wm - wm_recon > 0).
    B: wm is worse than not predicting, against copy_last read by the same search oracle AND
    against copy_last's exact |theta_t - theta_{t+k}| (the search can miss copy_last's block).
    confirmed = A and B; partial = A only; rejected = not A."""
    a = wm_minus_recon["lo"] > 0
    b_search, b_exact = wm_minus_copy_search["lo"] > 0, wm_minus_copy_exact["lo"] > 0
    b = b_search and b_exact
    verdict = "confirmed" if a and b else ("partial" if a else "rejected")
    return {"verdict": verdict, "A_rollout_adds_error": bool(a), "B_worse_than_copy_last": bool(b),
            "B_search": bool(b_search), "B_exact": bool(b_exact)}


def pick_gallery(tags: Sequence[str], tag: str, k: int = 4, seed: int = 0) -> list[int]:
    """Up to k episode indices with this tag, chosen by default_rng(seed), returned sorted."""
    idx = [i for i, t in enumerate(tags) if t == tag]
    if len(idx) <= k:
        return idx
    return sorted(int(i) for i in np.random.default_rng(seed).choice(idx, k, replace=False))

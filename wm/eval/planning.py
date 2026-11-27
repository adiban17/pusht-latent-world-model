"""Planning evaluation (CLAUDE.md §9 Planning; Phase 6 pre-registered rules in PROGRESS.md).

Per method: success (block CoG < 20 px AND |Δθ| < 20°, primary), position-only success (CoG
< 20 px after any step; secondary, headline on pairs whose start CoG error is >= 20 px), both
split into rotation pairs (|goal angle − start angle| >= 20°) and translation pairs, with
Wilson 95% CIs; median final errors, mean steps-to-success, seconds per episode.
Rules: `select_bc` (which BC goes to the test), `select_setting` (plan_sweep -> frozen config),
`calibrate_tau` (imagined-success threshold for Phase 7's failure tags).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from wm.envs.pusht import SUCCESS_ANG_DEG, SUCCESS_POS_PX, block_cog, wrap_angles_rad

ROTATION_DEG = 20.0   # a pair is a "rotation pair" if |goal angle - start angle| >= 20 deg


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (n = 0 -> (0, 1))."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(flags: list[bool]) -> dict[str, Any]:
    k, n = int(sum(flags)), len(flags)
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": [lo, hi]}   # None: empty group


def is_rotation_pair(pair: dict[str, Any]) -> bool:
    return abs(pair["turn_deg"]) >= ROTATION_DEG


def summarize(episodes: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of one method's episodes (same order as `pairs`)."""
    assert len(episodes) == len(pairs)
    succ = [e["success"] for e in episodes]
    pos = [e["pos_success"] for e in episodes]
    far = [e["start_pos_px"] >= SUCCESS_POS_PX for e in episodes]
    rot = [is_rotation_pair(p) for p in pairs]
    out: dict[str, Any] = {
        "success": rate(succ),
        "pos_success": rate([p for p, f in zip(pos, far) if f]),      # headline secondary (subset)
        "pos_success_all": rate(pos),
        "groups": {},
        "median_final_pos_px": float(np.median([e["final_pos_px"] for e in episodes])),
        "median_final_ang_deg": float(np.median([abs(e["final_ang_deg"]) for e in episodes])),
        "mean_steps_to_success": (float(np.mean([e["steps"] for e in episodes if e["success"]]))
                                  if any(succ) else None),
        # Phase 6b: direct test of the angle-insensitivity hypothesis (episodes that reached position)
        "pos_reached_n": int(sum(pos)),
        "pos_reached_median_final_ang_deg": (float(np.median([abs(e["final_ang_deg"]) for e in episodes
                                                              if e["pos_success"]])) if any(pos) else None),
        "pos_reached_failed_n": int(sum(p and not s_ for p, s_ in zip(pos, succ))),
        "pos_reached_failed_median_final_ang_deg": (
            float(np.median([abs(e["final_ang_deg"]) for e in episodes if e["pos_success"] and not e["success"]]))
            if any(p and not s_ for p, s_ in zip(pos, succ)) else None),
        "seconds_per_episode": float(np.mean([e["seconds"] for e in episodes])),
        "policy_s_per_step": float(sum(e["policy_s"] for e in episodes) / max(1, sum(e["steps"] for e in episodes))),
        "env_s_per_step": float(sum(e["env_s"] for e in episodes) / max(1, sum(e["steps"] for e in episodes))),
    }
    for g, want in (("rotation", True), ("translation", False)):
        idx = [i for i, r in enumerate(rot) if r == want]
        out["groups"][g] = {"pairs": len(idx), "success": rate([succ[i] for i in idx]),
                            "pos_success": rate([pos[i] for i in idx if far[i]]),
                            "pos_success_all": rate([pos[i] for i in idx])}
    return out


def select_bc(bc: dict[str, Any], bc_clean: dict[str, Any]) -> tuple[str, str]:
    """(selected variant, reason). More successes; then more position-only successes (subset);
    then lower median final pos px; still tied -> 'bc' (the spec default)."""
    keys = (("success", lambda s: s["success"]["k"], True),
            ("position-only success", lambda s: s["pos_success"]["k"], True),
            ("median final pos px", lambda s: s["median_final_pos_px"], False))
    for label, f, higher in keys:
        a, b = f(bc), f(bc_clean)
        if a != b:
            win = "bc" if (a > b) == higher else "bc_clean"
            return win, f"{label}: bc {a:g} vs bc_clean {b:g}"
    return "bc", "tied on every criterion -> bc (spec default)"


def select_setting(rows: dict[str, dict[str, Any]], costs: dict[str, float], default: str = "default",
                   margin: int = 3, floor: int = 2) -> dict[str, Any]:
    """plan_sweep freeze rule. A variant replaces the default only with >= margin more successes;
    several -> the most, ties -> the cheaper (N x rounds x H). If every setting has <= floor
    successes, the same rule runs on position-only success (subset)."""
    metric = "success"
    if max(r["success"]["k"] for r in rows.values()) <= floor:
        metric = "pos_success"
    k = {name: r[metric]["k"] for name, r in rows.items()}
    better = [n for n in rows if n != default and k[n] >= k[default] + margin]
    if not better:
        return {"selected": default, "metric": metric,
                "reason": f"no variant has >= {margin} more {metric} than default ({k[default]})"}
    best = sorted(better, key=lambda n: (-k[n], costs[n], n))[0]
    return {"selected": best, "metric": metric,
            "reason": f"{best}: {metric} {k[best]} vs default {k[default]} (margin {margin})"}


def success_matrix(states: np.ndarray) -> np.ndarray:
    """[M, M] bool: is_success(states[i], states[j]) for every pair (vectorised)."""
    cog = block_cog(states)
    d = np.linalg.norm(cog[:, None] - cog[None], axis=-1)
    a = np.degrees(np.abs(wrap_angles_rad(states[:, None, 4].astype(np.float64) - states[None, :, 4])))
    return (d < SUCCESS_POS_PX) & (a < SUCCESS_ANG_DEG)


def calibrate_tau(z: np.ndarray, states: np.ndarray, episode_of: np.ndarray, percentile: float = 95.0,
                  max_pairs: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Imagined-success threshold τ: the `percentile` of ||E(o_i) − E(o_j)||² over frame pairs
    (i < j) of the same val episode whose states satisfy is_success. For context it also gives
    the median over same-episode pairs that are not successes."""
    succ_d, fail_d = [], []
    for e in np.unique(episode_of):
        idx = np.flatnonzero(episode_of == e)
        m = success_matrix(states[idx])
        iu = np.triu_indices(len(idx), 1)
        d = ((z[idx][:, None] - z[idx][None]) ** 2).sum(-1)[iu]
        succ_d.append(d[m[iu]])
        fail_d.append(d[~m[iu]])
    succ, fail = np.concatenate(succ_d), np.concatenate(fail_d)
    rng = np.random.default_rng(seed)
    if len(succ) > max_pairs:
        succ = rng.choice(succ, max_pairs, replace=False)
    if len(succ) == 0:
        return {"tau": None, "n_success_pairs": 0, "percentile": percentile}
    return {"tau": float(np.percentile(succ, percentile)), "percentile": percentile,
            "n_success_pairs": int(len(succ)), "median_success": float(np.median(succ)),
            "median_non_success": float(np.median(fail)) if len(fail) else None,
            "definition": "pct of ||E_wm(o_i) - E_wm(o_j)||^2 over same-val-episode frame pairs with "
                          "is_success(s_i, s_j); imagined success = terminal ||z_H - z_g||^2 < tau"}


def same_episode_pairs(states: np.ndarray, episode_of: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered pairs (i, j), i != j, of frames from the same episode -> (i, j, is_success(s_i, s_j))."""
    ii, jj, ss = [], [], []
    for e in np.unique(episode_of):
        idx = np.flatnonzero(episode_of == e)
        m = success_matrix(states[idx])
        a, b = np.nonzero(~np.eye(len(idx), dtype=bool))
        ii.append(idx[a]); jj.append(idx[b]); ss.append(m[a, b])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss)


def calibrate_tau_pixel(distance, states: np.ndarray, episode_of: np.ndarray, percentile: float = 95.0,
                        max_pairs: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Phase 6b τ_pixel: the `percentile` of distance(i, j) = weighted pixel distance between
    D_wm(E_wm(o_i)) and o_j (weight from o_j, pusher pixels removed) over ordered same-val-episode
    pairs i != j whose states satisfy is_success. `distance(i [P], j [P]) -> [P]` (numpy)."""
    i, j, ok = same_episode_pairs(states, episode_of)
    rng = np.random.default_rng(seed)
    succ, fail = np.flatnonzero(ok), np.flatnonzero(~ok)
    if len(succ) == 0:
        return {"tau": None, "n_success_pairs": 0, "percentile": percentile}
    succ = rng.choice(succ, min(max_pairs, len(succ)), replace=False)
    fail = rng.choice(fail, min(max_pairs, len(fail)), replace=False) if len(fail) else fail
    ds = distance(i[succ], j[succ])
    df = distance(i[fail], j[fail]) if len(fail) else np.zeros(0)
    return {"tau": float(np.percentile(ds, percentile)), "percentile": percentile,
            "n_success_pairs": int(len(ds)), "median_success": float(np.median(ds)),
            "median_non_success": float(np.median(df)) if len(df) else None,
            "definition": "pct of mean(w_j * (clip(D_wm(E_wm(o_i))) - o_j)^2) over ordered same-val-episode "
                          "frame pairs with is_success(s_i, s_j); w_j = 1 + 20 * (fg AND NOT pusher) from o_j; "
                          "imagined success = terminal pixel distance < tau_pixel"}


def select_cost_arm(latent: dict[str, Any], pixel: dict[str, Any], margin: int = 3) -> tuple[bool, str]:
    """Phase 6b rule: cost_space = pixel joins Phase 7 (as "post-hoc diagnostic, chosen on val")
    only with >= margin more successes than the latent arm; the latent default stays the headline."""
    a, b = latent["success"]["k"], pixel["success"]["k"]
    joins = b >= a + margin
    return joins, (f"pixel {b} vs latent {a} successes (needs >= {a + margin}): "
                   + ("joins Phase 7 as 'post-hoc diagnostic, chosen on val'" if joins else "does not join Phase 7"))


def compare_runs(ours: list[dict[str, Any]], ref: list[dict[str, Any]], ours_setup: dict[str, Any],
                 ref_setup: dict[str, Any]) -> dict[str, Any]:
    """Per-pair agreement of two runs' success flags (episode records), if their setups match
    (pairs, budget, seed); otherwise {'comparable': False, 'reason': ...}."""
    diff = [k for k in ("budget", "seed") if ours_setup.get(k) != ref_setup.get(k)]
    if [r["pair"] for r in ours] != [r["pair"] for r in ref]:
        diff.append("pairs")
    if diff:
        return {"comparable": False, "reason": "different " + ", ".join(diff)}
    agree = sum(a["success"] == b["success"] for a, b in zip(ours, ref))
    return {"comparable": True, "n": len(ours), "agree": int(agree),
            "ours_success": int(sum(a["success"] for a in ours)), "ref_success": int(sum(b["success"] for b in ref)),
            "only_ours": int(sum(a["success"] and not b["success"] for a, b in zip(ours, ref))),
            "only_ref": int(sum(b["success"] and not a["success"] for a, b in zip(ours, ref)))}

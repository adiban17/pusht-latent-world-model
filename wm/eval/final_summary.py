"""summary.md for the final evaluation (Phase 7): every number is formatted from final_eval.json
and imagined_angle.json; nothing is typed by hand. Used by final_eval and by
scripts/make_final_summary.py (regenerates results/final/summary.md locally)."""
from __future__ import annotations

from typing import Any


def _pct(r: dict[str, Any] | None) -> str:
    """Pooled rate {mean, lo, hi} -> '31.5% [25.0, 38.5]'."""
    if not r or r.get("mean") is None:
        return "n/a"
    return f"{100 * r['mean']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def _wil(r: dict[str, Any] | None) -> str:
    """Wilson rate {k, n, ci} -> '13/50 [15.9, 39.6]'."""
    if not r or not r.get("n"):
        return "0/0"
    return f"{r['k']}/{r['n']} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"


def _num(x: Any, d: int = 1, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:.{d}f}{suffix}"


def _ci(s: dict[str, float], d: int = 1, sign: bool = False) -> str:
    f = f"+.{d}f" if sign else f".{d}f"
    return f"{s['mean']:{f}} [{s['lo']:{f}}, {s['hi']:{f}}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def main_table(f: dict[str, Any]) -> list[str]:
    rows = []
    for m in f["methods"]:
        p, s0 = f["pooled"][m], f["per_seed"][m][0]
        g = p["groups"]
        im = f["imagined"].get(m)
        t = f["timing"].get(m)
        rows.append([
            f"**{m}**" if m == "cem_wm" else m, str(p["seeds"]), _pct(p["success"]), _wil(s0["success"]),
            _pct(p["pos_success"]),
            f"{_pct(g['rotation']['success'])} (n {g['rotation']['pairs']})",
            f"{_pct(g['translation']['success'])} (n {g['translation']['pairs']}"
            + (", low power)" if g["translation"]["low_power"] else ")"),
            f"{_num(p['median_final_pos_px'])} / {_num(p['median_final_ang_deg'])}",
            f"{_num(p['pos_reached_median_final_ang_deg'])} (n {p['pos_reached_n']})",
            f"{im['believed']} ({im['believed_success_failed']} failed)" if im else "–",
            _num(p["seconds_per_episode"], 2), _num(t["call_s"], 3) if t else "–"])
    return _table(["method", "seeds", "success, pooled [95% CI]", "success seed 0 [Wilson]",
                   "position-only, pooled", "turning pairs: success", "non-turning pairs: success",
                   "median final pos px / angle °", "median final angle ° once position reached",
                   "imagined success (of them failed)", "s / episode", "s / planning call"], rows)


def primary_table(f: dict[str, Any]) -> list[str]:
    rows = []
    for r in f["primary_tests"]:
        d = r["diff_bootstrap"]
        rows.append([f"{r['a']} vs {r['b']}", r["metric"], f"{100 * r['rate_a_seed0']:.1f}% vs {100 * r['rate_b_seed0']:.1f}%",
                     f"{r['a_only']} / {r['b_only']}", f"{r['p']:.3g}", f"{r['p_holm']:.3g}", r["claim"],
                     f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] pp "
                     f"(seeds {r['seeds_a']} vs {r['seeds_b']})"])
    return _table(["comparison", "metric", "seed 0 rates", "discordant (a only / b only)", "McNemar p",
                   "Holm p", "claim", "paired bootstrap difference a − b"], rows)


def bins_table(t: dict[str, Any]) -> list[str]:
    header = ["method"] + [f"{l} (n {n})" for l, n in zip(t["labels"], t["pairs"])]
    rows = [[m] + [f"{_pct(b['success'])} / {_pct(b['pos_success'])}" for b in bins]
            for m, bins in t["methods"].items()]
    return _table(header, rows)


def tags_table(f: dict[str, Any]) -> list[str]:
    tags = f["tags"]
    rows = []
    for m in f["methods"]:
        p = f["pooled"][m]
        im = f["imagined"].get(m)
        rows.append([m] + [f"{p['tags'][t]} ({p['tags_seed0'][t]})" for t in tags]
                    + [f"{im['believed_success_failed']}" if im else "–"])
    return _table(["method"] + [f"{t}: all seeds (seed 0)" for t in tags] + ["believed success but failed"], rows)


def angle_section(a: dict[str, Any] | None) -> list[str]:
    if not a:
        return ["`pending`: no `imagined_angle.json` (run `eval_imagined_angle`)."]
    ks = a["ks"]
    out = [f"Source: `imagined_angle.json` ({a['mode']}, {a['platform']}, git {a['git']}); "
           f"{a['windows']['n']} test windows; template oracle, candidates every {a['step_deg']:g}°.", ""]
    for reader, title in (("search", "Search reader (primary)"), ("truepos", "True-position reader (secondary)")):
        rows = []
        for r in list(a["rows"]) + ["copy_last_exact"]:
            s = a["metrics"][f"angle_{reader}"].get(r)
            if s is None:
                continue
            cells = []
            for j, _ in enumerate(ks):
                c = _ci({key: s[key][j] for key in ("mean", "lo", "hi")})
                if reader == "search" and r in a["metrics"]["edge_share"]:
                    c += f" (edge {100 * a['metrics']['edge_share'][r][j]:.0f}%)"
                cells.append(c)
            rows.append([r] + cells)
        out += [f"**{title}**, mean angle error ° [95% CI]" + (" (share of windows at the shift limit)"
                                                              if reader == "search" else ""), ""]
        out += _table(["row"] + [f"k = {k}" for k in ks], rows) + [""]
    rows = [[r] + [f"{a['metrics']['shift_px'][r]['mean'][j]:.1f} / {a['metrics']['shift_px'][r]['median'][j]:.1f}"
                   for j, _ in enumerate(ks)] for r in a["rows"]]
    out += ["**Oracle-found shift** (world px, mean / median)", ""] + _table(["row"] + [f"k = {k}" for k in ks], rows) + [""]
    rows = []
    for name, s in a["paired"].items():
        if name.startswith("search/"):
            rows.append([name.split("/", 1)[1]] + [_ci({key: s[key][j] for key in ("mean", "lo", "hi")}, sign=True)
                                                   for j, _ in enumerate(ks)])
    out += ["**Paired differences** (search reader, °, wm − other)", ""] + _table(["difference"] + [f"k = {k}" for k in ks], rows)
    v = a["verdict"]
    out += ["", f"**Verdict (pre-registered, k = {v['k']}): imagined rollouts lose the angle: {v['verdict']}** "
                f"(A: rollout adds error beyond wm_recon = {v['A_rollout_adds_error']}; B: worse than copy_last = "
                f"{v['B_worse_than_copy_last']}, search {v['B_search']}, exact {v['B_exact']})."]
    return out


def sweep_section(f: dict[str, Any]) -> list[str]:
    sw = f.get("sweep")
    if not sw:
        return ["`pending`"]
    methods = sw["methods"]
    rows = [[str(o["offset"]), str(o["n"]), f"{o['candidates']}", _num(o["median_move_px"]), _num(o["median_abs_turn_deg"])]
            + [f"{_wil(o['summaries'][m]['success'])} / {_wil(o['summaries'][m]['pos_success'])}" for m in methods]
            for o in sw["offsets"]]
    return _table(["goal offset", "pairs", "candidates", "median move px", "median |turn| °"]
                  + [f"{m}: success / position-only (seed 0)" for m in methods], rows)


def write_summary(f: dict[str, Any], a: dict[str, Any] | None) -> str:
    """The whole summary.md."""
    L = [f"# Final evaluation summary ({f['mode']}, {f['platform']})", "",
         f"Generated from `final_eval.json` (git {f['git']}, config_hash {f['config_hash']}) and "
         f"`imagined_angle.json`. Test pairs: {f['pairs']['n']} ({f['pairs']['artifact']}); budget {f['budget']}; "
         f"seeds {f['seeds']} (cem_pixel {f['pixel_seeds']}); BC arm = {f['bc_selected']['variant']} "
         f"(`bc_selected` from {f['bc_selected']['source']}). Frozen CEM config: N {f['cem']['candidates']}, "
         f"H {f['cem']['horizon']}, rounds {f['cem']['rounds']}, K {f['cem']['elites']}, beta {f['cem']['beta']:g}.",
         "", "Success = block CoG < 20 px AND |angle| < 20°, checked after every step. Position-only = CoG < 20 px "
         "after any step (pairs starting >= 20 px away). Pooled rates average over a method's seeds with a "
         "pair-clustered bootstrap 95% CI.", "", "## Main table", ""]
    L += main_table(f)
    L += ["", f"τ (imagined-success threshold, val-calibrated per encoder): "
              + ", ".join(f"{k} {_num(v.get('tau'), 3)}" for k, v in f["tau"].items()), "",
          "## Primary tests (pre-registered)", "",
          f"Exact McNemar on seed 0, Holm over the {len(f['primary_tests'])} tests (claim if Holm p < {f['alpha']:g}); "
          "effect = paired bootstrap over pairs of the per-pair mean over each method's seeds. Everything else "
          "is secondary/exploratory.", ""]
    L += primary_table(f)
    L += ["", "## Success vs goal difficulty (secondary)", "", "Cells: success / position-only, pooled [95% CI].", ""]
    for key in ("move_px", "turn_deg"):
        L += [f"**{key}**" + (" (absolute)" if key == "turn_deg" else ""), ""] + bins_table(f["difficulty"][key]) + [""]
    L += ["## Failure tags (from the final state)", ""] + tags_table(f) + [""]
    L += ["Gallery (seed 0, up to 4 episodes per tag picked by `default_rng(0)`; columns start | goal | final"
          " | decoded terminal imagined frame of the last plan (cem_wm only)): "
          + "; ".join(f"`gallery/{m}_{t}.png` pairs {v}" for m, tags in f["gallery"].items()
                      for t, v in tags.items() if v), ""]
    L += ["## Imagined angle (template oracle on imagined frames)", ""] + angle_section(a) + [""]
    L += ["## Goal-distance sweep (exploratory)", ""] + sweep_section(f) + [""]
    L += ["## Speed", ""]
    L += _table(["method", "s / episode (mean)", "s / full-size planning call"],
                [[m, _num(f["pooled"][m]["seconds_per_episode"], 2),
                  _num(f["timing"][m]["call_s"], 3) if m in f["timing"] else "–"] for m in f["methods"]])
    return "\n".join(L) + "\n"

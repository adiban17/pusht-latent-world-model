"""README rendering (Phase 8): every number in README.md is formatted from a committed results JSON
(or, for settings, a config file / code constant), never typed by hand.

`docs/README.template.md` holds the prose. Placeholders:
  {{f:<fact>}}   one value from `facts()`; each fact records its source (file: key path).
  {{t:<table>}}  a whole table from `tables()`; each table records the keys it reads.
Static text may contain digits only in ALLOWED_TOKENS, `k = 1|5|10`, code, link targets and the
References section (`untraced_numbers` enforces it). `render` + `sources_markdown` are used by the
`make_report` job and `scripts/make_readme_tables.py`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wm.envs import pusht  # noqa: E402  (code constants: image size, success thresholds)
from wm.eval import final_summary as fs  # noqa: E402

FILES = {
    "final": "results/final/final_eval.json",
    "angle": "results/final/imagined_angle.json",
    "roll": "results/rollouts.json",
    "pv": "results/plan_val.json",
    "ps": "results/plan_sweep.json",
    "pca": "results/plan_cost_ablation.json",
    "aa": "results/angle_audit.json",
    "abl": "results/ablation_wm_fixes.json",
    "col": "results/ablation_collapse_local.json",
    "data": "configs/data.yaml",
    "wm": "configs/wm.yaml",
    "pairs": "configs/pairs.yaml",
}
TEMPLATE = "docs/README.template.md"
README = "README.md"
SOURCES = "docs/readme_sources.md"
FIGURES = ("results/final/plots/success.png", "results/final/plots/imagined_angle_vs_k.png",
           "results/final/gallery/cem_wm_position_only.png")
# Digits allowed in static template text (not results): hardware, precision, image sizes.
ALLOWED_TOKENS = {"T4", "fp16", "64×64", "96×96", "4×4", "L2", "M1", "PushT-v0", "README.template.md"}
K10 = 9  # index of k = 10 in rollouts.json's per-k lists (ks = 1..10)


@dataclass(frozen=True)
class Fact:
    value: str
    source: str


class Results:
    """The committed JSON / YAML files, with key-path access that remembers the source."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)
        self.data: dict[str, Any] = {}
        for key, rel in FILES.items():
            text = (self.root / rel).read_text()
            self.data[key] = yaml.safe_load(text) if rel.endswith(".yaml") else json.loads(text)

    def get(self, key: str, path: str) -> tuple[Any, str]:
        """Value at 'a.b[3].c' in file `key`, and its source string 'file: a.b[3].c'."""
        obj: Any = self.data[key]
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
            obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
        return obj, f"{FILES[key]}: {path}"


# ---------- formatting ----------
def f1(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace("-", "−")


def pp(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "−")


def signed(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def intc(x: float) -> str:
    return f"{int(round(x)):,}"


def minutes(s: float) -> str:
    return f"{s / 60:.1f}"


def pval(p: float) -> str:
    return f"{p:.3g}"


def facts(r: Results) -> dict[str, Fact]:
    """Every single number the README prose uses, with its source."""
    F: dict[str, Fact] = {}

    def add(name: str, key: str, path: str, fmt: Callable[[Any], str] = str) -> Any:
        v, src = r.get(key, path)
        F[name] = Fact(str(fmt(v)), src)
        return v

    def derived(name: str, value: str, how: str) -> None:
        F[name] = Fact(value, f"derived: {how}")

    def const(name: str, value: Any, where: str) -> None:
        F[name] = Fact(str(value), where)

    # --- data ---
    ds = "checkpoints.wm.summary.data_summary"
    n_tr = add("train_eps", "roll", f"{ds}.train_episodes")
    n_va = add("val_eps", "roll", f"{ds}.val_episodes")
    add("train_windows", "roll", f"{ds}.train_windows", intc)
    add("val_windows", "roll", f"{ds}.val_windows_total", intc)
    steps = add("steps_per_ep", "data", "steps_per_episode")
    derived("frames_per_ep", str(steps + 1), "steps_per_episode + 1 (final frame stored)")
    derived("episodes", str(n_tr + n_va), "train_episodes + val_episodes")
    derived("transitions", intc((n_tr + n_va) * steps), "(train_episodes + val_episodes) × steps_per_episode")
    derived("train_transitions", intc(n_tr * steps), "train_episodes × steps_per_episode")
    derived("val_transitions", intc(n_va * steps), "val_episodes × steps_per_episode")
    test_frames = add("test_frames", "roll", "judges.probe_v1.test_all.frames", intc)
    n_test = test_frames // (steps + 1)
    derived("test_eps", str(n_test), "test_all.frames / (steps_per_episode + 1)")
    derived("test_transitions", intc(n_test * steps), "test episodes × steps_per_episode")
    add("test_windows", "roll", "windows.available", intc)
    tr, va, te = (r.get("data", k)[0] for k in ("train_seeds", "val_seeds", "test_seeds"))
    derived("train_seeds", f"{tr[0]}–{va[0] - 1}", "configs/data.yaml train_seeds minus val_seeds")
    derived("val_seeds", f"{va[0]}–{va[1]}", "configs/data.yaml val_seeds")
    derived("test_seeds", f"{te[0]}–{te[1]}", "configs/data.yaml test_seeds")
    add("noise_sigmas", "data", "expert.noise_sigmas", lambda v: ", ".join(f"{x:g}" for x in v))
    add("random_eps", "data", "expert.random_eps", lambda v: ", ".join(f"{x:g}" for x in v))
    add("subgoal_every", "data", "expert.subgoal_every")
    const("img", pusht.IMG_SIZE, "wm/envs/pusht.py: IMG_SIZE")
    const("succ_px", f"{pusht.SUCCESS_POS_PX:g}", "wm/envs/pusht.py: SUCCESS_POS_PX")
    const("succ_deg", f"{pusht.SUCCESS_ANG_DEG:g}", "wm/envs/pusht.py: SUCCESS_ANG_DEG")

    # --- model + training ---
    for part in ("encoder", "decoder", "dynamics", "total", "planner"):
        add(f"params_{part}", "roll", f"checkpoints.wm.summary.params.{part}", intc)
    v, src = r.get("roll", "checkpoints.wm.summary.params.total")
    F["params_m"] = Fact(f"{v / 1e6:.1f}M", f"derived: {src} / 1e6")
    add("params_pixel", "roll", "checkpoints.pixel.summary.params.total", intc)
    add("params_bc", "final", "checkpoints.bc_clean.summary.params.total", intc)
    add("latent_dim", "wm", "model.latent_dim")
    add("hidden_dim", "wm", "model.hidden_dim")
    add("context", "wm", "window.context")
    H = add("horizon", "wm", "window.horizon")
    add("fg_alpha", "wm", "fg_alpha", lambda v: f"{v:g}")
    for term in ("rec", "lat", "img", "var", "proprio"):
        add(f"lambda_{term}", "wm", f"loss.{term}", lambda v: f"{v:g}")
    add("train_steps", "roll", "checkpoints.wm.summary.step", intc)
    add("batch", "wm", "train.batch_size")
    add("train_wm_min", "roll", "checkpoints.wm.summary.wall_clock_s", minutes)

    # --- planner ---
    for k, path in (("N", "candidates"), ("H_cem", "horizon"), ("rounds", "rounds"), ("K", "elites"),
                    ("sigma0", "sigma_init"), ("dclip", "delta_clip")):
        add(k, "final", f"cem.{path}", lambda v: f"{v:g}")
    add("beta", "final", "cem.beta", lambda v: f"{v:g}")
    add("budget", "final", "budget")
    add("seeds", "final", "seeds", lambda v: ", ".join(map(str, v)))
    add("n_seeds", "final", "seeds", len)
    add("pixel_seeds", "final", "pixel_seeds", lambda v: ", ".join(map(str, v)))
    add("seed0", "final", "seeds[0]")
    add("pos_pairs", "final", "pooled.cem_wm.pos_success.n_pairs")
    add("bin_edge1", "final", "config.bins.move_px[1]")
    add("pmix_gate_angle", "roll", "checkpoints.probe_mix.summary.accept_thresholds.real_angle_deg", lambda v: f"{v:g}")
    add("sw10_n", "final", "sweep.offsets[0].n")
    add("n_test_pairs", "final", "pairs.n")
    add("n_val_pairs", "pv", "pairs.n")
    add("n_turning", "final", "pairs.rotation_pairs")
    add("n_nonturning", "final", "pairs.translation_pairs")
    add("goal_offset", "pairs", "offset")
    add("min_move", "pairs", "min_move_px", lambda v: f"{v:g}")
    add("min_turn", "pairs", "min_turn_deg", lambda v: f"{v:g}")
    add("tau_wm", "final", "tau.wm.tau", lambda v: f"{v:.2f}")
    add("n_tests", "final", "primary_tests", len)
    add("alpha", "final", "alpha", lambda v: f"{v:g}")
    add("boot", "final", "bootstrap.resamples", intc)

    # --- final results ---
    for m in ("cem_wm", "cem_wm_1step", "cem_pixel", "bc_clean", "random", "noop"):
        for metric in ("success", "pos_success"):
            for stat in ("mean", "lo", "hi"):
                add(f"{m}_{metric}_{stat}", "final", f"pooled.{m}.{metric}.{stat}", pct)
        add(f"{m}_spe", "final", f"pooled.{m}.seconds_per_episode", lambda v: f"{v:.2f}")
        add(f"{m}_pos_reached_ang", "final", f"pooled.{m}.pos_reached_median_final_ang_deg", f1)
        add(f"{m}_tag_position_only", "final", f"pooled.{m}.tags.position_only")
    for m in ("cem_wm", "cem_wm_1step", "cem_pixel"):
        add(f"{m}_call", "final", f"timing.{m}.call_s", lambda v: f"{v:.3f}")
        add(f"{m}_believed_failed", "final", f"imagined.{m}.believed_success_failed")
        n, _ = r.get("final", f"imagined.{m}.n")
        succ, src = r.get("final", f"pooled.{m}.tags.success")
        derived(f"{m}_failed", str(n - succ), f"imagined.{m}.n − {src.split(': ')[1]}")
    bf, _ = r.get("final", "imagined.cem_wm.believed_success_failed")
    failed = int(F["cem_wm_failed"].value)
    derived("cem_wm_model_error_pct", f"{100 * bf / failed:.0f}",
            "imagined.cem_wm.believed_success_failed / (imagined.cem_wm.n − pooled.cem_wm.tags.success)")
    cw, _ = r.get("final", "timing.cem_wm.call_s")
    cp, _ = r.get("final", "timing.cem_pixel.call_s")
    derived("speedup", f"{cp / cw:.0f}", "timing.cem_pixel.call_s / timing.cem_wm.call_s (final_eval.json)")
    for i, t in enumerate(r.get("final", "primary_tests")[0]):
        tag = f"pt_{t['b']}_{'pos' if t['metric'] == 'pos_success' else 'succ'}"
        base = f"primary_tests[{i}]"
        add(f"{tag}_diff", "final", f"{base}.diff_bootstrap.mean", pp)
        add(f"{tag}_lo", "final", f"{base}.diff_bootstrap.lo", pp)
        add(f"{tag}_hi", "final", f"{base}.diff_bootstrap.hi", pp)
        add(f"{tag}_holm", "final", f"{base}.p_holm", pval)
        add(f"{tag}_claim", "final", f"{base}.claim")

    # --- imagined angle (search reader, k = 1 / 5 / 10) ---
    ks = r.get("angle", "ks")[0]
    for row in ("real", "wm_recon", "wm", "wm_1step", "pixel", "copy_last", "copy_last_exact"):
        for j, k in enumerate(ks):
            add(f"ang_{row}_k{k}", "angle", f"metrics.angle_search.{row}.mean[{j}]", f1)
    jk = ks.index(10)
    for d in ("wm_minus_wm_recon", "wm_minus_copy_last", "wm_minus_copy_last_exact", "wm_minus_wm_1step",
              "wm_minus_pixel"):
        for stat in ("mean", "lo", "hi"):
            add(f"angd_{d}_{stat}", "angle", f"paired.search/{d}.{stat}[{jk}]", signed)
    add("ang_verdict", "angle", "verdict.verdict")
    add("ang_step", "angle", "step_deg", lambda v: f"{v:g}")
    add("ang_windows", "angle", "windows.n")

    # --- rollouts (Phase 5 / 5b) ---
    for stat in ("mean", "lo", "hi"):
        add(f"nmse_red_1step_{stat}", "roll", f"paired.nmse_reduction_wm_via_frames_vs_wm_1step.{stat}[{K10}]", pct)
        add(f"nmse_red_pixel_{stat}", "roll", f"paired.nmse_reduction_wm_via_frames_vs_pixel.{stat}[{K10}]", pct)
        add(f"blockd_pixel_{stat}", "roll", f"paired.latent/block_px_diff_wm_minus_pixel.{stat}[{K10}]", signed)
    for m in ("wm", "wm_1step", "copy_last"):
        add(f"phat_{m}", "roll", f"metrics.p_hat_pusher_px.{m}.mean[{K10}]", f1)
    for m in ("wm", "pixel", "wm_1step", "copy_last", "real"):
        add(f"lat_block_{m}", "roll", f"metrics.latent/block_px.{m}.mean[{K10}]", f1)
    add("latent_judge_block", "roll", "judges.latent.test_all.block_px_mean", f1)
    add("latent_judge_angle", "roll", "angle_in_wm_latent.angle_deg_mean", f1)
    add("latent_judge_verdict", "roll", "angle_in_wm_latent.verdict")
    add("rollout_windows", "roll", "windows.n")

    # --- judges / angle audit ---
    for split in ("train", "val", "test"):
        add(f"v1_angle_{split}", "aa", f"probe_v1.{split}.block_deg_mean", f1)
    add("v1_block_test", "aa", "probe_v1.test.block_px_mean", f1)
    add("oracle_real", "aa", "template_oracle.real.angle_deg_mean", f1)
    add("oracle_wm_recon", "aa", "template_oracle.wm_recon.angle_deg_mean", f1)
    add("rerender_diff", "aa", "rerender.mean_abs_diff", lambda v: f"{v:.2f}")
    pm = "checkpoints.probe_mix.summary"
    add("pmix_angle", "roll", f"{pm}.test.real.block_deg_mean", f1)
    add("pmix_block", "roll", f"{pm}.test.real.block_px_mean", f1)
    add("pmix_wmrecon_pusher", "roll", f"{pm}.test.wm_recon.pusher_px_mean", f1)

    # --- what went wrong: collapse, pusher, planning on val, cost ablation ---
    add("col_base_zstd", "col", "runs.base.z_std", lambda v: f"{v:.4f}")
    add("col_b_zstd", "col", "runs.b.z_std", lambda v: f"{v:.4f}")
    add("col_steps", "col", "train.steps")
    add("abl_v1_pusher", "abl", "runs.v1.rows[0].ridge_real_val.pusher_px", f1)
    add("abl_ab_phat", "abl", "runs.AB.rows[0].p_hat_pusher_px_k10", f1)
    add("abl_ab_phat_copy", "abl", "runs.AB.rows[0].copy_last_pusher_px_k10", f1)
    add("abl_steps", "abl", "runs.v1.rows[0].steps", intc)
    add("pv_cem_wm", "pv", "summaries.cem_wm.success.k")
    add("pv_cem_wm_pos", "pv", "summaries.cem_wm.pos_success.k")
    add("pv_bc", "pv", "summaries.bc.success.k")
    add("pv_bc_clean", "pv", "summaries.bc_clean.success.k")
    add("pv_imag", "pv", "imagined_success.cem_wm.imagined_success")
    add("pv_imag_fail", "pv", "imagined_success.cem_wm.imagined_success_real_fail")
    add("ps_default", "ps", "summaries.default.success.k")
    best = max(v["success"]["k"] for v in r.get("ps", "summaries")[0].values())
    derived("ps_best", str(best), "max over plan_sweep.json summaries.*.success.k")
    add("sweep_margin", "ps", "selection.rule", lambda v: re.search(r">= (\d+) more", v).group(1))
    add("pca_latent", "pca", "summaries.cem_wm.success.k")
    add("pca_pixel", "pca", "summaries.cem_wm_pixelcost.success.k")
    add("pca_latent_pos", "pca", "summaries.cem_wm.pos_success.k")
    add("pca_pixel_pos", "pca", "summaries.cem_wm_pixelcost.pos_success.k")
    add("pca_pixel_imag", "pca", "imagined_success.cem_wm_pixelcost.imagined_success")
    add("pca_pixel_imag_fail", "pca", "imagined_success.cem_wm_pixelcost.imagined_success_real_fail")
    add("pca_agree", "pca", "reproduction.agree")
    add("pca_spe", "pca", "summaries.cem_wm_pixelcost.seconds_per_episode", lambda v: f"{v:.2f}")

    # --- exploratory: bins + sweep ---
    mv = r.get("final", "difficulty.move_px")[0]
    for m in ("cem_wm", "bc_clean", "cem_pixel"):
        for j, lab in ((0, "short"), (len(mv["labels"]) - 1, "long")):
            add(f"bin_{m}_{lab}", "final", f"difficulty.move_px.methods.{m}[{j}].success.mean", pct)
    add("bin_short_label", "final", "difficulty.move_px.labels[0]")
    add("bin_long_label", "final", f"difficulty.move_px.labels[{len(mv['labels']) - 1}]")
    for i, o in enumerate(r.get("final", "sweep.offsets")[0]):
        off = o["offset"]
        add(f"sw{off}_offset", "final", f"sweep.offsets[{i}].offset")
        add(f"sw{off}_move", "final", f"sweep.offsets[{i}].median_move_px", f1)
        for m in ("cem_wm", "bc_clean"):
            add(f"sw{off}_{m}", "final", f"sweep.offsets[{i}].summaries.{m}.success.k")
    return F


# ---------- tables ----------
@dataclass(frozen=True)
class Table:
    markdown: str
    source: str


def _ci3(s: dict[str, list[float]], j: int, fmt: Callable[[float], str]) -> str:
    return f"{fmt(s['mean'][j])} [{fmt(s['lo'][j])}, {fmt(s['hi'][j])}]"


def tables(r: Results) -> dict[str, Table]:
    """Whole tables. The final-eval ones reuse summary.md's formatters, so they match it exactly."""
    f, a, roll = r.data["final"], r.data["angle"], r.data["roll"]
    T: dict[str, Table] = {}
    T["main"] = Table("\n".join(fs.main_table(f)),
                      f"{FILES['final']}: pooled, per_seed[*][0], imagined, timing (= summary.md Main table)")
    T["primary"] = Table("\n".join(fs.primary_table(f)), f"{FILES['final']}: primary_tests (= summary.md)")
    T["tags"] = Table("\n".join(fs.tags_table(f)), f"{FILES['final']}: pooled.*.tags, imagined")
    T["bins_move"] = Table("\n".join(fs.bins_table(f["difficulty"]["move_px"])), f"{FILES['final']}: difficulty.move_px")
    T["bins_turn"] = Table("\n".join(fs.bins_table(f["difficulty"]["turn_deg"])), f"{FILES['final']}: difficulty.turn_deg")
    T["sweep"] = Table("\n".join(fs.sweep_section(f)), f"{FILES['final']}: sweep.offsets")

    ks = a["ks"]
    rows = []
    for row in list(a["rows"]) + ["copy_last_exact"]:
        s = a["metrics"]["angle_search"][row]
        rows.append([row] + [_ci3(s, j, f1) for j in range(len(ks))])
    T["angle"] = Table("\n".join(fs._table(["row"] + [f"k = {k}" for k in ks], rows)),
                       f"{FILES['angle']}: metrics.angle_search.<row>.{{mean,lo,hi}}")
    rows = [[name.split("/", 1)[1].replace("_", " ")] + [_ci3(s, j, signed) for j in range(len(ks))]
            for name, s in a["paired"].items() if name.startswith("search/")]
    T["angle_paired"] = Table("\n".join(fs._table(["difference (°)"] + [f"k = {k}" for k in ks], rows)),
                              f"{FILES['angle']}: paired.search/*")

    sel = [roll["ks"].index(k) for k in (1, 5, 10)]
    nm = roll["metrics"]["nmse"]
    lat = roll["metrics"]["latent/block_px"]
    ph = roll["metrics"]["p_hat_pusher_px"]
    labels = {"wm_via_frames": "wm (E(D(ẑ)), headline)", "wm": "wm (direct ẑ)", "wm_1step": "wm_1step",
              "pixel": "pixel U-Net", "copy_last": "copy_last", "real": "real frames (floor)",
              "wm_recon": "wm_recon (floor)"}
    rows = []
    for m in ("wm_via_frames", "wm", "wm_1step", "pixel", "copy_last"):
        rows.append([labels[m]]
                    + [_ci3(nm[m], j, lambda x: f"{x:.3f}") for j in sel]
                    + [_ci3(lat[m], sel[-1], f1) if m in lat else "–"]
                    + [_ci3(ph[m], sel[-1], f1) if m in ph else "–"])
    rows.append([labels["real"], "–", "–", "–", _ci3(lat["real"], sel[-1], f1), "–"])
    T["rollouts"] = Table("\n".join(fs._table(
        ["method", "NMSE k = 1", "NMSE k = 5", "NMSE k = 10", "block px k = 10 (latent judge)",
         "pusher px k = 10 (p̂)"], rows)),
        f"{FILES['roll']}: metrics.nmse, metrics.latent/block_px, metrics.p_hat_pusher_px")

    T["speed"] = Table("\n".join(fs._table(
        ["method", "s / episode (mean)", "s / full-size planning call"],
        [[m, f"{f['pooled'][m]['seconds_per_episode']:.2f}",
          f"{f['timing'][m]['call_s']:.3f}" if m in f["timing"] else "–"] for m in f["methods"]])),
        f"{FILES['final']}: pooled.*.seconds_per_episode, timing.*.call_s")

    jobs = [  # (job, file key, key path)
        ("train_wm", "roll", "checkpoints.wm.summary.wall_clock_s"),
        ("train_wm_1step", "roll", "checkpoints.wm_1step.summary.wall_clock_s"),
        ("train_pixel", "roll", "checkpoints.pixel.summary.wall_clock_s"),
        ("train_bc", "pv", "checkpoints.bc.summary.wall_clock_s"),
        ("train_bc_clean", "pv", "checkpoints.bc_clean.summary.wall_clock_s"),
        ("train_probe", "aa", "checkpoints.probe.summary.wall_clock_s"),
        ("wm_ablate", "abl", "wall_clock_s"),
        ("angle_audit", "aa", "wall_clock_s"),
        ("train_probe_mix", "roll", "checkpoints.probe_mix.summary.wall_clock_s"),
        ("train_latent_probe", "roll", "checkpoints.latent.summary.wall_clock_s"),
        ("eval_rollouts", "roll", "wall_clock_s"),
        ("plan_val", "pv", "wall_clock_s"),
        ("plan_sweep", "ps", "wall_clock_s"),
        ("plan_cost_ablation", "pca", "wall_clock_s"),
        ("eval_imagined_angle", "angle", "wall_clock_s"),
        ("final_eval", "final", "wall_clock_s"),
    ]
    rows = []
    for job, key, path in jobs:
        v, src = r.get(key, path)
        rows.append([f"`{job}`", minutes(v), f"`{src}`"])
    for job in ("env_check", "collect_data", "diagnose_wm", "render_judge_data", "make_pairs"):
        rows.append([f"`{job}`", "not recorded", "no results JSON"])
    T["runtimes"] = Table("\n".join(fs._table(["job", "T4 minutes", "source"], rows)),
                          "wall_clock_s of each results JSON / checkpoint summary (source column)")
    return T


# ---------- rendering + checks ----------
PLACEHOLDER = re.compile(r"\{\{([ft]):([A-Za-z0-9_]+)\}\}")


def render(template: str, F: dict[str, Fact], T: dict[str, Table]) -> tuple[str, set[str], list[str]]:
    """(rendered text, used placeholder names, unresolved placeholders)."""
    used: set[str] = set()
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        kind, name = m.group(1), m.group(2)
        pool = F if kind == "f" else T
        if name not in pool:
            missing.append(f"{kind}:{name}")
            return m.group(0)
        used.add(f"{kind}:{name}")
        return pool[name].value if kind == "f" else pool[name].markdown

    out = PLACEHOLDER.sub(sub, template)
    header = ("<!-- Generated by `python run.py --job make_report` from docs/README.template.md and the "
              "committed results JSONs. Edit the template, not this file. -->\n")
    return header + out, used, missing


def untraced_numbers(template: str) -> list[str]:
    """Digit-bearing tokens in the template's static text that no source backs (should be empty)."""
    text = template.split("\n## References", 1)[0]
    for pat in (r"```.*?```", r"`[^`]*`", r"<!--.*?-->", r"\]\([^)]*\)", r"\{\{[ft]:[A-Za-z0-9_]+\}\}",
                r"k = (?:1|5|10)\b", r"^#+ .*$", r"Phase \d+[ab]?", r"[\w-]*1step[\w-]*", r"\b\w*v1\b"):
        text = re.sub(pat, " ", text, flags=re.S | re.M)
    bad = []
    for tok in re.findall(r"\S*\d\S*", text):
        core = tok.strip(".,;:()[]*_\"'")
        if core not in ALLOWED_TOKENS:
            bad.append(tok)
    return bad


# Resume claims (Aditya, Phase 8): each must appear in the README as rendered facts.
RESUME_CLAIMS: list[tuple[str, list[str]]] = [
    ("4.2M parameters (CNN encoder + GRU dynamics + decoder)", ["params_m", "params_total"]),
    ("~40 min training on a T4", ["train_wm_min"]),
    ("20,000-transition PushT dataset (200 episodes × 100 actions)", ["transitions", "episodes", "steps_per_ep"]),
    ("pusher tracking over 10-step imagined rollouts: 2.4 px (wm) vs 11 px (wm_1step)", ["phat_wm", "phat_wm_1step"]),
    ("10-step latent prediction error 60% lower than teacher forcing (95% CI 51–68%)",
     ["nmse_red_1step_mean", "nmse_red_1step_lo", "nmse_red_1step_hi"]),
    ("CEM success +9.5 pp and position-only +21.7 pp vs wm_1step; 200 held-out goals × 3 seeds; Holm-corrected McNemar",
     ["pt_cem_wm_1step_succ_diff", "pt_cem_wm_1step_pos_diff", "n_test_pairs", "n_seeds",
      "pt_cem_wm_1step_succ_holm", "pt_cem_wm_1step_pos_holm"]),
    ("latent CEM 20× faster per planning call than pixel-space CEM", ["speedup", "cem_wm_call", "cem_pixel_call"]),
    ("BC 67.5% vs cem_wm 34.7%", ["bc_clean_success_mean", "cem_wm_success_mean"]),
    ("imagined orientation error 60° vs 5.7° reconstruction at k = 10", ["ang_wm_k10", "ang_wm_recon_k10"]),
    ("92% of cem_wm failures are model errors (361/392)",
     ["cem_wm_model_error_pct", "cem_wm_believed_failed", "cem_wm_failed"]),
]


def sources_markdown(F: dict[str, Fact], T: dict[str, Table], used: set[str]) -> str:
    """docs/readme_sources.md: resume claims, then every rendered fact and table with its source."""
    L = ["# README number sources", "",
         "Generated by `make_report` together with README.md. Every number in README.md is one of the "
         "facts or tables below, formatted from the file and key shown.", "", "## Resume claims", "",
         "| resume claim | README value(s) | fact | source |", "| --- | --- | --- | --- |"]
    for claim, names in RESUME_CLAIMS:
        for i, n in enumerate(names):
            L.append(f"| {claim if i == 0 else ''} | {F[n].value} | `{n}` | `{F[n].source}` |")
    L += ["", "## Facts used in README.md", "", "| fact | value | source |", "| --- | --- | --- |"]
    for n in sorted(k.split(":", 1)[1] for k in used if k.startswith("f:")):
        L.append(f"| `{n}` | {F[n].value} | `{F[n].source}` |")
    L += ["", "## Tables used in README.md", "", "| table | source |", "| --- | --- |"]
    for n in sorted(k.split(":", 1)[1] for k in used if k.startswith("t:")):
        L.append(f"| `{n}` | {T[n].source} |")
    return "\n".join(L) + "\n"


def build(root: str | Path = ".") -> dict[str, Any]:
    """Render README.md and docs/readme_sources.md text (nothing is written)."""
    root = Path(root)
    r = Results(root)
    F, T = facts(r), tables(r)
    template = (root / TEMPLATE).read_text()
    readme, used, missing = render(template, F, T)
    claims_missing = [n for _, names in RESUME_CLAIMS for n in names
                      if f"f:{n}" not in used]
    return {"readme": readme, "sources": sources_markdown(F, T, used), "used": used, "missing": missing,
            "untraced": untraced_numbers(template), "claims_missing": claims_missing,
            "figures_missing": [p for p in FIGURES if not (root / p).exists()], "facts": F, "tables": T}

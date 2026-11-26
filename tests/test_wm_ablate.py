import json

import pytest

from wm.jobs import wm_ablate as job
from wm.repro import REPO_ROOT, load_yaml

ABL = load_yaml(REPO_ROOT / "configs/wm_ablate.yaml")
BASE = load_yaml(REPO_ROOT / "configs/wm.yaml")


def test_variant_config_merge():
    v1, a, ab = (job.variant_config(ABL, BASE, n) for n in ("v1", "A", "AB"))
    assert job.flags_of(v1) == {"fg_weight": False, "proprio": False, "keep_best": False}
    assert job.flags_of(a) == {"fg_weight": True, "proprio": False, "keep_best": False}
    assert job.flags_of(ab) == {"fg_weight": True, "proprio": True, "keep_best": True}
    assert a["fg_alpha"] == 20.0 and a["fg_threshold"] == pytest.approx(25 / 255)
    assert ab["loss"]["proprio"] == 1.0 and "proprio" not in v1["loss"]
    for cfg in (v1, a, ab):
        assert cfg["train"]["steps"] == 5000 and cfg["train"]["val_every"] == 250
        assert cfg["train"]["batch_size"] == 64 and cfg["seed"] == BASE["seed"]
        assert cfg["smoke"]["local"]["steps"] == 50
    # v1 is the base model + the ablation schedule only
    assert {k: v for k, v in v1.items() if k not in ("train", "smoke")} == \
           {k: v for k, v in BASE.items() if k not in ("train", "smoke")}
    assert "fg_weight" not in BASE                      # base config untouched


def row(label, block, block_copy, ang, ang_copy, pusher_ridge=50.0, p_hat=None):
    return {"row": label, "imagined_k10": {"block_px": block, "angle_deg": ang, "pusher_px": 60.0},
            "copy_last_k10": {"block_px": block_copy, "angle_deg": ang_copy, "pusher_px": 70.0},
            "ridge_real_val": {"block_px": 5.0, "angle_deg": 5.0, "pusher_px": pusher_ridge},
            "p_hat_pusher_px_k10": p_hat}


def test_select_winner_rule():
    rows = [row("v1", 29.5, 29.8, 25, 20),             # (a) only: angle worse than copy-last
            row("A", 25.0, 30.0, 15, 20),               # (a) and (b)
            row("A+B", 22.0, 30.0, 18, 20, p_hat=12.0)]  # (a) and (b), lowest block px
    assert job.select_winner(rows)["winner"] == "A+B"
    # tie within 1 px -> lower pusher error (p_hat for B rows, ridge pusher otherwise)
    rows[1] = row("A", 22.5, 30.0, 15, 20, pusher_ridge=10.0)
    sel = job.select_winner(rows)
    assert sel["winner"] == "A" and "tie" in sel["note"]
    # nobody satisfies (b): fall back to (a) and say so
    only_a = [row("v1", 29.0, 30.0, 25, 20), row("A", 27.0, 30.0, 30, 20)]
    sel = job.select_winner(only_a)
    assert sel["winner"] == "A" and "NO row satisfies both" in sel["note"]
    # nobody beats copy-last
    sel = job.select_winner([row("v1", 31.0, 30.0, 10, 20)])
    assert sel["winner"] is None


def test_table_line_formats_missing_p_hat():
    r = {**row("v1", 29.5, 29.8, 25, 20), "nmse_k5": 0.1, "copy_last_nmse_k5": 0.2, "nmse_k10": 0.3,
         "copy_last_nmse_k10": 0.4, "copy_last_pusher_px_k10": None, "recon_fg_mse_val": 0.01, "it_per_s": 9.5}
    line = job.table_line(r)
    assert "n/a (n/a)" in line and "29.5 (29.8)" in line and "25.0 (20.0)" in line


def test_smoke_runs_then_resume_skips_finished_variants(capsys, isolated_dirs, monkeypatch):
    """Smoke on tiny data; a rerun after an interrupted run trains only the missing variants."""
    import importlib.util

    import numpy as np

    from wm.data.collect import collect_split
    from wm.store import LocalStore

    spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    arrays, _ = collect_split([0, 1], data_cfg["expert"], steps=20)
    d = isolated_dirs / "npz"
    d.mkdir()
    np.savez_compressed(d / "pusht_train.npz", **arrays)
    LocalStore().push("pusht-data-smoke", d, {"complete": True, "config_hash": "x", "git": "t"})
    small = {"batch_size": 2, "steps": 4, "warmup_steps": 1, "val_every": 2, "log_every": 100, "val_windows": 2}
    real_load = job.load_yaml
    monkeypatch.setattr(run, "load_job_config", lambda entry: {**real_load(REPO_ROOT / entry["config"]),
                                                               "smoke": {"local": small, "remote": small}}
                        if entry.get("config") == "configs/wm_ablate.yaml" else
                        (real_load(REPO_ROOT / entry["config"]) if entry.get("config") else {}))

    # interrupt after the first variant
    calls = {"n": 0}
    orig_eval = job.evaluate

    def failing_eval(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated kill")
        return orig_eval(*a, **k)

    monkeypatch.setattr(job, "evaluate", failing_eval)
    assert run.main(["--job", "wm_ablate"]) == 1
    meta = json.loads((isolated_dirs / "store" / "results-wm-ablate-smoke" / "meta.json").read_text())
    assert meta["complete"] is False and meta["done"] == ["v1"]
    capsys.readouterr()

    monkeypatch.setattr(job, "evaluate", orig_eval)
    assert run.main(["--job", "wm_ablate"]) == 0
    out = capsys.readouterr().out
    assert "runs already finished: v1" in out and "[v1] flags" not in out and "[AB] flags" in out
    for label in ("v1", "A", "A+B", "A+B+C"):
        assert f" {label:<6} = " in out
    res = json.loads((isolated_dirs / "outputs" / "wm_ablate" / "artifact" / job.RESULTS_NAME).read_text())
    assert [r["row"] for r in res["table"]] == ["v1", "A", "A+B", "A+B+C"] and res["complete"]
    assert res["table"][3]["checkpoint"].startswith("best@") and res["table"][3]["best_step"] in (2, 4)
    assert res["table"][2]["p_hat_pusher_px_k10"] is not None and res["table"][0]["p_hat_pusher_px_k10"] is None
    assert str(REPO_ROOT / "results") not in out          # smoke never writes into results/

    assert run.main(["--job", "wm_ablate"]) == 0
    assert "already finished, skipping" in capsys.readouterr().out

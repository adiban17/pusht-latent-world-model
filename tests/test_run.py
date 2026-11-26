import importlib.util

import pytest

from wm.repro import REPO_ROOT

spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_registry_well_formed():
    reg = run.load_registry()
    assert set(reg["jobs"]) == {
        "env_check", "collect_data", "train_wm", "train_wm_1step", "train_pixel", "train_bc",
        "train_probe", "eval_rollouts", "plan_val", "plan_sweep", "final_eval", "make_report",
    }
    for name, entry in reg["jobs"].items():
        assert set(entry) == {"phase", "module", "config", "needs", "makes"}, name
        assert entry["module"] == f"wm.jobs.{name}"
    assert reg["groups"]["train_baselines"]["jobs"] == [
        "train_wm_1step", "train_pixel", "train_bc", "train_probe"]
    for member in reg["groups"]["train_baselines"]["jobs"]:
        assert member in reg["jobs"]


def test_full_refused_locally(capsys):
    assert run.main(["--job", "env_check", "--mode", "full"]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_unknown_job(capsys):
    assert run.main(["--job", "nope"]) == 2
    assert "unknown job" in capsys.readouterr().out


def test_unbuilt_job(capsys):
    assert run.main(["--job", "train_wm"]) == 2
    assert "not built yet (Phase 3)" in capsys.readouterr().out


def test_collect_data_smoke_then_skip(capsys, isolated_dirs):
    import json

    import numpy as np

    code = run.main(["--job", "collect_data"])
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and code == 0
    assert "train_transitions: 200" in out and "test_transitions: 100" in out
    job_dir = isolated_dirs / "outputs" / "collect_data"
    with np.load(job_dir / "pusht_train.npz") as d:
        assert d["frames"].shape == (202, 64, 64, 3)
        assert d["episode_seeds"].tolist() == [0, 1]
    assert len(list(job_dir.glob("filmstrip_*.png"))) == 3
    assert len(list(job_dir.glob("episode_*.gif"))) == 3
    meta = json.loads((isolated_dirs / "store" / "pusht-data-smoke" / "meta.json").read_text())
    assert meta["complete"] is True

    assert run.main(["--job", "collect_data"]) == 0
    assert "already finished, skipping" in capsys.readouterr().out


def test_env_check_runs_then_skips_then_force(capsys, isolated_dirs):
    assert run.main(["--job", "env_check", "--mode", "smoke"]) == 0
    out = capsys.readouterr().out
    assert "PASTE THIS TO CLAUDE" in out and "[FAIL]" not in out
    assert (isolated_dirs / "outputs" / "env_check" / "frame.png").exists()
    assert (isolated_dirs / "outputs" / "env_check" / "REPORT.md").exists()

    assert run.main(["--job", "env_check"]) == 0
    assert "already finished, skipping" in capsys.readouterr().out

    assert run.main(["--job", "env_check", "--force"]) == 0
    assert "PASTE THIS TO CLAUDE" in capsys.readouterr().out
